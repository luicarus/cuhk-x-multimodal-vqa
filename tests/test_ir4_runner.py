from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from test_cached_inputs import project  # small, self-contained CPU fixture
from cuhkx.config import InputError, load_config
from cuhkx.data.validate import check_inputs
from cuhkx.cli import main, prepare_run
from cuhkx.inference.prompt import allowed_answer_outputs, build_mcq_prompt, detect_prompt_leakage, parse_model_answer
from cuhkx.inference.qwen import token_constraint
from cuhkx.inference.runner import run_predictions


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, images, prompt, **kwargs):
        self.calls.append(([image.getpixel((200, 200))[0] for image in images], prompt, kwargs))
        value = next(self.outputs)
        if isinstance(value, BaseException):
            raise value
        return value

    def metadata(self):
        return {"backend": "CPU_SIMULATION", "model_loaded": False}


def setup_run(project):
    config = load_config(project)
    config["baseline"]["model"]["revision"] = "a" * 40
    source = {"model_id": config["baseline"]["model"]["id"], "revision": "a" * 40,
              "receipt_sha256": "fake-only", "method": "CPU_SIMULATION"}
    return config, source


def run(config, source, backend, **kwargs):
    return run_predictions(config, "test", kwargs.pop("limit", None), "unit", source,
                           lambda: backend, execution_mode=kwargs.pop("execution_mode", "simulation"), **kwargs)


def never_load():
    raise AssertionError("completed run must not load a backend")


def test_complete_resume_and_exact_images(project):
    config, source = setup_run(project)
    backend = FakeBackend(["A", "B"])
    result = run(config, source, backend)
    assert result["status"] == "PASS" and result["execution_mode"] == "simulation"
    assert len(backend.calls) == 2
    colors, prompt, args = backend.calls[0]
    assert all(abs(a-b) <= 2 for a,b in zip(colors, [30,90,150,210]))
    assert "absent/video.mp4" not in prompt and args["allowed_outputs"] == ("A", "B")
    assert args["max_new_tokens"] == 8
    resumed = run_predictions(config, "test", None, "unit", source, never_load,
                              resume=True, execution_mode="simulation")
    assert resumed == result


def test_run_records_per_request_latency(project):
    """Every executed request must contribute a timing sample.

    The numbers drive the infra comparison, so a run that silently reported no
    timing would make the bench table read as "not measured" with no clue why.
    """
    config, source = setup_run(project)
    backend = FakeBackend(["A", "B"])
    result = run(config, source, backend)

    latency = result["latency"]
    assert latency["requests"] == 2
    assert latency["warmup_skipped"] == 0
    for phase in ("total_ms", "image_ms", "generate_ms", "overhead_ms"):
        assert latency["latency_ms"][phase]["count"] == 2, phase
    assert latency["latency_ms"]["total_ms"]["mean_ms"] > 0
    # Phases must not exceed the request they came from.
    assert latency["latency_ms"]["total_ms"]["min_ms"] >= (
        latency["latency_ms"]["generate_ms"]["min_ms"])
    assert result["latency"]["steady_state_valid_rate"] == 1.0
    assert result["latency"]["steady_state_throughput_rps"] > 0


def test_latency_survives_a_failed_request(project):
    """A backend error must still produce a sample, or throughput reads high."""
    config, source = setup_run(project)
    backend = FakeBackend([RuntimeError("boom")])
    result = run(config, source, backend, fail_fast=True)

    assert result["status"] == "FAIL"
    assert result["latency"]["requests"] == 1
    assert result["latency"]["steady_state_valid_rate"] == 0.0


def test_interruption_and_stable_limit(project):
    config, source = setup_run(project)
    backend = FakeBackend(["A", KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        run(config, source, backend)
    checkpoint = project / "outputs/unit/checkpoint.jsonl"
    assert len(checkpoint.read_text().splitlines()) == 1
    restarted = FakeBackend(["B"])
    result = run(config, source, restarted, resume=True)
    assert result["status"] == "PASS" and result["resumed_valid"] == 1
    assert len(restarted.calls) == 1
    with pytest.raises(InputError, match="signature"):
        run(config, source, FakeBackend([]), limit=1, resume=True)


@pytest.mark.parametrize("value,status", [("not-an-answer", "invalid"), (RuntimeError("CUDA OOM"), "failed")])
def test_retry_failed_only(project, value, status):
    config, source = setup_run(project)
    first = run(config, source, FakeBackend(["A", value]))
    assert first["status"] == "FAIL" and first["counts"][status] == 1
    retry = FakeBackend(["B"])
    second = run(config, source, retry, resume=True)
    assert second["status"] == "PASS" and len(retry.calls) == 1
    records = [json.loads(line) for line in (project / "outputs/unit/checkpoint.jsonl").read_text().splitlines()]
    assert [r["attempts"] for r in records] == [1,2]


@pytest.mark.parametrize("filename", ["predictions.csv", "checkpoint.jsonl", "audit.jsonl"])
def test_completed_tamper_rejected(project, filename):
    config, source = setup_run(project)
    run(config, source, FakeBackend(["A", "B"]))
    path = project / "outputs/unit" / filename
    path.write_text(path.read_text().replace('"prediction": "A"', '"prediction": "B"').replace("q0,A", "q0,B") + "\n")
    with pytest.raises((InputError, ValueError)):
        run(config, source, FakeBackend([]), resume=True)


def test_false_pass_and_missing_state_rejected(project):
    config, source = setup_run(project)
    run(config, source, FakeBackend(["invalid"]))
    path = project / "outputs/unit/run_summary.json"
    summary = json.loads(path.read_text())
    summary["status"] = "PASS"
    path.write_text(json.dumps(summary))
    with pytest.raises(InputError, match="falsely"):
        run(config, source, FakeBackend([]), resume=True)
    (path.parent / "resume_state.json").unlink()
    with pytest.raises(InputError, match="no resume state"):
        run(config, source, FakeBackend([]), resume=True)


def test_changed_source_config_and_mode_rejected(project):
    config, source = setup_run(project)
    run(config, source, FakeBackend(["A", "B"]))
    with pytest.raises(InputError, match="signature"):
        run(config, {**source, "receipt_sha256": "changed"}, FakeBackend([]), resume=True)
    with pytest.raises(InputError, match="signature"):
        run(config, source, FakeBackend([]), resume=True, execution_mode="cloud")
    config["baseline"]["runtime"]["cpu_memory_gib"] = 12
    with pytest.raises(InputError, match="signature"):
        run(config, source, FakeBackend([]), resume=True)


def test_complete_run_portable_after_data_move(project, tmp_path):
    config, source = setup_run(project)
    run(config, source, FakeBackend(["A", "B"]))
    moved = tmp_path / "moved_data"
    shutil.move(str(project / "data"), moved)
    config["data_root"] = str(moved)
    assert run(config, source, FakeBackend([]), resume=True)["status"] == "PASS"


def test_prepared_check_can_transition_to_prediction(project):
    config, source = setup_run(project)
    prepare_run(config, check_inputs(config, "test", 1), "unit")
    assert run(config, source, FakeBackend(["A"]), limit=1)["status"] == "PASS"


def test_predict_unpinned_fails_without_gpu_import(project, capsys):
    assert main(["predict", "--project-root", str(project), "--dataset", "test", "--run-id", "x",
                 "--weights-dir", str(project / "absent")]) == 2
    assert "not pinned" in capsys.readouterr().err


def test_prompt_categories_and_parser():
    row = {"category": "sequence", "question": "Order?", "A": "first", "B": "second",
           "C": "", "D": "", "path": "Testing/private/file.mp4", "answer": "SECRET"}
    prompt = build_mcq_prompt(row)
    assert "complete permutation of AB" in prompt and "SECRET" not in prompt
    assert not detect_prompt_leakage(prompt, row)
    assert allowed_answer_outputs("sequence", "AB") == ("AB", "BA")
    assert allowed_answer_outputs("multi", "AB") == ("A", "B", "AB")
    for category in ("single", "combination", "object_interaction", "emotion"):
        assert allowed_answer_outputs(category, "AB") == ("A", "B")
    assert parse_model_answer("BA", category="sequence", valid_options="AB").prediction == "BA"
    assert parse_model_answer("BA", category="multi", valid_options="AB").prediction == "AB"
    assert not parse_model_answer("A", category="sequence", valid_options="AB").is_valid


class Tokens(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Tokens(result) if isinstance(key, slice) else result
    def tolist(self):
        return list(self)


def test_constraint_preserves_multi_answer_prefix_and_budget():
    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, **kwargs):
            return [ord(x) for x in text]
    fn = token_constraint(Tokenizer(), ["A", "AB", "B"], 1, 8)
    assert fn(0, Tokens([999])) == [65,66]
    assert fn(0, Tokens([999,65])) == [0,66]
    assert fn(0, Tokens([999,65,66])) == [0]
    with pytest.raises(InputError, match="illegal"):
        fn(0, Tokens([999,67]))
    with pytest.raises(InputError, match="budget"):
        token_constraint(Tokenizer(), ["AB"], 1, 2)


def test_resume_after_projection_write_interruption(project, monkeypatch):
    from cuhkx.inference import runner
    config, source = setup_run(project)
    original = runner.atomic_write

    def interrupted(path, payload):
        if path.name == "predictions.csv":
            raise OSError("simulated interrupted output write")
        original(path, payload)

    monkeypatch.setattr(runner, "atomic_write", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        run(config, source, FakeBackend(["A", "B"]))
    monkeypatch.setattr(runner, "atomic_write", original)
    resumed = run_predictions(config, "test", None, "unit", source, never_load,
                              resume=True, execution_mode="simulation")
    assert resumed["status"] == "PASS" and resumed["resumed_valid"] == 2


def test_prompt_change_invalidates_checkpoint(project, monkeypatch):
    from cuhkx.inference import runner
    config, source = setup_run(project)
    run(config, source, FakeBackend(["A", "B"]))
    original = runner.build_mcq_prompt
    monkeypatch.setattr(runner, "build_mcq_prompt", lambda row: original(row) + "\nchanged")
    with pytest.raises(InputError, match="signature"):
        run(config, source, FakeBackend([]), resume=True)


def test_cloud_generation_arguments_without_gpu(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from PIL import Image
    from cuhkx.inference.qwen import QwenBackend

    seen = {}

    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=SimpleNamespace(shape=(1, 2)))
            self.input_ids = [[111, 222]]
        def to(self, device):
            seen["device"] = device
            return self

    class Processor:
        tokenizer = Tokenizer()
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            return "rendered"
        def __call__(self, **kwargs):
            seen["processor"] = kwargs
            return Batch()
        def batch_decode(self, rows, **kwargs):
            assert rows == [[65, 0]]
            return ["A"]

    def vision(messages):
        content = messages[0]["content"]
        assert len(content) == 5 and content[-1]["text"] == "prompt"
        assert all(item["resized_height"] == item["resized_width"] == 280 for item in content[:4])
        return [item["image"] for item in content[:4]], None

    def generate(**kwargs):
        seen["generation"] = kwargs
        assert kwargs["prefix_allowed_tokens_fn"](0, Tokens([111, 222])) == [65, 66]
        return [[111, 222, 65, 0]]

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=vision))
    backend = QwenBackend.__new__(QwenBackend)
    backend.processor = Processor()
    backend.image_size = 280
    backend.device = "mock-cloud-device"
    backend.torch = SimpleNamespace(inference_mode=nullcontext)
    backend.model = SimpleNamespace(generate=generate)
    images = [Image.new("RGB", (448, 448)) for _ in range(4)]
    try:
        assert backend.generate(images, "prompt", allowed_outputs=("A", "B"), max_new_tokens=8) == "A"
        assert seen["generation"]["max_new_tokens"] == 8
        assert seen["generation"]["do_sample"] is False and seen["generation"]["num_beams"] == 1
        assert seen["processor"]["videos"] is None
        with pytest.raises(InputError, match="four"):
            backend.generate(images[:2], "prompt", allowed_outputs=("A", "B"), max_new_tokens=8)
    finally:
        for image in images:
            image.close()
