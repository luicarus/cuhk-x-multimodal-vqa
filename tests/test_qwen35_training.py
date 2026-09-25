from pathlib import Path
import re

import pytest
from PIL import Image

from cuhkx.config import load_qwen35_config
from cuhkx.training.dataset import load_training_config
from cuhkx.training.qwen35_support import qwen35_text_targets
from cuhkx.training.trainer import configure_qwen35_fp16_scaler, lora_gradient_diagnostics


PROJECT = Path(__file__).resolve().parents[1]


def test_qwen35_training_profile_reuses_complete_ir8_data_contract():
    config = load_qwen35_config()
    settings = load_training_config(config)
    assert settings["profile"] == "qwen35_4b_qlora_v1"
    assert settings["splits"] == {"train": [0, 1, 2], "dev": [3], "confirm": [4]}


def test_qwen35_targets_cover_hybrid_text_decoder_without_vision_modules():
    names = [
        "model.visual.blocks.0.attn.q_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ]
    selected = qwen35_text_targets(names)
    assert selected == [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.v_proj",
    ]


def test_training_optimizer_and_loader_are_config_driven_and_signed():
    """Both execution settings must come from config and reach the signature.

    The optimizer backend used to be hardcoded to adamw_torch inside the trainer
    while the contract was built only from the YAML, so switching optimizers
    would have changed how a run trained without changing its signature -- a run
    could then be resumed or compared against an adapter trained differently.
    """
    from cuhkx.training.dataset import OPTIMIZERS

    source = (PROJECT / "src/cuhkx/training/trainer.py").read_text(encoding="utf-8")
    # No hardcoded backend survives.
    assert 'optim="adamw_torch"' not in source
    assert 'optim=opt["optim"]' in source
    assert "dataloader_num_workers=opt[\"dataloader_num_workers\"]" in source
    assert "dataloader_num_workers=0" not in source
    # Both are restated in the contract, which is what the signature hashes.
    assert '"training_execution"' in source
    assert 'settings["optimizer"]["optim"]' in source
    assert 'settings["optimizer"]["dataloader_num_workers"]' in source

    # Paged 8-bit is the backend the measured QLoRA speedup comes from.
    assert "paged_adamw_8bit" in OPTIMIZERS
    assert "adamw_torch" in OPTIMIZERS


def test_training_configs_declare_the_execution_settings():
    """Both lanes share one validator, so both YAMLs must carry the new keys."""
    import yaml

    for name in ("training_qwen35.yaml", "training.yaml"):
        value = yaml.safe_load((PROJECT / "configs" / name).read_text(encoding="utf-8"))
        opt = value["optimizer"]
        assert opt["optim"] == "paged_adamw_8bit", name
        assert opt["dataloader_num_workers"] == 2, name


def test_training_validator_rejects_bad_execution_settings():
    from cuhkx.config import require
    from cuhkx.training.dataset import OPTIMIZERS

    with pytest.raises(ValueError, match="unsupported optimizer"):
        require("sgd" in OPTIMIZERS, "unsupported optimizer 'sgd'")
    # The worker bound is what the validator enforces.
    for value in (-1, 9, "2", 1.5):
        assert not (type(value) is int and 0 <= value <= 8), value


def test_training_profiler_times_steps_and_excludes_warmup():
    """Step timing, dataloader share and throughput come out of one run."""
    from cuhkx.training.profiling import TrainingProfiler

    profiler = TrainingProfiler(warmup=1, samples_per_step=2)
    profiler.start()
    for _ in range(4):
        profiler.begin_batch()
        profiler.begin_step()
        profiler.end_step(samples=2)
    summary = profiler.stop()

    assert summary["available"] is True
    assert summary["steps"] == 4
    assert summary["warmup_skipped"] == 1
    assert summary["step_ms"]["count"] == 4
    assert summary["steady_state_step_ms"]["count"] == 3
    assert summary["samples_per_step"] == 2
    assert summary["steady_state_samples_per_s"] > 0
    # Forward and backward are not separable under gradient checkpointing, and
    # the summary says so rather than implying a split it cannot measure.
    assert "checkpointing" in summary["note"]


def test_training_profiler_reports_nothing_rather_than_zero():
    """A run with no observed step must not look like a very fast one."""
    from cuhkx.training.profiling import TrainingProfiler

    summary = TrainingProfiler().start().stop()
    assert summary["available"] is False
    assert "reason" in summary
    assert "step_ms" not in summary


def test_training_profiler_degrades_when_telemetry_fails():
    """Telemetry is diagnostic: a broken monitor must not hide step timing."""
    import cuhkx.inference.profiling as inference_profiling
    from cuhkx.training.profiling import TrainingProfiler

    class Exploding:
        def __init__(self, **kwargs):
            raise RuntimeError("no NVML in this environment")

    original = inference_profiling.GpuMemoryPeakMonitor
    inference_profiling.GpuMemoryPeakMonitor = Exploding
    try:
        profiler = TrainingProfiler()
        profiler.start()
        profiler.begin_step()
        profiler.end_step(samples=1)
        summary = profiler.stop()
    finally:
        inference_profiling.GpuMemoryPeakMonitor = original

    assert profiler.monitor is None
    assert "no NVML" in profiler.error
    # The step timing still landed.
    assert summary["available"] is True
    assert summary["step_ms"]["count"] == 1


def test_training_profiling_is_not_part_of_the_verified_contract():
    """Adding a metric must never invalidate an already finished run."""
    source = (PROJECT / "src/cuhkx/training/trainer.py").read_text(encoding="utf-8")
    assert '"training_profile":training_profile' in source
    # It is written into result.json only, never into the signed contract.
    contract_block = source.split("contract = {", 1)[1].split("signature = fingerprint", 1)[0]
    assert "training_profile" not in contract_block


def test_training_profiler_is_packaged_for_both_lanes():
    """Both releases ship the trainer, so both must ship its profiler."""
    for script in ("scripts/package_qwen35_training.py", "scripts/package_training.py"):
        source = (PROJECT / script).read_text(encoding="utf-8")
        assert "src/cuhkx/training/profiling.py" in source, script
        # The profiler imports the NVML sampler from the inference module.
        assert "src/cuhkx/inference/profiling.py" in source, script


def test_qwen35_train_lock_accepts_linux_markupsafe_wheel():
    lock = (PROJECT / "requirements/train_qwen35.lock.txt").read_text(encoding="utf-8")
    assert "--index-url https://pypi.org/simple" in lock
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in lock
    assert "peft==0.18.0" in lock
    # vLLM 0.19.1 pins the torch/torchvision pair.
    assert "torchvision==0.25.0+cu128" in lock
    assert "torch==2.10.0+cu128" in lock
    assert "vllm==0.19.1" in lock
    assert "-r qwen35.in" in (PROJECT / "requirements/train_qwen35.in").read_text(encoding="utf-8")
    start = lock.index("markupsafe==3.0.3")
    block = lock[start:lock.index("\nmdurl==", start)]
    assert "0bf2a864d67e76e5c9a34dc26ec616a66b9888e25e7b9460e1c76d3293bd9dbf" in block


def test_qwen35_inference_lock_pairs_vllm_with_transformers_five():
    lock = (PROJECT / "requirements/qwen35.lock.txt").read_text(encoding="utf-8")
    for pin in ("vllm==0.19.1", "torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
                "transformers==5.17.0"):
        assert pin in lock, pin
    # The structured-output backends are what enforce the closed answer space.
    assert "xgrammar==" in lock
    # This lane must never drift into the 7B lane's Transformers 4.x toolchain.
    baseline = (PROJECT / "requirements/cloud.lock.txt").read_text(encoding="utf-8")
    assert "vllm==" not in baseline
    # The 7B lane keeps its own cu126 toolchain and must never be swept along.
    assert "torch==2.7.1+cu126" in baseline
    assert "transformers==4.57.6" in baseline


def _pinned_packages(lock_text):
    return {match.group(1): match.group(2)
            for match in re.finditer(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s\\]+)", lock_text, re.M)}


@pytest.mark.parametrize("name", ["qwen35.lock.txt", "train_qwen35.lock.txt"])
def test_qwen35_locks_keep_one_cuda_major_version(name):
    """A lock must not mix CUDA 12 and CUDA 13 packages.

    The kernel extension shipped in the vLLM wheel hard-links one libcudart
    major version: 0.19.1 links libcudart.so.12, while 0.20.2 and later link
    libcudart.so.13. A CUDA 13 wheel on this CUDA 12.8 host fails at
    `import vllm` with "libcudart.so.13: cannot open shared object file".
    Resolving successfully is not evidence that the result is runnable, so the
    invariant is guarded directly rather than inferred from metadata.
    """
    pinned = _pinned_packages((PROJECT / "requirements" / name).read_text(encoding="utf-8"))
    cuda13 = sorted(package for package in pinned if "cu13" in package)
    cuda12 = sorted(package for package in pinned if package.endswith("-cu12"))
    assert not cuda13, f"{name} pulls CUDA 13 packages alongside a cu12 torch: {cuda13}"
    assert cuda12, f"{name} is missing the expected CUDA 12 runtime packages"
    assert "torch==2.10.0+cu128" in (PROJECT / "requirements" / name).read_text(encoding="utf-8")
    # The CUDA 13 cutlass stack is exactly what breaks the import.
    assert "nvidia-cutlass-dsl-libs-cu13" not in pinned


@pytest.mark.parametrize("name", ["qwen35.lock.txt", "train_qwen35.lock.txt"])
def test_qwen35_locks_pin_vllm_from_the_cuda12_binary_line(name):
    """Only vLLM releases whose wheel links libcudart.so.12 are usable here.

    The cutover was established by extracting DT_NEEDED from the published
    wheels: 0.19.1 links libcudart.so.12, 0.20.2 and 0.21.0 link
    libcudart.so.13. This host is CUDA 12.8, so anything from 0.20.0 up fails
    at import. Pinning the exact version keeps an innocuous-looking bump from
    silently reintroducing the CUDA 13 link.
    """
    pinned = _pinned_packages((PROJECT / "requirements" / name).read_text(encoding="utf-8"))
    assert pinned.get("vllm") == "0.19.1"
    # 0.19.1 pins this torch/torchvision pair.
    assert pinned.get("torch") == "2.10.0+cu128"
    assert pinned.get("torchvision") == "0.25.0+cu128"


def test_qwen35_training_notebook_keeps_cuda_package_index():
    import json

    notebook = json.loads((PROJECT / "notebooks/qwen35-4b-qlora-vllm.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "--index-url" in source
    assert "https://download.pytorch.org/whl/cu128" in source
    assert "import json, peft, transformers, vllm, torch" in source
    assert source.count('cloud("verify-run", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG)') == 3


def test_qwen35_notebook_runs_vllm_on_two_gpus():
    import json

    notebook = json.loads((PROJECT / "notebooks/qwen35-4b-qlora-vllm.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    # Using both GPUs is the point of this lane; the notebook selects the topology
    # through PARALLEL_MODE so the tensor- and data-parallel rounds are both
    # reproducible from one file.
    assert 'PARALLEL_MODE = "dp"' in source
    assert 'DATA_PARALLEL = 2 if PARALLEL_MODE == "dp" else 1' in source
    assert 'TENSOR_PARALLEL = 1 if PARALLEL_MODE == "dp" else 2' in source
    assert '"--backend", "vllm"' in source
    assert '"--tensor-parallel-size", str(TENSOR_PARALLEL)' in source
    assert '"--data-parallel-size", str(DATA_PARALLEL)' in source
    # vLLM cannot train, so training must stay on the Transformers trainer.
    training_calls = [line for line in source.splitlines() if 'cloud("train"' in line]
    assert training_calls and all("vllm" not in line for line in training_calls)
    # Both GPUs must be visible before vLLM starts.
    assert "device_count" in source
    # The smoke gate must prove the topology it actually ran, not just that vLLM
    # was used: a silent fallback to one GPU would otherwise pass.
    assert "data_parallel_size" in source


def test_qwen35_notebooks_gate_on_telemetry_and_prefix_cache():
    """Worker telemetry must be present; a missing cache counter only warns.

    Worker memory was silently empty in an earlier round because the RPC could
    not serialize its function, so that one is a hard gate: an empty report means
    the telemetry is broken. The prefix-cache counter is different -- it depends
    on the engine build exposing the field at all, and aborting a 682-request run
    over a diagnostic would be worse than the missing number. It warns and dumps
    the real field inventory instead.
    """
    import json

    for name in ("qwen35-4b-vllm.ipynb", "qwen35-4b-qlora-vllm.ipynb"):
        notebook = json.loads((PROJECT / "notebooks" / name).read_text(encoding="utf-8"))
        source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert 'workers = backend_metadata.get("workers") or {}' in source, name
        assert 'if not workers.get("workers")' in source, name
        assert 'prefix = backend_metadata.get("prefix_cache") or {}' in source, name
        # Missing counters warn rather than raise, and the run records what the
        # engine actually reported.
        assert 'if prefix.get("reported_requests"):' in source, name
        assert 'metrics_shape' in source, name


def test_qwen35_vllm_backend_keeps_the_constrained_answer_space():
    source = (PROJECT / "src/cuhkx/inference/qwen35_vllm.py").read_text(encoding="utf-8")
    # The reference backend constrains decoding with a prefix automaton; vLLM
    # has no such hook and must constrain the same language via structured output.
    assert "StructuredOutputsParams" in source
    assert "choice=" in source
    assert "enable_thinking" in source
    assert "tensor_parallel_size" in source
    # Two T4s have no NVLink, so eager mode and no custom all-reduce are required.
    assert "enforce_eager=True" in source
    assert "disable_custom_all_reduce=True" in source
    assert "NCCL_P2P_DISABLE" in source


def test_qwen35_vllm_passes_literal_answers_to_choice():
    """`choice` takes answer strings, not regexes.

    vLLM builds the grammar from `choice` by escaping each element, so a regex
    such as r"\\s*B" becomes a literal to match: the model then answers with the
    characters "\\s*B" instead of "B", and every prediction is rejected as an
    invalid answer. Assert on the value actually handed to SamplingParams.
    """
    from cuhkx.inference.qwen35_vllm import Qwen35VLLMBackend

    captured = {}

    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            captured["structured_outputs"] = kwargs
            self.kwargs = kwargs

    class SamplingParams:
        def __init__(self, **kwargs):
            captured["sampling"] = kwargs

    class Engine:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat"] = kwargs
            return [type("R", (), {"outputs": [type("O", (), {"text": "B"})()]})()]

    backend = Qwen35VLLMBackend.__new__(Qwen35VLLMBackend)
    backend.SamplingParams = SamplingParams
    backend.StructuredOutputsParams = StructuredOutputsParams
    backend.engine = Engine()
    backend.adapter_request = None
    backend.image_size = 280
    backend.tensor_parallel_size = 2
    backend.attention_backend = "TRITON_ATTN"

    allowed = ("A", "B", "C", "D")
    images = [Image.new("RGB", (280, 280)) for _ in range(4)]
    try:
        assert backend.generate(images, "prompt", allowed_outputs=allowed,
                                max_new_tokens=8) == "B"
    finally:
        for image in images:
            image.close()
    assert captured["structured_outputs"]["choice"] == list(allowed)
    # No regex or grammar may be smuggled in through the choice path.
    for value in captured["structured_outputs"]["choice"]:
        assert "\\" not in value and "*" not in value and "|" not in value


def test_qwen35_vllm_never_builds_a_regex_answer_grammar():
    source = (PROJECT / "src/cuhkx/inference/qwen35_vllm.py").read_text(encoding="utf-8")
    assert "_choice_pattern" not in source
    assert "re.escape" not in source
    assert "regex=" not in source


def test_qwen35_vllm_defaults_to_triton_attention():
    """FlashInfer needs the driver's libcuda.so, which Kaggle does not ship.

    FlashInfer JIT-compiles SM 7.5 kernels through nvcc and links `-lcuda`; the
    container has no driver stubs, so the very first forward pass dies with
    "cannot find -lcuda". TRITON_ATTN needs neither nvcc nor a link step.
    """
    source = (PROJECT / "src/cuhkx/inference/qwen35_vllm.py").read_text(encoding="utf-8")
    assert 'attention_backend: str = "TRITON_ATTN"' in source
    assert 'attention_config={"backend": attention_backend}' in source
    # The alternative must stay reachable for hosts that do have driver stubs.
    assert '"FLASHINFER"' in source
    # The chosen backend is part of the run metadata, not just an internal default.
    assert '"attention_backend": self.attention_backend' in source
    assert "cannot find -lcuda" in source


def test_qwen35_vllm_enables_engine_stats_for_ttft():
    """LLM forces disable_log_stats=True unless the caller overrides it.

    The output processor then sets RequestStateStats to None, so the finished
    output carries metrics=None and first_token_latency is never produced. A run
    with this unset reported no TTFT at all, which is the symptom this guards.
    """
    source = (PROJECT / "src/cuhkx/inference/qwen35_vllm.py").read_text(encoding="utf-8")
    assert "disable_log_stats=False" in source
    # VLLM_NO_USAGE_STATS only silences the remote usage reporter.
    assert '"VLLM_NO_USAGE_STATS"' in source


def test_qwen35_vllm_samples_memory_inside_worker_processes():
    """Parent-process allocator counters are always zero under vLLM.

    The engine and each tensor-parallel rank run in separate processes, so
    torch.cuda.memory_allocated() read from the parent returns 0. An earlier
    version reported used_mib correctly but allocated_mib 0 for this reason.
    """
    source = (PROJECT / "src/cuhkx/inference/qwen35_vllm.py").read_text(encoding="utf-8")
    assert "collective_rpc" in source
    assert "def _worker_telemetry(worker)" in source
    # The worker function must read its own device, not an inherited one.
    assert "torch.cuda.current_device()" in source
    assert "available_kv_cache_memory_bytes" in source
    assert "num_gpu_blocks" in source


def test_qwen35_vllm_telemetry_failure_never_raises():
    """Telemetry is diagnostic; it must not be able to fail a prediction run."""
    from cuhkx.inference.qwen35_vllm import Qwen35VLLMBackend

    class BrokenEngine:
        def collective_rpc(self, method, *args, **kwargs):
            raise RuntimeError("worker died")

    backend = Qwen35VLLMBackend.__new__(Qwen35VLLMBackend)
    backend.engine = BrokenEngine()

    result = backend.worker_memory()
    assert result["workers"] == []
    assert "worker died" in result["error"]


def test_qwen35_vllm_worker_telemetry_returns_plain_types():
    """The result is pickled across a process boundary, so no tensors."""
    from cuhkx.inference.qwen35_vllm import _worker_telemetry

    class FakeCuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(index):
            return "Tesla T4"

        @staticmethod
        def mem_get_info(index):
            return (4 * 1024**3, 16 * 1024**3)

        @staticmethod
        def memory_allocated(index):
            return 1024**3

        @staticmethod
        def memory_reserved(index):
            return 2 * 1024**3

        @staticmethod
        def max_memory_allocated(index):
            return 3 * 1024**3

        @staticmethod
        def max_memory_reserved(index):
            return 4 * 1024**3

    class FakeTorch:
        cuda = FakeCuda()

    class CacheConfig:
        num_gpu_blocks = 1000
        block_size = 528

    class Worker:
        available_kv_cache_memory_bytes = 5 * 1024**3
        cache_config = CacheConfig()

    import sys
    original = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch()
    try:
        report = _worker_telemetry(Worker())
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original

    assert report["device_index"] == 0
    assert report["used_mib"] == 12288.0
    assert report["allocated_mib"] == 1024.0
    assert report["kv_cache_mib"] == 5120.0
    assert report["kv_cache_blocks"] == 1000
    assert report["kv_cache_tokens"] == 1000 * 528
    # Everything must be JSON-serializable.
    import json
    json.dumps(report)


def test_qwen35_vllm_worker_telemetry_omits_missing_kv_cache():
    """A worker that never finished profiling reports memory without KV cache."""
    from cuhkx.inference.qwen35_vllm import _worker_telemetry

    class FakeCuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(index):
            return "Tesla T4"

        @staticmethod
        def mem_get_info(index):
            return (1 * 1024**3, 16 * 1024**3)

        @staticmethod
        def memory_allocated(index):
            return 0

        @staticmethod
        def memory_reserved(index):
            return 0

        @staticmethod
        def max_memory_allocated(index):
            return 0

        @staticmethod
        def max_memory_reserved(index):
            return 0

    class FakeTorch:
        cuda = FakeCuda()

    class Bare:
        pass

    import sys
    original = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch()
    try:
        report = _worker_telemetry(Bare())
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original

    assert "kv_cache_mib" not in report
    assert "kv_cache_tokens" not in report


def test_qwen35_vllm_allows_worker_rpc_serialization():
    """Without the pickle fallback the worker RPC fails and telemetry is empty.

    vLLM's default msgpack serializer rejects callables, so collective_rpc on a
    function raised "Object of type <class 'function'> is not serializable" and
    every worker figure was lost. The env var is what makes the channel accept
    it; this test pins the setting so a later cleanup cannot drop it silently.
    """
    from cuhkx.inference.qwen35_vllm import VLLM_ENGINE_ENV

    assert VLLM_ENGINE_ENV.get("VLLM_ALLOW_INSECURE_SERIALIZATION") == "1"


def test_qwen35_vllm_rejects_tensor_and_data_parallel_together():
    """Two GPUs are either split inside one replica or divided between replicas."""
    from cuhkx.inference.qwen35_vllm import Qwen35VLLMBackend

    with pytest.raises(ValueError, match="cannot both exceed 1"):
        Qwen35VLLMBackend(
            {"frames": {"input_image_size": 280}, "model": {}},
            Path("/nonexistent"), tensor_parallel_size=2, data_parallel_size=2)


def test_qwen35_vllm_round_robin_covers_every_request_once():
    """Every request must land on exactly one replica, in input order."""
    from cuhkx.inference.qwen35_vllm import Qwen35VLLMBackend

    backend = Qwen35VLLMBackend.__new__(Qwen35VLLMBackend)
    backend.engines = [object(), object()]

    for size in (1, 2, 3, 4, 5, 16, 682):
        assignments = backend._round_robin(size)
        covered = []
        for _, start, end in assignments:
            covered.extend(range(start, end))
        assert covered == list(range(size)), size
        # Contiguous blocks, so each replica gets one engine call per dispatch.
        for _, start, end in assignments:
            assert end > start
        assert sum(end - start for _, start, end in assignments) == size
        assert len(assignments) <= 2


def test_qwen35_vllm_prefix_cache_rate_is_measured_not_assumed():
    """The hit rate must come from engine counters, and be absent if unreported."""
    from cuhkx.inference.qwen35_vllm import Qwen35VLLMBackend

    backend = Qwen35VLLMBackend.__new__(Qwen35VLLMBackend)
    backend.prefix_cache_requests = 0
    backend.prefix_cache_missing = 0
    backend.prefix_cache_cached_tokens = 0
    backend.prefix_cache_prompt_tokens = 0
    backend.prefix_cache_summary = {}

    # Engine reports the split for two requests.
    backend._record_prefix_cache({"num_cached_tokens": 30, "num_prompt_tokens": 1000})
    backend._record_prefix_cache({"num_cached_tokens": 10, "num_prompt_tokens": 1000})
    backend._record_prefix_summary()
    assert backend.prefix_cache_summary["cached_tokens"] == 40
    assert backend.prefix_cache_summary["prompt_tokens"] == 2000
    assert backend.prefix_cache_summary["hit_rate"] == 0.02
    assert backend.prefix_cache_summary["reported_requests"] == 2

    # A run whose engine never reports the field must say so, rather than
    # reporting a hit rate of zero as if it had been measured.
    bare = Qwen35VLLMBackend.__new__(Qwen35VLLMBackend)
    bare.prefix_cache_requests = 0
    bare.prefix_cache_missing = 0
    bare.prefix_cache_cached_tokens = 0
    bare.prefix_cache_prompt_tokens = 0
    bare.prefix_cache_summary = {}
    bare._record_prefix_cache({"reported": True, "first_token_latency": 0.1})
    bare._record_prefix_summary()
    assert bare.prefix_cache_summary["reported_requests"] == 0
    assert bare.prefix_cache_summary["unreported_requests"] == 1
    assert bare.prefix_cache_summary["hit_rate"] is None


def test_qwen35_vllm_engine_metrics_reads_cached_tokens():
    """The prefix-cache count must be found under whichever name the build uses."""
    from cuhkx.inference.qwen35_vllm import _engine_metrics

    class Stats:
        first_token_latency = 0.5
        num_prompt_tokens = 1200
        num_cached_tokens = 48

    class Output:
        metrics = Stats()

    metrics = _engine_metrics(Output())
    assert metrics["reported"] is True
    assert metrics["num_cached_tokens"] == 48
    assert metrics["num_prompt_tokens"] == 1200

    # The first data-parallel smoke run found num_cached_tokens absent on every
    # request, so the alternative spellings and the nested containers are tried
    # before giving up.
    class Alternative:
        num_prompt_tokens = 500
        num_prefix_cached_tokens = 12

    class AltOutput:
        metrics = Alternative()

    assert _engine_metrics(AltOutput())["num_cached_tokens"] == 12

    class Nested:
        num_prompt_tokens = 500
        class kv_cache_metrics:
            num_cached_tokens = 7

    class NestedOutput:
        metrics = Nested()

    assert _engine_metrics(NestedOutput())["num_cached_tokens"] == 7

    # Absent must stay absent, so "not measured" never reads as "measured zero".
    class NoneReported:
        num_prompt_tokens = 500

    class NoneOutput:
        metrics = NoneReported()

    assert "num_cached_tokens" not in _engine_metrics(NoneOutput())


def test_qwen35_vllm_records_the_real_metrics_shape():
    """An absent counter must diagnose itself instead of failing silently."""
    from cuhkx.inference.qwen35_vllm import describe_engine_metrics

    class Stats:
        num_prompt_tokens = 900
        first_token_latency = 0.25

    class Output:
        metrics = Stats()

    shape = describe_engine_metrics(Output())
    assert "num_prompt_tokens" in shape["fields"]
    assert shape["values"]["num_prompt_tokens"] == 900
    assert shape["metrics_is_none"] is False

    class NoStats:
        metrics = None

    assert describe_engine_metrics(NoStats())["metrics_is_none"] is True


def test_cli_records_data_parallel_size_in_engine_options():
    """DP changes the topology, so it belongs in the signed contract."""
    from cuhkx.cli import engine_options

    class Args:
        backend = "vllm"
        tensor_parallel_size = 1
        data_parallel_size = 2
        gpu_memory_utilization = 0.8
        attention_backend = "TRITON_ATTN"
        max_num_seqs = 16
        adapter_dir = None

    options = engine_options(Args(), "qwen35")
    assert options["data_parallel_size"] == 2
    assert options["tensor_parallel_size"] == 1
    assert options["max_num_seqs"] == 16


def test_cli_rejects_tensor_and_data_parallel_together():
    from cuhkx.cli import resolve_backend

    class Args:
        backend = "vllm"
        tensor_parallel_size = 2
        data_parallel_size = 2
        adapter_dir = None

    with pytest.raises(ValueError, match="not both"):
        resolve_backend(Args(), "qwen35")


def test_qwen35_vllm_replica_process_is_not_daemonic():
    """A daemonic replica cannot spawn vLLM's own workers.

    vLLM builds several processes beneath the replica process, and
    multiprocessing refuses to let a daemon have children:

        AssertionError: daemonic processes are not allowed to have children

    The first data-parallel smoke run died exactly there, before any request was
    served. Daemon mode would have auto-reaped the replicas, so the cleanup it
    provided has to come from close() plus the atexit registry instead.
    """
    source = (PROJECT / "src/cuhkx/inference/qwen35_vllm.py").read_text(encoding="utf-8")
    assert "daemon=False" in source
    assert "daemon=True" not in source
    # The replacement for daemon-mode cleanup.
    assert "atexit.register" in source
    assert "def _terminate_replicas" in source


def test_runner_releases_the_backend_after_a_run():
    """Data-parallel replicas must not outlive the run that started them."""
    source = (PROJECT / "src/cuhkx/inference/runner.py").read_text(encoding="utf-8")
    # Anchor on the call site, not the helper definition that appears earlier.
    call = source.index("        _release_backend(backend)\n        return summary")
    # Cleanup happens after the summary is written, since metadata() samples the
    # worker memory that only exists while the engine is alive.
    assert source.index("write_json(summary_path, summary)") < call
    assert "def _release_backend(backend)" in source


def test_qwen35_notebooks_pin_the_attention_backend():
    import json

    for name in ("qwen35-4b-vllm.ipynb", "qwen35-4b-qlora-vllm.ipynb"):
        notebook = json.loads((PROJECT / "notebooks" / name).read_text(encoding="utf-8"))
        source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert 'ATTENTION_BACKEND = "TRITON_ATTN"' in source, name
        assert '"--attention-backend", ATTENTION_BACKEND' in source, name
        # A silent fallback to FlashInfer would reintroduce the linker failure.
        assert 'backend_metadata.get("attention_backend") != ATTENTION_BACKEND' in source, name


def test_qwen35_vllm_is_not_shipped_to_the_baseline_lane():
    # The 7B lane runs torch 2.7.1 + Transformers 4.57.6 and must never see vLLM.
    for script in ("scripts/package_cloud.py", "scripts/package_training.py",
                   "scripts/build_training_notebook.py"):
        source = (PROJECT / script).read_text(encoding="utf-8")
        assert "qwen35_vllm" not in source, script
    training_lock = (PROJECT / "requirements/train.lock.txt").read_text(encoding="utf-8")
    assert "vllm==" not in training_lock


def test_qwen35_trainer_has_transformers_5_warmup_compatibility():
    source = (PROJECT / "src/cuhkx/training/trainer.py").read_text(encoding="utf-8")
    assert '"warmup_ratio" if "warmup_ratio" in inspect.signature(TrainingArguments).parameters else "warmup_steps"' in source


def test_scaled_lora_gradient_overflow_is_reported_not_conflated_with_missing_gradients():
    class FiniteResult:
        def __init__(self, value):
            self.value = value

        def all(self):
            return self.value

    class Gradient:
        def __init__(self, finite, magnitude):
            self.finite, self.magnitude = finite, magnitude

        def detach(self):
            return self

        def abs(self):
            return self

        def sum(self):
            return self.magnitude

    class Parameter:
        requires_grad = True

        def __init__(self, gradient):
            self.grad = gradient

    class Model:
        def named_parameters(self):
            return [("layer.lora_A.default.weight", Parameter(Gradient(True, 2))),
                    ("layer.lora_B.default.weight", Parameter(Gradient(False, 0)))]

    class Torch:
        @staticmethod
        def isfinite(gradient):
            return FiniteResult(gradient.finite)

    diagnostics = lora_gradient_diagnostics(Model(), Torch())
    assert diagnostics["gradient_tensors"] == 2
    assert diagnostics["nonfinite_tensors"] == 1
    assert diagnostics["finite_nonzero_tensors"] == 1

    class MissingModel:
        def named_parameters(self):
            return [("layer.lora_A.default.weight", Parameter(None))]

    with pytest.raises(ValueError, match="gradients are missing"):
        lora_gradient_diagnostics(MissingModel(), Torch())


def test_qwen35_fp16_scaler_starts_below_the_observed_overflow_range():
    existing = object()

    class FakeGradScalerKwargs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    accelerator_args = configure_qwen35_fp16_scaler(
        {"kwargs_handlers": [existing]}, FakeGradScalerKwargs)

    assert accelerator_args["kwargs_handlers"][0] is existing
    assert accelerator_args["kwargs_handlers"][1].kwargs == {"init_scale": 1.0, "growth_interval": 16}
