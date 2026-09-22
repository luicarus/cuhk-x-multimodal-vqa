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
    # Dual-GPU tensor parallelism is the point of this lane.
    assert 'TENSOR_PARALLEL = 2' in source
    assert '"--backend", "vllm"' in source
    assert '"--tensor-parallel-size", str(TENSOR_PARALLEL)' in source
    # vLLM cannot train, so training must stay on the Transformers trainer.
    training_calls = [line for line in source.splitlines() if 'cloud("train"' in line]
    assert training_calls and all("vllm" not in line for line in training_calls)
    # Both GPUs must be visible before vLLM starts.
    assert "device_count" in source


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
