from pathlib import Path

import pytest

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
    assert "--extra-index-url https://download.pytorch.org/whl/cu126" in lock
    assert "peft==0.18.0" in lock
    # vLLM 0.24.0 pins the torch/torchvision pair and needs transformers>=5.5.3.
    assert "torchvision==0.26.0+cu126" in lock
    assert "torch==2.11.0+cu126" in lock
    assert "vllm==0.24.0" in lock
    assert "-r qwen35.in" in (PROJECT / "requirements/train_qwen35.in").read_text(encoding="utf-8")
    start = lock.index("markupsafe==3.0.3")
    block = lock[start:lock.index("\nmdurl==", start)]
    assert "0bf2a864d67e76e5c9a34dc26ec616a66b9888e25e7b9460e1c76d3293bd9dbf" in block


def test_qwen35_inference_lock_pairs_vllm_with_transformers_five():
    lock = (PROJECT / "requirements/qwen35.lock.txt").read_text(encoding="utf-8")
    for pin in ("vllm==0.24.0", "torch==2.11.0+cu126", "torchvision==0.26.0+cu126",
                "transformers==5.17.0"):
        assert pin in lock, pin
    # The structured-output backends are what enforce the closed answer space.
    assert "xgrammar==" in lock
    # This lane must never drift into the 7B lane's Transformers 4.x toolchain.
    baseline = (PROJECT / "requirements/cloud.lock.txt").read_text(encoding="utf-8")
    assert "vllm==" not in baseline
    assert "torch==2.7.1+cu126" in baseline


def test_qwen35_training_notebook_keeps_cuda_package_index():
    import json

    notebook = json.loads((PROJECT / "notebooks/qwen35-4b-qlora-vllm.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "--index-url" in source
    assert "https://download.pytorch.org/whl/cu126" in source
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
