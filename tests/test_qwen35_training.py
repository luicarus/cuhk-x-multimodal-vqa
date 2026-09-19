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
    assert "torchvision==0.22.1+cu126" in lock
    assert "-r qwen35.in" in (PROJECT / "requirements/train_qwen35.in").read_text(encoding="utf-8")
    start = lock.index("markupsafe==3.0.3")
    block = lock[start:lock.index("\nmdurl==", start)]
    assert "0bf2a864d67e76e5c9a34dc26ec616a66b9888e25e7b9460e1c76d3293bd9dbf" in block


def test_qwen35_training_notebook_keeps_cuda_package_index():
    import json

    notebook = json.loads((PROJECT / "notebooks/qwen35-4b-qlora-full-v1.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "--index-url" in source
    assert "https://download.pytorch.org/whl/cu126" in source
    assert "import json, peft, transformers" in source
    assert source.count('cloud("verify-run", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG)') == 3


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
