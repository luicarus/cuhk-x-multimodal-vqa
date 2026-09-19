import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuhkx.config import InputError
from cuhkx.training.trainer import fit_and_validate, lora_state


class Parameter:
    def __init__(self, values):self.values=np.array(values,dtype=np.float32)
    def detach(self):return self
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return self.values


class Model:
    def __init__(self,b=0):
        self.parameters={"model.layers.0.self_attn.q_proj.lora_A.default.weight":Parameter([0.2,0.3]),
                         "model.layers.0.self_attn.q_proj.lora_B.default.weight":Parameter([b,b])}
    def named_parameters(self):return self.parameters.items()


class Trainer:
    def __init__(self,load_best=False):
        self.model=Model()
        self.load_best=load_best
        self.state=SimpleNamespace(global_step=0,max_steps=4)
        self.new_steps=0
    def train(self,resume_from_checkpoint=None):
        if resume_from_checkpoint:
            self.state.global_step=json.loads((Path(resume_from_checkpoint)/"trainer_state.json").read_text())["global_step"]
        while self.state.global_step<4:
            self.state.global_step+=1;self.new_steps+=1
        # First (warmup) checkpoint has no update; the last checkpoint has learned B.
        self.model=Model(0 if self.load_best else 0.1)
        return SimpleNamespace(metrics={"train_loss":0 if not self.new_steps else 0.5})


def test_warmup_tie_reproduces_zero_adapter_and_last_selection_fixes_it():
    warmup=math.ceil(4*0.03)
    assert warmup==1 and 0/warmup==0
    old=Trainer(load_best=True)
    with pytest.raises(InputError,match="zero-initialized"):
        fit_and_validate(old,None)
    fixed=Trainer(load_best=False)
    _,check=fit_and_validate(fixed,None)
    assert check["optimizer_steps_this_run"]==4
    assert check["adapter_state"]["nonzero_b_tensors"]==1


def test_completed_checkpoint_exports_without_demanding_more_steps(tmp_path):
    (tmp_path/"trainer_state.json").write_text(json.dumps({"global_step":4}))
    trainer=Trainer(load_best=False)
    _,check=fit_and_validate(trainer,tmp_path)
    assert trainer.new_steps==0 and check["optimizer_steps_this_run"]==0
    assert check["resumed_from_step"]==4 and check["adapter_state"]["nonzero_b_tensors"]>0


def test_partial_resume_counts_only_new_steps(tmp_path):
    (tmp_path/"trainer_state.json").write_text(json.dumps({"global_step":2}))
    _,check=fit_and_validate(Trainer(),tmp_path)
    assert check["optimizer_steps_this_run"]==2


def test_already_loaded_complete_adapter_is_valid_without_new_changes(tmp_path):
    (tmp_path/"trainer_state.json").write_text(json.dumps({"global_step":4}))
    trainer=Trainer()
    trainer.model=Model(0.1)
    _,check=fit_and_validate(trainer,tmp_path)
    assert check["completed_resume"] is True


def test_completed_but_zero_checkpoint_is_still_rejected(tmp_path):
    (tmp_path/"trainer_state.json").write_text(json.dumps({"global_step":4}))
    with pytest.raises(InputError,match="zero-initialized"):
        fit_and_validate(Trainer(load_best=True),tmp_path)


def test_live_snapshot_detects_replacement_and_rejects_nonfinite():
    model=Model();before=lora_state(model)
    model.parameters[next(k for k in model.parameters if '.lora_B.' in k)]=Parameter([0.5,0.5])
    assert lora_state(model)["fingerprint"]!=before["fingerprint"]
    with pytest.raises(InputError,match="non-finite"):
        lora_state(Model(float('nan')))


def test_production_training_keeps_best_selection_and_smoke_uses_last():
    path=Path(__file__).resolve().parents[1]/"src/cuhkx/training/trainer.py"
    call=next(n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
              if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="TrainingArguments")
    expr=next(k.value for k in call.keywords if k.arg=="load_best_model_at_end")
    code=compile(ast.Expression(expr),"selection_policy","eval")
    assert eval(code,{"smoke_steps":4}) is False
    assert eval(code,{"smoke_steps":None}) is True
