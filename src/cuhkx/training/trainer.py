"""Single-GPU QLoRA, answer-only loss and generative development evaluation."""
from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import re
import sys
import time

from PIL import Image

from cuhkx.config import inside, require
from cuhkx.data.validate import fingerprint
from cuhkx.evaluation.metric import available_option_letters, evaluate_records
from cuhkx.inference.prompt import build_mcq_prompt, allowed_answer_outputs, parse_model_answer
from cuhkx.inference.storage import run_lock, write_json
from cuhkx.inference.weights import verify_weights, file_hash
from cuhkx.training.adapter import text_targets, save_adapter_receipt, verify_adapter
from cuhkx.training.collator import CompletionCollator
from cuhkx.training.dataset import prepare_data, require_ready, TrainingDataset
from cuhkx.training.profiling import TrainingProfiler


CHECKPOINT_FILES = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "scaler.pt"}
QWEN35_FP16_SCALER = {"init_scale": 1.0, "growth_interval": 16}


def lora_state(model):
    """Read the live adapter, including parameters frozen/replaced during reload."""
    import numpy as np
    hashes, nonzero_b = {}, 0
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        values = parameter.detach().float().cpu().numpy()
        require(bool(np.isfinite(values).all()), "non-finite LoRA weights")
        hashes[name] = file_hash_tensor(parameter)
        if ".lora_B." in name and np.count_nonzero(values):
            nonzero_b += 1
    require(hashes and any(".lora_B." in name for name in hashes), "live LoRA tensors missing")
    return {"fingerprint":fingerprint(hashes), "tensors":len(hashes), "nonzero_b_tensors":nonzero_b}


def fit_and_validate(trainer, checkpoint):
    """A completed resume need not take new steps, but its adapter must be learned."""
    initial = lora_state(trainer.model)
    start_step = 0
    if checkpoint is not None:
        saved = json.loads((Path(checkpoint)/"trainer_state.json").read_text())
        start_step = int(saved["global_step"])
        require(start_step >= 0, "invalid checkpoint step")
    trained = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    final = lora_state(trainer.model)
    end_step = trainer.state.global_step
    require(end_step > 0 and end_step >= start_step, "training made no valid optimizer progress")
    completed_resume = checkpoint is not None and end_step == start_step and end_step >= trainer.state.max_steps
    require(final["nonzero_b_tensors"] > 0 and (initial["fingerprint"] != final["fingerprint"] or completed_resume),
            "Selected LoRA is still the zero-initialized adapter; inspect latest checkpoint lora_B tensors")
    return trained, {"resumed_from_step":start_step, "optimizer_steps_this_run":end_step-start_step,
                     "completed_resume":completed_resume,"adapter_state":final}


def checkpoint_receipt(path, signature):
    path = Path(path)
    needed = CHECKPOINT_FILES
    require(all((path/name).is_file() for name in needed), "incomplete resumable training checkpoint")
    value = {"training_signature": signature, "files": {name: file_hash(path/name) for name in sorted(needed)}}
    write_json(path / "checkpoint_receipt.json", value)


def lora_gradient_diagnostics(model, torch):
    """Classify scaled LoRA gradients without pre-empting AMP overflow handling."""
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                 if ".lora_" in name and parameter.requires_grad and parameter.grad is not None]
    require(gradients, "LoRA gradients are missing")
    nonfinite = [name for name, gradient in gradients if not bool(torch.isfinite(gradient).all())]
    finite_nonzero = [name for name, gradient in gradients
                      if name not in nonfinite and bool(gradient.detach().abs().sum() > 0)]
    return {"gradient_tensors": len(gradients), "nonfinite_tensors": len(nonfinite),
            "nonfinite_examples": nonfinite[:5], "finite_nonzero_tensors": len(finite_nonzero)}


def configure_qwen35_fp16_scaler(accelerator_args, grad_scaler_kwargs):
    """Add a conservative public GradScaler policy while preserving Trainer handlers."""
    handlers = list(accelerator_args.get("kwargs_handlers", ()))
    handlers.append(grad_scaler_kwargs(**QWEN35_FP16_SCALER))
    accelerator_args["kwargs_handlers"] = handlers
    return accelerator_args


def latest_checkpoint(output, signature):
    complete = []
    for path in output.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path/"checkpoint_receipt.json").is_file():
            value = json.loads((path/"checkpoint_receipt.json").read_text())
            require(value["training_signature"] == signature, "training checkpoint signature differs")
            require(set(value["files"]) == CHECKPOINT_FILES, "checkpoint receipt omits required recovery state")
            for name, digest in value["files"].items():
                file = inside(path, name)
                require(file.is_file() and file_hash(file) == digest, "training checkpoint content changed")
            complete.append((int(match[1]), path))
    return max(complete, default=(0, None))[1]


def train(config, settings, weights_dir, run_id, *, resume=False, gpu=0, smoke_steps=None):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) is not None, "invalid training run-id")
    require(type(gpu) is int and gpu >= 0, "invalid GPU index")
    require(smoke_steps is None or 1 <= smoke_steps <= 20, "smoke steps must be 1..20")
    prepared = prepare_data(config, settings)
    require_ready(prepared, ("train", "dev"))
    qwen35 = config["baseline"]["model"]["id"] == "Qwen/Qwen3.5-4B"
    if qwen35:
        from cuhkx.inference.qwen35_weights import verify_qwen35_weights
        base_source = verify_qwen35_weights(weights_dir, config["baseline"]["model"])
    else:
        base_source = verify_weights(weights_dir, config["baseline"]["model"])
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "first training lane is single-process/single-GPU")
    if "torch" in sys.modules:
        require(not sys.modules["torch"].cuda.is_initialized(), "start training in a fresh process before CUDA initialization")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    import importlib.metadata
    from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if qwen35:
        from accelerate.utils import GradScalerKwargs
        from transformers import AutoModelForMultimodalLM
        from cuhkx.training.qwen35_support import Qwen35CompletionCollator, qwen35_text_targets
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        from cuhkx.inference.qwen import QwenBackend

    require(torch.cuda.is_available(), "training requires a cloud CUDA GPU")
    train_samples = prepared["samples"]["train"][:16] if smoke_steps else prepared["samples"]["train"]
    dev_samples = prepared["samples"]["dev"][:16] if smoke_steps else prepared["samples"]["dev"]
    package_names = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
    if not qwen35:
        package_names = (*package_names, "qwen-vl-utils")
    versions = {name: importlib.metadata.version(name) for name in package_names}
    contract = {"schema_version": 1, "profile": settings, "baseline": config["baseline"], "base_source": base_source,
                "data_signature": prepared["data_signature"], "software": versions, "smoke_steps": smoke_steps,
                "train_ids": [s["qa"]["qa_id"] for s in train_samples], "dev_ids": [s["qa"]["qa_id"] for s in dev_samples],
                "prompts": fingerprint([build_mcq_prompt(s["qa"]) for s in train_samples+dev_samples]),
                # Both settings reach TrainingArguments from code rather than from
                # the YAML alone, so they are restated here: the signature must
                # change when the optimizer backend or the loader parallelism
                # changes, or a differently-trained adapter could be accepted as
                # a resume of this run.
                "training_execution": {"optim": settings["optimizer"]["optim"],
                                       "dataloader_num_workers": settings["optimizer"]["dataloader_num_workers"],
                                       "gradient_checkpointing": True,
                                       "precision": "fp16"},
                "fp16_grad_scaler": QWEN35_FP16_SCALER if qwen35 else "accelerate_default"}
    signature = fingerprint(contract)
    output = inside(Path(config["project_root"])/"artifacts/training", run_id)
    with run_lock(output):
        state = output/"training_state.json"
        if state.exists():
            require(resume and json.loads(state.read_text()) == {"signature":signature,"contract":contract}, "training run exists or contract differs")
        else:
            require({p.name for p in output.iterdir()} <= {".run.lock"}, "existing training files lack a contract")
            write_json(state, {"signature":signature,"contract":contract})
        if (output/"result.json").exists():
            result = json.loads((output/"result.json").read_text())
            adapter = verify_adapter(output/"adapter", base_source)
            require(result["training_signature"] == signature and adapter["training_signature"] == signature and
                    result["adapter_receipt"] == adapter["receipt_sha256"], "finished training result changed")
            return result
        checkpoint = latest_checkpoint(output, signature) if resume else None
        set_seed(settings["optimizer"]["seed"])
        if qwen35:
            processor = AutoProcessor.from_pretrained(str(weights_dir), local_files_only=True,
                                                      trust_remote_code=False, use_fast=False)
        else:
            processor = AutoProcessor.from_pretrained(str(weights_dir), local_files_only=True, trust_remote_code=False,
                                                      min_pixels=280**2, max_pixels=280**2)
        processor.tokenizer.padding_side = "right"
        model_kwargs = dict(local_files_only=True, trust_remote_code=False, dtype=torch.float16,
                            device_map={"": 0}, low_cpu_mem_usage=True,
                            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16))
        if not qwen35:
            model_kwargs["attn_implementation"] = "sdpa"
        model_cls = AutoModelForMultimodalLM if qwen35 else Qwen2_5_VLForConditionalGeneration
        model = model_cls.from_pretrained(str(weights_dir), **model_kwargs)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant":False})
        targets = (qwen35_text_targets([name for name,_ in model.named_modules()]) if qwen35
                   else text_targets([name for name,_ in model.named_modules()]))
        lora = settings["lora"]
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=lora["r"], lora_alpha=lora["alpha"],
                                               lora_dropout=lora["dropout"], target_modules=targets, bias="none"))
        trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
        require(trainable and all(".lora_" in name and not any(s in name.split(".") for s in ("visual","vision_model","merger")) for name,_ in trainable), "unexpected trainable parameters")

        class ReceiptCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                checkpoint_receipt(output/f"checkpoint-{state.global_step}", signature)

        # Step timing and device memory, recorded beside the run rather than in
        # its contract, so a new metric can never invalidate a verified run.
        profiler = TrainingProfiler(
            samples_per_step=1,
            scope="training_after_model_load").start()

        class ProfilingCallback(TrainerCallback):
            """Time each optimizer step without touching the training loop.

            ``on_step_begin`` fires before the batch is fetched for that step, so
            the span it opens covers the dataloader wait; the wait is closed as
            soon as the step body starts, which is what makes the dataloader
            share measurable instead of assumed.
            """

            def on_train_begin(self, args, state, control, **kwargs):
                profiler.begin_batch()

            def on_step_begin(self, args, state, control, **kwargs):
                profiler.begin_step()

            def on_step_end(self, args, state, control, **kwargs):
                profiler.end_step(samples=1)
                # Re-open the wait for the next batch; the last step leaves one
                # pending, which summary() simply ignores.
                profiler.begin_batch()

        class AnswerTrainer(Trainer):
            checked_gradients = False
            gradient_microbatches = 0
            scaled_overflow_microbatches = 0
            def _build_accelerator_args(self, **kwargs):
                accelerator_args = super()._build_accelerator_args(**kwargs)
                return (configure_qwen35_fp16_scaler(accelerator_args, GradScalerKwargs)
                        if qwen35 else accelerator_args)

            def training_step(self, model, inputs, num_items_in_batch=None):
                loss = super().training_step(model, inputs, num_items_in_batch)
                require(bool(torch.isfinite(loss).all()), "non-finite training loss")
                diagnostics = lora_gradient_diagnostics(model, torch)
                self.gradient_microbatches += 1
                if diagnostics["nonfinite_tensors"]:
                    require(getattr(self.accelerator, "scaler", None) is not None,
                            "non-finite LoRA gradients without an AMP GradScaler")
                    self.scaled_overflow_microbatches += 1
                    if self.scaled_overflow_microbatches == 1:
                        print(json.dumps({"amp_scaled_gradient_overflow": diagnostics,
                                          "grad_scale": float(self.accelerator.scaler.get_scale())}), flush=True)
                elif not self.checked_gradients:
                    require(diagnostics["finite_nonzero_tensors"] > 0, "LoRA gradients are all zero")
                    self.checked_gradients = True
                return loss

            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
                if qwen35:
                    from cuhkx.inference.qwen35 import Qwen35Backend
                    backend = Qwen35Backend.__new__(Qwen35Backend)
                else:
                    backend = QwenBackend.__new__(QwenBackend)
                backend.processor, backend.model, backend.torch = processor, self.model, torch
                backend.device, backend.image_size = torch.device("cuda:0"), 280
                was_training = self.model.training
                self.model.eval()
                references, predictions = [], []
                started = time.perf_counter()
                try:
                    for sample in dataset:
                        images = []
                        try:
                            for path in sample["frame_paths"]:
                                with Image.open(path) as image:
                                    image.load(); images.append(image.copy())
                            qa = sample["qa"]
                            allowed = allowed_answer_outputs(qa["category"], available_option_letters(qa))
                            raw = backend.generate(images, build_mcq_prompt(qa), allowed_outputs=allowed, max_new_tokens=8)
                            parsed = parse_model_answer(raw, category=qa["category"], valid_options=available_option_letters(qa))
                            require(parsed.is_valid and raw in allowed, "invalid answer during development evaluation")
                            references.append({**qa,"answer":sample["answer"]})
                            predictions.append({"qa_id":qa["qa_id"],"prediction":parsed.prediction})
                        finally:
                            for image in images:image.close()
                finally:
                    self.model.train(was_training)
                result = evaluate_records(references,predictions)
                metrics = {f"{metric_key_prefix}_accuracy":result["overall_accuracy"], f"{metric_key_prefix}_runtime":time.perf_counter()-started}
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args,self.state,self.control,metrics)
                return metrics

        opt = settings["optimizer"]
        warmup_name = "warmup_ratio" if "warmup_ratio" in inspect.signature(TrainingArguments).parameters else "warmup_steps"
        arguments = TrainingArguments(output_dir=str(output), per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=opt["gradient_accumulation_steps"],
            learning_rate=opt["learning_rate"], num_train_epochs=opt["epochs"], max_steps=smoke_steps or -1,
            **{warmup_name: opt["warmup_ratio"]}, max_grad_norm=opt["max_grad_norm"], optim=opt["optim"],
            fp16=True, bf16=False, seed=opt["seed"], data_seed=opt["seed"], report_to=[],
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
            remove_unused_columns=False, label_names=["labels"], eval_strategy="epoch", save_strategy="epoch",
            # A smoke epoch is one optimizer step. Its warmup checkpoint may still
            # be the zero adapter and win/tie dev accuracy: smoke must retain last.
            save_total_limit=2, save_only_model=False, load_best_model_at_end=(smoke_steps is None),
            metric_for_best_model="accuracy", greater_is_better=True, logging_steps=1 if smoke_steps else 10,
            # Collation is CPU-heavy here (four JPEG decodes plus two processor
            # calls per sample); with 0 workers it sits on the critical path.
            dataloader_num_workers=opt["dataloader_num_workers"], push_to_hub=False)
        trainer = AnswerTrainer(model=model,args=arguments,
            train_dataset=TrainingDataset(train_samples,config["data_root"]),
            eval_dataset=TrainingDataset(dev_samples,config["data_root"]),
            data_collator=(Qwen35CompletionCollator(processor, config["data_root"], opt["max_sequence_length"])
                           if qwen35 else CompletionCollator(processor,opt["max_sequence_length"])),
            processing_class=processor,callbacks=[ReceiptCallback(), ProfilingCallback()])
        trained, update_check = fit_and_validate(trainer, checkpoint)
        training_profile = profiler.stop()
        adapter_dir = output/"adapter"
        trainer.model.save_pretrained(adapter_dir,safe_serialization=True)
        save_adapter_receipt(adapter_dir,base_source,signature,prepared["data_signature"],lora,targets,"smoke" if smoke_steps else "sft")
        adapter = verify_adapter(adapter_dir,base_source)
        metrics = dict(trained.metrics)
        if update_check["optimizer_steps_this_run"] == 0:
            # HF's zero-loss/high-throughput values describe an empty resumed loop.
            for key in ("train_loss", "train_samples_per_second", "train_steps_per_second"):
                metrics.pop(key, None)
        result = {"status":"PASS","training_signature":signature,"purpose":"smoke" if smoke_steps else "sft",
                  "steps":trainer.state.global_step,"best_dev_accuracy":trainer.state.best_metric if smoke_steps is None else None,
                  "checkpoint_selection":"last" if smoke_steps else "best_dev_accuracy",
                  "update_check":update_check,"implementation_revision":"smoke-last-v1",
                  "gradient_diagnostics":{"microbatches":trainer.gradient_microbatches,
                    "scaled_overflow_microbatches":trainer.scaled_overflow_microbatches,
                    "finite_nonzero_seen":trainer.checked_gradients},
                  # Advisory evidence, not part of the verified contract: nothing
                  # in verify/result checking reads it, so a metric change cannot
                  # invalidate an already finished training run.
                  "training_profile":training_profile,
                  "metrics":metrics,"adapter_receipt":adapter["receipt_sha256"]}
        write_json(output/"result.json",result)
        return result


def file_hash_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()
