"""CPU input checks and cloud-only IR4 + 7B prediction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from cuhkx.config import load_config, load_qwen35_config, require
from cuhkx.data.validate import check_inputs


def prepare_run(config: dict, result: dict, run_id: str) -> Path:
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) is not None, "invalid run-id")
    root = Path(config["project_root"]) / "outputs"
    output = (root / run_id).resolve()
    require(output.is_relative_to(root.resolve()), "run directory escapes outputs")
    payloads = {
        "resolved_config.json": {"schema_version": 1, "phase": "prepared_only", "run_id": run_id,
                                 "config": config, "dataset": result["dataset"],
                                 "target_ids": result["target_ids"], "input_signature": result["input_signature"]},
        "input_check.json": result,
    }
    encoded = {name: (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
               for name, value in payloads.items()}
    for name, payload in encoded.items():
        path = output / name
        if path.exists():
            require(path.read_bytes() == payload, "run inputs/config differ; use a new run-id")
    if output.exists():
        require(all((output / name).exists() for name in encoded) or
                not any(output.iterdir()), "incomplete/unknown existing run; use a new run-id")
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in encoded.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
    return output


def add_inference_backend(command):
    """Attach the generation-engine selector shared by predict/evaluate-training."""
    command.add_argument("--backend", choices=("transformers", "vllm"), default=None,
                         help="Generation engine. Defaults to vLLM for the Qwen3.5 lane "
                              "and to Transformers for the Qwen2.5-VL-7B lane.")
    command.add_argument("--tensor-parallel-size", type=int, default=2,
                         help="vLLM tensor parallelism across visible GPUs (Kaggle 2x T4 default)")
    command.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                         help="Fraction of each GPU vLLM may reserve")
    command.add_argument("--attention-backend", default=None,
                         choices=("TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION"),
                         help="vLLM attention backend. Defaults to TRITON_ATTN because "
                              "FlashInfer JIT-compiles SM 7.5 kernels and fails to link "
                              "libcuda.so on Kaggle (no driver stubs).")
    command.add_argument("--max-num-seqs", type=int, default=1,
                         help="vLLM scheduler concurrency. Above 1 the runner submits "
                              "requests in batches so the engine can interleave them.")


def default_backend(profile):
    """vLLM is the accelerated default for the lane that ships it."""
    return "vllm" if profile == "qwen35" else "transformers"


def resolve_backend(args, profile):
    """Pick the generation engine, rejecting combinations that cannot work."""
    chosen = getattr(args, "backend", None) or default_backend(profile)
    if chosen == "vllm":
        require(profile == "qwen35",
                "vLLM is only wired for the Qwen3.5 lane; the 7B lane uses NF4 on Transformers")
        require(getattr(args, "adapter_dir", None) is None or profile == "qwen35",
                "unsupported adapter/backend combination")
    return chosen


def default_attention_backend():
    """FlashInfer cannot link on Kaggle; Triton needs no nvcc or driver stubs."""
    return "TRITON_ATTN"


def engine_options(args, profile):
    """Engine settings that must appear in the signed run contract."""
    if resolve_backend(args, profile) != "vllm":
        return {}
    return {"tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "attention_backend": getattr(args, "attention_backend", None)
            or default_attention_backend(),
            "max_num_seqs": args.max_num_seqs}


def build_backend(profile, config, args, run_output):
    """Construct the selected generation engine for one run."""
    backend = resolve_backend(args, profile)
    weights = args.weights_dir.resolve()
    adapter = args.adapter_dir.resolve() if args.adapter_dir else None
    if backend == "vllm":
        from cuhkx.inference.qwen35_vllm import Qwen35VLLMBackend
        return Qwen35VLLMBackend(
            config["baseline"], weights, adapter=adapter,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            attention_backend=getattr(args, "attention_backend", None)
            or default_attention_backend(),
            max_num_seqs=args.max_num_seqs,
        )
    if profile == "qwen35":
        from cuhkx.inference.qwen35 import Qwen35Backend
        return Qwen35Backend(config["baseline"], weights, adapter=adapter)
    from cuhkx.inference.qwen import QwenBackend
    kwargs = {"adapter": adapter} if adapter is not None else {}
    return QwenBackend(config["baseline"], weights, run_output / "offload", **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("check", help="Verify cached inputs on CPU; never load a model")
    check.add_argument("--dataset", choices=("test", "pilot"), required=True)
    check.add_argument("--project-root", type=Path, help="Project containing configs; defaults to installed source checkout")
    check.add_argument("--data-root", type=Path, help="Portable data directory; defaults to <project-root>/data")
    check.add_argument("--limit", type=int, help="Fix the first N ordered QA before any resume filtering")
    check.add_argument("--run-id", help="Also save resolved config and fixed target IDs; does not start inference")
    check.add_argument("--profile", choices=("baseline", "qwen35"), default="baseline")
    check.add_argument("--json", action="store_true")
    predict = subcommands.add_parser("predict", help="Run pinned 7B weights on a cloud CUDA GPU")
    predict.add_argument("--dataset", choices=("test", "pilot"), required=True)
    predict.add_argument("--project-root", type=Path)
    predict.add_argument("--data-root", type=Path)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--weights-dir", type=Path, required=True)
    predict.add_argument("--resume", action="store_true")
    predict.add_argument("--adapter-dir", type=Path, help="Optional verified LoRA adapter; omitted for the baseline")
    predict.add_argument("--profile", choices=("baseline", "qwen35"), default="baseline")
    add_inference_backend(predict)
    fetch = subcommands.add_parser("fetch-weights", help="Download pinned weights and record their content provenance")
    fetch.add_argument("--project-root", type=Path)
    fetch.add_argument("--data-root", type=Path)
    fetch.add_argument("--weights-dir", type=Path, required=True)
    qfetch = subcommands.add_parser("fetch-qwen35-weights", help="Download pinned Qwen3.5-4B weights")
    qfetch.add_argument("--project-root", type=Path)
    qfetch.add_argument("--data-root", type=Path)
    qfetch.add_argument("--weights-dir", type=Path, required=True)
    qfetch.add_argument("--revision", required=True)
    for name in ("training-check", "train", "evaluate-training"):
        command = subcommands.add_parser(name)
        command.add_argument("--project-root", type=Path)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--training-config", type=Path)
        command.add_argument("--profile", choices=("baseline", "qwen35"), default="baseline")
        if name != "training-check":
            command.add_argument("--weights-dir", type=Path, required=True)
            command.add_argument("--run-id", required=True)
            command.add_argument("--resume", action="store_true")
        if name == "train":
            command.add_argument("--gpu", type=int, default=0)
            command.add_argument("--smoke-steps", type=int)
        if name == "evaluate-training":
            command.add_argument("--split", choices=("dev", "confirm"), required=True)
            command.add_argument("--adapter-dir", type=Path)
            add_inference_backend(command)
    for name, help_text in (("verify-run", "Verify a completed run without loading weights"),
                            ("evaluate", "Score a verified pilot run on CPU"),
                            ("submit", "Export a verified full test run; does not upload")):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--project-root", type=Path)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--run-id", required=True)
        command.add_argument("--profile", choices=("baseline", "qwen35"), default="baseline")
        if name == "verify-run":
            command.add_argument("--training-config", type=Path)
    args = parser.parse_args(argv)
    try:
        profile = getattr(args, "profile", "baseline")
        if args.command == "fetch-qwen35-weights":
            config = load_qwen35_config(args.project_root, args.data_root)
            require(re.fullmatch(r"[0-9a-f]{40}", args.revision) is not None, "Qwen3.5 revision must be a 40-character lowercase commit SHA")
            config["baseline"]["model"]["revision"] = args.revision
        elif profile == "qwen35":
            config = load_qwen35_config(args.project_root, args.data_root,
                                        require_revision=args.command in ("predict", "verify-run", "submit"))
        else:
            config = load_config(args.project_root, args.data_root, require_revision=args.command in ("predict", "fetch-weights", "train", "evaluate-training"))
        if args.command == "fetch-qwen35-weights":
            from cuhkx.inference.qwen35_weights import fetch_qwen35_weights
            source = fetch_qwen35_weights(args.weights_dir, config["baseline"]["model"])
            print(json.dumps({"status":"PASS","model_id":source["model_id"],"revision":source["revision"],
                              "receipt_sha256":source["receipt_sha256"]}, indent=2))
            return 0
        if args.command in ("training-check", "train", "evaluate-training"):
            from cuhkx.training.dataset import load_training_config, prepare_data
            settings = load_training_config(config, args.training_config)
            if args.command == "training-check":
                data = prepare_data(config,settings)
                result = {k:data[k] for k in ("status","coverage","cache_summary","missing_indices","data_signature")}
            elif args.command == "train":
                from cuhkx.training.trainer import train
                result = train(config,settings,args.weights_dir.resolve(),args.run_id,
                               resume=args.resume,gpu=args.gpu,smoke_steps=args.smoke_steps)
            else:
                from cuhkx.training.evaluate import evaluate_training
                result = evaluate_training(config,settings,args.split,args.run_id,args.weights_dir.resolve(),
                                           adapter=args.adapter_dir,resume=args.resume,
                                           backend=resolve_backend(args, profile),
                                           tensor_parallel_size=args.tensor_parallel_size,
                                           gpu_memory_utilization=args.gpu_memory_utilization,
                                           attention_backend=args.attention_backend
                                           or default_attention_backend(),
                                           max_num_seqs=args.max_num_seqs)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result["status"] == "PASS" else 2
        if args.command == "verify-run":
            from cuhkx.inference.runner import output_directory, verify_completed_run
            from cuhkx.inference.storage import run_lock
            with run_lock(output_directory(config, args.run_id)):
                prepared = None
                state = json.loads((output_directory(config,args.run_id)/"resume_state.json").read_text(encoding="utf-8"))
                if state["contract"]["dataset"] in ("dev","confirm"):
                    from cuhkx.training.dataset import load_training_config
                    from cuhkx.training.evaluate import evaluation_input
                    prepared,_ = evaluation_input(config,load_training_config(config,args.training_config),state["contract"]["dataset"])
                _, targets, _, summary, contract = verify_completed_run(config, args.run_id, prepared_input=prepared)
            print(json.dumps({"status": "PASS", "dataset": contract["dataset"], "target_qa": len(targets),
                              "execution_mode": contract["execution_mode"], "signature": summary["signature"]}, indent=2))
            return 0
        if args.command in ("evaluate", "submit"):
            from cuhkx.submission.export import evaluate_run, export_submission
            result = (evaluate_run if args.command == "evaluate" else export_submission)(config, args.run_id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "fetch-weights":
            from cuhkx.inference.weights import fetch_weights
            source = fetch_weights(args.weights_dir, config["baseline"]["model"])
            print(json.dumps({"status": "PASS", "model_id": source["model_id"],
                              "revision": source["revision"], "receipt_sha256": source["receipt_sha256"]}, indent=2))
            return 0
        if args.command == "predict":
            from cuhkx.inference.runner import output_directory, run_predictions

            if profile == "qwen35":
                from cuhkx.inference.qwen35_weights import verify_qwen35_weights
                source = verify_qwen35_weights(args.weights_dir, config["baseline"]["model"])
                if args.adapter_dir is not None:
                    from cuhkx.training.adapter import verify_adapter
                    source["adapter"] = verify_adapter(args.adapter_dir, source)
            else:
                from cuhkx.inference.weights import verify_weights
                source = verify_weights(args.weights_dir, config["baseline"]["model"])
                if args.adapter_dir is not None:
                    from cuhkx.training.adapter import verify_adapter
                    source["adapter"] = verify_adapter(args.adapter_dir, source)

            def backend_factory():
                return build_backend(profile, config, args, output_directory(config, args.run_id))

            result = run_predictions(config, args.dataset, args.limit, args.run_id, source,
                                     backend_factory, resume=args.resume,
                                     engine=resolve_backend(args, profile),
                                     engine_options=engine_options(args, profile))
            print(json.dumps({k: v for k, v in result.items() if k != "target_ids"}, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 1
        result = check_inputs(config, args.dataset, args.limit)
        output = prepare_run(config, result, args.run_id) if args.run_id else None
        compact = {key: value for key, value in result.items() if key not in ("target_ids", "selected_frames")}
        if output is not None:
            compact["prepared_run"] = str(output)
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, yaml.YAMLError) as error:
        import os
        if os.environ.get("CUHKX_TRACEBACK") == "1":
            import traceback
            traceback.print_exc()
        label = "Input check" if args.command == "check" else args.command
        print(f"{label} failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
