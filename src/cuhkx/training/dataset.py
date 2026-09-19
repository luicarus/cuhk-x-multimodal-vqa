"""Read existing frame caches and subject folds; never extract or split randomly."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from cuhkx.config import inside, keys, read_yaml, require, same
from cuhkx.data.inputs import QA_FIELDS, read_csv
from cuhkx.data.validate import fingerprint, sha256
from cuhkx.evaluation.metric import available_option_letters, canonicalize_answer
from cuhkx.inference.prompt import build_mcq_prompt, detect_prompt_leakage


def load_training_config(config, path=None):
    project = Path(config["project_root"])
    if path is None:
        filename = ("training_qwen35.yaml" if config["baseline"]["model"]["id"] == "Qwen/Qwen3.5-4B"
                    else "training.yaml")
        path = project / "configs" / filename
    if not path.is_absolute():
        path = project / path
    value = read_yaml(path)
    keys(value, {"schema_version", "profile", "qa", "folds", "pilot", "fold_version", "splits", "caches", "lora", "optimizer"}, "training")
    same(value["schema_version"], 1, "training schema")
    expected_profile = "qwen35_4b_qlora_v1" if config["baseline"]["model"]["id"] == "Qwen/Qwen3.5-4B" else "ir4_7b_qlora_v1"
    same(value["profile"], expected_profile, "training profile")
    same(value["fold_version"], "subject_grouped_v1", "fold version")
    same(value["splits"], {"train": [0, 1, 2], "dev": [3], "confirm": [4]}, "training splits")
    data = Path(config["data_root"])
    for field in ("qa", "folds", "pilot"):
        inside(data, value[field])
    require(isinstance(value["caches"], list) and value["caches"], "no cache bindings")
    for cache in value["caches"]:
        keys(cache, {"frames_root", "frame_index"}, "cache")
        inside(data, cache["frames_root"])
        inside(data, cache["frame_index"])
    keys(value["lora"], {"r", "alpha", "dropout"}, "LoRA")
    require(type(value["lora"]["r"]) is int and 0 < value["lora"]["r"] <= 64, "invalid LoRA rank")
    require(type(value["lora"]["alpha"]) is int and value["lora"]["alpha"] > 0, "invalid LoRA alpha")
    require(type(value["lora"]["dropout"]) in (int, float) and 0 <= value["lora"]["dropout"] < 1, "invalid dropout")
    opt = value["optimizer"]
    keys(opt, {"learning_rate", "epochs", "gradient_accumulation_steps", "warmup_ratio", "max_grad_norm", "seed", "max_sequence_length"}, "optimizer")
    for field in ("epochs", "gradient_accumulation_steps", "max_sequence_length", "seed"):
        require(type(opt[field]) is int and opt[field] > 0, f"invalid {field}")
    require(1 <= opt["epochs"] <= 2, "first training cycle supports one or two epochs")
    require(0 < opt["learning_rate"] <= 0.001 and 0 <= opt["warmup_ratio"] < 1 and opt["max_grad_norm"] > 0, "invalid optimizer parameters")
    return value


def prepare_data(config, training):
    data = Path(config["data_root"])
    qpath, fpath, ppath = [inside(data, training[k]) for k in ("qa", "folds", "pilot")]
    fields, qa = read_csv(qpath)
    require(set(fields) == {*QA_FIELDS, "answer"}, "training QA schema must include exactly one answer column")
    _, folds = read_csv(fpath)
    _, pilot = read_csv(ppath)
    by_id = {row["qa_id"]: row for row in qa}
    by_fold = {row["qa_id"]: row for row in folds}
    pilot_ids = {row["qa_id"] for row in pilot}
    require(len(by_id) == len(qa) and len(by_fold) == len(folds) and len(pilot_ids) == len(pilot), "duplicate QA IDs")
    require(set(by_id) == set(by_fold) and pilot_ids <= set(by_id), "QA/fold/pilot IDs do not align")
    subjects, clips, rows = {}, {}, {name: [] for name in training["splits"]}
    for row in qa:
        qid = row["qa_id"]
        require(bool(qid.strip()) and qid == qid.strip(), "blank QA ID")
        fold = by_fold[qid]
        same(fold["fold_version"], training["fold_version"], "fold version")
        number = int(fold["fold"])
        require(number in range(5) and fold["subject_id"] and fold["clip_key"].startswith("train:"), "invalid subject/clip/fold")
        for groups, name in ((subjects, "subject_id"), (clips, "clip_key")):
            previous = groups.setdefault(fold[name], number)
            require(previous == number, f"{name} leaks across folds: {fold[name]}")
        require(fold["source"] == row["source"] and fold["category"] == row["category"], "QA/fold fields differ")
        original_path = row["path"].replace("\\", "/").strip().rstrip("/")
        if original_path.lower().endswith(".mp4"):
            original_path = "/".join(original_path.split("/")[:-2])
        require(fold["clip_key"] == "train:" + original_path, "QA path/fold clip mismatch")
        answer = canonicalize_answer(row["answer"], row["category"], available_option_letters(row))
        clean = {key: row[key] for key in QA_FIELDS}
        require(not detect_prompt_leakage(build_mcq_prompt(clean), clean), f"prompt leakage: {qid}")
        if qid in pilot_ids:
            require(number == 4, "pilot must remain in fold 4")
            continue
        split = next(name for name, numbers in training["splits"].items() if number in numbers)
        rows[split].append({"qa": clean, "answer": answer, "subject_id": fold["subject_id"],
                            "clip_key": fold["clip_key"], "fold": number})
    linked, hashes, missing_indices = {}, {}, []
    asset_manifest = json.loads(inside(data, config["datasets"]["asset_manifest"]).read_text(encoding="utf-8"))
    registered = {}
    for entry in asset_manifest["files"]:
        if entry["path"].startswith("data/"):
            relative = entry["path"][5:]
            require(relative not in registered, "duplicate registered asset")
            registered[relative] = entry
    images_decoded = 0

    def track(path):
        digest = sha256(path)
        relative = path.relative_to(data).as_posix()
        if relative in registered:
            expected = registered[relative]
            require(path.stat().st_size == expected["bytes"] and digest == expected["sha256"], f"registered training asset changed: {relative}")
        hashes[relative] = digest
        return digest

    for path in (qpath, fpath, ppath):
        track(path)
    seen_clips = set()
    for binding in training["caches"]:
        index_path = inside(data, binding["frame_index"])
        if not index_path.is_file():
            missing_indices.append(binding["frame_index"])
            continue
        track(index_path)
        root = inside(data, binding["frames_root"])
        _, entries = read_csv(index_path)
        for entry in entries:
            require(entry["status"] == "ok" and entry["split"] == "train" and entry["modality"] == "IR", "cache must be successful training IR")
            clip = entry["clip_key"]
            require(clip not in seen_clips, "duplicate clip in cache bindings")
            seen_clips.add(clip)
            meta_path = inside(root, entry["metadata_path"])
            track(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for field in ("clip_key", "sample_id", "source", "modality", "split", "status", "config_hash", "protocol_version"):
                require(meta[field] == entry[field], f"cache metadata mismatch: {field}")
            protocol = config["baseline"]["frames"]
            require(meta["config_hash"] == protocol["config_hash"] and meta["protocol_version"] == protocol["protocol_version"], "incompatible frame protocol")
            require(meta["actual_frames"] == meta["requested_frames"] == meta["config"]["num_frames"] == 8, "eight frames required")
            require(meta["config"]["image_size"] == 448 and meta["config"]["resize_mode"] == "letterbox", "wrong cached image protocol")
            require(meta["normalized_positions"] == [(i+0.5)/8 for i in range(8)], "wrong temporal sampling")
            for field in ("normalized_positions", "target_timestamps_seconds", "actual_timestamps_seconds"):
                values = [float(v) for v in entry[field].split(";")]
                require(len(values) == len(meta[field]) == 8 and all(abs(a-b) < 1e-6 for a,b in zip(values,meta[field])), "timeline mismatch")
            require(meta["actual_timestamps_seconds"] == sorted(meta["actual_timestamps_seconds"]), "unordered frames")
            require(meta["source_frame_indices"] == [int(v) for v in entry["source_frame_indices"].split(";")], "source frame indices mismatch")
            images = [inside(root, name) for name in entry["frame_paths"].split(";")]
            require(len(images) == len(set(images)) == 8, "cache frame count differs")
            require(images == [inside(meta_path.parent, name) for name in meta["frame_files"]], "frame paths differ from metadata")
            for path in images:
                track(path)
                with Image.open(path) as image:
                    require(image.format == "JPEG" and image.mode == "RGB" and image.size == (448,448), "bad cached image")
                    image.load()
                images_decoded += 1
            ids = entry["qa_ids"].split(";")
            require(ids == meta["qa_ids"] and len(ids) == meta["qa_count"] == int(entry["qa_count"]), "cache QA list differs")
            for qid in ids:
                require(qid in by_id and qid not in linked, "unknown/duplicate cached QA")
                require(by_fold[qid]["clip_key"] == clip and by_id[qid]["source"] == entry["source"], "cache is joined to the wrong QA clip/source")
                linked[qid] = [path.relative_to(data).as_posix() for i,path in enumerate(images) if i in protocol["indices"]]
    coverage = {}
    available = {}
    for split, samples in rows.items():
        missing = [sample["qa"]["qa_id"] for sample in samples if sample["qa"]["qa_id"] not in linked]
        available[split] = [{**sample, "frame_paths": linked[sample["qa"]["qa_id"]]} for sample in samples if sample["qa"]["qa_id"] in linked]
        coverage[split] = {"expected_qa": len(samples), "available_qa": len(available[split]),
                           "missing_qa": len(missing), "missing_examples": missing[:10]}
    identity = {"fold_version": training["fold_version"], "inputs": hashes,
                "splits": {k: [s["qa"]["qa_id"] for s in v] for k,v in rows.items()}, "pilot_excluded": sorted(pilot_ids)}
    return {"status": "PASS" if all(not c["missing_qa"] for c in coverage.values()) else "INCOMPLETE",
            "coverage": coverage, "missing_indices": missing_indices,
            "cache_summary": {"linked_qa":len(linked), "clips":len(seen_clips), "images_decoded":images_decoded},
            "data_signature": fingerprint(identity), "identity": identity, "samples": available}


def require_ready(prepared, splits):
    for split in splits:
        c = prepared["coverage"][split]
        require(c["expected_qa"] > 0 and c["missing_qa"] == 0, f"{split}: missing {c['missing_qa']} of {c['expected_qa']} QA frame caches; supply existing IR8 caches first")


class TrainingDataset:
    def __init__(self, samples, data_root):
        self.samples, self.data_root = samples, Path(data_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {**sample, "frame_paths": [inside(self.data_root, path) for path in sample["frame_paths"]]}
