"""Portable cached-input validation. No raw-video or GPU dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from cuhkx.config import inside, require
from cuhkx.data.inputs import load_qa, read_csv, select_targets


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def check_inputs(config: dict, dataset: str, limit: int | None = None) -> dict:
    data = Path(config["data_root"])
    binding = config["datasets"]["datasets"][dataset]
    protocol = config["baseline"]["frames"]
    manifest_path = inside(data, config["datasets"]["asset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1, "unsupported asset manifest schema")
    allowed_asset_ids = {config["baseline"]["baseline_id"], config.get("asset_baseline_id", "")}
    require(manifest.get("baseline_id") in allowed_asset_ids, "asset baseline mismatch")
    files = {}
    for entry in manifest["files"]:
        # P1 manifest paths are project-relative. Only data/ assets are needed here.
        path = entry["path"]
        if path.startswith("data/"):
            relative = path.removeprefix("data/")
            require(relative not in files, f"duplicate manifest asset: {relative}")
            inside(data, relative)
            files[relative] = entry
    checked: dict[str, str] = {}

    def verified(path: Path) -> Path:
        require(path.is_relative_to(data), f"asset escapes data root: {path}")
        relative = path.relative_to(data).as_posix()
        if relative not in checked:
            require(relative in files, f"asset missing from manifest: {relative}")
            entry = files[relative]
            require(path.is_file() and path.stat().st_size == entry["bytes"], f"missing/size-changed asset: {relative}")
            digest = sha256(path)
            require(digest == entry["sha256"], f"asset hash mismatch: {relative}")
            checked[relative] = digest
        return path

    all_qa = load_qa(verified(inside(data, binding["qa"])), binding["expected_qa"])
    targets = select_targets(all_qa, limit)
    fields, index = read_csv(verified(inside(data, binding["frame_index"])))
    require({"sample_id", "qa_ids", "qa_count", "frame_paths", "metadata_path", "status", "modality",
             "config_hash", "protocol_version", "clip_key", "source", "split", "source_frame_indices",
             "normalized_positions", "target_timestamps_seconds", "actual_timestamps_seconds"} <= set(fields), "frame index columns missing")
    by_qa = {}
    samples = set()
    for row in index:
        require(row["sample_id"] and row["sample_id"] not in samples, "duplicate/empty cache sample ID")
        samples.add(row["sample_id"])
        # Do not filter out unsuccessful rows or silently intersect available QA IDs.
        for qa_id in row["qa_ids"].split(";"):
            require(qa_id and qa_id not in by_qa, f"duplicate/empty QA linkage: {qa_id}")
            by_qa[qa_id] = row
    require(all(row["qa_id"] in by_qa for row in targets), "requested QA has no cache entry")
    root = inside(data, binding["frames_root"])
    cache = {}
    all_images: set[Path] = set()
    selected_by_qa = {}
    for qa in targets:
        row = by_qa[qa["qa_id"]]
        sid = row["sample_id"]
        require(row["source"] == qa["source"], f"QA/cache source mismatch: {qa['qa_id']}")
        if sid not in cache:
            for field, expected in {"status": "ok", "modality": protocol["modality"],
                                    "config_hash": protocol["config_hash"], "protocol_version": protocol["protocol_version"],
                                    "split": binding["split"]}.items():
                require(row[field] == expected, f"invalid cache {field}: {sid}")
            meta_path = verified(inside(root, row["metadata_path"]))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for field in ("sample_id", "clip_key", "source", "split", "status", "modality", "config_hash", "protocol_version"):
                require(meta[field] == row[field], f"metadata mismatch {field}: {sid}")
            require(meta["qa_ids"] == row["qa_ids"].split(";") and
                    len(meta["qa_ids"]) == meta["qa_count"] == int(row["qa_count"]), f"metadata QA mismatch: {sid}")
            count = protocol["cache_num_frames"]
            require(meta["config"]["num_frames"] == meta["requested_frames"] == meta["actual_frames"] == count, f"wrong frame count: {sid}")
            require(meta["config"]["image_size"] == protocol["cache_image_size"] and
                    meta["config"]["resize_mode"] == protocol["cache_resize"], f"wrong image protocol: {sid}")
            for field in ("normalized_positions", "target_timestamps_seconds", "actual_timestamps_seconds"):
                values = [float(v) for v in row[field].split(";")]
                require(len(values) == len(meta[field]) == count and
                        all(abs(a - b) <= 0.000001 for a, b in zip(values, meta[field])), f"timeline mismatch: {sid}/{field}")
            require(meta["normalized_positions"] == [(i + 0.5) / count for i in range(count)], f"wrong sampling: {sid}")
            require(meta["actual_timestamps_seconds"] == sorted(meta["actual_timestamps_seconds"]), f"unordered timestamps: {sid}")
            require([int(x) for x in row["source_frame_indices"].split(";")] == meta["source_frame_indices"], f"frame indices mismatch: {sid}")
            paths = [verified(inside(root, value)) for value in row["frame_paths"].split(";")]
            require(len(paths) == len(set(paths)) == count, f"wrong cached frame list: {sid}")
            require(paths == [inside(meta_path.parent, value) for value in meta["frame_files"]], f"metadata frame paths mismatch: {sid}")
            require(not all_images.intersection(paths), f"frame reused by multiple clips: {sid}")
            for path in paths:
                with Image.open(path) as image:
                    require(image.format == "JPEG" and image.mode == "RGB" and
                            image.size == (protocol["cache_image_size"],) * 2, f"wrong image format: {path.name}")
                    image.load()
            all_images.update(paths)
            cache[sid] = [paths[i].relative_to(data).as_posix() for i in protocol["indices"]]
        selected_by_qa[qa["qa_id"]] = cache[sid]
    if dataset == "test":
        template_fields, template = read_csv(verified(inside(data, config["submission"]["template"])))
        require(template_fields == config["submission"]["columns"], "template schema mismatch")
        ids = [row["qa_id"] for row in template]
        require(len(ids) == len(set(ids)) and set(ids) == {row["qa_id"] for row in all_qa}, "template QA set mismatch")
    target_ids = [row["qa_id"] for row in targets]
    signature = {"baseline": config["baseline"], "dataset": dataset, "target_ids": target_ids,
                 "qa_content": targets, "selected_frames": selected_by_qa,
                 "verified_assets": checked}
    return {"status": "PASS", "kind": "cached_input_check", "dataset": dataset,
            "dataset_qa": len(all_qa), "target_qa": len(targets), "target_ids": target_ids,
            "cache_clips_checked": len(cache), "images_decoded": len(all_images),
            "assets_verified": len(checked), "input_indices": protocol["indices"],
            "selected_frames": selected_by_qa, "input_signature": fingerprint(signature),
            "model_revision_pinned": config["baseline"]["model"]["revision"] is not None,
            "gpu_model_loaded": False, "raw_videos_accessed": 0}
