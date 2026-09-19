"""Answer-free inputs and stable target selection before checkpoint filtering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from cuhkx.config import require
from cuhkx.evaluation.metric import QA_CATEGORIES, available_option_letters


QA_FIELDS = ("qa_id", "source", "path", "category", "question", "A", "B", "C", "D")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        require(bool(fields) and len(fields) == len(set(fields)), f"empty/duplicate CSV columns: {path}")
        rows = list(reader)
    require(bool(rows), f"empty CSV: {path}")
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows), f"malformed CSV: {path}")
    return fields, rows


def load_qa(path: Path, expected_count: int) -> list[dict[str, str]]:
    fields, rows = read_csv(path)
    require(tuple(fields) == QA_FIELDS, "inference QA must contain exactly the answer-free field whitelist")
    ids = [row["qa_id"] for row in rows]
    require(all(q.strip() == q and q for q in ids) and len(ids) == len(set(ids)), "empty or duplicate QA IDs")
    require(len(rows) == expected_count, f"QA count {len(rows)} differs from expected {expected_count}")
    for row in rows:
        require(row["category"] in QA_CATEGORIES, f"unknown category: {row['qa_id']}")
        require(bool(row["question"].strip()) and bool(available_option_letters(row)), f"empty question/options: {row['qa_id']}")
    return rows


def select_targets(rows: Sequence[dict[str, str]], limit: int | None = None) -> list[dict[str, str]]:
    require(limit is None or (type(limit) is int and 0 < limit <= len(rows)), "limit must be within the dataset size")
    return list(rows if limit is None else rows[:limit])


def pending_targets(targets: Sequence[dict[str, str]], completed_ids: set[str]) -> list[dict[str, str]]:
    """Only call after the same run's checkpoint signature has been verified."""
    ids = {row["qa_id"] for row in targets}
    require(completed_ids <= ids, "checkpoint contains IDs outside the fixed target subset")
    return [row for row in targets if row["qa_id"] not in completed_ids]
