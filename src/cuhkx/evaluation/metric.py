"""Locally reconstructed CUHK-X exact-match metric and answer-format rules."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


ANSWER_LETTERS = "ABCD"
QA_CATEGORIES = (
    "single",
    "multi",
    "combination",
    "sequence",
    "object_interaction",
    "emotion",
)
SINGLE_ANSWER_CATEGORIES = frozenset({"single", "combination", "object_interaction", "emotion"})


class AnswerFormatError(ValueError):
    """Raised when an answer cannot represent a legal choice for its question."""


class MetricInputError(ValueError):
    """Raised when references and predictions cannot be aligned safely."""


@dataclass(frozen=True)
class AnswerAssessment:
    raw: str
    canonical: str | None
    is_valid: bool
    error: str | None


def available_option_letters(row: Mapping[str, Any]) -> str:
    """Return option labels whose text is non-empty, in canonical A-D order."""
    return "".join(letter for letter in ANSWER_LETTERS if str(row.get(letter, "") or "").strip())


def canonicalize_answer(
    value: Any,
    category: str,
    valid_options: str = ANSWER_LETTERS,
) -> str:
    """Validate and canonicalize one CUHK-X answer.

    Multi-answer choices are treated as a set and sorted into A-D order. Sequence
    choices retain their order and must be a complete permutation of all available
    options. Other categories require exactly one available option letter.
    """
    normalized_category = str(category).strip().casefold()
    if normalized_category not in QA_CATEGORIES:
        raise AnswerFormatError(f"unknown category: {category!r}")

    ordered_options = "".join(
        letter for letter in ANSWER_LETTERS if letter in str(valid_options).upper()
    )
    if not ordered_options:
        raise AnswerFormatError("question has no available options")

    raw = "" if value is None else str(value)
    answer = raw.strip().upper()
    if not answer:
        raise AnswerFormatError("prediction is empty")
    if re.fullmatch(r"[A-D]+", answer) is None:
        raise AnswerFormatError("prediction must contain only letters A-D")
    if len(set(answer)) != len(answer):
        raise AnswerFormatError("prediction contains duplicate option letters")

    unavailable = sorted(set(answer) - set(ordered_options))
    if unavailable:
        raise AnswerFormatError(f"prediction selects unavailable option(s): {''.join(unavailable)}")

    if normalized_category in SINGLE_ANSWER_CATEGORIES:
        if len(answer) != 1:
            raise AnswerFormatError(f"{normalized_category} requires exactly one option")
        return answer

    if normalized_category == "multi":
        return "".join(letter for letter in ANSWER_LETTERS if letter in answer)

    if len(answer) != len(ordered_options) or set(answer) != set(ordered_options):
        raise AnswerFormatError(
            f"sequence requires a permutation of all options: {ordered_options}"
        )
    return answer


def assess_answer(
    value: Any,
    category: str,
    valid_options: str = ANSWER_LETTERS,
) -> AnswerAssessment:
    """Return a non-raising answer-format assessment."""
    raw = "" if value is None else str(value)
    try:
        canonical = canonicalize_answer(raw, category, valid_options)
    except AnswerFormatError as error:
        return AnswerAssessment(raw=raw, canonical=None, is_valid=False, error=str(error))
    return AnswerAssessment(raw=raw, canonical=canonical, is_valid=True, error=None)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _index_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    id_field: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    ids = [str(row.get(id_field, "") or "").strip() for row in rows]
    empty_count = sum(not qa_id for qa_id in ids)
    duplicates = sorted(qa_id for qa_id, count in Counter(ids).items() if qa_id and count > 1)
    if empty_count:
        raise MetricInputError(f"{label} contains {empty_count} empty {id_field} value(s)")
    if duplicates:
        raise MetricInputError(f"{label} contains duplicate {id_field}: {duplicates[:5]}")
    return {qa_id: row for qa_id, row in zip(ids, rows, strict=True)}


def _bucket(correct: int, total: int) -> dict[str, int | float]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def evaluate_records(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    answer_field: str = "answer",
    prediction_field: str = "prediction",
    id_field: str = "qa_id",
) -> dict[str, Any]:
    """Evaluate predictions by QA ID with category-aware exact match.

    Invalid predictions receive zero credit and are reported. Invalid reference
    answers and structural ID mismatches raise ``MetricInputError`` because a score
    would be unsafe to interpret.
    """
    reference_index = _index_unique_rows(references, id_field, "references")
    prediction_index = _index_unique_rows(predictions, id_field, "predictions")
    reference_ids = set(reference_index)
    prediction_ids = set(prediction_index)
    if reference_ids != prediction_ids:
        missing = sorted(reference_ids - prediction_ids)
        unexpected = sorted(prediction_ids - reference_ids)
        raise MetricInputError(
            "prediction IDs do not match references: "
            f"missing={missing[:5]} ({len(missing)}), "
            f"unexpected={unexpected[:5]} ({len(unexpected)})"
        )

    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    source_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    source_category_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    clip_scores: dict[str, list[int]] = defaultdict(list)
    invalid_examples: list[dict[str, str]] = []
    correct_total = 0
    invalid_total = 0

    for reference in references:
        qa_id = str(reference.get(id_field, "") or "").strip()
        prediction = prediction_index[qa_id]
        category = str(reference.get("category", "") or "").strip().casefold()
        source = str(reference.get("source", "") or "").strip() or "<missing>"
        valid_options = available_option_letters(reference)
        try:
            expected = canonicalize_answer(reference.get(answer_field), category, valid_options)
        except AnswerFormatError as error:
            raise MetricInputError(f"invalid reference answer for {qa_id}: {error}") from error

        assessed = assess_answer(prediction.get(prediction_field), category, valid_options)
        is_correct = int(assessed.is_valid and assessed.canonical == expected)
        correct_total += is_correct
        if not assessed.is_valid:
            invalid_total += 1
            if len(invalid_examples) < 20:
                invalid_examples.append(
                    {
                        "qa_id": qa_id,
                        "category": category,
                        "prediction": assessed.raw,
                        "error": assessed.error or "invalid prediction",
                    }
                )

        for bucket in (
            category_counts[category],
            source_counts[source],
            source_category_counts[(source, category)],
        ):
            bucket[0] += is_correct
            bucket[1] += 1
        clip_key = str(reference.get("path", "") or qa_id).strip() or qa_id
        clip_scores[clip_key].append(is_correct)

    total = len(references)
    return {
        "metric": "cuhkx_local_question_accuracy",
        "total": total,
        "correct": correct_total,
        "overall_accuracy": correct_total / total if total else 0.0,
        "invalid_predictions": invalid_total,
        "invalid_rate": invalid_total / total if total else 0.0,
        "invalid_examples": invalid_examples,
        "clip_macro_accuracy": mean(sum(values) / len(values) for values in clip_scores.values())
        if clip_scores
        else 0.0,
        "clips": len(clip_scores),
        "per_category": {
            category: _bucket(*category_counts[category])
            for category in QA_CATEGORIES
            if category in category_counts
        },
        "per_source": {
            source: _bucket(*counts) for source, counts in sorted(source_counts.items())
        },
        "per_source_category": {
            f"{source}/{category}": _bucket(*counts)
            for (source, category), counts in sorted(source_category_counts.items())
        },
    }


def evaluate_csv(
    ground_truth_path: Path,
    predictions_path: Path,
    *,
    answer_field: str = "answer",
    prediction_field: str = "prediction",
) -> dict[str, Any]:
    """Read two CSV files and evaluate their rows by ``qa_id``."""
    reference_fields, references = _read_csv(ground_truth_path)
    prediction_fields, predictions = _read_csv(predictions_path)
    required_reference_fields = {"qa_id", "category", answer_field, "A", "B", "C", "D"}
    missing_reference_fields = sorted(required_reference_fields - set(reference_fields))
    if missing_reference_fields:
        raise MetricInputError(f"ground truth is missing columns: {missing_reference_fields}")
    required_prediction_fields = {"qa_id", prediction_field}
    missing_prediction_fields = sorted(required_prediction_fields - set(prediction_fields))
    if missing_prediction_fields:
        raise MetricInputError(f"predictions are missing columns: {missing_prediction_fields}")
    return evaluate_records(
        references,
        predictions,
        answer_field=answer_field,
        prediction_field=prediction_field,
    )


def _print_human_report(result: Mapping[str, Any]) -> None:
    print(f"CUHK-X accuracy: {result['overall_accuracy']:.6f}")
    print(f"Correct: {result['correct']}/{result['total']}")
    print(f"Invalid predictions: {result['invalid_predictions']} ({result['invalid_rate']:.2%})")
    print(f"Clip-macro accuracy: {result['clip_macro_accuracy']:.6f}")
    print("Per category:")
    for category, stats in result["per_category"].items():
        print(f"  {category}: {stats['accuracy']:.6f} ({stats['correct']}/{stats['total']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/raw/Training/training_qa.csv"),
    )
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Return a non-zero exit code when any prediction has an invalid format.",
    )
    args = parser.parse_args(argv)

    try:
        result = evaluate_csv(
            args.ground_truth,
            args.predictions,
            answer_field=args.answer_column,
            prediction_field=args.prediction_column,
        )
    except (FileNotFoundError, MetricInputError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human_report(result)
    return 1 if args.fail_on_invalid and result["invalid_predictions"] else 0


__all__ = [
    "ANSWER_LETTERS",
    "QA_CATEGORIES",
    "SINGLE_ANSWER_CATEGORIES",
    "AnswerAssessment",
    "AnswerFormatError",
    "MetricInputError",
    "assess_answer",
    "available_option_letters",
    "canonicalize_answer",
    "evaluate_csv",
    "evaluate_records",
    "main",
]
