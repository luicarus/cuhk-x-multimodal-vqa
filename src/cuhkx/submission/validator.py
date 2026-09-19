"""Strict validation for CUHK-X Kaggle submission CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from cuhkx.evaluation.metric import QA_CATEGORIES, assess_answer, available_option_letters


SUBMISSION_COLUMNS = ("qa_id", "prediction")
REQUIRED_TEST_COLUMNS = (
    "qa_id",
    "source",
    "path",
    "category",
    "question",
    "A",
    "B",
    "C",
    "D",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_submission_records(
    test_rows: Sequence[Mapping[str, Any]],
    template_rows: Sequence[Mapping[str, Any]],
    submission_rows: Sequence[Mapping[str, Any]],
    *,
    test_fields: Sequence[str] = REQUIRED_TEST_COLUMNS,
    template_fields: Sequence[str] = SUBMISSION_COLUMNS,
    submission_fields: Sequence[str] = SUBMISSION_COLUMNS,
) -> dict[str, Any]:
    """Validate schema, IDs, order, and category-aware prediction formats."""
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def add_check(
        code: str,
        passed: bool,
        observed: Any,
        expected: Any,
        detail: str,
    ) -> None:
        checks.append(
            {
                "code": code,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )
        if not passed:
            issues.append({"code": code, "detail": detail})

    expected_test_fields = set(REQUIRED_TEST_COLUMNS)
    missing_test_fields = sorted(expected_test_fields - set(test_fields))
    add_check(
        "test_schema",
        not missing_test_fields,
        list(test_fields),
        f"contains {list(REQUIRED_TEST_COLUMNS)}",
        f"test QA missing columns: {missing_test_fields}" if missing_test_fields else "complete",
    )
    add_check(
        "template_columns",
        tuple(template_fields) == SUBMISSION_COLUMNS,
        list(template_fields),
        list(SUBMISSION_COLUMNS),
        "sample submission columns must be exact and ordered",
    )
    add_check(
        "submission_columns",
        tuple(submission_fields) == SUBMISSION_COLUMNS,
        list(submission_fields),
        list(SUBMISSION_COLUMNS),
        "candidate columns must exactly match sample_submission.csv",
    )

    unknown_categories = sorted(
        {str(row.get("category", "") or "").strip().casefold() for row in test_rows}
        - set(QA_CATEGORIES)
    )
    add_check(
        "test_categories",
        not unknown_categories,
        unknown_categories,
        list(QA_CATEGORIES),
        "test QA categories must use the known competition taxonomy",
    )
    invalid_option_layout: list[str] = []
    duplicate_option_rows: list[str] = []
    for row in test_rows:
        qa_id = str(row.get("qa_id", "") or "").strip()
        source = str(row.get("source", "") or "").strip()
        category = str(row.get("category", "") or "").strip().casefold()
        options = [str(row.get(letter, "") or "").strip() for letter in "ABCD"]
        d_is_optional = source == "HARn" and category == "single"
        if any(not option for option in options[:3]) or (not options[3] and not d_is_optional):
            invalid_option_layout.append(qa_id)
        normalized_options = [" ".join(option.casefold().split()) for option in options if option]
        if len(normalized_options) != len(set(normalized_options)):
            duplicate_option_rows.append(qa_id)
    add_check(
        "test_option_layout",
        not invalid_option_layout,
        len(invalid_option_layout),
        0,
        f"invalid option-layout examples: {invalid_option_layout[:10]}",
    )
    add_check(
        "test_unique_options",
        not duplicate_option_rows,
        len(duplicate_option_rows),
        0,
        f"duplicate option examples: {duplicate_option_rows[:10]}",
    )

    test_ids = [str(row.get("qa_id", "") or "").strip() for row in test_rows]
    template_ids = [str(row.get("qa_id", "") or "").strip() for row in template_rows]
    submission_ids = [str(row.get("qa_id", "") or "").strip() for row in submission_rows]

    for label, ids in (
        ("test", test_ids),
        ("template", template_ids),
        ("submission", submission_ids),
    ):
        blank_count = sum(not qa_id for qa_id in ids)
        duplicates = sorted(qa_id for qa_id, count in Counter(ids).items() if qa_id and count > 1)
        add_check(
            f"{label}_nonempty_ids",
            blank_count == 0,
            blank_count,
            0,
            f"{label} contains {blank_count} blank qa_id value(s)",
        )
        add_check(
            f"{label}_unique_ids",
            not duplicates,
            len(duplicates),
            0,
            f"{label} duplicate qa_id examples: {duplicates[:10]}",
        )

    add_check(
        "template_matches_test_order",
        template_ids == test_ids,
        len(template_ids),
        len(test_ids),
        "sample_submission qa_id values must match test_qa.csv in exact order",
    )
    add_check(
        "submission_row_count",
        len(submission_rows) == len(template_rows),
        len(submission_rows),
        len(template_rows),
        "candidate must contain exactly one row per template row",
    )

    missing_ids = sorted(set(template_ids) - set(submission_ids))
    unexpected_ids = sorted(set(submission_ids) - set(template_ids))
    add_check(
        "submission_id_set",
        not missing_ids and not unexpected_ids,
        {"missing": len(missing_ids), "unexpected": len(unexpected_ids)},
        {"missing": 0, "unexpected": 0},
        f"missing examples={missing_ids[:10]}; unexpected examples={unexpected_ids[:10]}",
    )
    first_order_mismatch = next(
        (
            index
            for index, (actual, expected) in enumerate(
                zip(submission_ids, template_ids, strict=False), start=1
            )
            if actual != expected
        ),
        None,
    )
    order_matches = submission_ids == template_ids
    add_check(
        "submission_id_order",
        order_matches,
        "exact" if order_matches else f"first mismatch at data row {first_order_mismatch}",
        "exact template order",
        "candidate qa_id order must match sample_submission.csv",
    )

    test_index: dict[str, Mapping[str, Any]] = {}
    for qa_id, row in zip(test_ids, test_rows, strict=True):
        if qa_id and qa_id not in test_index:
            test_index[qa_id] = row

    invalid_predictions: list[dict[str, str]] = []
    canonical_predictions = 0
    category_counts: Counter[str] = Counter()
    for row, qa_id in zip(submission_rows, submission_ids, strict=True):
        test_row = test_index.get(qa_id)
        if test_row is None:
            continue
        category = str(test_row.get("category", "") or "").strip().casefold()
        category_counts[category] += 1
        raw_prediction = "" if row.get("prediction") is None else str(row.get("prediction"))
        assessed = assess_answer(
            raw_prediction,
            category,
            available_option_letters(test_row),
        )
        if not assessed.is_valid:
            if len(invalid_predictions) < 25:
                invalid_predictions.append(
                    {
                        "qa_id": qa_id,
                        "category": category,
                        "prediction": raw_prediction,
                        "error": assessed.error or "invalid prediction",
                    }
                )
            continue
        if raw_prediction != assessed.canonical:
            if len(invalid_predictions) < 25:
                invalid_predictions.append(
                    {
                        "qa_id": qa_id,
                        "category": category,
                        "prediction": raw_prediction,
                        "error": f"non-canonical; expected {assessed.canonical}",
                    }
                )
            continue
        canonical_predictions += 1

    invalid_count = len(submission_rows) - canonical_predictions
    add_check(
        "prediction_format",
        invalid_count == 0,
        invalid_count,
        0,
        "predictions must be uppercase, separator-free, category-valid, and canonical; "
        f"examples={invalid_predictions[:5]}",
    )

    valid = all(check["status"] == "PASS" for check in checks)
    return {
        "status": "PASS" if valid else "FAIL",
        "valid": valid,
        "rows": len(submission_rows),
        "expected_rows": len(template_rows),
        "canonical_predictions": canonical_predictions,
        "invalid_predictions": invalid_count,
        "invalid_prediction_examples": invalid_predictions,
        "category_rows": dict(sorted(category_counts.items())),
        "checks": checks,
        "issues": issues,
    }


def validate_submission_files(
    submission_path: Path,
    test_qa_path: Path = Path("data/raw/Testing/test_qa.csv"),
    template_path: Path = Path("data/raw/Testing/sample_submission.csv"),
) -> dict[str, Any]:
    """Read and validate one candidate submission against the official local files."""
    submission_path = submission_path.resolve()
    test_qa_path = test_qa_path.resolve()
    template_path = template_path.resolve()
    test_fields, test_rows = _read_csv(test_qa_path)
    template_fields, template_rows = _read_csv(template_path)
    submission_fields, submission_rows = _read_csv(submission_path)
    report = validate_submission_records(
        test_rows,
        template_rows,
        submission_rows,
        test_fields=test_fields,
        template_fields=template_fields,
        submission_fields=submission_fields,
    )
    report["inputs"] = {
        "submission": {
            "path": str(submission_path),
            "bytes": submission_path.stat().st_size,
            "sha256": _sha256(submission_path),
        },
        "test_qa": {
            "path": str(test_qa_path),
            "bytes": test_qa_path.stat().st_size,
            "sha256": _sha256(test_qa_path),
        },
        "template": {
            "path": str(template_path),
            "bytes": template_path.stat().st_size,
            "sha256": _sha256(template_path),
        },
    }
    return report


def _print_human_report(report: Mapping[str, Any]) -> None:
    print(f"Submission validation: {report['status']}")
    print(f"Rows: {report['rows']}/{report['expected_rows']}")
    print(
        f"Canonical predictions: {report['canonical_predictions']}; "
        f"invalid: {report['invalid_predictions']}"
    )
    for check in report["checks"]:
        print(f"  {check['status']:4} {check['code']}: {check['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_path",
        nargs="?",
        type=Path,
        help="Candidate CSV path (may also be provided with --submission).",
    )
    parser.add_argument(
        "--submission",
        dest="submission_option",
        type=Path,
        help="Candidate CSV path.",
    )
    parser.add_argument("--test-qa", type=Path, default=Path("data/raw/Testing/test_qa.csv"))
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/raw/Testing/sample_submission.csv"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.submission_path is not None and args.submission_option is not None:
        parser.error("provide the candidate either positionally or with --submission, not both")
    submission_path = args.submission_option or args.submission_path
    if submission_path is None:
        parser.error("a candidate submission CSV is required")

    try:
        report = validate_submission_files(submission_path, args.test_qa, args.template)
    except FileNotFoundError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human_report(report)
    return 0 if report["valid"] else 1


__all__ = [
    "REQUIRED_TEST_COLUMNS",
    "SUBMISSION_COLUMNS",
    "main",
    "validate_submission_files",
    "validate_submission_records",
]
