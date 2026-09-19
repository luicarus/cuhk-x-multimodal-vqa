from __future__ import annotations

from copy import deepcopy

import pytest

from cuhkx.submission.validator import (
    REQUIRED_TEST_COLUMNS,
    SUBMISSION_COLUMNS,
    validate_submission_records,
)


def _test_rows() -> list[dict[str, str]]:
    base = {
        "source": "HAU",
        "path": "clip",
        "question": "Question?",
        "A": "one",
        "B": "two",
        "C": "three",
        "D": "four",
    }
    return [
        {**base, "qa_id": "q1", "source": "HARn", "category": "single", "D": ""},
        {**base, "qa_id": "q2", "category": "multi"},
        {**base, "qa_id": "q3", "category": "sequence"},
        {**base, "qa_id": "q4", "category": "emotion"},
    ]


def _template_rows() -> list[dict[str, str]]:
    return [
        {"qa_id": "q1", "prediction": "A"},
        {"qa_id": "q2", "prediction": "AC"},
        {"qa_id": "q3", "prediction": "ABDC"},
        {"qa_id": "q4", "prediction": "D"},
    ]


def _validate(
    rows: list[dict[str, str]],
    *,
    submission_fields: tuple[str, ...] = SUBMISSION_COLUMNS,
) -> dict[str, object]:
    return validate_submission_records(
        _test_rows(),
        _template_rows(),
        rows,
        test_fields=REQUIRED_TEST_COLUMNS,
        template_fields=SUBMISSION_COLUMNS,
        submission_fields=submission_fields,
    )


def _failed_codes(report: dict[str, object]) -> set[str]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return {check["code"] for check in checks if check["status"] == "FAIL"}


def test_valid_submission_passes() -> None:
    report = _validate(_template_rows())

    assert report["valid"] is True
    assert report["status"] == "PASS"
    assert report["invalid_predictions"] == 0


@pytest.mark.parametrize(
    ("qa_id", "prediction"),
    [
        ("q1", "D"),
        ("q1", "a"),
        ("q1", " A"),
        ("q2", "CA"),
        ("q2", "ACC"),
        ("q2", "A C"),
        ("q3", "ABC"),
        ("q3", "AABC"),
        ("q4", ""),
        ("q4", "E"),
    ],
)
def test_invalid_or_noncanonical_predictions_fail(qa_id: str, prediction: str) -> None:
    rows = _template_rows()
    next(row for row in rows if row["qa_id"] == qa_id)["prediction"] = prediction

    report = _validate(rows)

    assert report["valid"] is False
    assert "prediction_format" in _failed_codes(report)


def test_submission_order_must_match_template() -> None:
    rows = _template_rows()
    rows[0], rows[1] = rows[1], rows[0]

    report = _validate(rows)

    assert "submission_id_order" in _failed_codes(report)


def test_submission_ids_must_be_unique_and_complete() -> None:
    rows = _template_rows()
    rows[3] = deepcopy(rows[0])

    report = _validate(rows)
    failed = _failed_codes(report)

    assert "submission_unique_ids" in failed
    assert "submission_id_set" in failed


def test_submission_row_count_must_match_template() -> None:
    report = _validate(_template_rows()[:-1])

    assert "submission_row_count" in _failed_codes(report)


def test_submission_columns_are_exact_and_ordered() -> None:
    report = _validate(
        _template_rows(),
        submission_fields=("prediction", "qa_id", "Unnamed: 0"),
    )

    assert "submission_columns" in _failed_codes(report)


def test_duplicate_prediction_values_are_allowed() -> None:
    rows = _template_rows()
    rows[0]["prediction"] = "A"
    rows[1]["prediction"] = "A"
    rows[2]["prediction"] = "ABCD"
    rows[3]["prediction"] = "A"

    report = _validate(rows)

    assert report["valid"] is True
