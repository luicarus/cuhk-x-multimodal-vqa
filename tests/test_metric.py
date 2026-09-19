from __future__ import annotations

import pytest

from cuhkx.evaluation.metric import (
    AnswerFormatError,
    MetricInputError,
    canonicalize_answer,
    evaluate_records,
)


@pytest.mark.parametrize(
    ("value", "category", "valid_options", "expected"),
    [
        (" a ", "single", "ABC", "A"),
        ("CA", "multi", "ABCD", "AC"),
        ("A", "multi", "ABCD", "A"),
        ("CADB", "sequence", "ABCD", "CADB"),
        ("D", "emotion", "ABCD", "D"),
    ],
)
def test_canonicalize_answer(value: str, category: str, valid_options: str, expected: str) -> None:
    assert canonicalize_answer(value, category, valid_options) == expected


@pytest.mark.parametrize(
    ("value", "category", "valid_options"),
    [
        ("", "single", "ABCD"),
        ("AA", "single", "ABCD"),
        ("AB", "single", "ABCD"),
        ("D", "single", "ABC"),
        ("A,C", "multi", "ABCD"),
        ("ACC", "multi", "ABCD"),
        ("ABC", "sequence", "ABCD"),
        ("AABC", "sequence", "ABCD"),
        ("ABCE", "sequence", "ABCD"),
        ("A", "unknown", "ABCD"),
    ],
)
def test_canonicalize_answer_rejects_invalid_values(
    value: str, category: str, valid_options: str
) -> None:
    with pytest.raises(AnswerFormatError):
        canonicalize_answer(value, category, valid_options)


def _reference(
    qa_id: str,
    category: str,
    answer: str,
    *,
    path: str,
    source: str = "HAU",
    d_option: str = "four",
) -> dict[str, str]:
    return {
        "qa_id": qa_id,
        "source": source,
        "path": path,
        "category": category,
        "A": "one",
        "B": "two",
        "C": "three",
        "D": d_option,
        "answer": answer,
    }


def test_evaluate_records_is_category_aware_and_question_weighted() -> None:
    references = [
        _reference("q1", "multi", "AC", path="clip-1"),
        _reference("q2", "sequence", "ABDC", path="clip-1"),
        _reference("q3", "single", "B", path="clip-2", source="HARn", d_option=""),
        _reference("q4", "emotion", "D", path="clip-3"),
    ]
    predictions = [
        {"qa_id": "q1", "prediction": "CA"},
        {"qa_id": "q2", "prediction": "ACBD"},
        {"qa_id": "q3", "prediction": "D"},
        {"qa_id": "q4", "prediction": " d "},
    ]

    result = evaluate_records(references, predictions)

    assert result["correct"] == 2
    assert result["total"] == 4
    assert result["overall_accuracy"] == pytest.approx(0.5)
    assert result["invalid_predictions"] == 1
    assert result["invalid_rate"] == pytest.approx(0.25)
    assert result["clip_macro_accuracy"] == pytest.approx(0.5)
    assert result["per_category"]["multi"]["accuracy"] == 1.0
    assert result["per_category"]["sequence"]["accuracy"] == 0.0
    assert result["per_source"]["HARn"]["accuracy"] == 0.0


def test_evaluate_records_rejects_missing_or_unexpected_ids() -> None:
    references = [_reference("q1", "single", "A", path="clip-1")]

    with pytest.raises(MetricInputError, match="prediction IDs do not match"):
        evaluate_records(references, [{"qa_id": "q2", "prediction": "A"}])


def test_evaluate_records_rejects_duplicate_prediction_ids() -> None:
    references = [_reference("q1", "single", "A", path="clip-1")]
    predictions = [
        {"qa_id": "q1", "prediction": "A"},
        {"qa_id": "q1", "prediction": "B"},
    ]

    with pytest.raises(MetricInputError, match="duplicate qa_id"):
        evaluate_records(references, predictions)


def test_evaluate_records_rejects_invalid_reference_answer() -> None:
    references = [_reference("q1", "single", "D", path="clip-1", d_option="")]

    with pytest.raises(MetricInputError, match="invalid reference answer"):
        evaluate_records(references, [{"qa_id": "q1", "prediction": "A"}])
