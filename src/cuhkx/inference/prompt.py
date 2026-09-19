"""IR-only cuhkx_mcq_v2 prompt and answer parsing, migrated verbatim."""
from __future__ import annotations
import itertools
import re
from dataclasses import dataclass
from typing import Any, Mapping
from cuhkx.config import InputError as InferenceInputError
from cuhkx.evaluation.metric import QA_CATEGORIES, SINGLE_ANSWER_CATEGORIES, assess_answer, available_option_letters
PROMPT_VERSION = "cuhkx_mcq_v2"
MODALITY_LABELS = {"IR": "infrared"}
SENSITIVE_PROMPT_FIELDS = ("path", "clip_key", "action_name", "subject_id", "trial_id", "source_relative_path")

@dataclass(frozen=True)
class ParsedPrediction:
    """Auditable answer extracted from raw model text."""

    raw_output: str
    candidate: str | None
    prediction: str | None
    is_valid: bool
    parse_method: str | None
    error: str | None


def _normalize_category(value: Any) -> str:
    category = str(value or "").strip().casefold()
    if category not in QA_CATEGORIES:
        raise InferenceInputError(f"unknown QA category: {value!r}")
    return category


def allowed_answer_outputs(category: str, valid_options: str) -> tuple[str, ...]:
    """Enumerate the complete category-valid output space for constrained decoding."""
    normalized = _normalize_category(category)
    letters = tuple(valid_options)
    if not letters:
        raise InferenceInputError("question has no available options")
    if normalized in SINGLE_ANSWER_CATEGORIES:
        return letters
    if normalized == "multi":
        return tuple(
            "".join(values)
            for count in range(1, len(letters) + 1)
            for values in itertools.combinations(letters, count)
        )
    return tuple("".join(values) for values in itertools.permutations(letters))


def build_mcq_prompt(row: Mapping[str, Any], *, modality: str = "IR") -> str:
    """Build a category-aware prompt using only question and option text."""
    category = _normalize_category(row.get("category"))
    valid_options = available_option_letters(row)
    if not valid_options:
        raise InferenceInputError("question has no available options")
    modality_label = MODALITY_LABELS.get(modality)
    if modality_label is None:
        raise InferenceInputError(f"unsupported modality: {modality!r}")
    question = str(row.get("question", "") or "").strip()
    if not question:
        raise InferenceInputError("question text is empty")
    option_lines = [f"{letter}. {str(row[letter]).strip()}" for letter in valid_options]

    if category in SINGLE_ANSWER_CATEGORIES:
        output_rule = "Output exactly one available option letter."
    elif category == "multi":
        output_rule = (
            "Output all selected option letters once, in alphabetical order, with no separators."
        )
    else:
        output_rule = (
            "This is a temporal ordering task, not a single-choice task. Output a complete "
            f"permutation of {valid_options} in the observed temporal order. Your answer must "
            "contain every available option letter exactly once, with no separators. Even if "
            "some actions are uncertain, provide the full best-estimate permutation."
        )

    return "\n".join(
        [
            f"Prompt protocol: {PROMPT_VERSION}",
            f"The following {modality_label} frames are in chronological order, earliest first.",
            "Answer the multiple-choice question using only visual evidence from these frames.",
            "",
            f"Question: {question}",
            "Options:",
            *option_lines,
            "",
            f"Category: {category}",
            output_rule,
            "Do not provide an explanation or punctuation.",
            "Final answer:",
        ]
    )


def detect_prompt_leakage(prompt: str, row: Mapping[str, Any]) -> list[str]:
    """Detect accidental inclusion of private path/identity fields in a prompt."""
    findings: list[str] = []
    normalized_prompt = prompt.casefold()
    for field in SENSITIVE_PROMPT_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if len(value) >= 3 and value.casefold() in normalized_prompt:
            findings.append(field)
    structural_markers = {
        "raw_video_extension": r"\.mp4\b",
        "windows_path": r"[a-zA-Z]:\\",
        "dataset_directory": r"\b(?:Training|Testing)[/\\]",
        "test_clip_id": r"\bLM_test_\d+\b",
        "subject_path_id": r"\buser\d+[/\\]",
    }
    for label, pattern in structural_markers.items():
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            findings.append(label)
    return sorted(set(findings))


def _compact_letters(value: str) -> str | None:
    compact = re.sub(r"[\s,;/|]+", "", value.strip().upper())
    return compact if compact and re.fullmatch(r"[A-D]+", compact) else None


def parse_model_answer(
    raw_output: str,
    *,
    category: str,
    valid_options: str,
) -> ParsedPrediction:
    """Parse conservative answer formats and validate them with the competition metric."""
    raw = str(raw_output or "").strip()
    if not raw:
        return ParsedPrediction(raw, None, None, False, None, "model output is empty")

    candidate = _compact_letters(raw)
    method = "exact"
    if candidate is None:
        patterns = (
            ("boxed", r"\\boxed\s*\{\s*([A-D\s,;/|]+?)\s*\}"),
            (
                "labeled",
                r"(?:final\s+answer|answer|答案)\s*(?:is\s*)?[:：]?\s*([A-D](?:[\s,;/|]*[A-D])*)\b",
            ),
        )
        parsed: list[tuple[str, str]] = []
        for parse_method, pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
                value = _compact_letters(match.group(1))
                if value is not None:
                    parsed.append((parse_method, value))
        unique = {value for _, value in parsed}
        if len(unique) == 1:
            candidate = next(iter(unique))
            method = next(parse_method for parse_method, value in parsed if value == candidate)
        elif len(unique) > 1:
            return ParsedPrediction(
                raw,
                None,
                None,
                False,
                "ambiguous",
                f"model output contains conflicting answer candidates: {sorted(unique)}",
            )
        else:
            return ParsedPrediction(
                raw,
                None,
                None,
                False,
                None,
                "model output does not match an accepted answer-only format",
            )

    assessment = assess_answer(candidate, category, valid_options)
    return ParsedPrediction(
        raw_output=raw,
        candidate=candidate,
        prediction=assessment.canonical,
        is_valid=assessment.is_valid,
        parse_method=method,
        error=assessment.error,
    )

