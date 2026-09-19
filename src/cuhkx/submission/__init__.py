"""Prediction normalization and Kaggle submission generation."""

from cuhkx.submission.validator import (
    validate_submission_files,
    validate_submission_records,
)


__all__ = ["validate_submission_files", "validate_submission_records"]
