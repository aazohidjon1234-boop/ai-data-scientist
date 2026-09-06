"""Domain exceptions mapped to HTTP responses by main.py."""
from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class ValidationError(AppError):
    status_code = 400
    code = "validation_error"


class DatasetNotFoundError(AppError):
    status_code = 404
    code = "dataset_not_found"


class ReportNotFoundError(AppError):
    status_code = 404
    code = "report_not_found"


class SampleNotFoundError(AppError):
    status_code = 404
    code = "sample_not_found"


class ProcessingError(AppError):
    """A tool/model step failed while the pipeline is running."""

    status_code = 500
    code = "processing_error"
