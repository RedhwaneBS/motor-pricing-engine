"""Explicit domain exception hierarchy for the MTPL pricing pipeline."""

from __future__ import annotations


class MtplPricingError(Exception):
    """Base exception for all errors raised by ``mtpl_pricing``."""


class ConfigurationError(MtplPricingError):
    """Raised when configuration loading or validation fails."""


class DataIngestionError(MtplPricingError):
    """Raised when acquiring, reading, or writing dataset files fails."""


class DatasetIntegrityError(DataIngestionError):
    """Raised when a raw dataset file fails SHA-256 integrity verification."""


class DataValidationError(MtplPricingError):
    """Base exception for schema, quality, and domain validation failures."""


class SchemaValidationError(DataValidationError):
    """Raised when expected columns, types, or target-isolation rules are violated."""


class DataQualityError(DataValidationError):
    """Raised when business invariants, bounds, or consistency checks fail."""


class UnknownCategoryError(DataValidationError):
    """Raised when an unseen categorical level appears during transformation or inference."""

    def __init__(
        self,
        column: str,
        unknown_values: list[str],
        allowed_values: list[str],
    ) -> None:
        self.column = column
        self.unknown_values = unknown_values
        self.allowed_values = allowed_values
        message = (
            f"Unknown category level(s) in column '{column}': {unknown_values}. "
            f"Allowed levels: {allowed_values}."
        )
        super().__init__(message)


class ModelTrainingError(MtplPricingError):
    """Raised when GLM estimation fails to converge or encounters invalid inputs."""


class ArtifactError(MtplPricingError):
    """Base exception for model artifact persistence and compatibility failures."""


class ArtifactSerializationError(ArtifactError):
    """Raised when saving or loading serialized model bundles fails."""


class ArtifactCompatibilityError(ArtifactError):
    """Raised when a model bundle manifest is missing, malformed, or incompatible."""


class InferenceError(MtplPricingError):
    """Raised when prediction inputs or model outputs violate inference invariants."""
