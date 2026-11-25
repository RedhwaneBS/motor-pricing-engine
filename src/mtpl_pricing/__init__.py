"""Production-grade MTPL GLM pricing pipeline on the Belgian beMTPL97 dataset."""

from __future__ import annotations

from mtpl_pricing.exceptions import (
    ArtifactCompatibilityError,
    ArtifactError,
    ArtifactSerializationError,
    ConfigurationError,
    DataIngestionError,
    DataQualityError,
    DatasetIntegrityError,
    DataValidationError,
    InferenceError,
    ModelTrainingError,
    MtplPricingError,
    SchemaValidationError,
    UnknownCategoryError,
)

__version__ = "0.1.0"

__all__ = [
    "ArtifactCompatibilityError",
    "ArtifactError",
    "ArtifactSerializationError",
    "ConfigurationError",
    "DataIngestionError",
    "DataQualityError",
    "DataValidationError",
    "DatasetIntegrityError",
    "InferenceError",
    "ModelTrainingError",
    "MtplPricingError",
    "SchemaValidationError",
    "UnknownCategoryError",
    "__version__",
]
