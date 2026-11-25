"""Atomic model bundle persistence, manifest generation, and schema compatibility checks."""

from __future__ import annotations

import json
import logging
import os
import pickle
import platform
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any, Final

import joblib

from mtpl_pricing import __version__
from mtpl_pricing.config import ArtifactsConfig
from mtpl_pricing.data import compute_file_sha256
from mtpl_pricing.evaluation import atomic_write_json
from mtpl_pricing.exceptions import ArtifactCompatibilityError, ArtifactSerializationError
from mtpl_pricing.features import FeatureSpec
from mtpl_pricing.models import PricingModel

LOGGER = logging.getLogger(__name__)

SCHEMA_FORMAT_VERSION: Final[str] = "1.0.0"
PACKAGE_NAME: Final[str] = "mtpl-glm-pricing"
TRACKED_DEPENDENCIES: Final[tuple[str, ...]] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "joblib",
    "pyreadr",
    "matplotlib",
    "requests",
)


@dataclass(frozen=True)
class PricingBundle:
    """Self-contained immutable artifact bundle saved via ``joblib``."""

    schema_format_version: str
    package_name: str
    package_version: str
    created_at_utc: str
    dataset_sha256: str
    random_seed: int
    feature_spec: FeatureSpec
    pricing_model: PricingModel


def _collect_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in TRACKED_DEPENDENCIES:
        try:
            versions[pkg] = dist_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def build_manifest(
    *,
    bundle: PricingBundle,
    bundle_filename: str,
    bundle_sha256: str,
    validation_metrics: dict[str, int | float] | None = None,
) -> dict[str, Any]:
    """Construct the JSON manifest recording environment, schema, base levels, and metrics."""
    spec = bundle.feature_spec
    freq = bundle.pricing_model.frequency_model
    sev = bundle.pricing_model.severity_model

    return {
        "schema_format_version": bundle.schema_format_version,
        "package_name": bundle.package_name,
        "package_version": bundle.package_version,
        "created_at_utc": bundle.created_at_utc,
        "bundle_filename": bundle_filename,
        "bundle_sha256": bundle_sha256,
        "python_version": platform.python_version(),
        "dependencies": _collect_dependency_versions(),
        "dataset_sha256": bundle.dataset_sha256,
        "random_seed": bundle.random_seed,
        "feature_schema": {
            "raw_required_columns": list(spec.raw_required_columns),
            "exposure_column": spec.exposure_column,
            "derive_zone_from_postcode": spec.derive_zone_from_postcode,
            "categorical_columns": [c.column for c in spec.categorical_specs],
            "base_levels": {c.column: c.base_level for c in spec.categorical_specs},
            "categories": {c.column: list(c.levels) for c in spec.categorical_specs},
            "numeric_columns": list(spec.numeric_columns),
            "spline": {
                "column": spec.spline_spec.column,
                "knots": list(spec.spline_spec.knots),
                "reference_value": spec.spline_spec.reference_value,
                "feature_names": list(spec.spline_spec.feature_names),
            },
            "design_columns": list(spec.design_columns),
            "unknown_category_policy": spec.unknown_category_policy,
            "bounds": {
                "max_exposure": spec.max_exposure,
                "min_ageph": spec.min_ageph,
                "max_ageph": spec.max_ageph,
            },
        },
        "model_summary": {
            "frequency": {
                "family": freq.family,
                "link": freq.link,
                "n_obs": freq.n_obs,
                "df_resid": freq.df_resid,
                "deviance": freq.deviance,
                "pearson_dispersion": freq.pearson_dispersion,
                "converged": freq.converged,
            },
            "severity": {
                "family": sev.family,
                "link": sev.link,
                "n_obs": sev.n_obs,
                "df_resid": sev.df_resid,
                "deviance": sev.deviance,
                "pearson_dispersion": sev.pearson_dispersion,
                "scale": sev.scale,
                "converged": sev.converged,
            },
        },
        "validation_metrics": validation_metrics or {},
    }


def create_pricing_bundle(
    *,
    feature_spec: FeatureSpec,
    pricing_model: PricingModel,
    dataset_sha256: str,
    random_seed: int,
) -> PricingBundle:
    """Create a timestamped ``PricingBundle`` ready for atomic persistence."""
    return PricingBundle(
        schema_format_version=SCHEMA_FORMAT_VERSION,
        package_name=PACKAGE_NAME,
        package_version=__version__,
        created_at_utc=datetime.now(UTC).isoformat(),
        dataset_sha256=dataset_sha256,
        random_seed=random_seed,
        feature_spec=feature_spec,
        pricing_model=pricing_model,
    )


def _atomic_dump_joblib(obj: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp.joblib",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            joblib.dump(obj, fh, compress=3)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, destination)
    except (OSError, TypeError, ValueError, pickle.PicklingError) as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise ArtifactSerializationError(
            f"Failed saving joblib artifact atomically to '{destination}': {exc}"
        ) from exc


def save_artifacts(
    *,
    bundle: PricingBundle,
    artifacts_cfg: ArtifactsConfig,
    validation_metrics: dict[str, int | float] | None = None,
) -> tuple[Path, Path]:
    """Atomically save the ``joblib`` bundle and its companion ``manifest.json``."""
    bundle_path = artifacts_cfg.output_dir / artifacts_cfg.bundle_filename
    manifest_path = artifacts_cfg.output_dir / artifacts_cfg.manifest_filename

    _atomic_dump_joblib(bundle, bundle_path)
    bundle_sha256 = compute_file_sha256(bundle_path)
    manifest = build_manifest(
        bundle=bundle,
        bundle_filename=artifacts_cfg.bundle_filename,
        bundle_sha256=bundle_sha256,
        validation_metrics=validation_metrics,
    )
    atomic_write_json(manifest, manifest_path)
    LOGGER.info("Saved artifact bundle (%s) and manifest (%s)", bundle_path, manifest_path)
    return bundle_path, manifest_path


def _major_version(version_str: str) -> str:
    return version_str.split(".", 1)[0]


def _validate_manifest_and_bundle_compatibility(
    manifest: dict[str, Any],
    bundle: PricingBundle,
) -> None:
    required_top = {
        "schema_format_version",
        "package_name",
        "package_version",
        "bundle_filename",
        "bundle_sha256",
        "dataset_sha256",
        "feature_schema",
    }
    missing = sorted(required_top - set(manifest.keys()))
    if missing:
        raise ArtifactCompatibilityError(f"Manifest missing required field(s): {missing}")

    if manifest["schema_format_version"] != SCHEMA_FORMAT_VERSION:
        raise ArtifactCompatibilityError(
            f"Incompatible artifact schema_format_version: {manifest['schema_format_version']} "
            f"(expected {SCHEMA_FORMAT_VERSION})."
        )
    if manifest["package_name"] != PACKAGE_NAME:
        raise ArtifactCompatibilityError(
            f"Unexpected package_name in manifest: {manifest['package_name']}."
        )
    if _major_version(str(manifest["package_version"])) != _major_version(__version__):
        raise ArtifactCompatibilityError(
            f"Incompatible major package version: artifact={manifest['package_version']}, "
            f"runtime={__version__}."
        )

    feat_schema = manifest["feature_schema"]
    if not isinstance(feat_schema, dict):
        raise ArtifactCompatibilityError("Manifest 'feature_schema' must be a JSON object.")

    spec = bundle.feature_spec
    if bundle.package_version != manifest["package_version"]:
        raise ArtifactCompatibilityError("Manifest package_version disagrees with bundle payload.")
    if bundle.dataset_sha256 != manifest["dataset_sha256"]:
        raise ArtifactCompatibilityError("Manifest dataset_sha256 disagrees with bundle payload.")

    if list(spec.design_columns) != feat_schema.get("design_columns"):
        raise ArtifactCompatibilityError(
            "Bundle design_columns disagree with manifest feature_schema."
        )
    expected_bases = {c.column: c.base_level for c in spec.categorical_specs}
    if expected_bases != feat_schema.get("base_levels"):
        raise ArtifactCompatibilityError(
            "Bundle base_levels disagree with manifest feature_schema."
        )
    expected_cats = {c.column: list(c.levels) for c in spec.categorical_specs}
    if expected_cats != feat_schema.get("categories"):
        raise ArtifactCompatibilityError(
            "Bundle categorical levels disagree with manifest feature_schema."
        )

    freq_cols = bundle.pricing_model.frequency_model.feature_names
    sev_cols = bundle.pricing_model.severity_model.feature_names
    if freq_cols != spec.design_columns or sev_cols != spec.design_columns:
        raise ArtifactCompatibilityError(
            "Model feature_names inside bundle do not match FeatureSpec design_columns."
        )


def load_artifacts(
    artifacts_dir: Path,
    *,
    bundle_filename: str = "pricing_bundle.joblib",
    manifest_filename: str = "manifest.json",
) -> tuple[PricingBundle, dict[str, Any]]:
    """Load and verify ``manifest.json`` + ``pricing_bundle.joblib`` integrity and schema."""
    manifest_path = artifacts_dir / manifest_filename
    if not manifest_path.is_file():
        raise ArtifactCompatibilityError(f"Missing artifact manifest: {manifest_path}")

    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"Failed parsing artifact manifest '{manifest_path}': {exc}"
        ) from exc

    if not isinstance(raw_manifest, dict):
        raise ArtifactCompatibilityError("Artifact manifest must be a JSON object.")

    effective_bundle_name = str(raw_manifest.get("bundle_filename") or bundle_filename)
    bundle_path = artifacts_dir / effective_bundle_name
    if not bundle_path.is_file():
        raise ArtifactSerializationError(f"Missing serialized model bundle: {bundle_path}")

    expected_sha = raw_manifest.get("bundle_sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ArtifactCompatibilityError("Manifest missing valid 'bundle_sha256'.")

    actual_sha = compute_file_sha256(bundle_path)
    if actual_sha.lower() != expected_sha.lower():
        raise ArtifactCompatibilityError(
            f"Corrupted or mismatched model bundle '{bundle_path}': "
            f"expected sha256={expected_sha}, got {actual_sha}."
        )

    try:
        loaded = joblib.load(bundle_path)
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
        ImportError,
        AttributeError,
    ) as exc:
        raise ArtifactSerializationError(
            f"Failed deserializing model bundle '{bundle_path}': {exc}"
        ) from exc

    if not isinstance(loaded, PricingBundle):
        raise ArtifactCompatibilityError(
            f"Deserialized artifact has unexpected type {type(loaded).__name__}, "
            "expected PricingBundle."
        )

    _validate_manifest_and_bundle_compatibility(raw_manifest, loaded)
    LOGGER.info("Loaded and verified artifact bundle from %s", artifacts_dir)
    return loaded, raw_manifest
