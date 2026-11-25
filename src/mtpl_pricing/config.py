"""Typed, immutable configuration loader and validator for TOML configs."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mtpl_pricing.exceptions import ConfigurationError

UnknownCategoryPolicy = Literal["raise", "fallback_to_base"]
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
VALID_UNKNOWN_POLICIES = frozenset({"raise", "fallback_to_base"})


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project execution metadata."""

    name: str
    random_seed: int
    log_level: str


@dataclass(frozen=True)
class DataConfig:
    """Data acquisition, integrity, and quality validation settings."""

    source_url: str
    expected_sha256: str
    raw_path: Path
    processed_path: Path
    rda_object_name: str
    download_timeout_seconds: int
    allow_network_download: bool
    max_exposure: float
    min_ageph: int
    max_ageph: int
    consistency_rtol: float
    consistency_atol: float


@dataclass(frozen=True)
class SplitConfig:
    """Train/test splitting settings."""

    test_size: float
    stratify_column: str
    min_stratum_count: int


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering, base-level selection, and leakage-guard settings."""

    id_column: str
    exposure_column: str
    target_columns: tuple[str, ...]
    derive_zone_from_postcode: bool
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    spline_column: str
    spline_knot_quantiles: tuple[float, ...]
    spline_reference_age: float
    unknown_category_policy: UnknownCategoryPolicy


@dataclass(frozen=True)
class ModelConfig:
    """GLM estimation parameters."""

    frequency_family: str
    severity_family: str
    link: str
    max_iter: int
    tol: float
    confidence_level: float


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and reporting output paths and parameters."""

    n_lift_bins: int
    reports_dir: Path
    figures_dir: Path
    metrics_filename: str
    relativities_filename: str


@dataclass(frozen=True)
class ArtifactsConfig:
    """Artifact bundle persistence locations."""

    output_dir: Path
    bundle_filename: str
    manifest_filename: str


@dataclass(frozen=True)
class PricingConfig:
    """Root validated configuration object for the pricing pipeline."""

    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    features: FeatureConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig
    base_dir: Path


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigurationError(f"Missing or invalid configuration section '[{name}]'.")
    return section


def _require_str(section: dict[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must be a non-empty string."
        )
    return value.strip()


def _require_bool(section: dict[str, Any], key: str, section_name: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"Configuration '{section_name}.{key}' must be a boolean.")
    return value


def _require_int(
    section: dict[str, Any],
    key: str,
    section_name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Configuration '{section_name}.{key}' must be an integer.")
    if min_value is not None and value < min_value:
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must be >= {min_value}, got {value}."
        )
    if max_value is not None and value > max_value:
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must be <= {max_value}, got {value}."
        )
    return value


def _require_float(
    section: dict[str, Any],
    key: str,
    section_name: str,
    *,
    gt: float | None = None,
    lt: float | None = None,
    ge: float | None = None,
) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"Configuration '{section_name}.{key}' must be a number.")
    num = float(value)
    if gt is not None and not (num > gt):
        raise ConfigurationError(f"Configuration '{section_name}.{key}' must be > {gt}, got {num}.")
    if lt is not None and not (num < lt):
        raise ConfigurationError(f"Configuration '{section_name}.{key}' must be < {lt}, got {num}.")
    if ge is not None and not (num >= ge):
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must be >= {ge}, got {num}."
        )
    return num


def _require_str_tuple(
    section: dict[str, Any],
    key: str,
    section_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must be a list of non-empty strings."
        )
    if not allow_empty and len(value) == 0:
        raise ConfigurationError(f"Configuration '{section_name}.{key}' cannot be empty.")
    if len(set(value)) != len(value):
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must not contain duplicates: {value}."
        )
    return tuple(value)


def _require_float_tuple(section: dict[str, Any], key: str, section_name: str) -> tuple[float, ...]:
    value = section.get(key)
    if not isinstance(value, list) or len(value) < 3:
        raise ConfigurationError(
            f"Configuration '{section_name}.{key}' must be a list of at least 3 numbers."
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigurationError(
                f"Configuration '{section_name}.{key}' entries must be numeric."
            )
        result.append(float(item))
    return tuple(result)


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _infer_base_dir(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "configs":
        return resolved.parent.parent
    return resolved.parent


def parse_config_dict(raw: dict[str, Any], base_dir: Path) -> PricingConfig:
    """Validate and convert a parsed TOML dictionary into a ``PricingConfig``.

    Args:
        raw: Parsed TOML mapping.
        base_dir: Root directory used to resolve relative paths.

    Returns:
        Validated immutable ``PricingConfig`` instance.

    Raises:
        ConfigurationError: If any section, field, type, or domain invariant is invalid.
    """
    proj_raw = _require_section(raw, "project")
    log_level = _require_str(proj_raw, "log_level", "project").upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigurationError(
            f"Invalid 'project.log_level': {log_level}. Allowed: {sorted(VALID_LOG_LEVELS)}."
        )
    project = ProjectConfig(
        name=_require_str(proj_raw, "name", "project"),
        random_seed=_require_int(proj_raw, "random_seed", "project", min_value=0),
        log_level=log_level,
    )

    data_raw = _require_section(raw, "data")
    expected_sha = _require_str(data_raw, "expected_sha256", "data").lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ConfigurationError("'data.expected_sha256' must be a 64-character hex digest.")
    min_age = _require_int(data_raw, "min_ageph", "data", min_value=0)
    max_age = _require_int(data_raw, "max_ageph", "data", min_value=min_age + 1)
    data = DataConfig(
        source_url=_require_str(data_raw, "source_url", "data"),
        expected_sha256=expected_sha,
        raw_path=_resolve_path(_require_str(data_raw, "raw_path", "data"), base_dir),
        processed_path=_resolve_path(_require_str(data_raw, "processed_path", "data"), base_dir),
        rda_object_name=_require_str(data_raw, "rda_object_name", "data"),
        download_timeout_seconds=_require_int(
            data_raw, "download_timeout_seconds", "data", min_value=1
        ),
        allow_network_download=_require_bool(data_raw, "allow_network_download", "data"),
        max_exposure=_require_float(data_raw, "max_exposure", "data", gt=0.0),
        min_ageph=min_age,
        max_ageph=max_age,
        consistency_rtol=_require_float(data_raw, "consistency_rtol", "data", ge=0.0),
        consistency_atol=_require_float(data_raw, "consistency_atol", "data", ge=0.0),
    )

    split_raw = _require_section(raw, "split")
    split = SplitConfig(
        test_size=_require_float(split_raw, "test_size", "split", gt=0.0, lt=1.0),
        stratify_column=_require_str(split_raw, "stratify_column", "split"),
        min_stratum_count=_require_int(split_raw, "min_stratum_count", "split", min_value=2),
    )

    feat_raw = _require_section(raw, "features")
    target_cols = _require_str_tuple(feat_raw, "target_columns", "features")
    cat_cols = _require_str_tuple(feat_raw, "categorical_columns", "features")
    num_cols = _require_str_tuple(feat_raw, "numeric_columns", "features", allow_empty=True)
    spline_col = _require_str(feat_raw, "spline_column", "features")
    id_col = _require_str(feat_raw, "id_column", "features")
    expo_col = _require_str(feat_raw, "exposure_column", "features")

    # Enforce strict separation between explanatory features and target/identifier columns.
    all_explanatory = set(cat_cols) | set(num_cols) | {spline_col}
    forbidden = set(target_cols) | {id_col, expo_col}
    overlap = sorted(all_explanatory & forbidden)
    if overlap:
        raise ConfigurationError(
            f"Forbidden target/identifier column(s) configured as features: {overlap}."
        )
    if len(all_explanatory) != len(cat_cols) + len(num_cols) + 1:
        raise ConfigurationError(
            "Feature columns must not overlap across categorical, numeric, and spline specs."
        )

    quantiles = _require_float_tuple(feat_raw, "spline_knot_quantiles", "features")
    if any(not (0.0 < q < 1.0) for q in quantiles) or list(quantiles) != sorted(quantiles):
        raise ConfigurationError(
            "'features.spline_knot_quantiles' must be strictly inside (0, 1) and sorted."
        )
    if len(set(quantiles)) != len(quantiles):
        raise ConfigurationError("'features.spline_knot_quantiles' must be strictly increasing.")

    unknown_policy_str = _require_str(feat_raw, "unknown_category_policy", "features")
    if unknown_policy_str not in VALID_UNKNOWN_POLICIES:
        raise ConfigurationError(
            f"Invalid 'features.unknown_category_policy': {unknown_policy_str}. "
            f"Allowed: {sorted(VALID_UNKNOWN_POLICIES)}."
        )
    unknown_policy: UnknownCategoryPolicy = (
        "raise" if unknown_policy_str == "raise" else "fallback_to_base"
    )

    features = FeatureConfig(
        id_column=id_col,
        exposure_column=expo_col,
        target_columns=target_cols,
        derive_zone_from_postcode=_require_bool(feat_raw, "derive_zone_from_postcode", "features"),
        categorical_columns=cat_cols,
        numeric_columns=num_cols,
        spline_column=spline_col,
        spline_knot_quantiles=quantiles,
        spline_reference_age=_require_float(
            feat_raw, "spline_reference_age", "features", ge=float(min_age)
        ),
        unknown_category_policy=unknown_policy,
    )

    model_raw = _require_section(raw, "model")
    freq_family = _require_str(model_raw, "frequency_family", "model").lower()
    sev_family = _require_str(model_raw, "severity_family", "model").lower()
    link = _require_str(model_raw, "link", "model").lower()
    if freq_family != "poisson" or sev_family != "gamma" or link != "log":
        raise ConfigurationError(
            "Only Poisson frequency and Gamma severity GLMs with log link are supported."
        )
    model = ModelConfig(
        frequency_family=freq_family,
        severity_family=sev_family,
        link=link,
        max_iter=_require_int(model_raw, "max_iter", "model", min_value=1),
        tol=_require_float(model_raw, "tol", "model", gt=0.0),
        confidence_level=_require_float(model_raw, "confidence_level", "model", gt=0.0, lt=1.0),
    )

    eval_raw = _require_section(raw, "evaluation")
    evaluation = EvaluationConfig(
        n_lift_bins=_require_int(eval_raw, "n_lift_bins", "evaluation", min_value=2),
        reports_dir=_resolve_path(_require_str(eval_raw, "reports_dir", "evaluation"), base_dir),
        figures_dir=_resolve_path(_require_str(eval_raw, "figures_dir", "evaluation"), base_dir),
        metrics_filename=_require_str(eval_raw, "metrics_filename", "evaluation"),
        relativities_filename=_require_str(eval_raw, "relativities_filename", "evaluation"),
    )

    art_raw = _require_section(raw, "artifacts")
    artifacts = ArtifactsConfig(
        output_dir=_resolve_path(_require_str(art_raw, "output_dir", "artifacts"), base_dir),
        bundle_filename=_require_str(art_raw, "bundle_filename", "artifacts"),
        manifest_filename=_require_str(art_raw, "manifest_filename", "artifacts"),
    )

    return PricingConfig(
        project=project,
        data=data,
        split=split,
        features=features,
        model=model,
        evaluation=evaluation,
        artifacts=artifacts,
        base_dir=base_dir.resolve(),
    )


def load_config(config_path: Path, base_dir: Path | None = None) -> PricingConfig:
    """Load and validate a TOML configuration file into a ``PricingConfig``.

    Args:
        config_path: Path to the TOML configuration file.
        base_dir: Optional project root directory for resolving relative paths.

    Returns:
        Validated immutable ``PricingConfig``.

    Raises:
        ConfigurationError: If the file is missing, unreadable, malformed TOML, or invalid.
    """
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(
            f"Failed to parse TOML configuration '{config_path}': {exc}"
        ) from exc

    resolved_base = base_dir.resolve() if base_dir is not None else _infer_base_dir(config_path)
    return parse_config_dict(raw, resolved_base)


def configure_logging(level: str = "INFO") -> None:
    """Configure structured root logging for CLI and pipeline execution.

    Args:
        level: Standard Python logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ConfigurationError(f"Unsupported log level: {level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for existing_handler in root_logger.handlers:
            existing_handler.setFormatter(formatter)
