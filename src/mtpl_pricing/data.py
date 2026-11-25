"""Dataset ingestion, SHA-256 integrity verification, and strict data-quality validation."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyreadr
import requests

from mtpl_pricing.config import DataConfig, FeatureConfig, PricingConfig
from mtpl_pricing.exceptions import (
    DataIngestionError,
    DataQualityError,
    DatasetIntegrityError,
    SchemaValidationError,
)

LOGGER = logging.getLogger(__name__)

RAW_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "expo",
    "claim",
    "nclaims",
    "amount",
    "average",
    "coverage",
    "ageph",
    "sex",
    "bm",
    "power",
    "agec",
    "fuel",
    "use",
    "fleet",
    "postcode",
    "long",
    "lat",
)

NUMERIC_PORTFOLIO_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "expo",
    "nclaims",
    "amount",
    "ageph",
    "bm",
    "power",
    "agec",
    "postcode",
    "long",
    "lat",
)

RAW_CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "coverage",
    "sex",
    "fuel",
    "use",
    "fleet",
)


def compute_bytes_sha256(payload: bytes) -> str:
    """Compute lowercase hexadecimal SHA-256 digest for an in-memory payload."""
    return hashlib.sha256(payload).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest by streaming a file."""
    if not path.is_file():
        raise DataIngestionError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataIngestionError(f"Failed reading file for hashing '{path}': {exc}") from exc
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str) -> str:
    """Verify that ``path`` matches ``expected_sha256`` and return the actual digest."""
    actual = compute_file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for '{path}': expected {expected_sha256}, got {actual}."
        )
    return actual


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    """Normalize categorical/object/integer codes (e.g. fleet 0/1) into clean strings.

    Preserves missing values as ``pd.NA`` instead of converting them to ``"nan"``.
    """
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    mask = series.notna()
    if not mask.any():
        return out

    def _fmt(val: object) -> str:
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        if isinstance(val, (float, np.floating)) and float(val).is_integer():
            return str(int(val))
        return str(val).strip()

    out.loc[mask] = [_fmt(v) for v in series.loc[mask]]
    return out


def _coerce_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        raise SchemaValidationError(f"Column '{col}' must be numeric, got boolean.")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    try:
        return pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(
            f"Column '{col}' contains non-numeric values and cannot be coerced cleanly."
        ) from exc


def validate_portfolio_dataframe(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    feature_cfg: FeatureConfig,
) -> pd.DataFrame:
    """Validate schema, types, bounds, and mathematical consistency of raw ``beMTPL97`` data.

    This is a pure function: it returns a normalized copy without dropping rows silently
    and without performing any I/O.

    Args:
        df: Input portfolio DataFrame.
        data_cfg: Data validation bounds and tolerances.
        feature_cfg: Feature configuration (used to verify postcode/zone requirements).

    Returns:
        Normalized copy of the validated DataFrame.

    Raises:
        SchemaValidationError: If required columns are missing or types are invalid.
        DataQualityError: If any row violates uniqueness, range, or claim-amount consistency.
    """
    if df.empty:
        raise DataQualityError("Portfolio DataFrame is empty.")

    missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SchemaValidationError(f"Portfolio DataFrame missing required columns: {missing_cols}")

    out = df.copy()

    # Coerce and check finiteness on numeric portfolio columns.
    for col in NUMERIC_PORTFOLIO_COLUMNS:
        numeric = _coerce_numeric_column(out, col)
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            bad_count = int((numeric.isna() | ~np.isfinite(numeric)).sum())
            raise DataQualityError(
                f"Column '{col}' contains {bad_count} missing or non-finite value(s)."
            )
        out[col] = numeric

    # Check 'average' numeric type separately (NaN is expected when nclaims == 0).
    if pd.api.types.is_bool_dtype(out["average"]):
        raise SchemaValidationError("Column 'average' must be numeric, got boolean.")
    try:
        out["average"] = pd.to_numeric(out["average"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("Column 'average' contains non-numeric values.") from exc

    # Normalize and validate categorical columns.
    for col in RAW_CATEGORICAL_COLUMNS:
        norm = normalize_categorical_series(out[col])
        empty_mask = norm.isna() | (norm.astype(str).str.len() == 0)
        if empty_mask.any():
            raise DataQualityError(
                f"Categorical column '{col}' has {int(empty_mask.sum())} missing/empty value(s)."
            )
        out[col] = norm.astype(str)

    # Validate 'id' uniqueness and integer integrity.
    if not np.all(out["id"] % 1 == 0):
        raise DataQualityError("Column 'id' must contain integer values.")
    out["id"] = out["id"].astype(np.int64)
    if not out["id"].is_unique:
        dup_count = int(out["id"].duplicated().sum())
        raise DataQualityError(f"Column 'id' is not unique ({dup_count} duplicate row(s)).")

    # Validate exposure bounds: strictly positive and <= max_exposure.
    bad_expo = (out["expo"] <= 0.0) | (out["expo"] > data_cfg.max_exposure)
    if bad_expo.any():
        raise DataQualityError(
            f"Column 'expo' has {int(bad_expo.sum())} row(s) outside (0, {data_cfg.max_exposure}]."
        )

    # Validate nclaims as non-negative integers.
    if not np.all(out["nclaims"] % 1 == 0) or (out["nclaims"] < 0).any():
        raise DataQualityError("Column 'nclaims' must contain non-negative integers.")
    out["nclaims"] = out["nclaims"].astype(np.int64)

    # Validate claim indicator matches (nclaims > 0).
    claim_norm = normalize_categorical_series(out["claim"])
    if claim_norm.isna().any() or not set(claim_norm.unique()).issubset({"0", "1"}):
        raise DataQualityError("Column 'claim' must contain only binary values {0, 1}.")
    out["claim"] = claim_norm.astype(np.int64)
    claim_mismatch = (out["claim"] == 1) != (out["nclaims"] > 0)
    if claim_mismatch.any():
        raise DataQualityError(
            f"Inconsistent 'claim' vs 'nclaims > 0' on {int(claim_mismatch.sum())} row(s)."
        )

    # Validate claim / amount / average invariants.
    zero_claims = out["nclaims"] == 0
    bad_zero_amount = zero_claims & (out["amount"] != 0.0)
    if bad_zero_amount.any():
        raise DataQualityError(
            f"Found {int(bad_zero_amount.sum())} row(s) with nclaims == 0 and amount != 0."
        )
    bad_zero_avg = zero_claims & out["average"].notna() & (out["average"] != 0.0)
    if bad_zero_avg.any():
        raise DataQualityError(
            f"Found {int(bad_zero_avg.sum())} row(s) with nclaims == 0 and non-zero average."
        )

    pos_claims = out["nclaims"] > 0
    if pos_claims.any():
        bad_pos_amount = pos_claims & (out["amount"] <= 0.0)
        if bad_pos_amount.any():
            raise DataQualityError(
                f"Found {int(bad_pos_amount.sum())} row(s) with nclaims > 0 and amount <= 0."
            )
        bad_pos_avg_missing = pos_claims & (out["average"].isna() | ~np.isfinite(out["average"]))
        if bad_pos_avg_missing.any():
            raise DataQualityError(
                f"Found {int(bad_pos_avg_missing.sum())} row(s) with nclaims > 0 "
                "and missing/non-finite average."
            )
        expected_avg = out.loc[pos_claims, "amount"] / out.loc[pos_claims, "nclaims"]
        actual_avg = out.loc[pos_claims, "average"]
        consistent = np.isclose(
            actual_avg.to_numpy(),
            expected_avg.to_numpy(),
            rtol=data_cfg.consistency_rtol,
            atol=data_cfg.consistency_atol,
        )
        if not np.all(consistent):
            n_bad = int(np.sum(~consistent))
            raise DataQualityError(
                f"Found {n_bad} row(s) where 'average' disagrees with 'amount / nclaims'."
            )

    # Validate domain bounds on covariates.
    bad_age = (out["ageph"] < data_cfg.min_ageph) | (out["ageph"] > data_cfg.max_ageph)
    if bad_age.any():
        raise DataQualityError(
            f"Column 'ageph' has {int(bad_age.sum())} row(s) outside "
            f"[{data_cfg.min_ageph}, {data_cfg.max_ageph}]."
        )
    if (out["bm"] < 0).any() or (out["power"] <= 0).any() or (out["agec"] < 0).any():
        raise DataQualityError(
            "Explanatory bounds violated: require bm >= 0, power > 0, and agec >= 0."
        )
    if feature_cfg.derive_zone_from_postcode:
        bad_pc = (out["postcode"] % 1 != 0) | (out["postcode"] < 1000) | (out["postcode"] > 9999)
        if bad_pc.any():
            raise DataQualityError(
                f"Column 'postcode' has {int(bad_pc.sum())} invalid Belgian 4-digit code(s)."
            )
    out["postcode"] = out["postcode"].astype(np.int64)

    LOGGER.info(
        "Validated portfolio dataset: rows=%d, exposure=%.2f, nclaims=%d, "
        "avg_freq=%.4f, total_amount=%.2f",
        len(out),
        float(out["expo"].sum()),
        int(out["nclaims"].sum()),
        float(out["nclaims"].sum() / out["expo"].sum()),
        float(out["amount"].sum()),
    )
    return out


def atomic_write_bytes(destination: Path, payload: bytes) -> None:
    """Atomically write bytes to ``destination`` using temp file + fsync + replace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, destination)
    except OSError as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise DataIngestionError(f"Atomic write failed for '{destination}': {exc}") from exc


def atomic_write_dataframe_csv(df: pd.DataFrame, destination: Path) -> None:
    """Atomically serialize a DataFrame to UTF-8 CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            df.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, destination)
    except OSError as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise DataIngestionError(f"Failed writing processed CSV '{destination}': {exc}") from exc


def download_raw_dataset(
    url: str,
    destination: Path,
    expected_sha256: str,
    timeout_seconds: int,
) -> str:
    """Download the raw ``.rda`` file, verify SHA-256, and store atomically."""
    LOGGER.info("Downloading raw dataset from %s", url)
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataIngestionError(f"Network download failed for '{url}': {exc}") from exc

    payload = response.content
    actual_sha = compute_bytes_sha256(payload)
    if actual_sha.lower() != expected_sha256.lower():
        raise DatasetIntegrityError(
            f"Downloaded payload SHA-256 mismatch for '{url}': "
            f"expected {expected_sha256}, got {actual_sha}."
        )
    atomic_write_bytes(destination, payload)
    LOGGER.info("Stored verified raw dataset at %s (sha256=%s)", destination, actual_sha)
    return actual_sha


def read_rda_dataframe(raw_path: Path, object_name: str) -> pd.DataFrame:
    """Parse an R ``.rda`` archive via ``pyreadr`` and extract ``object_name``."""
    if not raw_path.is_file():
        raise DataIngestionError(f"Raw .rda file does not exist: {raw_path}")
    try:
        parsed = pyreadr.read_r(str(raw_path))
    except (pyreadr.PyreadrError, pyreadr.LibrdataError, OSError, ValueError, RuntimeError) as exc:
        raise DataIngestionError(f"Failed parsing .rda file '{raw_path}': {exc}") from exc

    if object_name not in parsed:
        raise DataIngestionError(
            f"Object '{object_name}' not found in '{raw_path}'. "
            f"Available objects: {list(parsed.keys())}."
        )
    obj = parsed[object_name]
    if not isinstance(obj, pd.DataFrame):
        raise DataIngestionError(
            f"Object '{object_name}' in '{raw_path}' is {type(obj).__name__}, expected DataFrame."
        )
    return obj


def load_processed_csv(csv_path: Path) -> pd.DataFrame:
    """Load processed CSV preserving categorical string representations cleanly."""
    if not csv_path.is_file():
        raise DataIngestionError(f"Processed CSV file does not exist: {csv_path}")
    dtype_overrides: dict[str, str] = {col: "string" for col in RAW_CATEGORICAL_COLUMNS}
    dtype_overrides["claim"] = "string"
    try:
        return pd.read_csv(csv_path, dtype=dtype_overrides, keep_default_na=True)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataIngestionError(f"Failed reading processed CSV '{csv_path}': {exc}") from exc


def ingest_dataset(config: PricingConfig) -> tuple[pd.DataFrame, str]:
    """Acquire, integrity-check, validate, and persist ``beMTPL97`` to processed CSV.

    Workflow:
    1. If ``raw_path`` exists, verify its SHA-256 (strictly immutable), parse ``.rda``,
       validate all quality invariants, and atomically write ``processed_path``.
    2. If ``raw_path`` is missing and network download is enabled, attempt download +
       SHA-256 verification. On network failure, cleanly fall back to ``processed_path``
       if it exists (logging a warning); otherwise raise ``DataIngestionError``.
    3. If network download is disabled and ``raw_path`` is missing, fall back to
       ``processed_path`` if present, else raise ``DataIngestionError``.

    Returns:
        Tuple of ``(validated_dataframe, dataset_sha256)``.
    """
    data_cfg = config.data
    raw_path = data_cfg.raw_path
    processed_path = data_cfg.processed_path

    if raw_path.exists():
        dataset_sha256 = verify_file_sha256(raw_path, data_cfg.expected_sha256)
        raw_df = read_rda_dataframe(raw_path, data_cfg.rda_object_name)
        validated = validate_portfolio_dataframe(raw_df, data_cfg, config.features)
        atomic_write_dataframe_csv(validated, processed_path)
        LOGGER.info("Exported validated CSV to %s", processed_path)
        return validated, dataset_sha256

    if data_cfg.allow_network_download:
        try:
            dataset_sha256 = download_raw_dataset(
                url=data_cfg.source_url,
                destination=raw_path,
                expected_sha256=data_cfg.expected_sha256,
                timeout_seconds=data_cfg.download_timeout_seconds,
            )
            raw_df = read_rda_dataframe(raw_path, data_cfg.rda_object_name)
            validated = validate_portfolio_dataframe(raw_df, data_cfg, config.features)
            atomic_write_dataframe_csv(validated, processed_path)
            LOGGER.info("Exported validated CSV to %s", processed_path)
            return validated, dataset_sha256
        except DatasetIntegrityError:
            raise
        except DataIngestionError as exc:
            if processed_path.is_file():
                LOGGER.warning(
                    "Raw download unavailable (%s); falling back to validated CSV at %s",
                    exc,
                    processed_path,
                )
                csv_df = load_processed_csv(processed_path)
                validated = validate_portfolio_dataframe(csv_df, data_cfg, config.features)
                return validated, compute_file_sha256(processed_path)
            raise DataIngestionError(
                f"Cannot ingest dataset: raw download failed ({exc}) and no local fallback exists "
                f"at '{processed_path}'. Place the verified .rda file at '{raw_path}'."
            ) from exc

    if processed_path.is_file():
        LOGGER.warning(
            "Network download disabled and raw file missing; reading fallback CSV at %s",
            processed_path,
        )
        csv_df = load_processed_csv(processed_path)
        validated = validate_portfolio_dataframe(csv_df, data_cfg, config.features)
        return validated, compute_file_sha256(processed_path)

    raise DataIngestionError(
        f"Dataset missing at '{raw_path}' (network disabled) "
        f"and no fallback CSV at '{processed_path}'."
    )
