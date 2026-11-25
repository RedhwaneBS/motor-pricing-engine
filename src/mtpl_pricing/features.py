"""Feature engineering: stratified splits, base levels, splines, and inference guards."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from mtpl_pricing.config import (
    FeatureConfig,
    PricingConfig,
    SplitConfig,
    UnknownCategoryPolicy,
)
from mtpl_pricing.data import normalize_categorical_series
from mtpl_pricing.exceptions import (
    DataQualityError,
    InferenceError,
    ModelTrainingError,
    SchemaValidationError,
    UnknownCategoryError,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoricalLevelSpec:
    """Fitted categorical encoding metadata with exposure-weighted base level."""

    column: str
    base_level: str
    levels: tuple[str, ...]
    exposure_by_level: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SplineSpec:
    """Fitted natural cubic spline specification for driver age."""

    column: str
    knots: tuple[float, ...]
    reference_value: float
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable fitted feature schema required for reproducible training and inference."""

    raw_required_columns: tuple[str, ...]
    exposure_column: str
    derive_zone_from_postcode: bool
    categorical_specs: tuple[CategoricalLevelSpec, ...]
    numeric_columns: tuple[str, ...]
    spline_spec: SplineSpec
    design_columns: tuple[str, ...]
    unknown_category_policy: UnknownCategoryPolicy
    max_exposure: float
    min_ageph: int
    max_ageph: int


def split_train_test(
    df: pd.DataFrame,
    split_cfg: SplitConfig,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform deterministic stratified train/test splitting.

    Rare strata with fewer than ``min_stratum_count`` rows are merged cleanly into
    a shared fallback stratum before stratification to prevent runtime errors on
    small subsets while preserving exact stratification on full datasets.

    Args:
        df: Validated portfolio DataFrame.
        split_cfg: Splitting configuration.
        random_seed: Deterministic PRNG seed.

    Returns:
        Tuple of ``(train_df, test_df)`` with freshly reset indices.

    Raises:
        SchemaValidationError: If ``stratify_column`` is missing.
        DataQualityError: If dataset is too small to split.
    """
    if split_cfg.stratify_column not in df.columns:
        raise SchemaValidationError(
            f"Stratification column '{split_cfg.stratify_column}' missing from DataFrame."
        )
    if len(df) < 5:
        raise DataQualityError(f"Cannot split dataset with only {len(df)} rows (need >= 5).")

    strata = df[split_cfg.stratify_column].astype(str).copy()
    counts = strata.value_counts()
    rare_levels = counts[counts < split_cfg.min_stratum_count].index.tolist()
    if rare_levels:
        LOGGER.warning(
            "Merging %d rare stratification level(s) %s (< %d rows) for stratified split.",
            len(rare_levels),
            rare_levels,
            split_cfg.min_stratum_count,
        )
        # Merge rare strata into the most frequent stratum so the fallback group itself is valid.
        most_frequent = str(counts.idxmax())
        strata = strata.where(~strata.isin(rare_levels), other=most_frequent)

    stratify_arg: pd.Series | None = strata if strata.nunique() > 1 else None
    if stratify_arg is None:
        LOGGER.warning(
            "Stratification column has single stratum; falling back to unstratified split."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=split_cfg.test_size,
        random_state=random_seed,
        stratify=stratify_arg,
    )
    train_out = train_df.reset_index(drop=True).copy()
    test_out = test_df.reset_index(drop=True).copy()

    LOGGER.info(
        "Split dataset deterministically (seed=%d, test_size=%.2f): train=%d, test=%d",
        random_seed,
        split_cfg.test_size,
        len(train_out),
        len(test_out),
    )
    return train_out, test_out


def derive_explanatory_columns(
    df: pd.DataFrame,
    *,
    derive_zone_from_postcode: bool,
) -> pd.DataFrame:
    """Derive deterministic explanatory columns (Belgian postal ``zone``) on a copy."""
    out = df.copy()
    if derive_zone_from_postcode:
        if "postcode" not in out.columns:
            raise SchemaValidationError(
                "Column 'postcode' is required to derive Belgian postal 'zone'."
            )
        pc = pd.to_numeric(out["postcode"], errors="coerce")
        if pc.isna().any() or not np.isfinite(pc.to_numpy()).all():
            raise DataQualityError("Column 'postcode' contains missing or non-numeric values.")
        if not np.all(pc % 1 == 0) or (pc < 1000).any() or (pc > 9999).any():
            raise DataQualityError(
                "Column 'postcode' values must be 4-digit integers in [1000, 9999]."
            )
        out["zone"] = (pc.astype(np.int64) // 1000).astype(str)
    return out


def select_exposure_weighted_base_level(
    df: pd.DataFrame,
    column: str,
    exposure_column: str,
) -> CategoricalLevelSpec:
    """Select the modal category with maximum cumulative exposure as reference level.

    Follows Goldburd et al. (*GLMs for Insurance Rating*, Ch. 3) to maximize statistical
    stability of the baseline intercept and relativities. Ties break deterministically
    by lexicographic order.
    """
    if column not in df.columns:
        raise SchemaValidationError(f"Categorical feature '{column}' not found in DataFrame.")

    levels_series = normalize_categorical_series(df[column])
    if levels_series.isna().any():
        raise DataQualityError(f"Categorical column '{column}' contains missing values.")

    expo_by_level = (
        pd.DataFrame(
            {"level": levels_series.astype(str), "expo": df[exposure_column].astype(float)}
        )
        .groupby("level", sort=True)["expo"]
        .sum()
    )
    max_expo = float(expo_by_level.max())
    # Deterministic lexicographic tie-break when exposures are identical.
    candidates = sorted(
        str(lvl)
        for lvl, val in expo_by_level.items()
        if np.isclose(float(val), max_expo, rtol=1e-12)
    )
    base_level = candidates[0]
    other_levels = sorted(str(lvl) for lvl in expo_by_level.index if str(lvl) != base_level)
    ordered_levels = (base_level, *other_levels)
    exposure_pairs = tuple((str(lvl), float(expo_by_level[lvl])) for lvl in ordered_levels)
    return CategoricalLevelSpec(
        column=column,
        base_level=base_level,
        levels=ordered_levels,
        exposure_by_level=exposure_pairs,
    )


def compute_natural_spline_basis(
    values: Sequence[float] | NDArray[np.float64] | pd.Series,
    knots: Sequence[float],
    reference_value: float,
) -> NDArray[np.float64]:
    """Compute a centered, scaled Harrell natural cubic spline basis.

    Properties:
    - Strictly linear below ``knots[0]`` and above ``knots[-1]`` (stable extrapolation).
    - Identically zero across all basis columns at ``reference_value`` so ``exp(eta) == 1.0``
      at the reference driver age and confidence bands pinch smoothly to zero there.
    - Scaled by ``knots[-1] - knots[0]`` so IRLS design matrices remain well-conditioned.

    Args:
        values: 1D array-like of numeric values.
        knots: Strictly increasing sequence of ``K >= 3`` knot locations.
        reference_value: Anchor point where every basis column evaluates to ``0.0``.

    Returns:
        2D float64 array of shape ``(len(values), K - 1)``.
    """
    arr: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Spline input must be 1D, got shape {arr.shape}.")
    knot_arr: NDArray[np.float64] = np.asarray(knots, dtype=np.float64)
    if len(knot_arr) < 3 or not np.all(np.diff(knot_arr) > 0):
        raise ModelTrainingError(
            f"Natural spline requires >= 3 strictly increasing knots, got {list(knot_arr)}."
        )

    span = float(knot_arr[-1] - knot_arr[0])
    scale = span * span
    k_last = float(knot_arr[-1])
    k_prev = float(knot_arr[-2])

    def _d_term(x_vals: NDArray[np.float64], k_j: float) -> NDArray[np.float64]:
        pos_j: NDArray[np.float64] = np.maximum(x_vals - k_j, 0.0) ** 3
        pos_last: NDArray[np.float64] = np.maximum(x_vals - k_last, 0.0) ** 3
        res: NDArray[np.float64] = (pos_j - pos_last) / ((k_last - k_j) * scale)
        return res

    ref_arr: NDArray[np.float64] = np.array([float(reference_value)], dtype=np.float64)
    d_prev_x = _d_term(arr, k_prev)
    d_prev_ref = float(_d_term(ref_arr, k_prev)[0])

    columns: list[NDArray[np.float64]] = [(arr - float(reference_value)) / span]
    for k_j in knot_arr[:-2]:
        col = (_d_term(arr, float(k_j)) - d_prev_x) - (
            float(_d_term(ref_arr, float(k_j))[0]) - d_prev_ref
        )
        columns.append(col)
    return np.column_stack(columns).astype(np.float64, copy=False)


def raw_required_columns_for_config(feature_cfg: FeatureConfig) -> tuple[str, ...]:
    """Return the exact raw input columns required for feature extraction and prediction."""
    cols: list[str] = [feature_cfg.exposure_column]
    for cat_col in feature_cfg.categorical_columns:
        if cat_col == "zone" and feature_cfg.derive_zone_from_postcode:
            if "postcode" not in cols:
                cols.append("postcode")
        elif cat_col not in cols:
            cols.append(cat_col)
    for num_col in feature_cfg.numeric_columns:
        if num_col not in cols:
            cols.append(num_col)
    if feature_cfg.spline_column not in cols:
        cols.append(feature_cfg.spline_column)
    return tuple(cols)


def validate_inference_dataframe(
    df: pd.DataFrame,
    *,
    raw_required_columns: tuple[str, ...],
    exposure_column: str,
    categorical_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    spline_column: str,
    derive_zone_from_postcode: bool,
    max_exposure: float,
    min_ageph: int,
    max_ageph: int,
) -> pd.DataFrame:
    """Validate explanatory features and exposure bounds prior to transformation or prediction.

    Explicitly ignores any target columns if present and rejects missing, non-finite,
    or out-of-domain values.
    """
    if df.empty:
        raise DataQualityError("Input DataFrame for transformation/prediction is empty.")

    missing = [col for col in raw_required_columns if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required input column(s): {missing}")

    work = derive_explanatory_columns(df, derive_zone_from_postcode=derive_zone_from_postcode)

    numeric_to_check = (exposure_column, spline_column, *numeric_columns)
    for col in numeric_to_check:
        series = work[col]
        if pd.api.types.is_bool_dtype(series):
            raise SchemaValidationError(f"Numeric feature '{col}' cannot be boolean.")
        try:
            numeric = pd.to_numeric(series, errors="raise").astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"Feature '{col}' contains non-numeric values during inference."
            ) from exc
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            bad_n = int((numeric.isna() | ~np.isfinite(numeric)).sum())
            raise DataQualityError(f"Feature '{col}' has {bad_n} missing or non-finite value(s).")
        work[col] = numeric

    bad_expo = (work[exposure_column] <= 0.0) | (work[exposure_column] > max_exposure)
    if bad_expo.any():
        raise DataQualityError(
            f"Exposure '{exposure_column}' must lie in (0, {max_exposure}]; "
            f"found {int(bad_expo.sum())} invalid row(s)."
        )

    bad_age = (work[spline_column] < min_ageph) | (work[spline_column] > max_ageph)
    if bad_age.any():
        raise DataQualityError(
            f"Column '{spline_column}' must lie in [{min_ageph}, {max_ageph}]; "
            f"found {int(bad_age.sum())} out-of-bounds row(s)."
        )

    for bound_col, allow_zero in (("bm", True), ("agec", True), ("power", False)):
        if bound_col in numeric_columns:
            invalid = (work[bound_col] < 0.0) if allow_zero else (work[bound_col] <= 0.0)
            if invalid.any():
                op = ">= 0" if allow_zero else "> 0"
                raise DataQualityError(
                    f"Feature '{bound_col}' must be {op}; "
                    f"found {int(invalid.sum())} invalid row(s)."
                )

    for cat_col in categorical_columns:
        norm = normalize_categorical_series(work[cat_col])
        empty_mask = norm.isna() | (norm.astype(str).str.len() == 0)
        if empty_mask.any():
            raise DataQualityError(
                f"Categorical feature '{cat_col}' has {int(empty_mask.sum())} missing value(s)."
            )
        work[cat_col] = norm.astype(str)

    return work


class FeatureTransformer:
    """Stateless-at-inference transformer encoding categoricals and driver-age splines.

    All fitted parameters reside inside an immutable ``FeatureSpec`` dataclass so
    serialization and compatibility checks remain transparent and deterministic.
    """

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self._spec = spec

    @property
    def spec(self) -> FeatureSpec:
        """Return the fitted immutable ``FeatureSpec``."""
        if self._spec is None:
            raise InferenceError("FeatureTransformer has not been fitted yet.")
        return self._spec

    def fit(self, df: pd.DataFrame, config: PricingConfig) -> FeatureTransformer:
        """Fit base levels and spline knot locations on training data.

        Args:
            df: Training portfolio DataFrame.
            config: Validated pipeline configuration.

        Returns:
            Self for method chaining.
        """
        feat_cfg = config.features
        data_cfg = config.data
        raw_req = raw_required_columns_for_config(feat_cfg)

        work = validate_inference_dataframe(
            df,
            raw_required_columns=raw_req,
            exposure_column=feat_cfg.exposure_column,
            categorical_columns=feat_cfg.categorical_columns,
            numeric_columns=feat_cfg.numeric_columns,
            spline_column=feat_cfg.spline_column,
            derive_zone_from_postcode=feat_cfg.derive_zone_from_postcode,
            max_exposure=data_cfg.max_exposure,
            min_ageph=data_cfg.min_ageph,
            max_ageph=data_cfg.max_ageph,
        )

        cat_specs: list[CategoricalLevelSpec] = []
        design_cols: list[str] = ["intercept"]
        for cat_col in feat_cfg.categorical_columns:
            cat_spec = select_exposure_weighted_base_level(
                work,
                column=cat_col,
                exposure_column=feat_cfg.exposure_column,
            )
            cat_specs.append(cat_spec)
            for lvl in cat_spec.levels[1:]:
                design_cols.append(f"{cat_col}[{lvl}]")
            LOGGER.info(
                "Selected base level for '%s': '%s' (levels=%s)",
                cat_col,
                cat_spec.base_level,
                list(cat_spec.levels),
            )

        raw_knots = np.quantile(
            work[feat_cfg.spline_column].to_numpy(dtype=np.float64),
            feat_cfg.spline_knot_quantiles,
        )
        # Deduplicate knots in case of narrow synthetic training slices.
        unique_knots = np.unique(np.round(raw_knots, decimals=8))
        if len(unique_knots) < 3:
            raise ModelTrainingError(
                f"Insufficient distinct quantile knots for '{feat_cfg.spline_column}': "
                f"{unique_knots}."
            )
        knots_tuple = tuple(float(k) for k in unique_knots)
        spline_feature_names = tuple(
            f"{feat_cfg.spline_column}_ns_{idx}" for idx in range(len(knots_tuple) - 1)
        )
        design_cols.extend(spline_feature_names)
        design_cols.extend(feat_cfg.numeric_columns)

        spline_spec = SplineSpec(
            column=feat_cfg.spline_column,
            knots=knots_tuple,
            reference_value=feat_cfg.spline_reference_age,
            feature_names=spline_feature_names,
        )
        self._spec = FeatureSpec(
            raw_required_columns=raw_req,
            exposure_column=feat_cfg.exposure_column,
            derive_zone_from_postcode=feat_cfg.derive_zone_from_postcode,
            categorical_specs=tuple(cat_specs),
            numeric_columns=feat_cfg.numeric_columns,
            spline_spec=spline_spec,
            design_columns=tuple(design_cols),
            unknown_category_policy=feat_cfg.unknown_category_policy,
            max_exposure=data_cfg.max_exposure,
            min_ageph=data_cfg.min_ageph,
            max_ageph=data_cfg.max_ageph,
        )
        LOGGER.info(
            "Fitted FeatureTransformer: %d design columns, spline knots=%s",
            len(self._spec.design_columns),
            [round(k, 2) for k in knots_tuple],
        )
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Transform raw inputs into the exact GLM design matrix and exposure vector.

        Raises:
            UnknownCategoryError: When unseen categorical levels are present and
                ``unknown_category_policy == 'raise'``.
        """
        spec = self.spec
        cat_cols = tuple(c.column for c in spec.categorical_specs)
        work = validate_inference_dataframe(
            df,
            raw_required_columns=spec.raw_required_columns,
            exposure_column=spec.exposure_column,
            categorical_columns=cat_cols,
            numeric_columns=spec.numeric_columns,
            spline_column=spec.spline_spec.column,
            derive_zone_from_postcode=spec.derive_zone_from_postcode,
            max_exposure=spec.max_exposure,
            min_ageph=spec.min_ageph,
            max_ageph=spec.max_ageph,
        )

        columns_data: dict[str, np.ndarray] = {
            "intercept": np.ones(len(work), dtype=np.float64),
        }

        for cat_spec in spec.categorical_specs:
            allowed = set(cat_spec.levels)
            series = work[cat_spec.column]
            observed_levels = set(series.unique().tolist())
            unknown = sorted(observed_levels - allowed)
            if unknown:
                if spec.unknown_category_policy == "raise":
                    raise UnknownCategoryError(
                        column=cat_spec.column,
                        unknown_values=unknown,
                        allowed_values=list(cat_spec.levels),
                    )
                unknown_mask = series.isin(unknown)
                LOGGER.warning(
                    "Unknown level(s) %s in '%s' on %d row(s); mapping to base level '%s'.",
                    unknown,
                    cat_spec.column,
                    int(unknown_mask.sum()),
                    cat_spec.base_level,
                )
                series = series.where(~unknown_mask, other=cat_spec.base_level)

            for lvl in cat_spec.levels[1:]:
                columns_data[f"{cat_spec.column}[{lvl}]"] = (series == lvl).to_numpy(
                    dtype=np.float64
                )

        spline_basis = compute_natural_spline_basis(
            work[spec.spline_spec.column].to_numpy(dtype=np.float64),
            knots=spec.spline_spec.knots,
            reference_value=spec.spline_spec.reference_value,
        )
        for idx, col_name in enumerate(spec.spline_spec.feature_names):
            columns_data[col_name] = spline_basis[:, idx]

        for num_col in spec.numeric_columns:
            columns_data[num_col] = work[num_col].to_numpy(dtype=np.float64)

        design_matrix = pd.DataFrame(columns_data, index=work.index)[list(spec.design_columns)]
        exposure = work[spec.exposure_column].to_numpy(dtype=np.float64)
        return design_matrix, exposure
