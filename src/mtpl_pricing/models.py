"""Frequency-Severity GLM modeling, pure premium pricing, and relativities extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import stats

from mtpl_pricing.config import ModelConfig, PricingConfig
from mtpl_pricing.exceptions import InferenceError, ModelTrainingError, SchemaValidationError
from mtpl_pricing.features import FeatureSpec

LOGGER = logging.getLogger(__name__)

ModelFamily = Literal["poisson", "gamma"]


@dataclass(frozen=True)
class FittedGLM:
    """Immutable, lightweight GLM state decoupled from training-data references."""

    family: ModelFamily
    link: Literal["log"]
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    std_errors: tuple[float, ...]
    p_values: tuple[float, ...]
    cov_matrix: tuple[tuple[float, ...], ...]
    pearson_dispersion: float
    scale: float
    deviance: float
    null_deviance: float
    pearson_chi2: float
    aic: float
    df_resid: int
    df_model: int
    n_obs: int
    converged: bool
    n_iter: int

    def coef_array(self) -> NDArray[np.float64]:
        """Return coefficient vector as a 1D float64 array."""
        return np.asarray(self.coefficients, dtype=np.float64)

    def cov_array(self) -> NDArray[np.float64]:
        """Return parameter covariance matrix as a 2D float64 array."""
        return np.asarray(self.cov_matrix, dtype=np.float64)

    def predict_linear_predictor(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Compute linear predictor ``X @ beta`` after strict column order validation."""
        expected = list(self.feature_names)
        if list(X.columns) != expected:
            raise SchemaValidationError(
                "Design matrix columns do not match fitted GLM schema. "
                f"Expected {expected}, got {list(X.columns)}."
            )
        X_mat: NDArray[np.float64] = X.to_numpy(dtype=np.float64)
        if not np.isfinite(X_mat).all():
            raise InferenceError("Design matrix contains missing or non-finite values.")
        return X_mat @ self.coef_array()

    def predict(
        self, X: pd.DataFrame, offset: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Predict expected response ``exp(X @ beta + offset)`` with numeric safety checks."""
        eta = self.predict_linear_predictor(X)
        if offset is not None:
            offset_arr: NDArray[np.float64] = np.asarray(offset, dtype=np.float64)
            if offset_arr.shape != (len(X),) or not np.isfinite(offset_arr).all():
                raise InferenceError("Prediction offset must be finite and match row count.")
            eta = eta + offset_arr
        if np.any(eta > 700.0):
            raise InferenceError(
                "Linear predictor overflow risk (eta > 700); check extreme inference covariates."
            )
        preds = cast(NDArray[np.float64], np.exp(eta))
        if not np.isfinite(preds).all() or np.any(preds <= 0.0):
            raise InferenceError("Model produced invalid (non-finite or non-positive) predictions.")
        return preds


@dataclass(frozen=True)
class PricingModel:
    """Paired Poisson frequency and Gamma severity GLMs forming a pure-premium model."""

    frequency_model: FittedGLM
    severity_model: FittedGLM
    confidence_level: float

    def predict_components(self, X: pd.DataFrame, exposure: np.ndarray) -> pd.DataFrame:
        """Predict frequency, annualized frequency, severity, and pure premium."""
        expo_arr = np.asarray(exposure, dtype=np.float64)
        if (
            expo_arr.shape != (len(X),)
            or not np.isfinite(expo_arr).all()
            or np.any(expo_arr <= 0.0)
        ):
            raise InferenceError(
                "Exposure vector must be strictly positive, finite, and match rows."
            )

        freq = self.frequency_model.predict(X, offset=np.log(expo_arr))
        annual_freq = self.frequency_model.predict(X, offset=None)
        sev = self.severity_model.predict(X, offset=None)
        pure_premium = freq * sev
        annual_pure_premium = annual_freq * sev

        result = pd.DataFrame(
            {
                "predicted_frequency": freq,
                "predicted_annual_frequency": annual_freq,
                "predicted_severity": sev,
                "predicted_pure_premium": pure_premium,
                "predicted_annual_pure_premium": annual_pure_premium,
            },
            index=X.index,
        )
        if not np.isfinite(result.to_numpy()).all():
            raise InferenceError("Predicted pricing components contain non-finite values.")
        return result


def _validate_design_matrix_for_fit(X: pd.DataFrame, min_rows: int) -> None:
    if len(X) < min_rows:
        raise ModelTrainingError(
            f"Insufficient observations ({len(X)}) to fit GLM (need >= {min_rows})."
        )
    if X.shape[1] == 0:
        raise ModelTrainingError("Design matrix has zero columns.")
    mat = X.to_numpy(dtype=np.float64)
    if not np.isfinite(mat).all():
        raise ModelTrainingError("Training design matrix contains missing or non-finite values.")
    if len(X) <= X.shape[1]:
        raise ModelTrainingError(
            f"Underdetermined system: {len(X)} observations <= {X.shape[1]} parameters."
        )
    rank = int(np.linalg.matrix_rank(mat))
    if rank < X.shape[1]:
        raise ModelTrainingError(
            f"Rank-deficient design matrix (rank={rank} < cols={X.shape[1]}). "
            "Check collinear features or empty categories in training split."
        )


def _extract_fitted_glm(
    result: sm.genmod.generalized_linear_model.GLMResultsWrapper,
    family: ModelFamily,
    feature_names: tuple[str, ...],
) -> FittedGLM:
    if not bool(getattr(result, "converged", False)):
        raise ModelTrainingError(f"{family.capitalize()} GLM failed to converge in IRLS.")

    params = np.asarray(result.params, dtype=np.float64)
    bse = np.asarray(result.bse, dtype=np.float64)
    pvals = np.asarray(result.pvalues, dtype=np.float64)
    cov = np.asarray(result.cov_params(), dtype=np.float64)
    if not (np.isfinite(params).all() and np.isfinite(bse).all() and np.isfinite(cov).all()):
        raise ModelTrainingError(
            f"{family.capitalize()} GLM produced non-finite parameter estimates."
        )

    df_resid = int(result.df_resid)
    if df_resid <= 0:
        raise ModelTrainingError(
            f"{family.capitalize()} GLM has non-positive residual df: {df_resid}."
        )
    pearson_chi2 = float(result.pearson_chi2)
    pearson_dispersion = pearson_chi2 / float(df_resid)

    return FittedGLM(
        family=family,
        link="log",
        feature_names=feature_names,
        coefficients=tuple(float(v) for v in params),
        std_errors=tuple(float(v) for v in bse),
        p_values=tuple(float(v) for v in pvals),
        cov_matrix=tuple(tuple(float(c) for c in row) for row in cov),
        pearson_dispersion=float(pearson_dispersion),
        scale=float(result.scale),
        deviance=float(result.deviance),
        null_deviance=float(result.null_deviance),
        pearson_chi2=pearson_chi2,
        aic=float(result.aic),
        df_resid=df_resid,
        df_model=int(result.df_model),
        n_obs=int(result.nobs),
        converged=bool(result.converged),
        n_iter=int(getattr(result, "fit_history", {}).get("iteration", 0)),
    )


def fit_frequency_glm(
    X: pd.DataFrame,
    nclaims: np.ndarray | pd.Series,
    exposure: np.ndarray | pd.Series,
    model_cfg: ModelConfig,
) -> FittedGLM:
    """Fit a log-link Poisson GLM for claim counts with ``offset=log(exposure)``."""
    _validate_design_matrix_for_fit(X, min_rows=X.shape[1] + 1)
    y = np.asarray(nclaims, dtype=np.float64)
    expo = np.asarray(exposure, dtype=np.float64)
    if y.shape != (len(X),) or expo.shape != (len(X),):
        raise ModelTrainingError("Target/exposure lengths do not match design matrix rows.")
    if np.any(y < 0.0) or not np.isfinite(y).all():
        raise ModelTrainingError("Frequency target 'nclaims' must be finite and >= 0.")
    if np.any(expo <= 0.0) or not np.isfinite(expo).all():
        raise ModelTrainingError("Frequency offset exposure must be finite and > 0.")

    glm = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(expo),
    )
    result = glm.fit(maxiter=model_cfg.max_iter, tol=model_cfg.tol)
    fitted = _extract_fitted_glm(result, "poisson", tuple(X.columns.tolist()))
    LOGGER.info(
        "Fitted Poisson Frequency GLM: n_obs=%d, deviance=%.2f, "
        "pearson_dispersion=%.4f, converged=%s",
        fitted.n_obs,
        fitted.deviance,
        fitted.pearson_dispersion,
        fitted.converged,
    )
    return fitted


def fit_severity_glm(
    X: pd.DataFrame,
    average_cost: np.ndarray | pd.Series,
    nclaims: np.ndarray | pd.Series,
    model_cfg: ModelConfig,
) -> FittedGLM:
    """Fit a log-link Gamma GLM on claim-bearing policies weighted by ``var_weights=nclaims``.

    Because ``average = amount / nclaims`` represents the policy-level average across ``nclaims``
    i.i.d. claims, its conditional variance scales as ``phi * mu^2 / nclaims``. Passing
    ``var_weights=nclaims`` yields exact equivalence with claim-level Gamma estimation.
    """
    counts = np.asarray(nclaims, dtype=np.float64)
    avg = np.asarray(average_cost, dtype=np.float64)
    if counts.shape != (len(X),) or avg.shape != (len(X),):
        raise ModelTrainingError("Severity target/weights lengths do not match design matrix rows.")

    claim_mask = counts > 0.0
    if not np.any(claim_mask):
        raise ModelTrainingError("Cannot fit severity GLM: zero claim-bearing policies present.")

    X_pos = X.loc[claim_mask]
    y_pos = avg[claim_mask]
    w_pos = counts[claim_mask]

    if np.any(y_pos <= 0.0) or not np.isfinite(y_pos).all():
        raise ModelTrainingError("Severity target 'average' must be strictly positive and finite.")
    _validate_design_matrix_for_fit(X_pos, min_rows=X_pos.shape[1] + 1)

    glm = sm.GLM(
        y_pos,
        X_pos,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=w_pos,
    )
    result = glm.fit(maxiter=model_cfg.max_iter, tol=model_cfg.tol)
    fitted = _extract_fitted_glm(result, "gamma", tuple(X.columns.tolist()))
    LOGGER.info(
        "Fitted Gamma Severity GLM: n_obs=%d, deviance=%.2f, scale=%.4f, converged=%s",
        fitted.n_obs,
        fitted.deviance,
        fitted.scale,
        fitted.converged,
    )
    return fitted


def fit_pricing_model(
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
    config: PricingConfig,
) -> PricingModel:
    """Fit the coupled Frequency-Severity GLM pricing model and verify portfolio balance."""
    expo = train_df[config.features.exposure_column].to_numpy(dtype=np.float64)
    nclaims = train_df["nclaims"].to_numpy(dtype=np.float64)
    average = train_df["average"].to_numpy(dtype=np.float64)
    amount = train_df["amount"].to_numpy(dtype=np.float64)

    freq_glm = fit_frequency_glm(X_train, nclaims=nclaims, exposure=expo, model_cfg=config.model)
    sev_glm = fit_severity_glm(
        X_train, average_cost=average, nclaims=nclaims, model_cfg=config.model
    )

    pricing_model = PricingModel(
        frequency_model=freq_glm,
        severity_model=sev_glm,
        confidence_level=config.model.confidence_level,
    )
    preds = pricing_model.predict_components(X_train, expo)

    freq_balance = float(nclaims.sum() / preds["predicted_frequency"].sum())
    pure_prem_balance = float(amount.sum() / preds["predicted_pure_premium"].sum())
    LOGGER.info(
        "Training portfolio balance check: frequency O/P=%.6f, pure_premium O/P=%.6f",
        freq_balance,
        pure_prem_balance,
    )
    return pricing_model


def _extract_model_relativities(
    glm: FittedGLM,
    feature_spec: FeatureSpec,
    z_crit: float,
) -> list[dict[str, object]]:
    coef_map = dict(zip(glm.feature_names, glm.coefficients, strict=True))
    se_map = dict(zip(glm.feature_names, glm.std_errors, strict=True))
    pval_map = dict(zip(glm.feature_names, glm.p_values, strict=True))

    rows: list[dict[str, object]] = []
    for cat_spec in feature_spec.categorical_specs:
        for lvl in cat_spec.levels:
            if lvl == cat_spec.base_level:
                rows.append(
                    {
                        "model": glm.family,
                        "feature": cat_spec.column,
                        "level": lvl,
                        "is_base_level": True,
                        "coefficient": 0.0,
                        "std_error": 0.0,
                        "p_value": 1.0,
                        "relativity": 1.0,
                        "ci_lower": 1.0,
                        "ci_upper": 1.0,
                    }
                )
            else:
                term = f"{cat_spec.column}[{lvl}]"
                beta = coef_map[term]
                se = se_map[term]
                rows.append(
                    {
                        "model": glm.family,
                        "feature": cat_spec.column,
                        "level": lvl,
                        "is_base_level": False,
                        "coefficient": beta,
                        "std_error": se,
                        "p_value": pval_map[term],
                        "relativity": float(np.exp(beta)),
                        "ci_lower": float(np.exp(beta - z_crit * se)),
                        "ci_upper": float(np.exp(beta + z_crit * se)),
                    }
                )

    for num_col in feature_spec.numeric_columns:
        beta = coef_map[num_col]
        se = se_map[num_col]
        rows.append(
            {
                "model": glm.family,
                "feature": num_col,
                "level": "per_unit",
                "is_base_level": False,
                "coefficient": beta,
                "std_error": se,
                "p_value": pval_map[num_col],
                "relativity": float(np.exp(beta)),
                "ci_lower": float(np.exp(beta - z_crit * se)),
                "ci_upper": float(np.exp(beta + z_crit * se)),
            }
        )
    return rows


def extract_relativities_table(
    pricing_model: PricingModel,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    """Build multiplicative relativities table ``exp(beta)`` with Wald confidence intervals."""
    alpha = 1.0 - pricing_model.confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    rows = _extract_model_relativities(pricing_model.frequency_model, feature_spec, z_crit)
    rows.extend(_extract_model_relativities(pricing_model.severity_model, feature_spec, z_crit))
    return pd.DataFrame(rows)


def compute_spline_relativity_curve(
    glm: FittedGLM,
    feature_spec: FeatureSpec,
    ages: np.ndarray,
    confidence_level: float,
) -> pd.DataFrame:
    """Evaluate spline relativity curve ``exp(B(age) @ beta)`` and delta-method CI band."""
    from mtpl_pricing.features import compute_natural_spline_basis

    spline_spec = feature_spec.spline_spec
    basis = compute_natural_spline_basis(
        ages,
        knots=spline_spec.knots,
        reference_value=spline_spec.reference_value,
    )
    name_to_idx = {name: idx for idx, name in enumerate(glm.feature_names)}
    indices = [name_to_idx[name] for name in spline_spec.feature_names]

    beta_sub = glm.coef_array()[indices]
    cov_sub = glm.cov_array()[np.ix_(indices, indices)]

    eta = basis @ beta_sub
    # Variance of linear combination B_i @ beta_sub: diag(B @ Cov @ B.T) computed row-wise.
    var_eta = np.maximum(np.sum((basis @ cov_sub) * basis, axis=1), 0.0)
    se_eta = np.sqrt(var_eta)

    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    return pd.DataFrame(
        {
            "ageph": np.asarray(ages, dtype=np.float64),
            "relativity": np.exp(eta),
            "ci_lower": np.exp(eta - z_crit * se_eta),
            "ci_upper": np.exp(eta + z_crit * se_eta),
        }
    )
