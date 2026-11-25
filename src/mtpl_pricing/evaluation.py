"""Actuarial evaluation metrics, Lorenz/lift diagnostics, and publication figures."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from mtpl_pricing.config import EvaluationConfig
from mtpl_pricing.data import normalize_categorical_series
from mtpl_pricing.exceptions import DataQualityError, InferenceError
from mtpl_pricing.features import FeatureSpec, derive_explanatory_columns
from mtpl_pricing.models import PricingModel, compute_spline_relativity_curve

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Out-of-sample actuarial performance and calibration metrics."""

    n_test_policies: int
    total_test_exposure: float
    total_observed_claims: int
    total_predicted_claims: float
    total_observed_amount: float
    total_predicted_amount: float
    poisson_deviance: float
    gamma_deviance: float
    frequency_observed_to_predicted: float
    severity_observed_to_predicted: float
    pure_premium_observed_to_predicted: float
    raw_gini: float
    oracle_gini: float
    normalized_gini: float
    frequency_pearson_dispersion: float
    severity_pearson_dispersion: float

    def to_dict(self) -> dict[str, int | float]:
        """Convert metrics dataclass into a JSON-serializable dictionary."""
        return asdict(self)


def compute_lorenz_curve(
    y_true: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute an exposure-weighted Lorenz curve and Gini coefficient.

    Rows sharing an identical ``score`` are aggregated before accumulating so that
    the resulting Lorenz curve and Gini index are invariant to row permutation.

    Args:
        y_true: Observed loss amounts (non-negative).
        score: Predicted annualized pure premium rates used for ranking.
        weight: Policy exposures (strictly positive).

    Returns:
        Tuple of ``(cum_exposure_share, cum_loss_share, gini)``.
    """
    y_arr = np.asarray(y_true, dtype=np.float64)
    s_arr = np.asarray(score, dtype=np.float64)
    w_arr = np.asarray(weight, dtype=np.float64)

    if not (len(y_arr) == len(s_arr) == len(w_arr)) or len(y_arr) == 0:
        raise DataQualityError("Lorenz inputs must be non-empty arrays of identical length.")
    if np.any(w_arr <= 0.0) or np.any(y_arr < 0.0):
        raise DataQualityError(
            "Lorenz computation requires positive weights and non-negative losses."
        )

    total_w = float(np.sum(w_arr))
    total_y = float(np.sum(y_arr))
    if total_y <= 0.0:
        raise DataQualityError(
            "Total observed claim amount must be positive to compute Lorenz curve."
        )

    grouped = (
        pd.DataFrame({"score": s_arr, "weight": w_arr, "loss": y_arr})
        .groupby("score", sort=True, as_index=False)[["weight", "loss"]]
        .sum()
    )
    cum_w = np.concatenate(([0.0], np.cumsum(grouped["weight"].to_numpy()) / total_w))
    cum_y = np.concatenate(([0.0], np.cumsum(grouped["loss"].to_numpy()) / total_y))
    area_under_lorenz = float(np.trapezoid(cum_y, cum_w))
    gini = 1.0 - 2.0 * area_under_lorenz
    return cum_w, cum_y, gini


def compute_normalized_gini(
    amount: np.ndarray,
    predicted_annual_pure_premium: np.ndarray,
    exposure: np.ndarray,
) -> tuple[float, float, float]:
    """Compute raw, oracle, and normalized exposure-weighted Gini indices."""
    amt = np.asarray(amount, dtype=np.float64)
    pred_rate = np.asarray(predicted_annual_pure_premium, dtype=np.float64)
    expo = np.asarray(exposure, dtype=np.float64)

    _, _, raw_gini = compute_lorenz_curve(amt, pred_rate, expo)
    _, _, oracle_gini = compute_lorenz_curve(amt, amt / expo, expo)
    if oracle_gini <= 0.0:
        raise InferenceError("Oracle Gini index is non-positive; cannot normalize.")
    return raw_gini, oracle_gini, raw_gini / oracle_gini


def compute_decile_lift_table(
    amount: np.ndarray,
    predicted_pure_premium: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute exposure-balanced calibration bins ordered by predicted annualized pure premium."""
    if n_bins < 2:
        raise DataQualityError(f"n_bins must be >= 2, got {n_bins}.")
    amt = np.asarray(amount, dtype=np.float64)
    pp = np.asarray(predicted_pure_premium, dtype=np.float64)
    expo = np.asarray(exposure, dtype=np.float64)
    rate = pp / expo

    order = np.argsort(rate, kind="mergesort")
    sorted_expo = expo[order]
    cum_expo = np.cumsum(sorted_expo)
    total_expo = float(cum_expo[-1])
    # Assign exposure-balanced bin indices 0 .. n_bins - 1 using midpoints to handle ties cleanly.
    mid_expo = cum_expo - 0.5 * sorted_expo
    bin_idx = np.clip((mid_expo / total_expo * n_bins).astype(int), 0, n_bins - 1)

    frame = pd.DataFrame(
        {
            "decile": bin_idx + 1,
            "exposure": sorted_expo,
            "observed_amount": amt[order],
            "predicted_amount": pp[order],
        }
    )
    grouped = frame.groupby("decile", sort=True, as_index=False).agg(
        policy_count=("exposure", "size"),
        exposure=("exposure", "sum"),
        observed_amount=("observed_amount", "sum"),
        predicted_amount=("predicted_amount", "sum"),
    )
    grouped["observed_rate"] = grouped["observed_amount"] / grouped["exposure"]
    grouped["predicted_rate"] = grouped["predicted_amount"] / grouped["exposure"]
    grouped["observed_to_predicted_ratio"] = np.where(
        grouped["predicted_amount"] > 0.0,
        grouped["observed_amount"] / grouped["predicted_amount"],
        np.nan,
    )
    return grouped


def compute_one_way_relativities(
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    """Compute univariate empirical frequency relativities relative to each factor's base level."""
    work = derive_explanatory_columns(
        train_df,
        derive_zone_from_postcode=feature_spec.derive_zone_from_postcode,
    )
    expo_col = feature_spec.exposure_column

    rows: list[dict[str, object]] = []
    for cat_spec in feature_spec.categorical_specs:
        col = cat_spec.column
        levels = normalize_categorical_series(work[col]).astype(str)
        agg = (
            pd.DataFrame(
                {
                    "level": levels,
                    "exposure": work[expo_col].to_numpy(dtype=np.float64),
                    "nclaims": work["nclaims"].to_numpy(dtype=np.float64),
                }
            )
            .groupby("level", sort=True)
            .sum()
        )
        agg["empirical_frequency"] = agg["nclaims"] / agg["exposure"]
        base_freq = float(agg.loc[cat_spec.base_level, "empirical_frequency"])
        for lvl in cat_spec.levels:
            emp_freq = float(agg.loc[lvl, "empirical_frequency"])
            rows.append(
                {
                    "feature": col,
                    "level": lvl,
                    "is_base_level": lvl == cat_spec.base_level,
                    "exposure": float(agg.loc[lvl, "exposure"]),
                    "empirical_frequency": emp_freq,
                    "one_way_relativity": emp_freq / base_freq if base_freq > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_evaluation_metrics(
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    pricing_model: PricingModel,
) -> EvaluationMetrics:
    """Evaluate out-of-sample Poisson/Gamma deviances, balance ratios, and Gini."""
    expo = test_df["expo"].to_numpy(dtype=np.float64)
    nclaims = test_df["nclaims"].to_numpy(dtype=np.float64)
    amount = test_df["amount"].to_numpy(dtype=np.float64)
    average = test_df["average"].to_numpy(dtype=np.float64)

    pred_freq = test_predictions["predicted_frequency"].to_numpy(dtype=np.float64)
    pred_sev = test_predictions["predicted_severity"].to_numpy(dtype=np.float64)
    pred_pp = test_predictions["predicted_pure_premium"].to_numpy(dtype=np.float64)
    pred_annual_pp = test_predictions["predicted_annual_pure_premium"].to_numpy(dtype=np.float64)

    poisson_dev = float(mean_poisson_deviance(nclaims, pred_freq))

    claim_mask = nclaims > 0.0
    if not np.any(claim_mask):
        raise DataQualityError("Test split has no claim-bearing policies for Gamma deviance.")
    gamma_dev = float(
        mean_gamma_deviance(
            average[claim_mask],
            pred_sev[claim_mask],
            sample_weight=nclaims[claim_mask],
        )
    )

    obs_claims = int(np.sum(nclaims))
    pred_claims_sum = float(np.sum(pred_freq))
    obs_amount_sum = float(np.sum(amount))
    pred_amount_sum = float(np.sum(pred_pp))
    pred_sev_weighted = float(np.sum(pred_sev[claim_mask] * nclaims[claim_mask]))

    raw_gini, oracle_gini, norm_gini = compute_normalized_gini(
        amount=amount,
        predicted_annual_pure_premium=pred_annual_pp,
        exposure=expo,
    )

    return EvaluationMetrics(
        n_test_policies=len(test_df),
        total_test_exposure=float(np.sum(expo)),
        total_observed_claims=obs_claims,
        total_predicted_claims=pred_claims_sum,
        total_observed_amount=obs_amount_sum,
        total_predicted_amount=pred_amount_sum,
        poisson_deviance=poisson_dev,
        gamma_deviance=gamma_dev,
        frequency_observed_to_predicted=float(obs_claims / pred_claims_sum),
        severity_observed_to_predicted=float(obs_amount_sum / pred_sev_weighted),
        pure_premium_observed_to_predicted=float(obs_amount_sum / pred_amount_sum),
        raw_gini=float(raw_gini),
        oracle_gini=float(oracle_gini),
        normalized_gini=float(norm_gini),
        frequency_pearson_dispersion=pricing_model.frequency_model.pearson_dispersion,
        severity_pearson_dispersion=pricing_model.severity_model.pearson_dispersion,
    )


def _save_figure_atomic(fig: Figure, output_path: Path, dpi: int = 160) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp.png",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fig.savefig(fh, format="png", dpi=dpi)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, output_path)
    except OSError as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise InferenceError(f"Failed writing figure '{output_path}': {exc}") from exc


def plot_one_way_vs_glm_relativities(
    one_way_df: pd.DataFrame,
    relativities_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot one-way empirical vs multivariate Poisson GLM frequency relativities."""
    glm_freq = relativities_df[
        (relativities_df["model"] == "poisson") & (relativities_df["level"] != "per_unit")
    ].copy()
    merged = one_way_df.merge(
        glm_freq[["feature", "level", "relativity", "ci_lower", "ci_upper"]],
        on=["feature", "level"],
        how="inner",
    )
    merged["label"] = merged["feature"] + " = " + merged["level"]
    merged = merged.iloc[::-1].reset_index(drop=True)

    fig_height = max(5.5, 0.34 * len(merged) + 1.8)
    fig = Figure(figsize=(10.5, fig_height))
    ax = fig.add_subplot(111)

    y_pos = np.arange(len(merged))
    ax.axvline(1.0, color="#6b7280", linestyle="--", linewidth=1.1, zorder=1)

    # Plot multivariate GLM point estimate + 95% confidence interval.
    x_glm = merged["relativity"].to_numpy(dtype=float)
    x_err_low = np.maximum(x_glm - merged["ci_lower"].to_numpy(dtype=float), 0.0)
    x_err_high = np.maximum(merged["ci_upper"].to_numpy(dtype=float) - x_glm, 0.0)
    ax.errorbar(
        x_glm,
        y_pos + 0.14,
        xerr=[x_err_low, x_err_high],
        fmt="D",
        color="#1d4ed8",
        ecolor="#93c5fd",
        elinewidth=2.0,
        capsize=3,
        markersize=5.5,
        label="Multivariate Poisson GLM (95% CI)",
        zorder=3,
    )

    ax.scatter(
        merged["one_way_relativity"].to_numpy(dtype=float),
        y_pos - 0.14,
        marker="o",
        color="#d97706",
        s=46,
        label="One-Way Empirical Frequency Relativity",
        zorder=4,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["label"], fontsize=9)
    ax.set_xlabel("Frequency Relativity vs Exposure-Weighted Base Level (Log Scale)", fontsize=10)
    ax.set_xscale("log")
    ax.set_title(
        "Composition Bias: One-Way Empirical vs. Multivariate Poisson GLM Relativities",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()

    _save_figure_atomic(fig, output_path)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path


def plot_age_spline_relativity(
    pricing_model: PricingModel,
    feature_spec: FeatureSpec,
    train_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot driver age (``ageph``) natural spline relativities with 95% CIs for freq & sev."""
    spline_col = feature_spec.spline_spec.column
    age_min = int(feature_spec.min_ageph)
    age_max = int(min(feature_spec.max_ageph, float(train_df[spline_col].max())))
    ages = np.linspace(float(age_min), float(age_max), 200)

    freq_curve = compute_spline_relativity_curve(
        pricing_model.frequency_model,
        feature_spec=feature_spec,
        ages=ages,
        confidence_level=pricing_model.confidence_level,
    )
    sev_curve = compute_spline_relativity_curve(
        pricing_model.severity_model,
        feature_spec=feature_spec,
        ages=ages,
        confidence_level=pricing_model.confidence_level,
    )

    age_expo = (
        train_df.groupby(train_df[spline_col].round().astype(int))[feature_spec.exposure_column]
        .sum()
        .sort_index()
    )

    fig = Figure(figsize=(12.5, 5.2))
    panels = (
        (1, freq_curve, "Frequency Relativity (Poisson GLM)", "#1d4ed8", "#bfdbfe"),
        (2, sev_curve, "Severity Relativity (Gamma GLM)", "#b45309", "#fde68a"),
    )
    ref_age = feature_spec.spline_spec.reference_value

    for idx, curve_df, title, line_color, fill_color in panels:
        ax = fig.add_subplot(1, 2, idx)
        ax_bg = ax.twinx()
        ax_bg.bar(
            age_expo.index.to_numpy(),
            age_expo.to_numpy(),
            width=0.85,
            color="#e5e7eb",
            alpha=0.75,
            zorder=0,
        )
        ax_bg.set_ylabel("Training Exposure (policy-years)", color="#6b7280", fontsize=8.5)
        ax_bg.tick_params(axis="y", labelcolor="#6b7280", labelsize=8)
        ax_bg.set_ylim(0.0, float(age_expo.max()) * 3.2)

        ax.set_zorder(ax_bg.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.fill_between(
            curve_df["ageph"],
            curve_df["ci_lower"],
            curve_df["ci_upper"],
            color=fill_color,
            alpha=0.55,
            label="95% Confidence Band",
        )
        ax.plot(
            curve_df["ageph"],
            curve_df["relativity"],
            color=line_color,
            linewidth=2.3,
            label="Natural Cubic Spline Relativity",
        )
        ax.axhline(1.0, color="#4b5563", linestyle="--", linewidth=1.0)
        ax.axvline(
            ref_age,
            color="#4b5563",
            linestyle=":",
            linewidth=1.1,
            label=f"Reference age = {ref_age:.0f}",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Policyholder Age (ageph)", fontsize=10)
        ax.set_ylabel("Multiplicative Relativity", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Non-Linear Driver Age Effect (Natural Cubic Spline, beMTPL97)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    _save_figure_atomic(fig, output_path)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path


def plot_lorenz_and_lift_chart(
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    lift_df: pd.DataFrame,
    metrics: EvaluationMetrics,
    output_path: Path,
) -> Path:
    """Plot out-of-sample exposure-weighted Lorenz curve and decile calibration lift chart."""
    expo = test_df["expo"].to_numpy(dtype=np.float64)
    amt = test_df["amount"].to_numpy(dtype=np.float64)
    pred_rate = test_predictions["predicted_annual_pure_premium"].to_numpy(dtype=np.float64)

    w_model, y_model, _ = compute_lorenz_curve(amt, pred_rate, expo)
    w_oracle, y_oracle, _ = compute_lorenz_curve(amt, amt / expo, expo)

    fig = Figure(figsize=(12.5, 5.2))

    # Left panel: Lorenz Curve
    ax_lor = fig.add_subplot(1, 2, 1)
    ax_lor.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#6b7280", label="Uniform Baseline")
    ax_lor.plot(
        w_model,
        y_model,
        color="#1d4ed8",
        linewidth=2.2,
        label=f"GLM Pure Premium (Norm. Gini = {metrics.normalized_gini:.3f})",
    )
    ax_lor.plot(
        w_oracle,
        y_oracle,
        color="#059669",
        linestyle="-.",
        linewidth=1.8,
        label=f"Oracle Curve (Gini = {metrics.oracle_gini:.3f})",
    )
    ax_lor.set_title("Test-Set Exposure-Weighted Lorenz Curve", fontsize=11, fontweight="bold")
    ax_lor.set_xlabel("Cumulative Exposure Share", fontsize=10)
    ax_lor.set_ylabel("Cumulative Observed Claim Amount Share", fontsize=10)
    ax_lor.set_xlim(0.0, 1.0)
    ax_lor.set_ylim(0.0, 1.02)
    ax_lor.grid(True, linestyle=":", alpha=0.5)
    ax_lor.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Right panel: Decile Lift / Calibration Chart
    ax_lift = fig.add_subplot(1, 2, 2)
    deciles = lift_df["decile"].to_numpy()
    width = 0.38
    ax_lift.bar(
        deciles - width / 2,
        lift_df["observed_rate"].to_numpy(),
        width=width,
        color="#f59e0b",
        label="Observed Pure Premium Rate",
    )
    ax_lift.bar(
        deciles + width / 2,
        lift_df["predicted_rate"].to_numpy(),
        width=width,
        color="#1d4ed8",
        label="Predicted Pure Premium Rate",
    )
    ax_lift.set_xticks(deciles)
    ax_lift.set_title(
        f"Test-Set Decile Calibration (O/P = {metrics.pure_premium_observed_to_predicted:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax_lift.set_xlabel("Predicted Risk Decile (Exposure-Weighted)", fontsize=10)
    ax_lift.set_ylabel("Annualized Pure Premium (EUR / policy-year)", fontsize=10)
    ax_lift.grid(axis="y", linestyle=":", alpha=0.5)
    ax_lift.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.tight_layout()
    _save_figure_atomic(fig, output_path)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path


def atomic_write_json(payload: dict[str, object], output_path: Path) -> None:
    """Atomically write formatted UTF-8 JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp.json",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, output_path)
    except OSError as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise InferenceError(f"Failed writing JSON file '{output_path}': {exc}") from exc


def generate_evaluation_reports(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    pricing_model: PricingModel,
    feature_spec: FeatureSpec,
    relativities_df: pd.DataFrame,
    eval_cfg: EvaluationConfig,
) -> tuple[EvaluationMetrics, dict[str, Path]]:
    """Compute all test diagnostics and atomically persist metrics, relativities, and 3 figures."""
    metrics = compute_evaluation_metrics(
        test_df=test_df,
        test_predictions=test_predictions,
        pricing_model=pricing_model,
    )
    one_way_df = compute_one_way_relativities(train_df=train_df, feature_spec=feature_spec)
    lift_df = compute_decile_lift_table(
        amount=test_df["amount"].to_numpy(dtype=np.float64),
        predicted_pure_premium=test_predictions["predicted_pure_premium"].to_numpy(
            dtype=np.float64
        ),
        exposure=test_df["expo"].to_numpy(dtype=np.float64),
        n_bins=eval_cfg.n_lift_bins,
    )

    metrics_path = eval_cfg.reports_dir / eval_cfg.metrics_filename
    relativities_path = eval_cfg.reports_dir / eval_cfg.relativities_filename
    fig_one_way = eval_cfg.figures_dir / "one_way_vs_glm_relativities.png"
    fig_spline = eval_cfg.figures_dir / "age_spline_relativity.png"
    fig_lorenz = eval_cfg.figures_dir / "lorenz_and_lift_chart.png"

    atomic_write_json(dict(metrics.to_dict()), metrics_path)

    eval_cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=relativities_path.parent,
            prefix=f".{relativities_path.name}.",
            suffix=".tmp.csv",
            delete=False,
        ) as fh:
            tmp_csv = Path(fh.name)
            relativities_df.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_csv, relativities_path)
    except OSError as exc:
        if tmp_csv is not None and tmp_csv.exists():
            tmp_csv.unlink(missing_ok=True)
        raise InferenceError(
            f"Failed writing relativities CSV '{relativities_path}': {exc}"
        ) from exc

    plot_one_way_vs_glm_relativities(one_way_df, relativities_df, fig_one_way)
    plot_age_spline_relativity(pricing_model, feature_spec, train_df, fig_spline)
    plot_lorenz_and_lift_chart(test_df, test_predictions, lift_df, metrics, fig_lorenz)

    LOGGER.info(
        "Evaluation complete: Poisson dev=%.4f, Gamma dev=%.4f, PurePrem O/P=%.4f, NormGini=%.4f",
        metrics.poisson_deviance,
        metrics.gamma_deviance,
        metrics.pure_premium_observed_to_predicted,
        metrics.normalized_gini,
    )
    outputs = {
        "metrics": metrics_path,
        "relativities": relativities_path,
        "one_way_vs_glm_relativities": fig_one_way,
        "age_spline_relativity": fig_spline,
        "lorenz_and_lift_chart": fig_lorenz,
    }
    return metrics, outputs
