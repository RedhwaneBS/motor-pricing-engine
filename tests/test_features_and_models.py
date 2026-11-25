"""Unit tests for feature engineering, natural splines, GLMs, and actuarial invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from tests.conftest import make_synthetic_portfolio

from mtpl_pricing.config import PricingConfig
from mtpl_pricing.evaluation import (
    compute_decile_lift_table,
    compute_evaluation_metrics,
    compute_lorenz_curve,
    compute_one_way_relativities,
)
from mtpl_pricing.features import (
    FeatureTransformer,
    compute_natural_spline_basis,
    select_exposure_weighted_base_level,
    split_train_test,
)
from mtpl_pricing.models import (
    compute_spline_relativity_curve,
    extract_relativities_table,
    fit_frequency_glm,
    fit_pricing_model,
    fit_severity_glm,
)


def test_stratified_split_is_deterministic_and_handles_rare_strata(
    temp_config: PricingConfig,
) -> None:
    df = make_synthetic_portfolio(300, seed=11)
    # Force a singleton rare stratum (nclaims = 9) to verify rare-stratum handling.
    df.loc[0, "nclaims"] = 9
    df.loc[0, "claim"] = 1
    df.loc[0, "amount"] = 1800.0
    df.loc[0, "average"] = 200.0

    tr1, te1 = split_train_test(df, temp_config.split, random_seed=42)
    tr2, te2 = split_train_test(df, temp_config.split, random_seed=42)

    assert tr1["id"].tolist() == tr2["id"].tolist()
    assert te1["id"].tolist() == te2["id"].tolist()
    assert set(tr1["id"]).isdisjoint(set(te1["id"]))
    assert len(tr1) + len(te1) == len(df)


def test_base_level_selection_uses_exposure_not_row_count() -> None:
    # 'A' has 3 policies totaling 0.30 exposure; 'B' has 2 policies totaling 1.80 exposure.
    df = pd.DataFrame(
        {
            "cov": ["A", "A", "A", "B", "B"],
            "expo": [0.10, 0.10, 0.10, 0.90, 0.90],
        }
    )
    spec = select_exposure_weighted_base_level(df, column="cov", exposure_column="expo")
    assert spec.base_level == "B"
    assert spec.levels == ("B", "A")


def test_natural_spline_basis_reference_zero_and_linear_tails() -> None:
    knots = (22.0, 35.0, 45.0, 58.0, 74.0)
    ref_age = 45.0

    ref_row = compute_natural_spline_basis([ref_age], knots=knots, reference_value=ref_age)
    assert np.allclose(ref_row, 0.0, atol=1e-12)

    # Below lower boundary knot (15, 18, 21) and above upper knot (76, 82, 88), second diff == 0.
    low_tails = compute_natural_spline_basis(
        [15.0, 18.0, 21.0], knots=knots, reference_value=ref_age
    )
    high_tails = compute_natural_spline_basis(
        [76.0, 82.0, 88.0], knots=knots, reference_value=ref_age
    )
    assert np.allclose(np.diff(low_tails, n=2, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.diff(high_tails, n=2, axis=0), 0.0, atol=1e-12)


def test_poisson_glm_balance_property_and_statsmodels_parity(
    synthetic_portfolio: pd.DataFrame,
    temp_config: PricingConfig,
) -> None:
    train_df, test_df = split_train_test(synthetic_portfolio, temp_config.split, random_seed=42)
    transformer = FeatureTransformer().fit(train_df, temp_config)
    X_train, train_expo = transformer.transform(train_df)

    pricing_model = fit_pricing_model(X_train, train_df, temp_config)
    train_preds = pricing_model.predict_components(X_train, train_expo)

    # Canonical log-link Poisson GLM with intercept guarantees exact claim balance on train.
    obs_claims = float(train_df["nclaims"].sum())
    pred_claims = float(train_preds["predicted_frequency"].sum())
    assert obs_claims == pytest.approx(pred_claims, rel=1e-6)

    # Verify exact numerical parity with statsmodels.GLM.predict.
    raw_sm = sm.GLM(
        train_df["nclaims"].to_numpy(dtype=float),
        X_train,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(train_expo),
    ).fit()
    assert np.allclose(
        train_preds["predicted_frequency"].to_numpy(),
        raw_sm.predict(X_train, offset=np.log(train_expo)),
        rtol=1e-8,
    )

    # Verify spline relativity equals exactly 1.0 (CI [1.0, 1.0]) at reference age.
    curve = compute_spline_relativity_curve(
        pricing_model.frequency_model,
        transformer.spec,
        ages=np.array([transformer.spec.spline_spec.reference_value, 25.0, 70.0]),
        confidence_level=0.95,
    )
    assert curve.loc[0, "relativity"] == pytest.approx(1.0, abs=1e-12)
    assert curve.loc[0, "ci_lower"] == pytest.approx(1.0, abs=1e-12)
    assert curve.loc[0, "ci_upper"] == pytest.approx(1.0, abs=1e-12)

    # Verify evaluation metrics and relativities table structure.
    X_test, test_expo = transformer.transform(test_df)
    test_preds = pricing_model.predict_components(X_test, test_expo)
    metrics = compute_evaluation_metrics(test_df, test_preds, pricing_model)
    assert metrics.poisson_deviance > 0.0
    assert metrics.gamma_deviance > 0.0
    assert 0.0 < metrics.normalized_gini <= 1.0

    rel_table = extract_relativities_table(pricing_model, transformer.spec)
    base_rows = rel_table[rel_table["is_base_level"]]
    assert np.allclose(base_rows["relativity"], 1.0)
    assert np.all(rel_table["ci_lower"] <= rel_table["relativity"] + 1e-12)
    assert np.all(rel_table["relativity"] <= rel_table["ci_upper"] + 1e-12)

    one_way = compute_one_way_relativities(train_df, transformer.spec)
    assert np.allclose(one_way.loc[one_way["is_base_level"], "one_way_relativity"], 1.0)


def test_gamma_severity_var_weights_equivalence_to_expanded_claims(
    temp_config: PricingConfig,
) -> None:
    # Policy-level average cost weighted by nclaims must yield identical coefficients to
    # claim-level expanded observations when individual claim amounts share the policy average.
    rng = np.random.default_rng(123)
    n_pol = 80
    X = pd.DataFrame(
        {
            "intercept": np.ones(n_pol),
            "x1": rng.normal(size=n_pol),
            "x2": rng.integers(0, 2, size=n_pol).astype(float),
        }
    )
    nclaims = rng.integers(1, 5, size=n_pol)
    average = np.exp(6.5 + 0.3 * X["x1"].to_numpy() - 0.2 * X["x2"].to_numpy()) * rng.gamma(
        shape=4.0, scale=0.25, size=n_pol
    )

    fitted_weighted = fit_severity_glm(
        X, average_cost=average, nclaims=nclaims, model_cfg=temp_config.model
    )

    X_expanded = X.loc[X.index.repeat(nclaims)].reset_index(drop=True)
    y_expanded = np.repeat(average, nclaims)
    ones_weights = np.ones(len(y_expanded), dtype=float)
    fitted_expanded = fit_severity_glm(
        X_expanded,
        average_cost=y_expanded,
        nclaims=ones_weights,
        model_cfg=temp_config.model,
    )

    assert np.allclose(fitted_weighted.coef_array(), fitted_expanded.coef_array(), rtol=1e-9)
    assert fitted_weighted.deviance == pytest.approx(fitted_expanded.deviance, rel=1e-9)

    freq_only = fit_frequency_glm(
        X, nclaims=nclaims, exposure=np.ones(n_pol), model_cfg=temp_config.model
    )
    assert freq_only.converged


def test_lorenz_gini_is_permutation_invariant_and_lift_sums_match() -> None:
    y = np.array([0.0, 500.0, 1200.0, 0.0, 300.0, 800.0])
    # Deliberate tie on score=100.0 across two different outcomes
    score = np.array([100.0, 200.0, 300.0, 100.0, 150.0, 250.0])
    expo = np.array([1.0, 0.5, 1.0, 0.8, 0.6, 1.0])

    _, _, gini_a = compute_lorenz_curve(y, score, expo)
    perm = np.array([3, 5, 0, 4, 1, 2])
    _, _, gini_b = compute_lorenz_curve(y[perm], score[perm], expo[perm])
    assert gini_a == pytest.approx(gini_b, abs=1e-12)

    lift = compute_decile_lift_table(y, score * expo, expo, n_bins=3)
    assert lift["exposure"].sum() == pytest.approx(expo.sum())
    assert lift["observed_amount"].sum() == pytest.approx(y.sum())
