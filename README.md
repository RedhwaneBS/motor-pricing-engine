# `mtpl-glm-pricing`

Production-grade Python package implementing an end-to-end Motor Third-Party Liability (MTPL) Frequency–Severity Generalized Linear Model (GLM) pricing pipeline on the historical Belgian portfolio `beMTPL97`.

> **Positioning & Historical Disclaimer**
> - This repository is designed by a software engineer to demonstrate **production Python engineering practices**—clean separation of pure domain logic from I/O, defensive data contracts, immutable artifact bundles with schema/integrity verification, structured logging, full static typing (`mypy --strict`), linting (`ruff`), and deterministic automated testing (`pytest`)—applied to classical non-life insurance ratemaking.
> - Actuarial methodology follows the foundational chapters of the Casualty Actuarial Society references: Werner & Modlin (*Basic Ratemaking*, Ch. 1, 2, 9, 10) and Goldburd, Khare & Tevet (*Generalized Linear Models for Insurance Rating*, Ch. 1, 2, 3).
> - **Historical scope**: `beMTPL97` is a **1997 Belgian portfolio** (see [`data/DATA_PROVENANCE.md`](data/DATA_PROVENANCE.md)). Its claim amounts, statutory Bonus-Malus (`bm`) scale (`0`–`22`), and inclusion of `sex` (subsequently prohibited in EU pricing by the 2011 ECJ *Test-Achats* ruling) **do not reflect contemporary Belgian or European market tariffs**.

---

## Architecture Overview

Pure statistical transformations and mathematical checks are strictly decoupled from disk, network, and CLI I/O:

```text
mtpl-glm-pricing/
├── configs/
│   └── default.toml                # Externalized, strongly typed pipeline configuration
├── data/
│   ├── DATA_PROVENANCE.md          # Origin, license, schema, and 1997 historical caveats
│   ├── raw/                        # Immutable raw beMTPL97.rda (SHA-256 verified)
│   └── processed/                  # Validated UTF-8 CSV export
├── artifacts/
│   ├── pricing_bundle.joblib       # Atomic serialized PricingBundle (FeatureSpec + GLMs)
│   └── manifest.json               # Package/Python/dep versions, dataset & bundle SHA-256, schema, metrics
├── reports/
│   ├── metrics.json                # Out-of-sample Poisson/Gamma deviances, O/P ratios, Gini
│   ├── relativities.csv            # Multiplicative relativities exp(beta) with 95% Wald CIs
│   └── figures/
│       ├── one_way_vs_glm_relativities.png
│       ├── age_spline_relativity.png
│       └── lorenz_and_lift_chart.png
├── src/mtpl_pricing/
│   ├── config.py                   # Frozen dataclasses + TOML validation + logging setup
│   ├── exceptions.py               # Explicit domain exception hierarchy (MtplPricingError)
│   ├── data.py                     # Ingestion (.rda -> CSV), SHA-256 checks, quality invariants
│   ├── features.py                 # Stratified split, exposure-weighted base levels, splines, guards
│   ├── models.py                   # Poisson frequency & var-weighted Gamma severity GLMs
│   ├── evaluation.py               # Actuarial metrics, Lorenz/lift curves, matplotlib figures
│   ├── artifacts.py                # Atomic joblib + manifest.json persistence & compatibility check
│   └── cli.py                      # Subcommands: ingest, train, evaluate, predict, run-pipeline
└── tests/
    ├── conftest.py
    ├── test_data_and_config.py
    ├── test_features_and_models.py
    ├── test_artifacts_and_inference.py
    └── test_cli_integration.py
```

---

## Engineering & Defensive Design Highlights

1. **Zero Target Leakage & Defensive Inference (`features.py`, `config.py`)**:
   - `parse_config_dict` rejects any configuration that places target columns (`claim`, `nclaims`, `amount`, `average`), `id`, or `expo` into explanatory features.
   - `FeatureTransformer.transform` validates required explanatory columns, finiteness, numeric types, exposure bounds (`0 < expo <= 1.0`), driver age bounds (`18 <= ageph <= 110`), and unseen categories at both training and `predict` time.
   - Unseen categories raise `UnknownCategoryError` by default (`unknown_category_policy = "raise"`) or optionally fall back to the exposure-weighted base level with a structured `WARNING` log (`"fallback_to_base"`).
2. **Data Integrity & Quality Contracts (`data.py`, `exceptions.py`)**:
   - `data/raw/beMTPL97.rda` is verified against its expected SHA-256 digest (`955a821a7a693bf18076c425e4d5a5a99889f3c89e1fbd99eca6239c11e963a6`) before parsing. Corrupted files fail loudly with `DatasetIntegrityError` rather than falling back silently.
   - Strict invariant checks verify `id` uniqueness, `expo > 0`, `nclaims >= 0`, `claim == (nclaims > 0)`, `nclaims == 0 => amount == 0`, and `nclaims > 0 => amount > 0` with `average ≈ amount / nclaims`.
3. **Lightweight Immutable Model State & Atomic Persistence (`models.py`, `artifacts.py`)**:
   - After `statsmodels` IRLS convergence checks, coefficients, covariance matrices, deviances, and Pearson dispersion are extracted into frozen `FittedGLM` dataclasses—eliminating training-matrix bloat and runtime coupling during inference.
   - Artifacts are written via temp-file + `fsync` + `os.replace`. On `load_artifacts`, `manifest.json` validates format version, major package version, bundle SHA-256 integrity, and feature-schema alignment before deserializing `pricing_bundle.joblib`.
4. **Observability & Code Hygiene**:
   - Structured `logging` throughout; enforced **zero `print()` calls** in `src/` via `ruff` rule `T20` and AST verification in tests.

---

## Actuarial Modeling Summary (`beMTPL97`)

- **Portfolio**: `163,212` policies (`145,216.82` policy-years of exposure, `20,215` total claims, empirical frequency `13.92%`, total claim cost `26,464,969.92` EUR).
- **Split**: Deterministic 80% / 20% split (`seed = 42`, stratified on `nclaims`): `130,569` train / `32,643` test policies.
- **Categorical Reference Selection (*Goldburd et al. Ch. 3*)**: Base levels selected by maximum cumulative training exposure (`coverage="TPL"`, `sex="male"`, `fuel="gasoline"`, `use="private"`, `fleet="0"`, `zone="7"` derived from `postcode // 1000`).
- **Driver Age (`ageph`) Spline**: Centered, scaled Harrell natural cubic spline (`knots = [25.4, 36.0, 46.0, 56.0, 73.0]`, linear tails outside boundary knots, anchored to relativity `1.0` at `reference_age = 45`).
- **Frequency GLM**: `Poisson(link="log")` on `nclaims` with `offset=log(expo)` (`deviance = 69,649.07`, Pearson dispersion $\hat{\phi} = 1.1660$, exact train O/P balance `1.000000`).
- **Severity GLM**: `Gamma(link="log")` on `nclaims > 0` (`14,621` train policies), target `average = amount / nclaims` with `var_weights = nclaims` (`deviance = 32,447.29`, scale $\hat{\phi} = 7.0445$). Because $\bar{Z}_i = \frac{1}{N_i} \sum_{k=1}^{N_i} Z_{ik}$ has variance $\phi \mu_i^2 / N_i$, weighting by `nclaims` preserves exact equivalence with claim-level Gamma estimation.

### Out-of-Sample Test Performance (`n = 32,643`)

| Metric | Observed Value |
| :--- | :---: |
| Mean Poisson Deviance | `0.5316` |
| Mean Gamma Deviance (claim-weighted) | `2.1221` |
| Frequency Observed / Predicted | `0.9987` |
| Severity Observed / Predicted | `1.0037` |
| Pure Premium Observed / Predicted | `1.0045` |
| Exposure-Weighted Normalized Gini | `0.2400` (`raw = 0.2320`, `oracle = 0.9670`) |

### Diagnostics & Visualizations

- **Composition Bias (`reports/figures/one_way_vs_glm_relativities.png`)**: Demonstrates why univariate tables mislead (*Werner & Modlin Ch. 9–10*)—for example, `sex=female` appears `+9%` higher one-way (`~1.09`) due to portfolio confounding, whereas the multivariate Poisson GLM places it near neutral (`1.022`, 95% CI `[0.986, 1.059]`).
- **Non-Linear Driver Age (`reports/figures/age_spline_relativity.png`)**: Captures decreasing frequency through middle age and the classic U-shaped severity profile (elevated under age 25 and rising sharply above age 65, with widening CIs where exposure thins).
- **Risk Discrimination & Calibration (`reports/figures/lorenz_and_lift_chart.png`)**: Exposure-weighted Lorenz curve alongside decile calibration across predicted pure-premium risk deciles.

---

## Quickstart & CLI Usage

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the Full Pipeline

```bash
mtpl-pricing run-pipeline
```

### Individual Commands

```bash
# 1. Download/verify beMTPL97.rda and export validated CSV
mtpl-pricing ingest --config configs/default.toml

# 2. Train GLMs and save atomic bundle + manifest.json
mtpl-pricing train --config configs/default.toml

# 3. Evaluate on test split and generate metrics, relativities, and figures
mtpl-pricing evaluate --config configs/default.toml

# 4. Score policies from a CSV (example on a 5-row slice from the processed portfolio)
head -n 6 data/processed/beMTPL97.csv > /tmp/sample_policies.csv
mtpl-pricing predict --config configs/default.toml --input /tmp/sample_policies.csv --output /tmp/scored_policies.csv
```

### Quality Checks & Tests

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov=mtpl_pricing --cov-report=term-missing
```
