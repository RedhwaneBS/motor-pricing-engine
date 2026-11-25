"""Shared pytest fixtures and deterministic synthetic MTPL data generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mtpl_pricing.config import PricingConfig, load_config


def make_synthetic_portfolio(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic, schema-compliant synthetic beMTPL97-like portfolio."""
    rng = np.random.default_rng(seed)

    policy_ids = np.arange(1, n_rows + 1, dtype=np.int64)
    expo = np.round(rng.uniform(0.25, 1.0, size=n_rows), 4)
    coverage = rng.choice(["TPL", "TPL+", "TPL++"], size=n_rows, p=[0.58, 0.28, 0.14])
    sex = rng.choice(["male", "female"], size=n_rows, p=[0.72, 0.28])
    fuel = rng.choice(["gasoline", "diesel"], size=n_rows, p=[0.68, 0.32])
    use = rng.choice(["private", "work"], size=n_rows, p=[0.90, 0.10])
    fleet = rng.choice(["0", "1"], size=n_rows, p=[0.95, 0.05])
    ageph = rng.integers(18, 86, size=n_rows)
    bm = rng.integers(0, 16, size=n_rows)
    power = rng.integers(25, 140, size=n_rows)
    agec = rng.integers(0, 20, size=n_rows)
    zone_digits = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n_rows)
    postcode = zone_digits * 1000 + rng.integers(0, 90, size=n_rows)
    longitude = 4.35 + (zone_digits - 5) * 0.15
    latitude = 50.80 + (zone_digits - 5) * 0.05

    # Construct genuine U-shaped age effect + covariate signal so Poisson/Gamma converge cleanly.
    age_centered = (ageph - 45.0) / 20.0
    log_rate = (
        -2.0
        + 0.35 * (age_centered**2)
        - 0.15 * age_centered
        + 0.05 * bm
        + 0.20 * (fuel == "diesel")
        - 0.12 * (coverage == "TPL+")
    )
    lam = expo * np.exp(log_rate)
    nclaims = rng.poisson(lam=lam).astype(np.int64)
    # Guarantee enough claim-bearing rows for both train and test splits.
    if int((nclaims > 0).sum()) < 40:
        nclaims[:40] = np.maximum(nclaims[:40], 1)

    claim = (nclaims > 0).astype(np.int64)
    amount = np.zeros(n_rows, dtype=np.float64)
    average = np.full(n_rows, np.nan, dtype=np.float64)

    pos_idx = np.where(nclaims > 0)[0]
    for idx in pos_idx:
        k = int(nclaims[idx])
        mean_sev = float(
            np.exp(6.8 + 0.25 * (age_centered[idx] ** 2) + 0.15 * (coverage[idx] == "TPL++"))
        )
        claims_costs = rng.gamma(shape=2.0, scale=mean_sev / 2.0, size=k)
        total_amt = float(np.sum(claims_costs))
        amount[idx] = total_amt
        average[idx] = total_amt / float(k)

    return pd.DataFrame(
        {
            "id": policy_ids,
            "expo": expo,
            "claim": claim,
            "nclaims": nclaims,
            "amount": amount,
            "average": average,
            "coverage": coverage,
            "ageph": ageph,
            "sex": sex,
            "bm": bm,
            "power": power,
            "agec": agec,
            "fuel": fuel,
            "use": use,
            "fleet": fleet,
            "postcode": postcode,
            "long": longitude,
            "lat": latitude,
        }
    )


def write_test_toml(
    tmp_path: Path,
    *,
    expected_sha256: str = "a" * 64,
    allow_network: bool = False,
    unknown_policy: str = "raise",
    extra_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a complete self-contained TOML config inside ``tmp_path``."""
    raw_path = (tmp_path / "data" / "raw" / "beMTPL97.rda").as_posix()
    proc_path = (tmp_path / "data" / "processed" / "beMTPL97.csv").as_posix()
    reports_dir = (tmp_path / "reports").as_posix()
    figures_dir = (tmp_path / "reports" / "figures").as_posix()
    artifacts_dir = (tmp_path / "artifacts").as_posix()

    cfg_text = f"""[project]
name = "mtpl-glm-pricing"
random_seed = 42
log_level = "INFO"

[data]
source_url = "https://example.invalid/beMTPL97.rda"
expected_sha256 = "{expected_sha256}"
raw_path = "{raw_path}"
processed_path = "{proc_path}"
rda_object_name = "beMTPL97"
download_timeout_seconds = 5
allow_network_download = {"true" if allow_network else "false"}
max_exposure = 1.0
min_ageph = 18
max_ageph = 110
consistency_rtol = 1e-6
consistency_atol = 1e-6

[split]
test_size = 0.2
stratify_column = "nclaims"
min_stratum_count = 2

[features]
id_column = "id"
exposure_column = "expo"
target_columns = ["claim", "nclaims", "amount", "average"]
derive_zone_from_postcode = true
categorical_columns = ["coverage", "sex", "fuel", "use", "fleet", "zone"]
numeric_columns = ["bm", "power", "agec"]
spline_column = "ageph"
spline_knot_quantiles = [0.05, 0.275, 0.50, 0.725, 0.95]
spline_reference_age = 45.0
unknown_category_policy = "{unknown_policy}"

[model]
frequency_family = "poisson"
severity_family = "gamma"
link = "log"
max_iter = 100
tol = 1e-8
confidence_level = 0.95

[evaluation]
n_lift_bins = 10
reports_dir = "{reports_dir}"
figures_dir = "{figures_dir}"
metrics_filename = "metrics.json"
relativities_filename = "relativities.csv"

[artifacts]
output_dir = "{artifacts_dir}"
bundle_filename = "pricing_bundle.joblib"
manifest_filename = "manifest.json"
"""
    if extra_overrides:
        for key, value in extra_overrides.items():
            cfg_text = cfg_text.replace(key, str(value))

    tmp_path.mkdir(parents=True, exist_ok=True)
    toml_path = tmp_path / "test_config.toml"
    toml_path.write_text(cfg_text, encoding="utf-8")
    return toml_path


@pytest.fixture()
def synthetic_portfolio() -> pd.DataFrame:
    """Return a clean 400-row synthetic portfolio."""
    return make_synthetic_portfolio(n_rows=400, seed=42)


@pytest.fixture()
def temp_config(tmp_path: Path) -> PricingConfig:
    """Return a validated ``PricingConfig`` rooted in ``tmp_path`` with network disabled."""
    toml_path = write_test_toml(tmp_path)
    return load_config(toml_path, base_dir=tmp_path)
