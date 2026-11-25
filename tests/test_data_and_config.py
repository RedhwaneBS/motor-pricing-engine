"""Tests for configuration validation, data quality invariants, ingestion, and code hygiene."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from tests.conftest import make_synthetic_portfolio, write_test_toml

from mtpl_pricing.config import PricingConfig, load_config
from mtpl_pricing.data import (
    atomic_write_dataframe_csv,
    compute_bytes_sha256,
    ingest_dataset,
    load_processed_csv,
    validate_portfolio_dataframe,
)
from mtpl_pricing.exceptions import (
    ConfigurationError,
    DataIngestionError,
    DataQualityError,
    DatasetIntegrityError,
    SchemaValidationError,
)


def test_load_default_config() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "configs" / "default.toml")
    assert cfg.project.random_seed == 42
    assert cfg.split.test_size == pytest.approx(0.2)
    assert cfg.features.unknown_category_policy == "raise"
    assert "nclaims" in cfg.features.target_columns


def test_config_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.toml")

    bad_toml = tmp_path / "broken.toml"
    bad_toml.write_text("[project\nname = 123", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Failed to parse TOML"):
        load_config(bad_toml)


def test_config_rejects_target_leakage_and_invalid_bounds(tmp_path: Path) -> None:
    toml_path = write_test_toml(
        tmp_path,
        extra_overrides={
            'numeric_columns = ["bm", "power", "agec"]': 'numeric_columns = ["bm", "nclaims"]',
        },
    )
    with pytest.raises(ConfigurationError, match="Forbidden target/identifier"):
        load_config(toml_path, base_dir=tmp_path)

    bad_split = write_test_toml(
        tmp_path,
        extra_overrides={"test_size = 0.2": "test_size = 1.5"},
    )
    with pytest.raises(ConfigurationError, match=r"split\.test_size"):
        load_config(bad_split, base_dir=tmp_path)

    bad_bool_int = write_test_toml(
        tmp_path,
        extra_overrides={"random_seed = 42": "random_seed = true"},
    )
    with pytest.raises(ConfigurationError, match="must be an integer"):
        load_config(bad_bool_int, base_dir=tmp_path)


def test_portfolio_validation_accepts_clean_and_csv_roundtrip(
    tmp_path: Path,
    synthetic_portfolio: pd.DataFrame,
    temp_config: PricingConfig,
) -> None:
    validated = validate_portfolio_dataframe(
        synthetic_portfolio,
        temp_config.data,
        temp_config.features,
    )
    assert len(validated) == len(synthetic_portfolio)

    csv_path = tmp_path / "roundtrip.csv"
    atomic_write_dataframe_csv(validated, csv_path)
    reloaded = load_processed_csv(csv_path)
    revalidated = validate_portfolio_dataframe(reloaded, temp_config.data, temp_config.features)
    assert set(revalidated["fleet"].unique()).issubset({"0", "1"})


@pytest.mark.parametrize(
    ("mutator_name", "exc_cls", "pattern"),
    [
        ("missing_col", SchemaValidationError, "missing required columns"),
        ("duplicate_id", DataQualityError, "not unique"),
        ("zero_expo", DataQualityError, "expo"),
        ("over_expo", DataQualityError, "expo"),
        ("negative_nclaims", DataQualityError, "nclaims"),
        ("claim_mismatch", DataQualityError, "Inconsistent 'claim'"),
        ("zero_claim_nonzero_amount", DataQualityError, "nclaims == 0 and amount != 0"),
        ("pos_claim_zero_amount", DataQualityError, "nclaims > 0 and amount <= 0"),
        ("average_mismatch", DataQualityError, "average"),
        ("age_out_of_bounds", DataQualityError, "ageph"),
    ],
)
def test_portfolio_validation_rejects_corrupt_invariants(
    synthetic_portfolio: pd.DataFrame,
    temp_config: PricingConfig,
    mutator_name: str,
    exc_cls: type[Exception],
    pattern: str,
) -> None:
    df = synthetic_portfolio.copy()
    zero_idx = int(df.index[df["nclaims"] == 0][0])
    pos_idx = int(df.index[df["nclaims"] > 0][0])

    if mutator_name == "missing_col":
        df = df.drop(columns=["coverage"])
    elif mutator_name == "duplicate_id":
        df.loc[1, "id"] = df.loc[0, "id"]
    elif mutator_name == "zero_expo":
        df.loc[0, "expo"] = 0.0
    elif mutator_name == "over_expo":
        df.loc[0, "expo"] = 1.5
    elif mutator_name == "negative_nclaims":
        df.loc[0, "nclaims"] = -1
    elif mutator_name == "claim_mismatch":
        df.loc[zero_idx, "claim"] = 1
    elif mutator_name == "zero_claim_nonzero_amount":
        df.loc[zero_idx, "amount"] = 100.0
    elif mutator_name == "pos_claim_zero_amount":
        df.loc[pos_idx, "amount"] = 0.0
    elif mutator_name == "average_mismatch":
        df.loc[pos_idx, "average"] = float(df.loc[pos_idx, "average"]) + 25.0
    elif mutator_name == "age_out_of_bounds":
        df.loc[0, "ageph"] = 14

    with pytest.raises(exc_cls, match=pattern):
        validate_portfolio_dataframe(df, temp_config.data, temp_config.features)


def test_ingest_falls_back_to_processed_csv_when_network_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = make_synthetic_portfolio(120, seed=7)
    toml_path = write_test_toml(tmp_path, allow_network=True)
    cfg = load_config(toml_path, base_dir=tmp_path)
    atomic_write_dataframe_csv(df, cfg.data.processed_path)

    def _fail_get(*args: object, **kwargs: object) -> object:
        raise requests.ConnectionError("Simulated offline environment")

    monkeypatch.setattr("mtpl_pricing.data.requests.get", _fail_get)

    with caplog.at_level("WARNING"):
        loaded_df, digest = ingest_dataset(cfg)

    assert len(loaded_df) == 120
    assert len(digest) == 64
    assert "falling back to validated CSV" in caplog.text


def test_ingest_rejects_corrupt_raw_file_without_fallback(
    tmp_path: Path,
    synthetic_portfolio: pd.DataFrame,
) -> None:
    toml_path = write_test_toml(tmp_path, expected_sha256="b" * 64, allow_network=False)
    cfg = load_config(toml_path, base_dir=tmp_path)
    cfg.data.raw_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.data.raw_path.write_bytes(b"corrupted-rda-bytes")
    atomic_write_dataframe_csv(synthetic_portfolio, cfg.data.processed_path)

    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        ingest_dataset(cfg)


def test_ingest_raises_clear_error_when_offline_and_no_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toml_path = write_test_toml(tmp_path, allow_network=True)
    cfg = load_config(toml_path, base_dir=tmp_path)

    mock_get = Mock(side_effect=requests.Timeout("Timed out"))
    monkeypatch.setattr("mtpl_pricing.data.requests.get", mock_get)

    with pytest.raises(DataIngestionError, match="no local fallback exists"):
        ingest_dataset(cfg)


def test_download_verifies_sha256_before_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"bad-payload"
    toml_path = write_test_toml(
        tmp_path,
        expected_sha256=hashlib.sha256(b"expected-payload").hexdigest(),
        allow_network=True,
    )
    cfg = load_config(toml_path, base_dir=tmp_path)

    resp = Mock()
    resp.content = payload
    resp.raise_for_status = Mock()
    monkeypatch.setattr("mtpl_pricing.data.requests.get", Mock(return_value=resp))

    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        ingest_dataset(cfg)
    assert not cfg.data.raw_path.exists()
    assert compute_bytes_sha256(payload) != cfg.data.expected_sha256


def test_ingest_downloads_verified_rda_and_exports_csv(
    tmp_path: Path,
    synthetic_portfolio: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"mock-rda-binary-content"
    digest = compute_bytes_sha256(payload)
    toml_path = write_test_toml(tmp_path, expected_sha256=digest, allow_network=True)
    cfg = load_config(toml_path, base_dir=tmp_path)

    resp = Mock()
    resp.content = payload
    resp.raise_for_status = Mock()
    monkeypatch.setattr("mtpl_pricing.data.requests.get", Mock(return_value=resp))
    monkeypatch.setattr(
        "mtpl_pricing.data.pyreadr.read_r",
        Mock(return_value={"beMTPL97": synthetic_portfolio}),
    )

    ingested, actual_sha = ingest_dataset(cfg)
    assert actual_sha == digest
    assert len(ingested) == len(synthetic_portfolio)
    assert cfg.data.raw_path.is_file()
    assert cfg.data.processed_path.is_file()


def test_source_tree_has_zero_print_calls_and_no_generic_except_exception() -> None:
    src_dir = Path(__file__).resolve().parent.parent / "src" / "mtpl_pricing"
    py_files = sorted(src_dir.glob("*.py"))
    assert len(py_files) >= 8

    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", (
                    f"Forbidden print() call in {path.name}:{node.lineno}"
                )
            if isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    raise AssertionError(
                        f"Forbidden generic 'except Exception:' in {path.name}:{node.lineno}"
                    )
