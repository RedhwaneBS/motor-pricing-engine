"""End-to-end integration tests for CLI commands using synthetic local data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import make_synthetic_portfolio, write_test_toml

from mtpl_pricing.cli import main
from mtpl_pricing.data import atomic_write_dataframe_csv


@pytest.mark.integration
def test_cli_train_evaluate_and_predict_end_to_end(tmp_path: Path) -> None:
    df = make_synthetic_portfolio(n_rows=450, seed=99)
    toml_path = write_test_toml(tmp_path, allow_network=False)

    processed_csv = tmp_path / "data" / "processed" / "beMTPL97.csv"
    atomic_write_dataframe_csv(df, processed_csv)

    # 1. Train
    assert main(["--config", str(toml_path), "train"]) == 0
    assert (tmp_path / "artifacts" / "pricing_bundle.joblib").is_file()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "normalized_gini" in manifest["validation_metrics"]

    # 2. Evaluate (metrics + relativities + 3 PNG figures)
    assert main(["--config", str(toml_path), "evaluate"]) == 0
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "relativities.csv").is_file()

    expected_figures = (
        "one_way_vs_glm_relativities.png",
        "age_spline_relativity.png",
        "lorenz_and_lift_chart.png",
    )
    for fig_name in expected_figures:
        fig_path = tmp_path / "reports" / "figures" / fig_name
        assert fig_path.is_file()
        # Verify valid PNG magic bytes.
        assert fig_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # 3. Predict on unlabeled new policies CSV
    unlabeled = df.drop(columns=["claim", "nclaims", "amount", "average", "long", "lat"]).head(10)
    input_csv = tmp_path / "new_policies.csv"
    output_csv = tmp_path / "scored_policies.csv"
    unlabeled.to_csv(input_csv, index=False)

    assert (
        main(
            [
                "--config",
                str(toml_path),
                "predict",
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
            ]
        )
        == 0
    )
    scored = pd.read_csv(output_csv)
    assert len(scored) == 10
    assert {"predicted_frequency", "predicted_severity", "predicted_pure_premium"}.issubset(
        scored.columns
    )
    assert (scored["predicted_pure_premium"] > 0.0).all()

    # 4. Predict fails cleanly with exit code 1 when an unknown category is supplied
    bad_input = unlabeled.iloc[[0]].copy()
    bad_input["coverage"] = "UNKNOWN_TIER"
    bad_csv = tmp_path / "bad_policy.csv"
    bad_input.to_csv(bad_csv, index=False)
    assert (
        main(
            [
                "--config",
                str(toml_path),
                "predict",
                "--input",
                str(bad_csv),
                "--output",
                str(tmp_path / "unused.csv"),
            ]
        )
        == 1
    )


@pytest.mark.integration
def test_cli_run_pipeline_offline_fallback(tmp_path: Path) -> None:
    df = make_synthetic_portfolio(n_rows=320, seed=17)
    toml_path = write_test_toml(tmp_path, allow_network=False)
    atomic_write_dataframe_csv(df, tmp_path / "data" / "processed" / "beMTPL97.csv")

    assert main(["--config", str(toml_path), "run-pipeline"]) == 0
    assert (tmp_path / "artifacts" / "manifest.json").is_file()
    assert (tmp_path / "reports" / "figures" / "lorenz_and_lift_chart.png").is_file()


@pytest.mark.integration
def test_cli_accepts_config_and_log_level_after_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = make_synthetic_portfolio(n_rows=320, seed=21)
    toml_path = write_test_toml(tmp_path, allow_network=False)
    atomic_write_dataframe_csv(df, tmp_path / "data" / "processed" / "beMTPL97.csv")

    assert main(["ingest", "--config", str(toml_path), "--log-level", "WARNING"]) == 0
    assert "[OK] ingest:" in capsys.readouterr().out

    assert main(["train", "--config", str(toml_path)]) == 0
    assert "[OK] train:" in capsys.readouterr().out

    assert main(["evaluate", "--config", str(toml_path)]) == 0
    assert "[OK] evaluate:" in capsys.readouterr().out

    sample_in = tmp_path / "sample_in.csv"
    sample_out = tmp_path / "sample_out.csv"
    df.drop(columns=["claim", "nclaims", "amount", "average"]).head(5).to_csv(
        sample_in, index=False
    )
    assert (
        main(
            [
                "predict",
                "--config",
                str(toml_path),
                "--log-level",
                "INFO",
                "--input",
                str(sample_in),
                "--output",
                str(sample_out),
            ]
        )
        == 0
    )
    assert "[OK] predict:" in capsys.readouterr().out
    assert len(pd.read_csv(sample_out)) == 5

    # Missing input file emits [FAIL] status line to stderr and returns exit code 1.
    assert (
        main(["predict", "--config", str(toml_path), "--input", str(tmp_path / "missing.csv")]) == 1
    )
    assert "[FAIL] predict: DataIngestionError" in capsys.readouterr().err

    # Missing input file emits [FAIL] status line to stderr and returns exit code 1.
    assert (
        main(["predict", "--config", str(toml_path), "--input", str(tmp_path / "missing.csv")])
        == 1
    )
    assert "[FAIL] predict: DataIngestionError" in capsys.readouterr().err
