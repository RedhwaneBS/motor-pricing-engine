"""Command-line interface for ingestion, training, evaluation, and inference."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from mtpl_pricing.artifacts import create_pricing_bundle, load_artifacts, save_artifacts
from mtpl_pricing.config import PricingConfig, configure_logging, load_config
from mtpl_pricing.data import (
    atomic_write_dataframe_csv,
    compute_file_sha256,
    ingest_dataset,
    load_processed_csv,
    validate_portfolio_dataframe,
)
from mtpl_pricing.evaluation import compute_evaluation_metrics, generate_evaluation_reports
from mtpl_pricing.exceptions import DataIngestionError, MtplPricingError
from mtpl_pricing.features import FeatureTransformer, split_train_test
from mtpl_pricing.models import extract_relativities_table, fit_pricing_model

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/default.toml")


def _load_validated_dataset(config: PricingConfig) -> tuple[pd.DataFrame, str]:
    """Load validated dataset from processed CSV if present, otherwise run full ingestion."""
    processed_path = config.data.processed_path
    if processed_path.is_file():
        df = load_processed_csv(processed_path)
        validated = validate_portfolio_dataframe(df, config.data, config.features)
        raw_path = config.data.raw_path
        dataset_sha = (
            compute_file_sha256(raw_path)
            if raw_path.is_file()
            else compute_file_sha256(processed_path)
        )
        return validated, dataset_sha
    return ingest_dataset(config)


def run_ingest(config: PricingConfig) -> tuple[pd.DataFrame, str]:
    """Execute Step A (ingestion, integrity verification, validation, and CSV export)."""
    return ingest_dataset(config)


def run_train(config: PricingConfig) -> tuple[Path, Path]:
    """Execute training, test-set metric calculation, and atomic artifact persistence."""
    df, dataset_sha256 = _load_validated_dataset(config)
    train_df, test_df = split_train_test(
        df,
        split_cfg=config.split,
        random_seed=config.project.random_seed,
    )

    transformer = FeatureTransformer().fit(train_df, config)
    X_train, _ = transformer.transform(train_df)
    pricing_model = fit_pricing_model(X_train, train_df, config)

    X_test, test_expo = transformer.transform(test_df)
    test_preds = pricing_model.predict_components(X_test, test_expo)
    metrics = compute_evaluation_metrics(test_df, test_preds, pricing_model)

    bundle = create_pricing_bundle(
        feature_spec=transformer.spec,
        pricing_model=pricing_model,
        dataset_sha256=dataset_sha256,
        random_seed=config.project.random_seed,
    )
    return save_artifacts(
        bundle=bundle,
        artifacts_cfg=config.artifacts,
        validation_metrics=dict(metrics.to_dict()),
    )


def run_evaluate(config: PricingConfig) -> dict[str, Path]:
    """Load persisted bundle, evaluate on deterministic test split, and write reports/figures."""
    bundle, manifest = load_artifacts(
        config.artifacts.output_dir,
        bundle_filename=config.artifacts.bundle_filename,
        manifest_filename=config.artifacts.manifest_filename,
    )
    df, _ = _load_validated_dataset(config)
    split_seed = int(manifest.get("random_seed", config.project.random_seed))
    train_df, test_df = split_train_test(df, split_cfg=config.split, random_seed=split_seed)

    transformer = FeatureTransformer(spec=bundle.feature_spec)
    X_test, test_expo = transformer.transform(test_df)
    test_preds = bundle.pricing_model.predict_components(X_test, test_expo)
    relativities_df = extract_relativities_table(bundle.pricing_model, bundle.feature_spec)

    _, output_paths = generate_evaluation_reports(
        train_df=train_df,
        test_df=test_df,
        test_predictions=test_preds,
        pricing_model=bundle.pricing_model,
        feature_spec=bundle.feature_spec,
        relativities_df=relativities_df,
        eval_cfg=config.evaluation,
    )
    return output_paths


def run_predict(
    config: PricingConfig,
    *,
    input_path: Path,
    output_path: Path | None,
    artifacts_dir: Path | None = None,
) -> pd.DataFrame:
    """Run defensive batch inference on an input CSV file."""
    if not input_path.is_file():
        raise DataIngestionError(f"Inference input CSV not found: {input_path}")

    resolved_artifacts_dir = artifacts_dir or config.artifacts.output_dir
    bundle, _ = load_artifacts(
        resolved_artifacts_dir,
        bundle_filename=config.artifacts.bundle_filename,
        manifest_filename=config.artifacts.manifest_filename,
    )

    try:
        dtype_overrides: dict[str, str] = {
            cat.column: "string"
            for cat in bundle.feature_spec.categorical_specs
            if cat.column != "zone" or not bundle.feature_spec.derive_zone_from_postcode
        }
        input_df = pd.read_csv(input_path, dtype=dtype_overrides, keep_default_na=True)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataIngestionError(f"Failed reading inference CSV '{input_path}': {exc}") from exc

    transformer = FeatureTransformer(spec=bundle.feature_spec)
    X_inf, exposure = transformer.transform(input_df)
    preds = bundle.pricing_model.predict_components(X_inf, exposure)

    result_df = input_df.copy()
    for col in preds.columns:
        result_df[col] = preds[col].to_numpy()

    if output_path is not None:
        atomic_write_dataframe_csv(result_df, output_path)
        LOGGER.info("Wrote %d predictions to %s", len(result_df), output_path)
    else:
        sys.stdout.write(result_df.to_csv(index=False))

    return result_df


def run_full_pipeline(config: PricingConfig) -> dict[str, Path]:
    """Execute end-to-end ingestion, training, artifact saving, and evaluation."""
    run_ingest(config)
    run_train(config)
    return run_evaluate(config)


def _add_common_options(
    target_parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool,
) -> None:
    """Attach shared ``--config`` and ``--log-level`` options to root and subparsers."""
    config_default: object = argparse.SUPPRESS if suppress_defaults else DEFAULT_CONFIG_PATH
    log_default: object = argparse.SUPPRESS if suppress_defaults else None
    target_parser.add_argument(
        "--config",
        type=Path,
        default=config_default,
        help="Path to TOML configuration file (default: configs/default.toml).",
    )
    target_parser.add_argument(
        "--log-level",
        type=str,
        default=log_default,
        help="Optional logging level override (DEBUG, INFO, WARNING, ERROR).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser supporting global flags before or after subcommands."""
    parser = argparse.ArgumentParser(
        prog="mtpl-pricing",
        description="Production MTPL GLM pricing pipeline on Belgian beMTPL97.",
    )
    _add_common_options(parser, suppress_defaults=False)

    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("ingest", "Download/verify raw .rda and export validated CSV."),
        ("train", "Fit frequency/severity GLMs and save model artifacts."),
        ("evaluate", "Evaluate test performance and generate figures."),
        ("run-pipeline", "Run ingest, train, and evaluate end-to-end."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_options(sub, suppress_defaults=True)

    predict_parser = subparsers.add_parser("predict", help="Score policies from an input CSV.")
    _add_common_options(predict_parser, suppress_defaults=True)
    predict_parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (writes CSV to stdout when omitted).",
    )
    predict_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional directory overriding configured artifacts location.",
    )
    return parser


def _emit_status(stream: TextIO, message: str) -> None:
    """Write a concise terminal status line without using ``print()``."""
    stream.write(f"{message}\n")
    stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit status code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        configure_logging(args.log_level or config.project.log_level)

        if args.command == "ingest":
            df, _ = run_ingest(config)
            _emit_status(
                sys.stdout,
                f"[OK] ingest: {len(df)} rows validated -> {config.data.processed_path}",
            )
        elif args.command == "train":
            bundle_path, manifest_path = run_train(config)
            _emit_status(
                sys.stdout,
                f"[OK] train: saved '{bundle_path.name}' and '{manifest_path.name}' "
                f"-> {bundle_path.parent}",
            )
        elif args.command == "evaluate":
            outputs = run_evaluate(config)
            _emit_status(
                sys.stdout,
                f"[OK] evaluate: wrote {len(outputs)} report artifact(s) "
                f"-> {config.evaluation.reports_dir}",
            )
        elif args.command == "run-pipeline":
            outputs = run_full_pipeline(config)
            _emit_status(
                sys.stdout,
                f"[OK] run-pipeline: completed ingest, train, evaluate "
                f"({len(outputs)} report files) -> {config.evaluation.reports_dir}",
            )
        elif args.command == "predict":
            preds_df = run_predict(
                config,
                input_path=args.input,
                output_path=args.output,
                artifacts_dir=args.artifacts_dir,
            )
            if args.output is not None:
                _emit_status(
                    sys.stdout,
                    f"[OK] predict: scored {len(preds_df)} row(s) -> {args.output}",
                )
            else:
                _emit_status(
                    sys.stderr,
                    f"[OK] predict: scored {len(preds_df)} row(s) to stdout",
                )
        else:  # pragma: no cover - guarded by argparse required subparser
            parser.error(f"Unknown command: {args.command}")
    except MtplPricingError as exc:
        LOGGER.error("Pipeline failed (%s): %s", type(exc).__name__, exc)
        _emit_status(sys.stderr, f"[FAIL] {args.command}: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
