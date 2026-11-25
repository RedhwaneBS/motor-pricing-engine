"""Tests for artifact persistence, manifest verification, and defensive inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import write_test_toml

from mtpl_pricing.artifacts import create_pricing_bundle, load_artifacts, save_artifacts
from mtpl_pricing.config import PricingConfig, load_config
from mtpl_pricing.data import compute_file_sha256
from mtpl_pricing.exceptions import (
    ArtifactCompatibilityError,
    ArtifactSerializationError,
    DataQualityError,
    InferenceError,
    SchemaValidationError,
    UnknownCategoryError,
)
from mtpl_pricing.features import FeatureTransformer, split_train_test
from mtpl_pricing.models import fit_pricing_model


def _train_and_save(
    portfolio: pd.DataFrame,
    cfg: PricingConfig,
) -> tuple[FeatureTransformer, pd.DataFrame, Path, Path]:
    train_df, test_df = split_train_test(portfolio, cfg.split, random_seed=cfg.project.random_seed)
    transformer = FeatureTransformer().fit(train_df, cfg)
    X_train, _ = transformer.transform(train_df)
    model = fit_pricing_model(X_train, train_df, cfg)
    bundle = create_pricing_bundle(
        feature_spec=transformer.spec,
        pricing_model=model,
        dataset_sha256="c" * 64,
        random_seed=cfg.project.random_seed,
    )
    bundle_path, manifest_path = save_artifacts(
        bundle=bundle,
        artifacts_cfg=cfg.artifacts,
        validation_metrics={"poisson_deviance": 0.55},
    )
    return transformer, test_df, bundle_path, manifest_path


def test_artifact_roundtrip_preserves_exact_predictions_and_manifest(
    synthetic_portfolio: pd.DataFrame,
    temp_config: PricingConfig,
) -> None:
    transformer, test_df, bundle_path, manifest_path = _train_and_save(
        synthetic_portfolio,
        temp_config,
    )
    assert bundle_path.is_file()
    assert manifest_path.is_file()

    loaded_bundle, manifest = load_artifacts(temp_config.artifacts.output_dir)
    assert manifest["package_name"] == "mtpl-glm-pricing"
    assert manifest["dataset_sha256"] == "c" * 64
    assert "base_levels" in manifest["feature_schema"]
    assert "dependencies" in manifest

    loaded_tf = FeatureTransformer(spec=loaded_bundle.feature_spec)
    X_test, test_expo = loaded_tf.transform(test_df)
    preds_loaded = loaded_bundle.pricing_model.predict_components(X_test, test_expo)

    X_orig, orig_expo = transformer.transform(test_df)
    assert list(X_test.columns) == list(X_orig.columns)
    assert np.allclose(X_test.to_numpy(), X_orig.to_numpy())
    assert np.allclose(test_expo, orig_expo)
    assert np.all(preds_loaded["predicted_pure_premium"] > 0.0)


def test_load_artifacts_detects_corrupt_bundle_and_incompatible_manifest(
    synthetic_portfolio: pd.DataFrame,
    temp_config: PricingConfig,
) -> None:
    _, _, bundle_path, manifest_path = _train_and_save(synthetic_portfolio, temp_config)
    art_dir = temp_config.artifacts.output_dir

    # 1. Tampered joblib bytes trigger SHA-256 mismatch before unpickling.
    orig_bytes = bundle_path.read_bytes()
    bundle_path.write_bytes(b"corrupted-binary-stream")
    with pytest.raises(ArtifactCompatibilityError, match="Corrupted or mismatched model bundle"):
        load_artifacts(art_dir)

    # 2. Matching SHA-256 of invalid joblib bytes triggers ArtifactSerializationError.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_sha256"] = compute_file_sha256(bundle_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactSerializationError, match="Failed deserializing"):
        load_artifacts(art_dir)

    # 3. Restore valid joblib, mutate schema_format_version -> ArtifactCompatibilityError.
    bundle_path.write_bytes(orig_bytes)
    manifest["bundle_sha256"] = compute_file_sha256(bundle_path)
    manifest["schema_format_version"] = "99.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ArtifactCompatibilityError, match="Incompatible artifact schema_format_version"
    ):
        load_artifacts(art_dir)

    # 3b. Major package version mismatch -> ArtifactCompatibilityError.
    bundle_path.write_bytes(orig_bytes)
    manifest["bundle_sha256"] = compute_file_sha256(bundle_path)
    manifest["schema_format_version"] = "1.0.0"
    manifest["package_version"] = "99.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="Incompatible major package version"):
        load_artifacts(art_dir)

    # 3c. Manifest design_columns mismatch -> ArtifactCompatibilityError.
    manifest["package_version"] = "0.1.0"
    manifest["feature_schema"]["design_columns"] = ["intercept"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="disagree with manifest"):
        load_artifacts(art_dir)

    # 4. Missing manifest -> ArtifactCompatibilityError.
    manifest_path.unlink()
    with pytest.raises(ArtifactCompatibilityError, match="Missing artifact manifest"):
        load_artifacts(art_dir)


def test_inference_validation_and_unknown_category_policies(
    tmp_path: Path,
    synthetic_portfolio: pd.DataFrame,
    temp_config: PricingConfig,
) -> None:
    _transformer, test_df, _, _ = _train_and_save(synthetic_portfolio, temp_config)
    loaded_bundle, _ = load_artifacts(temp_config.artifacts.output_dir)

    # Single-row unlabeled inference (target columns omitted) succeeds cleanly.
    unlabeled_row = (
        test_df.drop(columns=["id", "claim", "nclaims", "amount", "average", "long", "lat"])
        .iloc[[0]]
        .copy()
    )
    X_single, expo_single = FeatureTransformer(spec=loaded_bundle.feature_spec).transform(
        unlabeled_row
    )
    single_pred = loaded_bundle.pricing_model.predict_components(X_single, expo_single)
    assert len(single_pred) == 1

    # Negative or zero exposure raises DataQualityError.
    zero_expo_row = unlabeled_row.copy()
    zero_expo_row["expo"] = 0.0
    with pytest.raises(DataQualityError, match="Exposure"):
        FeatureTransformer(spec=loaded_bundle.feature_spec).transform(zero_expo_row)

    # Missing value raises DataQualityError.
    nan_row = unlabeled_row.copy()
    nan_row["bm"] = np.nan
    with pytest.raises(DataQualityError, match="missing or non-finite"):
        FeatureTransformer(spec=loaded_bundle.feature_spec).transform(nan_row)

    # Non-numeric string raises SchemaValidationError.
    bad_type_row = unlabeled_row.copy()
    bad_type_row["power"] = "not-a-number"
    with pytest.raises(SchemaValidationError, match="non-numeric"):
        FeatureTransformer(spec=loaded_bundle.feature_spec).transform(bad_type_row)

    # Unknown category with policy='raise' raises UnknownCategoryError.
    unknown_row = unlabeled_row.copy()
    unknown_row["coverage"] = "PLATINUM_VIP"
    with pytest.raises(UnknownCategoryError) as exc_info:
        FeatureTransformer(spec=loaded_bundle.feature_spec).transform(unknown_row)
    assert exc_info.value.column == "coverage"
    assert exc_info.value.unknown_values == ["PLATINUM_VIP"]

    # Unknown category with policy='fallback_to_base' maps to base level cleanly.
    fallback_toml = write_test_toml(tmp_path / "fb", unknown_policy="fallback_to_base")
    (tmp_path / "fb").mkdir(parents=True, exist_ok=True)
    fallback_cfg = load_config(fallback_toml, base_dir=tmp_path / "fb")
    fb_tf = FeatureTransformer().fit(synthetic_portfolio, fallback_cfg)
    X_unknown, _ = fb_tf.transform(unknown_row)

    base_cov = next(c.base_level for c in fb_tf.spec.categorical_specs if c.column == "coverage")
    base_row = unknown_row.copy()
    base_row["coverage"] = base_cov
    X_base, _ = fb_tf.transform(base_row)
    assert np.allclose(X_unknown.to_numpy(), X_base.to_numpy())

    # Unfitted transformer raises InferenceError.
    with pytest.raises(InferenceError, match="not been fitted"):
        _ = FeatureTransformer().spec
