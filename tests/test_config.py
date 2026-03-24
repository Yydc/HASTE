"""Tests for YAML config loading."""

import os
import pytest
import yaml


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


@pytest.fixture(params=["seedvig.yaml", "sadt.yaml", "mpddf.yaml"])
def config_path(request):
    return os.path.join(CONFIG_DIR, request.param)


def test_config_loads(config_path):
    """Each YAML config should load without errors."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_config_has_required_keys(config_path):
    """Each config must have essential keys."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required = [
        "dataset", "data_root", "num_channels", "num_bands",
        "batch_size", "epochs", "lr", "n_folds",
        "spatial_d_model", "temporal_d_model", "temporal_window",
        "lambda_pred", "lambda_band", "lambda_graph",
    ]
    for key in required:
        assert key in cfg, f"Missing key '{key}' in {config_path}"


def test_config_dataset_matches_filename(config_path):
    """Config 'dataset' value should match the filename."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    basename = os.path.basename(config_path).replace(".yaml", "")
    assert cfg["dataset"] == basename


def test_config_values_sensible(config_path):
    """Basic sanity checks on config values."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["num_channels"] > 0
    assert cfg["num_bands"] == 5
    assert cfg["batch_size"] > 0
    assert cfg["epochs"] > 0
    assert cfg["lr"] > 0
    assert cfg["n_folds"] > 0
    assert cfg["temporal_window"] > 0
