from __future__ import annotations

from echoquality.config import load_config


def test_experiment_config_extends_base() -> None:
    config = load_config("configs/experiments/camus10.yaml")
    assert config["experiment"]["name"] == "camus10"
    assert config["dataset"]["n_patients"] == 10
    assert config["recognition"]["quality_gate"]["models"] == ["LogisticRegression", "RandomForest"]
