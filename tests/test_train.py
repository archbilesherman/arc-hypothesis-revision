"""A09 CPU smoke, reproducibility, and checkpoint tests."""

import json
import math
from pathlib import Path

import pytest
import torch

from arc_revision.data.generator import GenerationConfig, save_dataset
from arc_revision.models.recurrent import SequentialGRUHypothesisClassifier
from scripts.train import load_config, run_experiment, validate_config


@pytest.fixture(scope="module")
def tiny_snapshot(tmp_path_factory):
    directory = tmp_path_factory.mktemp("a09") / "v1"
    save_dataset(
        directory,
        GenerationConfig(
            train_size=4,
            validation_size=2,
            test_size=0,
            ambiguous_fraction=0.5,
            seed=42,
        ),
    )
    return directory


def make_config(snapshot, output_dir, objective="baseline"):
    return {
        "experiment": {"name": objective, "objective": objective, "seed": 42},
        "data": {
            "snapshot_dir": str(snapshot),
            "train_size": 4,
            "validation_size": 2,
        },
        "model": {
            "demonstration_dim": 16,
            "hidden_dim": 16,
            "program_dim": 16,
            "gru_layers": 1,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "lambda_revision": 0.5 if objective == "revision" else 0.0,
            "device": "cpu",
        },
        "output_dir": str(output_dir),
    }


def test_configuration_yaml_parses():
    baseline = load_config(Path("configs/baseline.yaml"))
    revision = load_config(Path("configs/revision.yaml"))

    validate_config(baseline)
    validate_config(revision)

    for key in ("data", "model"):
        assert baseline[key] == revision[key]

    for key in ("batch_size", "epochs", "learning_rate", "device"):
        assert baseline["training"][key] == revision["training"][key]

    assert baseline["experiment"]["seed"] == revision["experiment"]["seed"]


@pytest.mark.parametrize("objective", ["baseline", "revision"])
def test_cpu_smoke_training_and_reload(tiny_snapshot, tmp_path, objective):
    config = make_config(tiny_snapshot, tmp_path / objective, objective)
    output = run_experiment(config)

    assert (output / "config.yaml").exists()
    assert (output / "metadata.json").exists()
    assert (output / "last.pt").exists()
    assert (output / "best.pt").exists()

    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1

    record = records[0]
    assert record["train"]["batches"] == 2
    assert record["train"]["examples"] == 4
    assert record["validation"]["examples"] == 2

    for split in ("train", "validation"):
        for metric in ("objective_loss", "final_ce", "accuracy"):
            assert math.isfinite(record[split][metric])
        assert 0 <= record[split]["accuracy"] <= 1

    checkpoint = torch.load(
        output / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["config"] == config

    model = SequentialGRUHypothesisClassifier(
        demonstration_dim=config["model"]["demonstration_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        program_dim=config["model"]["program_dim"],
        num_gru_layers=config["model"]["gru_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["optimizer_state_dict"]["state"]

    # Verify that optimizer updates changed the initial parameters.
    torch.manual_seed(config["experiment"]["seed"])
    fresh_model = SequentialGRUHypothesisClassifier(
        demonstration_dim=16,
        hidden_dim=16,
        program_dim=16,
        num_gru_layers=1,
    )
    assert any(
        not torch.equal(value, fresh_model.state_dict()[key])
        for key, value in model.state_dict().items()
        if value.is_floating_point()
    )


def test_seed_reproducibility_cpu(tiny_snapshot, tmp_path):
    first = run_experiment(
        make_config(tiny_snapshot, tmp_path / "first")
    )
    second = run_experiment(
        make_config(tiny_snapshot, tmp_path / "second")
    )

    assert (
        (first / "metrics.jsonl").read_text()
        == (second / "metrics.jsonl").read_text()
    )

    a = torch.load(
        first / "last.pt", map_location="cpu", weights_only=True
    )
    b = torch.load(
        second / "last.pt", map_location="cpu", weights_only=True
    )

    for key in a["model_state_dict"]:
        torch.testing.assert_close(
            a["model_state_dict"][key],
            b["model_state_dict"][key],
            rtol=0,
            atol=0,
        )


def test_invalid_config_rejected(tiny_snapshot, tmp_path):
    config = make_config(tiny_snapshot, tmp_path / "bad")
    config["training"]["lambda_revision"] = 0.5

    with pytest.raises(ValueError, match="Baseline"):
        run_experiment(config)

    assert not (tmp_path / "bad").exists()


def test_mismatched_snapshot_size_rejected(tiny_snapshot, tmp_path):
    config = make_config(tiny_snapshot, tmp_path / "bad-size")
    config["data"]["train_size"] = 20

    with pytest.raises(ValueError, match="snapshot contains"):
        run_experiment(config)

    assert not (tmp_path / "bad-size").exists()


def test_two_layer_gru_construction():
    model = SequentialGRUHypothesisClassifier(
        demonstration_dim=16,
        hidden_dim=16,
        program_dim=16,
        num_gru_layers=2,
    )

    assert model.recurrent.num_layers == 2