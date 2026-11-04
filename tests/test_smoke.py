"""End-to-end A10 smoke tests for baseline and revision experiments."""

import json
from pathlib import Path

import pytest
import torch

from arc_revision.data.dataset import ARCEpisodeDataset
from arc_revision.data.encoding import (
    decode_hypotheses,
    encode_hypotheses,
)
from arc_revision.data.generator import (
    GenerationConfig,
    generate_episode,
    save_dataset,
)
from arc_revision.data.hypotheses import apply_hypothesis
from scripts.evaluate import evaluate_checkpoint
from scripts.train import load_config, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_snapshot(tmp_path_factory):
    snapshot_dir = tmp_path_factory.mktemp("a10") / "v1"

    save_dataset(
        snapshot_dir,
        GenerationConfig(
            train_size=4,
            validation_size=2,
            test_size=8,
            ambiguous_fraction=0.5,
            seed=42,
        ),
    )

    return snapshot_dir


def _smoke_config(objective, snapshot_dir, output_dir):
    config = load_config(
        PROJECT_ROOT / "configs" / f"{objective}.yaml"
    )

    config["data"].update({
        "snapshot_dir": str(snapshot_dir),
        "train_size": 4,
        "validation_size": 2,
    })

    config["model"].update({
        "demonstration_dim": 16,
        "hidden_dim": 16,
        "program_dim": 16,
        "gru_layers": 1,
    })

    config["training"].update({
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
    })

    config["output_dir"] = str(output_dir)

    return config


@pytest.mark.parametrize("objective", ["baseline", "revision"])
def test_complete_training_evaluation_pipeline(
    tiny_snapshot,
    tmp_path,
    objective,
):
    config = _smoke_config(
        objective,
        tiny_snapshot,
        tmp_path / objective,
    )

    training_dir = run_experiment(config)

    checkpoint_path = training_dir / "best.pt"
    report_path = training_dir / "test_metrics.json"

    assert checkpoint_path.is_file()

    report = evaluate_checkpoint(
        checkpoint_path,
        device="cpu",
        output_path=report_path,
    )

    assert report_path.is_file()
    assert report == json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert report["objective"] == objective
    assert report["split"] == "test"
    assert report["checkpoint_epoch"] == 1

    overall = report["metrics"]["overall"]

    assert overall["examples"] == 8
    assert 0 <= overall["program_accuracy"] <= 1
    assert 0 <= overall["query_exact_match"] <= 1

    assert (
        overall["program_accuracy"]
        == overall["query_exact_match"]
    )

    ambiguity = report["metrics"]["by_ambiguity"]

    assert ambiguity["ambiguous"]["examples"] == 4
    assert ambiguity["unambiguous"]["examples"] == 4

    lengths = report["metrics"]["by_num_demos"]

    for num_demos in range(2, 6):
        assert lengths[str(num_demos)]["examples"] == 2

    for step in overall["consistency_by_step"].values():
        assert 0 <= step["valid_space_mass"] <= 1
        assert 0 <= step["uniform_prior_mass"] <= 1

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["config"] == config


def test_program_encoding_executes_ground_truth_query():
    episode = generate_episode(
        seed=101,
        num_demos=2,
        ambiguous=True,
    )

    encoded = encode_hypotheses(episode.hypotheses)
    decoded = decode_hypotheses(
        encoded.op_ids,
        encoded.op_params,
    )

    selected = decoded[episode.rule_index]
    predicted_grid = apply_hypothesis(
        episode.query_input, selected
    )

    torch.testing.assert_close(
        predicted_grid, episode.query_output
    )


def test_snapshot_query_and_candidate_order_preserved(
    tiny_snapshot,
):
    dataset = ARCEpisodeDataset.from_snapshot(
        tiny_snapshot, "test"
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        hypothesis = decode_hypotheses(
            sample.op_ids[
                sample.rule_index:sample.rule_index + 1
            ],
            sample.op_params[
                sample.rule_index:sample.rule_index + 1
            ],
        )[0]

        expected = apply_hypothesis(
            sample.query_input,
            hypothesis,
        )

        assert torch.equal(expected, sample.query_output)


def test_evaluation_rejects_changed_dataset_manifest(
    tiny_snapshot,
    tmp_path,
):
    config = _smoke_config(
        "baseline",
        tiny_snapshot,
        tmp_path / "manifest-check",
    )

    training_dir = run_experiment(config)
    checkpoint = training_dir / "best.pt"

    manifest_path = tiny_snapshot / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")

    try:
        manifest = json.loads(original)
        manifest["config"]["seed"] += 1
        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="manifest differs"):
            evaluate_checkpoint(checkpoint)
    finally:
        manifest_path.write_text(
            original,
            encoding="utf-8",
        )