"""Evaluate a trained ARC hypothesis classifier on the held-out test set."""

import argparse
import json
from pathlib import Path

import torch

from arc_revision.data.dataset import (
    ARCEpisodeDataset,
    make_dataloader,
)
from arc_revision.data.encoding import decode_hypotheses
from arc_revision.data.hypotheses import apply_hypothesis
from arc_revision.evaluation.metrics import (
    final_candidate_indices,
    valid_space_mass,
)
from arc_revision.models.recurrent import (
    SequentialGRUHypothesisClassifier,
)

def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    return torch.device(name)


def forward_batch(model, batch):
    return model(
        batch.demonstrations,
        batch.demo_mask,
        batch.op_ids,
        batch.op_params,
        batch.op_mask,
        batch.param_mask,
        batch.candidate_mask,
    )


def _new_stats() -> dict:
    return {
        "examples": 0,
        "program_correct": 0,
        "grid_correct": 0,
        "steps": {},
    }


def _add_result(
    stats: dict,
    program_ok: bool,
    grid_ok: bool,
    masses: list[float],
    uniform_masses: list[float],
) -> None:
    stats["examples"] += 1
    stats["program_correct"] += int(program_ok)
    stats["grid_correct"] += int(grid_ok)

    for step, (mass, uniform) in enumerate(
        zip(masses, uniform_masses), start=1
    ):
        key = str(step)
        entry = stats["steps"].setdefault(
            key,
            {"count": 0, "mass": 0.0, "uniform": 0.0},
        )
        entry["count"] += 1
        entry["mass"] += mass
        entry["uniform"] += uniform


def _summarize(stats: dict) -> dict:
    count = stats["examples"]

    if count == 0:
        return {
            "examples": 0,
            "program_accuracy": None,
            "query_exact_match": None,
            "consistency_by_step": {},
        }

    steps = {}
    for key, entry in sorted(
        stats["steps"].items(), key=lambda item: int(item[0])
    ):
        steps[key] = {
            "examples": entry["count"],
            "valid_space_mass": entry["mass"] / entry["count"],
            "uniform_prior_mass": entry["uniform"] / entry["count"],
        }

    return {
        "examples": count,
        "program_accuracy": stats["program_correct"] / count,
        "query_exact_match": stats["grid_correct"] / count,
        "consistency_by_step": steps,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    output_path: str | Path | None = None,
) -> dict:
    """Load an A09 checkpoint and evaluate its held-out test split."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    options = config["model"]

    snapshot_dir = Path(config["data"]["snapshot_dir"])
    manifest = json.loads(
        (snapshot_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    # Ensure this really is the same snapshot used during training.
    metadata_path = checkpoint_path.parent / "metadata.json"
    training_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if training_metadata["dataset_manifest"] != manifest:
        raise ValueError(
            "Dataset manifest differs from the training snapshot"
        )

    test_dataset = ARCEpisodeDataset.from_snapshot(
        snapshot_dir, "test"
    )

    if len(test_dataset) == 0:
        raise ValueError("Held-out test split is empty")

    device = select_device(device)

    model = SequentialGRUHypothesisClassifier(
        demonstration_dim=options["demonstration_dim"],
        hidden_dim=options["hidden_dim"],
        program_dim=options["program_dim"],
        num_gru_layers=options["gru_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = make_dataloader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    overall = _new_stats()
    by_ambiguity = {
        "ambiguous": _new_stats(),
        "unambiguous": _new_stats(),
    }
    by_demos = {
        str(count): _new_stats()
        for count in range(2, 6)
    }

    with torch.inference_mode():
        for batch in loader:
            model_batch = batch.to(device)

            # Only input demonstrations and candidate programs
            # enter the neural model.
            logits = forward_batch(model, model_batch)

            predicted = final_candidate_indices(
                logits,
                model_batch.demo_mask,
                model_batch.candidate_mask,
            ).cpu()

            # Ground-truth version-space masks are used strictly
            # after inference to measure consistency.
            masses = valid_space_mass(
                logits,
                model_batch.valid_masks,
                model_batch.demo_mask,
                model_batch.candidate_mask,
            ).cpu()

            for row in range(batch.batch_size):
                selected = int(predicted[row])
                num_demos = int(batch.num_demos[row])
                num_candidates = int(batch.num_candidates[row])

                # Decode exactly the selected executable program.
                hypothesis = decode_hypotheses(
                    batch.op_ids[row, selected:selected + 1],
                    batch.op_params[row, selected:selected + 1],
                )[0]

                predicted_grid = apply_hypothesis(
                    batch.query_input[row], hypothesis
                )

                program_ok = (
                    selected == int(batch.rule_index[row])
                )
                grid_ok = torch.equal(
                    predicted_grid, batch.query_output[row]
                )

                episode_masses = masses[
                    row, :num_demos
                ].tolist()

                uniform_masses = [
                    float(batch.valid_masks[row, step].sum())
                    / num_candidates
                    for step in range(num_demos)
                ]

                ambiguous = (
                    int(batch.valid_masks[row, 0].sum()) > 1
                )
                ambiguity_key = (
                    "ambiguous" if ambiguous else "unambiguous"
                )

                for stats in (
                    overall,
                    by_ambiguity[ambiguity_key],
                    by_demos[str(num_demos)],
                ):
                    _add_result(
                        stats,
                        program_ok,
                        grid_ok,
                        episode_masses,
                        uniform_masses,
                    )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "objective": config["experiment"]["objective"],
        "split": "test",
        "snapshot_dir": str(snapshot_dir),
        "test_sha256": manifest["splits"]["test"]["sha256"],
        "metrics": {
            "overall": _summarize(overall),
            "by_ambiguity": {
                name: _summarize(stats)
                for name, stats in by_ambiguity.items()
            },
            "by_num_demos": {
                name: _summarize(stats)
                for name, stats in by_demos.items()
            },
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        output_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        device=args.device,
        output_path=args.output,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()