"""Reproducible baseline and revision-aware ARC training."""

import argparse
import json
import math
import platform
import random
from pathlib import Path

import torch
import yaml

from arc_revision.data.dataset import ARCEpisodeDataset, make_dataloader
from arc_revision.models.recurrent import SequentialGRUHypothesisClassifier
from arc_revision.training.losses import final_only_loss, revision_aware_loss


def load_config(path: Path) -> dict:
    """Read an experiment YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def validate_config(config: dict) -> None:
    """Reject invalid experiments before loading data or writing output."""
    try:
        experiment = config["experiment"]
        data = config["data"]
        model = config["model"]
        training = config["training"]
        output_dir = config["output_dir"]
        objective = experiment["objective"]
        seed = experiment["seed"]
        snapshot_dir = data["snapshot_dir"]
        lambda_revision = training["lambda_revision"]
        device = training["device"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing configuration field: {error}") from error

    if objective not in ("baseline", "revision"):
        raise ValueError("objective must be 'baseline' or 'revision'")
    if not isinstance(experiment.get("name"), str) or not experiment["name"]:
        raise ValueError("experiment.name must be a nonempty string")
    if type(seed) is not int or seed < 0:
        raise ValueError("experiment.seed must be a nonnegative integer")
    if not isinstance(snapshot_dir, str) or not snapshot_dir:
        raise ValueError("data.snapshot_dir must be a nonempty string")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a nonempty string")

    for name in ("train_size", "validation_size"):
        value = data.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"data.{name} must be a positive integer")
    for name in ("demonstration_dim", "hidden_dim", "program_dim", "gru_layers"):
        value = model.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"model.{name} must be a positive integer")
    for name in ("batch_size", "epochs"):
        value = training.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    rate = training.get("learning_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("training.learning_rate must be positive and finite")
    if (type(lambda_revision) not in (int, float)
            or not math.isfinite(lambda_revision)
            or lambda_revision < 0):
        raise ValueError("lambda_revision must be finite and nonnegative")
    if objective == "baseline" and lambda_revision != 0:
        raise ValueError("Baseline must have lambda_revision=0")
    if objective == "revision" and lambda_revision == 0:
        raise ValueError("Revision-aware objective requires positive lambda_revision")
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("training.device must be cpu, cuda, or auto")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(name)


def forward_batch(model, batch):
    return model(
        batch.demonstrations, batch.demo_mask, batch.op_ids,
        batch.op_params, batch.op_mask, batch.param_mask,
        batch.candidate_mask,
    )


def objective_loss(logits, batch, objective: str, lambda_revision: float):
    if objective == "baseline":
        return final_only_loss(
            logits, batch.rule_index, batch.demo_mask, batch.candidate_mask
        )
    return revision_aware_loss(
        logits, batch.rule_index, batch.demo_mask,
        batch.candidate_mask, batch.soft_targets,
        lambda_revision=lambda_revision,
    )


def run_epoch(model, loader, optimizer, device, objective, lambda_revision):
    """One epoch; optimizer=None enables validation without gradient tracking."""
    training = optimizer is not None
    model.train(training)
    total_objective = total_ce = total_correct = total_examples = batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch)
            loss = objective_loss(logits, batch, objective, lambda_revision)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective loss")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()

            with torch.no_grad():
                ce = final_only_loss(
                    logits, batch.rule_index, batch.demo_mask,
                    batch.candidate_mask,
                )
                index = batch.demo_mask.sum(dim=1) - 1
                final_logits = logits[
                    torch.arange(batch.batch_size, device=device), index
                ].masked_fill(~batch.candidate_mask, float("-inf"))
                correct = (final_logits.argmax(dim=-1) == batch.rule_index).sum()

            total_objective += loss.item() * batch.batch_size
            total_ce += ce.item() * batch.batch_size
            total_correct += int(correct.item())
            total_examples += batch.batch_size
            batches += 1

    if not total_examples:
        raise ValueError("Dataset contained no batches")
    return {
        "objective_loss": total_objective / total_examples,
        "final_ce": total_ce / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
        "batches": batches,
    }


def run_experiment(config: dict) -> Path:
    """Train and evaluate a run; return its output directory."""
    validate_config(config)
    experiment = config["experiment"]
    data = config["data"]
    train = config["training"]
    seed_everything(experiment["seed"])
    device = select_device(train["device"])

    snapshot = Path(data["snapshot_dir"])
    training_set = ARCEpisodeDataset.from_snapshot(snapshot, "train")
    validation_set = ARCEpisodeDataset.from_snapshot(snapshot, "validation")
    for name, dataset in (("train", training_set), ("validation", validation_set)):
        expected = data[f"{name}_size"]
        if len(dataset) != expected:
            raise ValueError(f"{name} snapshot contains {len(dataset)} episodes, expected {expected}")

    train_loader = make_dataloader(
        training_set, batch_size=train["batch_size"],
        shuffle=True, seed=experiment["seed"], num_workers=0,
    )
    validation_loader = make_dataloader(
        validation_set, batch_size=train["batch_size"],
        shuffle=False, seed=experiment["seed"], num_workers=0,
    )

    options = config["model"]
    model = SequentialGRUHypothesisClassifier(
        demonstration_dim=options["demonstration_dim"],
        hidden_dim=options["hidden_dim"],
        program_dim=options["program_dim"],
        num_gru_layers=options["gru_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    metadata = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "dataset_manifest": manifest,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    best_ce = float("inf")
    metrics_file = output_dir / "metrics.jsonl"
    for epoch in range(1, train["epochs"] + 1):
        training_metrics = run_epoch(
            model, train_loader, optimizer, device,
            experiment["objective"], train["lambda_revision"],
        )
        validation_metrics = run_epoch(
            model, validation_loader, None, device,
            experiment["objective"], train["lambda_revision"],
        )
        record = {
            "epoch": epoch,
            "train": training_metrics,
            "validation": validation_metrics,
        }
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": record,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_metrics["final_ce"] < best_ce:
            best_ce = validation_metrics["final_ce"]
            torch.save(checkpoint, output_dir / "best.pt")
        print(
            f"Epoch {epoch}/{train['epochs']} | "
            f"train={training_metrics['objective_loss']:.4f} "
            f"val_ce={validation_metrics['final_ce']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.3f}",
            flush=True,
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    run_experiment(config)


if __name__ == "__main__":
    main()