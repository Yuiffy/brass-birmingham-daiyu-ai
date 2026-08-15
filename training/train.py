from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_model_checkpoint, save_checkpoint
from .data import SelfPlayDataset, TrainingBatch, collate_positions
from .model import BrassPolicyValueNet, ModelConfig, compute_losses


def main() -> None:
    args = build_parser().parse_args()
    train(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Fast Brass variable-action policy/value network."
    )
    parser.add_argument(
        "--shards", nargs="+", required=True, help="Self-play JSONL shards"
    )
    parser.add_argument("--output", required=True, help="Destination checkpoint")
    parser.add_argument("--resume", help="Checkpoint to resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or CUDA device"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--shared-win-weight", type=float, default=1.0)
    parser.add_argument("--vp-margin-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--state-hidden-dim", type=int, default=256)
    parser.add_argument("--action-embedding-dim", type=int, default=128)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--vp-margin-scale", type=float, default=100.0)
    return parser


def train(args: argparse.Namespace) -> None:
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    dataset = SelfPlayDataset(args.shards)
    assert dataset.schema is not None
    train_indices, validation_indices = _split_indices(
        len(dataset), args.validation_fraction, args.seed
    )
    loader_generator = torch.Generator().manual_seed(args.seed ^ 0x5A17)
    train_loader = _make_loader(
        Subset(dataset, train_indices),
        args,
        shuffle=True,
        generator=loader_generator,
        device=device,
    )
    validation_loader = (
        _make_loader(
            Subset(dataset, validation_indices),
            args,
            shuffle=False,
            generator=None,
            device=device,
        )
        if validation_indices
        else None
    )

    resumed_payload = None
    if args.resume:
        model, resumed_payload = load_model_checkpoint(
            args.resume,
            expected_schema=dataset.schema,
            map_location=device,
        )
    else:
        model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=dataset.schema.state_dim,
                action_dim=dataset.schema.action_dim,
                state_hidden_dim=args.state_hidden_dim,
                action_embedding_dim=args.action_embedding_dim,
                trunk_dim=args.trunk_dim,
                vp_margin_scale=args.vp_margin_scale,
            )
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    global_step = 0
    if resumed_payload is not None:
        optimizer_state = resumed_payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        training_metadata = resumed_payload["metadata"].get("training", {})
        global_step = int(training_metadata.get("global_step", 0))

    print(
        json.dumps(
            {
                "event": "training_start",
                "device": str(device),
                "positions": len(dataset),
                "training_positions": len(train_indices),
                "validation_positions": len(validation_indices),
                "feature_version": dataset.schema.version,
                "state_dim": dataset.schema.state_dim,
                "action_dim": dataset.schema.action_dim,
                "model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "starting_step": global_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    last_epoch = -1
    stop = False
    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        aggregate: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_losses(
                output,
                batch,
                vp_margin_scale=model.config.vp_margin_scale,
                shared_win_weight=args.shared_win_weight,
                victory_point_margin_weight=args.vp_margin_weight,
            )
            losses.total.backward()
            gradient_norm = clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            global_step += 1
            batches += 1
            metrics = losses.detached_metrics()
            metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
            for name, value in metrics.items():
                aggregate[name] += value

            if global_step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_step",
                            "epoch": epoch,
                            "global_step": global_step,
                            **metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        epoch_metrics = {
            name: value / max(1, batches) for name, value in aggregate.items()
        }
        validation_metrics = (
            _evaluate(
                model,
                validation_loader,
                device,
                args.shared_win_weight,
                args.vp_margin_weight,
            )
            if validation_loader is not None
            else {}
        )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "global_step": global_step,
                    "training": epoch_metrics,
                    "validation": validation_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stop:
            break

    save_checkpoint(
        args.output,
        model,
        dataset.schema,
        optimizer=optimizer,
        training_metadata={
            "global_step": global_step,
            "completed_epoch": last_epoch,
            "seed": args.seed,
            "positions": len(dataset),
            "training_positions": len(train_indices),
            "validation_positions": len(validation_indices),
            "shards": [str(Path(path).resolve()) for path in args.shards],
            "engine_revisions": sorted(dataset.engine_revisions),
        },
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_saved",
                "path": str(Path(args.output).resolve()),
                "global_step": global_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dataset.close()


@torch.no_grad()
def _evaluate(
    model: BrassPolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    shared_win_weight: float,
    vp_margin_weight: float,
) -> dict[str, float]:
    model.eval()
    aggregate: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        losses = compute_losses(
            model(batch),
            batch,
            vp_margin_scale=model.config.vp_margin_scale,
            shared_win_weight=shared_win_weight,
            victory_point_margin_weight=vp_margin_weight,
        )
        for name, value in losses.detached_metrics().items():
            aggregate[name] += value
        batches += 1
    return {name: value / max(1, batches) for name, value in aggregate.items()}


def _make_loader(
    dataset: Subset,
    args: argparse.Namespace,
    shuffle: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_positions,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def _split_indices(
    length: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed ^ 0x71D4)
    indices = torch.randperm(length, generator=generator).tolist()
    validation_count = int(round(length * validation_fraction))
    if length > 1 and validation_fraction > 0.0:
        validation_count = min(length - 1, max(1, validation_count))
    else:
        validation_count = 0
    return indices[validation_count:], indices[:validation_count]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative")
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient-clip must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in [0, 1)")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.max_steps < 0:
        raise ValueError("max-steps must be non-negative")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")


if __name__ == "__main__":
    main()
