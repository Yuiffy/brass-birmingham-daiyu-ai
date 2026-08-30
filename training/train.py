from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .checkpoint import (
    LEGACY_ACTOR_VP_HEAD_MARKER,
    load_model_checkpoint,
    save_checkpoint,
)
from .data import (
    ACTOR_VP_TARGET_ABSOLUTE,
    ACTOR_VP_RAILROAD_SCALE_DEFAULT,
    POLICY_TARGET_FIELD,
    POLICY_TARGET_EXPONENT_DEFAULT,
    SEARCH_POLICY_TARGET_FIELD,
    SUPPORTED_ACTOR_VP_TARGET_MODES,
    SUPPORTED_POLICY_TARGET_FIELDS,
    SelfPlayDataset,
    TrainingBatch,
    collate_positions,
    validate_policy_target_exponent,
)
from .model import (
    BrassPolicyValueNet,
    ModelConfig,
    compute_losses,
    expand_with_policy_adapter,
    with_actor_vp_target_mode,
)


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
    parser.add_argument(
        "--replay-shards",
        nargs="*",
        default=[],
        help=(
            "Optional old-champion/self-play shards kept in the replay pool; "
            "their target fraction is controlled by --replay-fraction"
        ),
    )
    parser.add_argument(
        "--human-shards",
        nargs="*",
        default=[],
        help=(
            "Optional audited human-policy shards; contaminated terminal values "
            "remain policy-only and their target fraction is controlled by "
            "--human-fraction"
        ),
    )
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.25,
        help="Replacement-sampling fraction for --replay-shards (default: 0.25)",
    )
    parser.add_argument(
        "--human-fraction",
        type=float,
        default=0.10,
        help="Replacement-sampling fraction for --human-shards (default: 0.10)",
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        help="Independent self-play JSONL shards used only for validation",
    )
    parser.add_argument(
        "--policy-target-field",
        choices=SUPPORTED_POLICY_TARGET_FIELDS,
        default=POLICY_TARGET_FIELD,
        help=(
            "Select the serialized policy target: strategy-reweighted policy_target "
            "or raw search visits search_policy_target"
        ),
    )
    parser.add_argument(
        "--search-policy-target-mix",
        type=float,
        default=0.0,
        help=(
            "Mix this fraction of raw search_policy_target into policy_target "
            "after selecting the strategy target"
        ),
    )
    parser.add_argument(
        "--policy-target-exponent",
        type=float,
        default=POLICY_TARGET_EXPONENT_DEFAULT,
        help=(
            "Raise normalized policy targets to this positive exponent and "
            "renormalize; values above 1 sharpen search preferences"
        ),
    )
    parser.add_argument(
        "--actor-vp-target-mode",
        choices=SUPPORTED_ACTOR_VP_TARGET_MODES,
        default=ACTOR_VP_TARGET_ABSOLUTE,
        help=(
            "Train the actor VP head on absolute final VP or future remaining VP; "
            "inference always exposes absolute final VP"
        ),
    )
    parser.add_argument("--output", required=True, help="Destination checkpoint")
    parser.add_argument("--resume", help="Checkpoint to resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or CUDA device"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many validation epochs without a significant loss "
            "improvement; 0 disables early stopping"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss reduction that resets early-stopping patience",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--shared-win-weight", type=float, default=1.0)
    parser.add_argument("--vp-margin-weight", type=float, default=0.25)
    parser.add_argument("--actor-vp-weight", type=float, default=0.25)
    parser.add_argument(
        "--final-vp-quality-weight",
        type=float,
        default=0.0,
        help=(
            "Favor higher-final-VP trajectories in all training losses; "
            "0 disables quality weighting"
        ),
    )
    parser.add_argument(
        "--actor-vp-head-only",
        action="store_true",
        help="Freeze the existing policy/value network and train only the actor VP head",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Deprecated; use independent --validation-shards (must remain zero)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--state-hidden-dim", type=int, default=256)
    parser.add_argument("--action-embedding-dim", type=int, default=128)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--vp-margin-scale", type=float, default=100.0)
    parser.add_argument("--actor-vp-scale", type=float, default=140.0)
    parser.add_argument(
        "--actor-vp-railroad-scale",
        type=float,
        default=ACTOR_VP_RAILROAD_SCALE_DEFAULT,
        help="Normalization scale for Railroad positions in the phase-conditioned target",
    )
    parser.add_argument(
        "--policy-adapter-dim",
        type=int,
        default=0,
        help=(
            "Add a zero-initialized residual policy adapter of this width to a "
            "resumed checkpoint"
        ),
    )
    parser.add_argument(
        "--policy-adapter-only",
        action="store_true",
        help="Freeze the base policy/value network and train only its policy adapter",
    )
    parser.add_argument(
        "--value-tower-only",
        action="store_true",
        help=(
            "Freeze the policy tower and train only an independent value encoder "
            "and its three value heads"
        ),
    )
    return parser


def train(args: argparse.Namespace) -> None:
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    dataset, validation_dataset = _load_datasets(
        args.shards,
        args.validation_shards,
        replay_shard_paths=args.replay_shards,
        human_shard_paths=args.human_shards,
        replay_fraction=args.replay_fraction,
        human_fraction=args.human_fraction,
        policy_target_field=args.policy_target_field,
        search_policy_target_mix=args.search_policy_target_mix,
        policy_target_exponent=args.policy_target_exponent,
        actor_vp_target_mode=args.actor_vp_target_mode,
    )
    try:
        _train_loaded_datasets(args, device, dataset, validation_dataset)
    finally:
        dataset.close()
        if validation_dataset is not None:
            validation_dataset.close()


def _train_loaded_datasets(
    args: argparse.Namespace,
    device: torch.device,
    dataset: SelfPlayDataset,
    validation_dataset: SelfPlayDataset | None,
) -> None:
    assert dataset.schema is not None
    loader_generator = torch.Generator().manual_seed(args.seed ^ 0x5A17)
    train_loader = _make_loader(
        dataset,
        args,
        shuffle=True,
        generator=loader_generator,
        device=device,
    )
    validation_loader = (
        _make_loader(
            validation_dataset,
            args,
            shuffle=False,
            generator=None,
            device=device,
        )
        if validation_dataset is not None
        else None
    )

    resumed_payload = None
    policy_adapter_expanded = False
    actor_vp_target_mode_changed = False
    policy_target_configuration_changed = False
    final_vp_quality_weight_changed = False
    if args.resume:
        model, resumed_payload = load_model_checkpoint(
            args.resume,
            expected_schema=dataset.schema,
            map_location=device,
        )
        if args.policy_adapter_dim > 0:
            previous_adapter_dim = model.config.policy_adapter_dim
            model = expand_with_policy_adapter(model, args.policy_adapter_dim)
            policy_adapter_expanded = previous_adapter_dim == 0
        if model.config.actor_vp_target_mode != args.actor_vp_target_mode:
            model = with_actor_vp_target_mode(model, args.actor_vp_target_mode)
            actor_vp_target_mode_changed = True
        policy_target_configuration_changed = _policy_target_configuration_changed(
            resumed_payload, args
        )
        final_vp_quality_weight_changed = _final_vp_quality_configuration_changed(
            resumed_payload, args
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
                actor_vp_scale=args.actor_vp_scale,
                actor_vp_railroad_scale=args.actor_vp_railroad_scale,
                actor_vp_target_mode=args.actor_vp_target_mode,
            )
        )
    if args.policy_adapter_only and model.config.policy_adapter_dim <= 0:
        raise ValueError("policy-adapter-only requires a model with a policy adapter")
    if args.value_tower_only and not model.config.separate_value_encoder:
        raise ValueError(
            "value-tower-only requires a model with a separate value encoder"
        )
    if args.actor_vp_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.actor_victory_points_head.parameters():
            parameter.requires_grad_(True)
    elif args.policy_adapter_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("policy_adapter_"))
    elif args.value_tower_only:
        value_tower_prefixes = (
            "value_state_encoder.",
            "shared_win_head.",
            "victory_point_margin_head.",
            "actor_victory_points_head.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(value_tower_prefixes))
    model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    global_step = 0
    optimizer_state_restored = _should_restore_optimizer_state(
        resumed_payload,
        actor_vp_head_only=args.actor_vp_head_only,
        policy_adapter_only=args.policy_adapter_only,
        value_tower_only=args.value_tower_only,
        architecture_expanded=policy_adapter_expanded,
        actor_vp_target_mode_changed=actor_vp_target_mode_changed,
        policy_target_configuration_changed=policy_target_configuration_changed,
        final_vp_quality_weight_changed=final_vp_quality_weight_changed,
    )
    if optimizer_state_restored:
        assert resumed_payload is not None
        _restore_optimizer_state(optimizer, resumed_payload, args)
    if resumed_payload is not None:
        training_metadata = resumed_payload["metadata"].get("training", {})
        global_step = int(training_metadata.get("global_step", 0))
    starting_global_step = global_step
    steps_this_run = 0

    print(
        json.dumps(
            {
                "event": "training_start",
                "device": str(device),
                "positions": len(dataset),
                "training_positions": len(dataset),
                "validation_positions": (
                    len(validation_dataset) if validation_dataset is not None else 0
                ),
                "feature_version": dataset.schema.version,
                "state_dim": dataset.schema.state_dim,
                "action_dim": dataset.schema.action_dim,
                "model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "policy_adapter_dim": model.config.policy_adapter_dim,
                "policy_adapter_only": args.policy_adapter_only,
                "value_tower_only": args.value_tower_only,
                "policy_target_field": args.policy_target_field,
                "search_policy_target_mix": args.search_policy_target_mix,
                "policy_target_exponent": args.policy_target_exponent,
                "policy_target_configuration_changed": policy_target_configuration_changed,
                "final_vp_quality_weight": args.final_vp_quality_weight,
                "final_vp_quality_weight_changed": final_vp_quality_weight_changed,
                "actor_vp_target_mode": args.actor_vp_target_mode,
                "actor_vp_target_mode_changed": actor_vp_target_mode_changed,
                "actor_vp_railroad_scale": model.config.actor_vp_railroad_scale,
                "sampling": dataset.sampling_metadata,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "starting_step": global_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    initial_validation_metrics = (
        _evaluate(
            model,
            validation_loader,
            device,
            args.shared_win_weight,
            args.vp_margin_weight,
            args.actor_vp_weight,
            args.final_vp_quality_weight,
        )
        if validation_loader is not None
        else {}
    )
    if initial_validation_metrics:
        print(
            json.dumps(
                {
                    "event": "validation_baseline",
                    "global_step": global_step,
                    "validation": initial_validation_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    best_model_state = (
        copy.deepcopy(model.state_dict()) if initial_validation_metrics else None
    )
    best_optimizer_state = (
        copy.deepcopy(optimizer.state_dict())
        if initial_validation_metrics
        else None
    )
    best_validation_metrics = dict(initial_validation_metrics)
    best_global_step = global_step
    best_epoch = -1
    last_validation_metrics = dict(initial_validation_metrics)
    early_stopping_reference_loss = initial_validation_metrics.get("loss")
    epochs_without_significant_improvement = 0

    last_epoch = -1
    stop = False
    stop_reason = "epochs_completed"
    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        aggregate: defaultdict[str, float] = defaultdict(float)
        batches = 0
        training_positions = 0
        for batch in train_loader:
            batch_positions = int(batch.states.shape[0])
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_losses(
                output,
                batch,
                vp_margin_scale=model.config.vp_margin_scale,
                actor_vp_scale=model.config.actor_vp_scale,
                actor_vp_railroad_scale=model.config.actor_vp_railroad_scale,
                actor_vp_target_mode=model.config.actor_vp_target_mode,
                shared_win_weight=args.shared_win_weight,
                victory_point_margin_weight=args.vp_margin_weight,
                actor_victory_points_weight=args.actor_vp_weight,
                final_vp_quality_weight=args.final_vp_quality_weight,
            )
            losses.total.backward()
            gradient_norm = clip_grad_norm_(
                trainable_parameters, args.gradient_clip
            )
            optimizer.step()
            global_step += 1
            steps_this_run += 1
            batches += 1
            training_positions += batch_positions
            metrics = losses.detached_metrics()
            metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
            for name, value in metrics.items():
                aggregate[name] += value * (
                    1 if name == "gradient_norm" else batch_positions
                )

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
            if args.max_steps and steps_this_run >= args.max_steps:
                stop = True
                stop_reason = "max_steps"
                break

        epoch_metrics = {
            name: value
            / max(1, batches if name == "gradient_norm" else training_positions)
            for name, value in aggregate.items()
        }
        validation_metrics = (
            _evaluate(
                model,
                validation_loader,
                device,
                args.shared_win_weight,
                args.vp_margin_weight,
                args.actor_vp_weight,
                args.final_vp_quality_weight,
            )
            if validation_loader is not None
            else {}
        )
        last_validation_metrics = validation_metrics
        if validation_metrics and (
            not best_validation_metrics
            or validation_metrics["loss"] < best_validation_metrics["loss"]
        ):
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_validation_metrics = dict(validation_metrics)
            best_global_step = global_step
            best_epoch = epoch
        early_stopping = None
        if args.early_stopping_patience > 0:
            current_loss = validation_metrics["loss"]
            assert early_stopping_reference_loss is not None
            significant_improvement = current_loss < (
                early_stopping_reference_loss - args.early_stopping_min_delta
            )
            if significant_improvement:
                early_stopping_reference_loss = current_loss
                epochs_without_significant_improvement = 0
            else:
                epochs_without_significant_improvement += 1
            early_stopping = {
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "reference_loss": early_stopping_reference_loss,
                "epochs_without_significant_improvement": (
                    epochs_without_significant_improvement
                ),
                "significant_improvement": significant_improvement,
            }
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "global_step": global_step,
                    "training": epoch_metrics,
                    "validation": validation_metrics,
                    "early_stopping": early_stopping,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            not stop
            and early_stopping is not None
            and epochs_without_significant_improvement
            >= args.early_stopping_patience
        ):
            stop = True
            stop_reason = "early_stopping"
            print(
                json.dumps(
                    {
                        "event": "training_early_stopped",
                        "epoch": epoch,
                        "global_step": global_step,
                        "patience": args.early_stopping_patience,
                        "min_delta": args.early_stopping_min_delta,
                        "epochs_without_significant_improvement": (
                            epochs_without_significant_improvement
                        ),
                        "selected_epoch": best_epoch,
                        "selected_validation": best_validation_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if stop:
            break

    last_global_step = global_step
    if best_model_state is not None and best_optimizer_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        global_step = best_global_step

    save_checkpoint(
        args.output,
        model,
        dataset.schema,
        optimizer=optimizer,
        training_metadata={
            "global_step": global_step,
            "starting_global_step": starting_global_step,
            "steps_this_run": steps_this_run,
            "completed_epoch": best_epoch if best_model_state is not None else last_epoch,
            "last_global_step": last_global_step,
            "last_completed_epoch": last_epoch,
            "selected_epoch": best_epoch if best_model_state is not None else last_epoch,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "shared_win_weight": args.shared_win_weight,
            "vp_margin_weight": args.vp_margin_weight,
            "actor_vp_weight": args.actor_vp_weight,
            "final_vp_quality_weight": args.final_vp_quality_weight,
            "actor_vp_head_only": args.actor_vp_head_only,
            "policy_adapter_dim": model.config.policy_adapter_dim,
            "policy_adapter_only": args.policy_adapter_only,
            "value_tower_only": args.value_tower_only,
            "policy_target_field": args.policy_target_field,
            "search_policy_target_mix": args.search_policy_target_mix,
            "policy_target_exponent": args.policy_target_exponent,
            "policy_target_configuration_changed": policy_target_configuration_changed,
            "actor_vp_target_mode": args.actor_vp_target_mode,
            "actor_vp_target_mode_changed": actor_vp_target_mode_changed,
            "actor_vp_railroad_scale": model.config.actor_vp_railroad_scale,
            "policy_adapter_expanded": policy_adapter_expanded,
            "optimizer_state_restored": optimizer_state_restored,
            "max_steps": args.max_steps,
            "stop_reason": stop_reason,
            "epochs_ran": last_epoch + 1,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_triggered": stop_reason == "early_stopping",
            "epochs_without_significant_improvement": (
                epochs_without_significant_improvement
            ),
            "early_stopping_reference_loss": early_stopping_reference_loss,
            "resume_checkpoint": (
                str(Path(args.resume).resolve()) if args.resume else None
            ),
            "positions": len(dataset),
            "training_positions": len(dataset),
            "validation_positions": (
                len(validation_dataset) if validation_dataset is not None else 0
            ),
            "shards": [str(Path(path).resolve()) for path in args.shards],
            "replay_shards": [
                str(Path(path).resolve()) for path in (args.replay_shards or [])
            ],
            "human_shards": [
                str(Path(path).resolve()) for path in (args.human_shards or [])
            ],
            "replay_fraction": args.replay_fraction,
            "human_fraction": args.human_fraction,
            "sampling": dataset.sampling_metadata,
            "engine_revisions": sorted(dataset.engine_revisions),
            "validation_shards": [
                str(Path(path).resolve()) for path in (args.validation_shards or [])
            ],
            "validation_engine_revisions": (
                sorted(validation_dataset.engine_revisions)
                if validation_dataset is not None
                else []
            ),
            "initial_validation_metrics": initial_validation_metrics,
            "selected_validation_metrics": best_validation_metrics,
            "last_validation_metrics": last_validation_metrics,
        },
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_saved",
                "path": str(Path(args.output).resolve()),
                "global_step": global_step,
                "selected_epoch": (
                    best_epoch if best_model_state is not None else last_epoch
                ),
                "selected_validation": best_validation_metrics,
                "stop_reason": stop_reason,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@torch.no_grad()
def _evaluate(
    model: BrassPolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    shared_win_weight: float,
    vp_margin_weight: float,
    actor_vp_weight: float = 0.25,
    final_vp_quality_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    aggregate: defaultdict[str, float] = defaultdict(float)
    positions = 0
    for batch in loader:
        batch_positions = int(batch.states.shape[0])
        batch = batch.to(device, non_blocking=device.type == "cuda")
        losses = compute_losses(
            model(batch),
            batch,
            vp_margin_scale=model.config.vp_margin_scale,
            actor_vp_scale=model.config.actor_vp_scale,
            actor_vp_railroad_scale=model.config.actor_vp_railroad_scale,
            actor_vp_target_mode=model.config.actor_vp_target_mode,
            shared_win_weight=shared_win_weight,
            victory_point_margin_weight=vp_margin_weight,
            actor_victory_points_weight=actor_vp_weight,
            final_vp_quality_weight=final_vp_quality_weight,
        )
        for name, value in losses.detached_metrics().items():
            aggregate[name] += value * batch_positions
        positions += batch_positions
    return {name: value / max(1, positions) for name, value in aggregate.items()}


def _make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    shuffle: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> DataLoader:
    sampling_weights = getattr(dataset, "sampling_weights", None)
    sampler = None
    if shuffle and sampling_weights is not None:
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_positions,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    resumed_payload: dict,
    args: argparse.Namespace,
) -> None:
    optimizer_state = resumed_payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        if resumed_payload.get(LEGACY_ACTOR_VP_HEAD_MARKER) is True:
            optimizer_state = _extend_legacy_optimizer_state(
                optimizer_state, optimizer.state_dict()
            )
        optimizer.load_state_dict(optimizer_state)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = args.learning_rate
        parameter_group["weight_decay"] = args.weight_decay


def _policy_target_configuration_changed(
    resumed_payload: dict, args: argparse.Namespace
) -> bool:
    previous_training = resumed_payload.get("metadata", {}).get("training", {})

    def same_float(raw_value: object, expected: float) -> bool:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value == expected

    return bool(
        previous_training.get("policy_target_field", POLICY_TARGET_FIELD)
        != args.policy_target_field
        or not same_float(
            previous_training.get(
                "search_policy_target_mix", 0.0
            ),
            args.search_policy_target_mix,
        )
        or not same_float(
            previous_training.get(
                "policy_target_exponent", POLICY_TARGET_EXPONENT_DEFAULT
            ),
            args.policy_target_exponent,
        )
    )


def _final_vp_quality_configuration_changed(
    resumed_payload: dict, args: argparse.Namespace
) -> bool:
    previous_training = resumed_payload.get("metadata", {}).get("training", {})
    try:
        previous = float(previous_training.get("final_vp_quality_weight", 0.0))
    except (TypeError, ValueError):
        return True
    return not math.isfinite(previous) or previous != args.final_vp_quality_weight


def _should_restore_optimizer_state(
    resumed_payload: dict | None,
    *,
    actor_vp_head_only: bool,
    policy_adapter_only: bool = False,
    value_tower_only: bool = False,
    architecture_expanded: bool = False,
    actor_vp_target_mode_changed: bool = False,
    policy_target_configuration_changed: bool = False,
    final_vp_quality_weight_changed: bool = False,
) -> bool:
    if (
        resumed_payload is None
        or actor_vp_head_only
        or policy_adapter_only
        or value_tower_only
        or architecture_expanded
        or actor_vp_target_mode_changed
        or policy_target_configuration_changed
        or final_vp_quality_weight_changed
    ):
        return False
    previous_training = resumed_payload.get("metadata", {}).get("training", {})
    return not bool(
        previous_training.get("actor_vp_head_only", False)
        or previous_training.get("policy_adapter_only", False)
        or previous_training.get("value_tower_only", False)
    )


def _extend_legacy_optimizer_state(saved: dict, current: dict) -> dict:
    migrated = copy.deepcopy(saved)
    saved_groups = migrated.get("param_groups")
    current_groups = current.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise ValueError("optimizer parameter groups are missing")
    if len(saved_groups) != len(current_groups):
        raise ValueError("legacy optimizer parameter-group count changed")

    old_to_new: dict[int, int] = {}
    for saved_group, current_group in zip(saved_groups, current_groups, strict=True):
        saved_params = saved_group.get("params")
        current_params = current_group.get("params")
        if not isinstance(saved_params, list) or not isinstance(current_params, list):
            raise ValueError("optimizer parameter identifiers are missing")
        if len(saved_params) >= len(current_params):
            raise ValueError("legacy optimizer has no room for the new actor VP head")
        old_to_new.update(zip(saved_params, current_params, strict=False))
        saved_group["params"] = list(current_params)

    saved_state = migrated.get("state")
    if not isinstance(saved_state, dict):
        raise ValueError("optimizer state is missing")
    migrated["state"] = {
        old_to_new[old_id]: value
        for old_id, value in saved_state.items()
        if old_id in old_to_new
    }
    return migrated


def _load_datasets(
    shard_paths: list[str] | tuple[str, ...],
    validation_shard_paths: list[str] | tuple[str, ...] | None,
    *,
    replay_shard_paths: list[str] | tuple[str, ...] | None = None,
    human_shard_paths: list[str] | tuple[str, ...] | None = None,
    replay_fraction: float = 0.25,
    human_fraction: float = 0.10,
    policy_target_field: str = POLICY_TARGET_FIELD,
    search_policy_target_mix: float = 0.0,
    policy_target_exponent: float = POLICY_TARGET_EXPONENT_DEFAULT,
    actor_vp_target_mode: str = ACTOR_VP_TARGET_ABSOLUTE,
) -> tuple[SelfPlayDataset, SelfPlayDataset | None]:
    teacher_paths = tuple(Path(path).resolve() for path in shard_paths)
    replay_paths = tuple(
        Path(path).resolve() for path in (replay_shard_paths or ())
    )
    human_paths = tuple(
        Path(path).resolve() for path in (human_shard_paths or ())
    )
    training_paths = teacher_paths + replay_paths + human_paths
    if len(set(training_paths)) != len(training_paths):
        duplicates = sorted(
            path
            for path in set(training_paths)
            if training_paths.count(path) > 1
        )
        raise ValueError(
            "training source shards overlap: "
            + ", ".join(str(path) for path in duplicates)
        )
    validation_paths = tuple(
        Path(path).resolve() for path in (validation_shard_paths or ())
    )
    overlap = set(training_paths) & set(validation_paths)
    if overlap:
        paths = ", ".join(str(path) for path in sorted(overlap))
        raise ValueError(f"training and validation shards overlap: {paths}")

    training_dataset = SelfPlayDataset(
        training_paths,
        policy_target_field=policy_target_field,
        search_policy_target_mix=search_policy_target_mix,
        policy_target_exponent=policy_target_exponent,
        actor_vp_target_mode=actor_vp_target_mode,
    )
    try:
        _configure_source_sampling(
            training_dataset,
            teacher_paths=teacher_paths,
            replay_paths=replay_paths,
            human_paths=human_paths,
            replay_fraction=replay_fraction,
            human_fraction=human_fraction,
        )
    except Exception:
        training_dataset.close()
        raise
    if not validation_paths:
        return training_dataset, None
    assert training_dataset.schema is not None
    try:
        validation_dataset = SelfPlayDataset(
            validation_paths,
            expected_schema=training_dataset.schema,
            policy_target_field=policy_target_field,
            search_policy_target_mix=search_policy_target_mix,
            policy_target_exponent=policy_target_exponent,
            actor_vp_target_mode=actor_vp_target_mode,
        )
        if training_dataset.engine_revisions != validation_dataset.engine_revisions:
            raise ValueError(
                "training and validation engine revisions differ: "
                f"{sorted(training_dataset.engine_revisions)} != "
                f"{sorted(validation_dataset.engine_revisions)}"
            )
        repeated_seeds = training_dataset.game_seeds & validation_dataset.game_seeds
        if repeated_seeds:
            preview = ", ".join(str(seed) for seed in sorted(repeated_seeds)[:8])
            raise ValueError(
                "training and validation shards reuse game seeds: " + preview
            )
    except Exception:
        training_dataset.close()
        if "validation_dataset" in locals():
            validation_dataset.close()
        raise
    return training_dataset, validation_dataset


def _configure_source_sampling(
    dataset: SelfPlayDataset,
    *,
    teacher_paths: tuple[Path, ...],
    replay_paths: tuple[Path, ...],
    human_paths: tuple[Path, ...],
    replay_fraction: float,
    human_fraction: float,
) -> None:
    """Set replacement-sampling quotas for the expert-iteration data pools."""
    for name, value in (
        ("replay_fraction", replay_fraction),
        ("human_fraction", human_fraction),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
        ):
            raise ValueError(
                f"{name.replace('_', '-')} must be finite and between 0 and 1"
            )

    effective_replay_fraction = float(replay_fraction) if replay_paths else 0.0
    effective_human_fraction = float(human_fraction) if human_paths else 0.0
    reserved_fraction = effective_replay_fraction + effective_human_fraction
    if not replay_paths and not human_paths:
        return
    if reserved_fraction >= 1.0:
        raise ValueError(
            "replay-fraction plus human-fraction must leave a positive teacher fraction"
        )
    if replay_paths and effective_replay_fraction <= 0.0:
        raise ValueError(
            "replay-fraction must be positive when replay shards are supplied"
        )
    if human_paths and effective_human_fraction <= 0.0:
        raise ValueError(
            "human-fraction must be positive when human shards are supplied"
        )

    position_counts = dataset.shard_position_counts
    groups = (
        ("teacher", teacher_paths, 1.0 - reserved_fraction),
        ("replay", replay_paths, effective_replay_fraction),
        ("human", human_paths, effective_human_fraction),
    )
    weights: dict[Path, float] = {}
    metadata_groups: list[dict[str, object]] = []
    for name, paths, fraction in groups:
        if not paths:
            continue
        positions = sum(position_counts.get(path, 0) for path in paths)
        if positions <= 0:
            raise ValueError(f"{name} source shards contain no positions")
        per_position_weight = fraction / positions
        for path in paths:
            if position_counts.get(path, 0) <= 0:
                raise ValueError(f"{name} shard contains no positions: {path}")
            weights[path] = per_position_weight
        metadata_groups.append(
            {
                "name": name,
                "paths": [str(path) for path in paths],
                "positions": positions,
                "fraction": fraction,
            }
        )

    dataset.configure_sampling_weights(
        weights,
        metadata={
            "sampler": "weighted_random_replacement",
            "epoch_samples": len(dataset),
            "groups": metadata_groups,
        },
    )


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
    for name in ("replay_fraction", "human_fraction"):
        value = float(getattr(args, name, 0.0))
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(
                f"{name.replace('_', '-')} must be finite and between 0 and 1"
            )
    replay_paths = tuple(getattr(args, "replay_shards", ()) or ())
    human_paths = tuple(getattr(args, "human_shards", ()) or ())
    if replay_paths and float(getattr(args, "replay_fraction", 0.0)) <= 0.0:
        raise ValueError(
            "replay-fraction must be positive when replay shards are supplied"
        )
    if human_paths and float(getattr(args, "human_fraction", 0.0)) <= 0.0:
        raise ValueError(
            "human-fraction must be positive when human shards are supplied"
        )
    if replay_paths or human_paths:
        replay_fraction = (
            float(getattr(args, "replay_fraction", 0.0)) if replay_paths else 0.0
        )
        human_fraction = (
            float(getattr(args, "human_fraction", 0.0)) if human_paths else 0.0
        )
        if replay_fraction + human_fraction >= 1.0:
            raise ValueError(
                "replay-fraction plus human-fraction must leave a positive teacher fraction"
            )
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early-stopping-min-delta must be non-negative")
    if args.early_stopping_patience > 0 and not args.validation_shards:
        raise ValueError("early stopping requires independent validation shards")
    if (
        args.early_stopping_patience == 0
        and args.early_stopping_min_delta != 0.0
    ):
        raise ValueError(
            "early-stopping-min-delta requires positive early-stopping-patience"
        )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if (
        not math.isfinite(args.actor_vp_railroad_scale)
        or args.actor_vp_railroad_scale <= 0.0
    ):
        raise ValueError("actor-vp-railroad-scale must be finite and positive")
    if (
        not math.isfinite(args.search_policy_target_mix)
        or not 0.0 <= args.search_policy_target_mix <= 1.0
    ):
        raise ValueError("search-policy-target-mix must be between 0 and 1")
    if (
        args.policy_target_field == SEARCH_POLICY_TARGET_FIELD
        and args.search_policy_target_mix != 0.0
    ):
        raise ValueError(
            "search-policy-target-mix requires --policy-target-field policy_target"
        )
    validate_policy_target_exponent(args.policy_target_exponent)
    if (
        not math.isfinite(args.final_vp_quality_weight)
        or not 0.0 <= args.final_vp_quality_weight <= 2.0
    ):
        raise ValueError("final-vp-quality-weight must be finite and between 0 and 2")
    if (
        args.final_vp_quality_weight > 0.0
        and args.actor_vp_target_mode != ACTOR_VP_TARGET_ABSOLUTE
    ):
        raise ValueError(
            "final-vp-quality-weight requires --actor-vp-target-mode absolute_final_vp"
        )
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative")
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient-clip must be positive")
    if args.actor_vp_head_only and args.actor_vp_weight <= 0.0:
        raise ValueError("actor-vp-head-only requires a positive actor-vp-weight")
    if args.policy_adapter_dim < 0:
        raise ValueError("policy-adapter-dim must be non-negative")
    if args.policy_adapter_dim > 0 and not args.resume:
        raise ValueError("policy-adapter-dim requires --resume")
    if args.policy_adapter_only and not args.resume:
        raise ValueError("policy-adapter-only requires --resume")
    if args.value_tower_only and not args.resume:
        raise ValueError("value-tower-only requires --resume")
    exclusive_modes = sum(
        bool(value)
        for value in (
            args.actor_vp_head_only,
            args.policy_adapter_only,
            args.value_tower_only,
        )
    )
    if exclusive_modes > 1:
        raise ValueError(
            "actor-vp-head-only, policy-adapter-only, and value-tower-only "
            "are mutually exclusive"
        )
    for name in ("shared_win_weight", "vp_margin_weight", "actor_vp_weight"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if args.value_tower_only and (
        args.shared_win_weight + args.vp_margin_weight + args.actor_vp_weight
        <= 0.0
    ):
        raise ValueError("value-tower-only requires a positive value loss weight")
    if args.validation_fraction != 0.0:
        raise ValueError(
            "validation-fraction leaks positions from the same game; "
            "use independent --validation-shards"
        )
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.max_steps < 0:
        raise ValueError("max-steps must be non-negative")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")


if __name__ == "__main__":
    main()
