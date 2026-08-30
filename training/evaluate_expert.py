"""Measure a checkpoint's policy agreement with human replay positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .data import SelfPlayDataset
from .inference import CheckpointEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate expert-action NLL and Top-k policy accuracy."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_expert_policy(
        Path(args.checkpoint), tuple(Path(path) for path in args.shards), args.device
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_expert_policy(
    checkpoint: str | Path,
    shard_paths: Iterable[str | Path],
    device: str = "cpu",
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in shard_paths)
    dataset = SelfPlayDataset(paths)
    try:
        assert dataset.schema is not None
        evaluator = CheckpointEvaluator(checkpoint, device)
        evaluator.schema.assert_compatible(dataset.schema, "expert evaluation")
        nll_values: list[float] = []
        weighted_nll_values: list[float] = []
        top1 = 0
        top3 = 0
        weighted_top1 = 0.0
        weighted_top3 = 0.0
        total_weight = 0.0
        root_counts: dict[str, int] = {}
        for index in range(len(dataset)):
            example = dataset[index]
            state = {
                "feature_version": dataset.schema.version,
                "features": [float(value) for value in example.state.tolist()],
            }
            legal = {
                "feature_version": dataset.schema.version,
                "actions": [
                    {
                        "index": action_index,
                        "key": example.action_keys[action_index],
                        "feature_indices": [
                            int(value)
                            for value in example.action_features[action_index].tolist()
                        ],
                    }
                    for action_index in range(len(example.action_features))
                ],
            }
            prediction = evaluator.predict(state, legal, dataset.schema)
            target_index = int(example.policy_target.argmax().item())
            probability = float(prediction.policy_probabilities[target_index])
            if not math.isfinite(probability) or probability <= 0.0:
                raise RuntimeError("expert action probability is not finite and positive")
            nll = -math.log(probability)
            weight = float(example.policy_loss_weight)
            nll_values.append(nll)
            weighted_nll_values.append(nll * weight)
            total_weight += weight
            ranked = sorted(
                range(len(prediction.policy_probabilities)),
                key=prediction.policy_probabilities.__getitem__,
                reverse=True,
            )
            is_top1 = int(ranked[0] == target_index)
            is_top3 = int(target_index in ranked[:3])
            top1 += is_top1
            top3 += is_top3
            weighted_top1 += weight * is_top1
            weighted_top3 += weight * is_top3
            root = example.action_keys[target_index].split("|", 1)[0]
            root_counts[root] = root_counts.get(root, 0) + 1
        count = len(nll_values)
        if count == 0 or total_weight <= 0.0:
            raise ValueError("expert shard contains no weighted positions")
        return {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_step": evaluator.checkpoint_step,
            "model_id": evaluator.model_id,
            "shards": [str(path.resolve()) for path in paths],
            "positions": count,
            "expert_nll": sum(nll_values) / count,
            "weighted_expert_nll": sum(weighted_nll_values) / total_weight,
            "top1_accuracy": top1 / count,
            "top3_accuracy": top3 / count,
            "weighted_top1_accuracy": weighted_top1 / total_weight,
            "weighted_top3_accuracy": weighted_top3 / total_weight,
            "selected_root_actions": dict(sorted(root_counts.items())),
        }
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
