from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_model_checkpoint
from .data import SelfPlayDataset, collate_positions
from .model import segmented_policy_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect model Top-N policy probabilities for one shard position."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    dataset = SelfPlayDataset([args.shard])
    if args.position < 0 or args.position >= len(dataset):
        raise IndexError(f"position {args.position} outside dataset of {len(dataset)}")
    assert dataset.schema is not None
    device = torch.device(args.device)
    model, payload = load_model_checkpoint(
        args.checkpoint,
        expected_schema=dataset.schema,
        map_location=device,
    )
    model.to(device).eval()
    example = dataset[args.position]
    batch = collate_positions([example]).to(device)
    with torch.no_grad():
        output = model(batch)
        probabilities = segmented_policy_probabilities(
            output.policy_logits, batch.action_counts
        )
        shared_win = torch.sigmoid(output.shared_win_logits[0]).item()
        vp_margin = (
            output.victory_point_margin_normalized[0].item()
            * model.config.vp_margin_scale
        )
    ranking = torch.argsort(probabilities, descending=True).tolist()
    recommendations = [
        {
            "rank": rank + 1,
            "action_index": action_index,
            "action_key": example.action_keys[action_index],
            "policy_probability": probabilities[action_index].item(),
            "search_policy_target": example.policy_target[action_index].item(),
        }
        for rank, action_index in enumerate(ranking[: args.top_k])
    ]
    print(
        json.dumps(
            {
                "checkpoint_step": payload["metadata"]
                .get("training", {})
                .get("global_step", 0),
                "game_index": example.game_index,
                "position_index": example.position_index,
                "predicted_shared_win": shared_win,
                "predicted_victory_point_margin": vp_margin,
                "target_shared_win": example.shared_win_target.item(),
                "target_victory_point_margin": example.victory_point_margin_target.item(),
                "recommendations": recommendations,
            },
            indent=2,
            sort_keys=True,
        )
    )
    dataset.close()


if __name__ == "__main__":
    main()
