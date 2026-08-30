from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluate import (
    MatchResult,
    _build_evaluation_policy,
    _splitmix64,
    play_match,
    summarize_gate,
)
from .reference_heuristic import (
    REFERENCE_COMMIT,
    REFERENCE_POLICY_VERSION,
    REFERENCE_SOURCE,
    ReferenceHeuristicPolicy,
)
from .schema import FeatureSchema


class _TrackedPolicy:
    def __init__(self, delegate: Any, label: str) -> None:
        self.delegate = delegate
        self.label = label
        self.counts: Counter[str] = Counter()

    def select_action(self, observation: dict, state_record: dict, legal_record: dict) -> int:
        index = self.delegate.select_action(observation, state_record, legal_record)
        self._record(legal_record, index)
        return index

    def select_action_with_search(
        self,
        game: Any,
        observation: dict,
        state_record: dict,
        legal_record: dict,
        num_players: int,
        simulations: int,
        exploration_constant: float,
        search_determinizations: int,
        inference_batch_size: int,
        search_seed: int,
    ) -> int:
        index = self.delegate.select_action_with_search(
            game,
            observation,
            state_record,
            legal_record,
            num_players,
            simulations,
            exploration_constant,
            search_determinizations,
            inference_batch_size,
            search_seed,
        )
        self._record(legal_record, index)
        return index

    def _record(self, legal_record: dict, index: int) -> None:
        actions = legal_record.get("actions", [])
        if not isinstance(actions, list) or not 0 <= index < len(actions):
            raise RuntimeError("policy returned an out-of-range action index")
        action = actions[index]
        root = str(action.get("key", "")).split("|", 1)[0]
        self.counts[root] += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the open-source Brass heuristic with the deployed champion."
    )
    champion_source = parser.add_mutually_exclusive_group(required=True)
    champion_source.add_argument("--champion")
    champion_source.add_argument("--champion-inference-url")
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026082801)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--search-simulations", type=int, default=64)
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument("--search-determinizations", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--minimum-games", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare_reference(args)
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite report {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def compare_reference(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    if args.minimum_games <= 1:
        raise ValueError("minimum-games must be at least two")
    if args.search_simulations < 0:
        raise ValueError("search-simulations must be non-negative")
    try:
        import fast_brass
    except ImportError as error:
        raise RuntimeError("fast_brass Python extension is required") from error

    probe = fast_brass.BrassRLGame(args.players, args.seed)
    raw_schema = probe.get_training_feature_schema()
    schema = FeatureSchema.from_schema_dict(raw_schema)
    champion = _build_evaluation_policy(
        checkpoint=args.champion,
        inference_url=args.champion_inference_url,
        raw_schema=raw_schema,
        schema=schema,
        device=args.device,
        timeout_seconds=args.inference_timeout_seconds,
    )
    reference = ReferenceHeuristicPolicy()
    tracked_champion = _TrackedPolicy(champion, "champion")
    tracked_reference = _TrackedPolicy(reference, "reference")

    results: list[MatchResult] = []
    for round_index in range(args.rounds):
        game_seed = _splitmix64(args.seed + round_index)
        for reference_seat in range(args.players):
            policies = [tracked_champion] * args.players
            policies[reference_seat] = tracked_reference
            result = play_match(
                fast_brass,
                policies,
                reference_seat,
                round_index,
                game_seed,
                args.max_actions,
                args.search_simulations,
                args.exploration,
                args.search_determinizations,
                args.inference_batch_size,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "event": "reference_comparison_game",
                        "game": len(results),
                        "reference_seat": reference_seat,
                        "reference_score_delta": result.candidate_score_delta,
                        "reference_vp_margin": result.candidate_victory_point_margin,
                        "victory_points": result.victory_points,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    summary = summarize_gate(results, 0.0, args.minimum_games)
    return {
        "reference": REFERENCE_SOURCE,
        "reference_commit": REFERENCE_COMMIT,
        "reference_policy_version": REFERENCE_POLICY_VERSION,
        "reference_random_perturbation": False,
        "reference_denial_bonus": False,
        "champion": champion.source,
        "champion_model_id": champion.model_id,
        "champion_checkpoint_step": champion.checkpoint_step,
        "players": args.players,
        "rounds": args.rounds,
        "base_seed": args.seed,
        "agent_mode": "reference_greedy_vs_champion_puct" if args.search_simulations else "reference_greedy_vs_champion_greedy",
        "search_simulations": args.search_simulations,
        "exploration_constant": args.exploration,
        "search_determinizations": args.search_determinizations if args.search_simulations else None,
        "inference_batch_size": args.inference_batch_size if args.search_simulations else None,
        "summary": asdict(summary),
        "action_counts": {
            "reference": dict(sorted(tracked_reference.counts.items())),
            "champion": dict(sorted(tracked_champion.counts.items())),
        },
        "games": [asdict(result) for result in results],
    }


if __name__ == "__main__":
    main()
