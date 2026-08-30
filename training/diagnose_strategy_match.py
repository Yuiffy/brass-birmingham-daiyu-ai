"""Trace paired strategy-prior matches on a shared checkpoint.

This is an analysis tool only. It records the selected root actions and the
static prior attached to them while both policies use the normal evaluation
search. It never writes a checkpoint or changes the deployed service.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluate import (
    MatchResult,
    _build_evaluation_policy,
    _splitmix64,
    play_match,
)
from .strategy_prior import (
    ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
    RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
    StrategyPrior,
    build_strategy_prior,
)
from .schema import FeatureSchema


@dataclass(frozen=True)
class TraceEvent:
    game_round: int
    v7_seat: int
    actor: int
    position_index: int
    phase: str
    turn_count: int
    actions_remaining: int
    action_key: str
    action_root: str
    action_family: str
    selected_industry: int | None
    selected_second_industry: int | None
    build_location: int | None
    road: int | None
    second_road: int | None
    sell_targets: tuple[int, ...]
    prior_probability: float
    prior_score: float
    prior_rank: int
    prior_top_key: str
    prior_top_score: float


class TracedPolicy:
    def __init__(self, delegate: Any, version: str, game_round: int, v7_seat: int) -> None:
        self.delegate = delegate
        self.version = version
        self.game_round = game_round
        self.v7_seat = v7_seat
        self.events: list[TraceEvent] = []

    def select_action(
        self, observation: dict[str, Any], state_record: dict[str, Any], legal_record: dict[str, Any]
    ) -> int:
        prior = build_strategy_prior(observation, legal_record, version=self.version)
        index = self.delegate.select_action(observation, state_record, legal_record)
        self._record(observation, legal_record, prior, index)
        return index

    def select_action_with_search(
        self,
        game: Any,
        observation: dict[str, Any],
        state_record: dict[str, Any],
        legal_record: dict[str, Any],
        num_players: int,
        simulations: int,
        exploration_constant: float,
        search_determinizations: int,
        inference_batch_size: int,
        search_seed: int,
    ) -> int:
        prior = build_strategy_prior(observation, legal_record, version=self.version)
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
        self._record(observation, legal_record, prior, index)
        return index

    def _record(
        self,
        observation: dict[str, Any],
        legal_record: dict[str, Any],
        prior: StrategyPrior,
        index: int,
    ) -> None:
        actions = legal_record.get("actions")
        if not isinstance(actions, list) or not 0 <= index < len(actions):
            raise RuntimeError("policy returned an out-of-range action index")
        action = actions[index]
        if not isinstance(action, dict):
            raise RuntimeError("legal action must be an object")
        ranking = sorted(
            range(len(prior.scores)),
            key=lambda item: (prior.scores[item], prior.probabilities[item], -item),
            reverse=True,
        )
        rank_by_index = {item: rank for rank, item in enumerate(ranking, start=1)}
        top_index = ranking[0]
        self.events.append(
            TraceEvent(
                game_round=self.game_round,
                v7_seat=self.v7_seat,
                actor=int(observation.get("decision_player", -1)),
                position_index=len(self.events),
                phase=prior.phase,
                turn_count=int(observation.get("turn_count", 0)),
                actions_remaining=int(observation.get("actions_remaining", 0)),
                action_key=str(action.get("key") or ""),
                action_root=_root(action),
                action_family=prior.action_families[index],
                selected_industry=_optional_int(action.get("selected_industry")),
                selected_second_industry=_optional_int(action.get("selected_second_industry")),
                build_location=_optional_int(action.get("build_location")),
                road=_optional_int(action.get("road")),
                second_road=_optional_int(action.get("second_road")),
                sell_targets=tuple(
                    _optional_int(value)
                    for value in action.get("sell_targets", [])
                    if _optional_int(value) is not None
                ),
                prior_probability=float(prior.probabilities[index]),
                prior_score=float(prior.scores[index]),
                prior_rank=rank_by_index[index],
                prior_top_key=str(actions[top_index].get("key") or ""),
                prior_top_score=float(prior.scores[top_index]),
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026082801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument("--determinizations", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    if args.simulations <= 0:
        raise ValueError("simulations must be positive")

    try:
        import fast_brass
    except ImportError as error:
        raise RuntimeError("fast_brass Python extension is required") from error

    probe = fast_brass.BrassRLGame(2, args.seed)
    raw_schema = probe.get_training_feature_schema()
    schema = FeatureSchema.from_schema_dict(raw_schema)
    v6_delegate = _build_evaluation_policy(
        checkpoint=args.checkpoint,
        inference_url=None,
        raw_schema=raw_schema,
        schema=schema,
        device=args.device,
        timeout_seconds=120.0,
        strategy_prior_strength=0.7,
        strategy_prior_version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
    )
    v7_delegate = _build_evaluation_policy(
        checkpoint=args.checkpoint,
        inference_url=None,
        raw_schema=raw_schema,
        schema=schema,
        device=args.device,
        timeout_seconds=120.0,
        strategy_prior_strength=0.7,
        strategy_prior_version=ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
    )

    results: list[MatchResult] = []
    games: list[dict[str, Any]] = []
    for game_round in range(args.rounds):
        game_seed = _splitmix64(args.seed + game_round)
        for v7_seat in range(2):
            v6 = TracedPolicy(v6_delegate, RESOURCE_AWARE_STRATEGY_PRIOR_VERSION, game_round, v7_seat)
            v7 = TracedPolicy(v7_delegate, ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION, game_round, v7_seat)
            policies = [v6, v6]
            policies[v7_seat] = v7
            result = play_match(
                fast_brass,
                policies,
                v7_seat,
                game_round,
                game_seed,
                args.max_actions,
                args.simulations,
                args.exploration,
                args.determinizations,
                args.inference_batch_size,
            )
            results.append(result)
            games.append(
                {
                    "result": asdict(result),
                    "v7_events": [asdict(event) for event in v7.events],
                    "v6_events": [asdict(event) for event in v6.events],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "strategy_diagnostic_game",
                        "round": game_round,
                        "v7_seat": v7_seat,
                        "victory_points": result.victory_points,
                        "score_delta": result.candidate_score_delta,
                        "vp_margin": result.candidate_victory_point_margin,
                        "v7_positions": len(v7.events),
                        "v6_positions": len(v6.events),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_id": v7_delegate.model_id,
        "checkpoint_step": v7_delegate.checkpoint_step,
        "base_seed": args.seed,
        "rounds": args.rounds,
        "simulations": args.simulations,
        "exploration": args.exploration,
        "determinizations": args.determinizations,
        "v6_version": RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        "v7_version": ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
        "games": games,
    }


def _root(action: dict[str, Any]) -> str:
    return str(action.get("key") or "").split("|", 1)[0]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = build_parser().parse_args()
    report = diagnose(args)
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite report {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
