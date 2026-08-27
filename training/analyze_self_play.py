from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = build_parser().parse_args()
    report = analyze_self_play_shards(tuple(Path(path) for path in args.shards))
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write_report(Path(args.output), serialized)
    print(serialized, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize score, action, and lifecycle quality in self-play shards."
    )
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output")
    return parser


def analyze_self_play_shards(shard_paths: tuple[Path, ...]) -> dict[str, Any]:
    if not shard_paths:
        raise ValueError("at least one self-play shard is required")

    metadata_values: dict[str, set[Any]] = {
        "model_id": set(),
        "checkpoint_step": set(),
        "simulations_per_decision": set(),
        "strategy_prior_version": set(),
        "strategy_prior_strength": set(),
        "card_choice_grouping_version": set(),
        "selection_temperature": set(),
        "score_utility_weight": set(),
    }
    all_scores: list[int] = []
    included_scores: list[int] = []
    final_income_levels: list[int] = []
    final_money: list[int] = []
    games = 0
    both_zero_games = 0
    positions = 0
    action_counts: Counter[str] = Counter()
    phase_action_counts: dict[str, Counter[str]] = {}
    build_industries: Counter[str] = Counter()
    loans = 0
    repeat_loans = 0
    network_before_first_build = 0

    for shard_path in shard_paths:
        path = shard_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"self-play shard does not exist: {path}")
        current_game: dict[str, Any] | None = None
        current_position_count = 0
        included_actors: set[int] = set()
        built_actors: set[int] = set()
        previous_action_by_actor: dict[int, str] = {}

        def finish_game() -> None:
            nonlocal current_game, current_position_count
            nonlocal games, both_zero_games
            if current_game is None:
                return
            expected_positions = _integer(
                current_game.get("positions"), "game.positions", minimum=0
            )
            if current_position_count != expected_positions:
                raise ValueError(
                    f"{path}: game {current_game.get('game_index')} declares "
                    f"{expected_positions} positions but contains {current_position_count}"
                )
            scores = _integer_list(current_game.get("victory_points"), "victory_points")
            income_levels = _integer_list(
                current_game.get("income_levels"), "income_levels"
            )
            money = _integer_list(current_game.get("money"), "money")
            if not (len(scores) == len(income_levels) == len(money)):
                raise ValueError(f"{path}: outcome vectors have different lengths")
            if any(actor >= len(scores) for actor in included_actors):
                raise ValueError(f"{path}: position actor is outside the score vector")
            all_scores.extend(scores)
            included_scores.extend(scores[actor] for actor in sorted(included_actors))
            final_income_levels.extend(income_levels)
            final_money.extend(money)
            both_zero_games += int(bool(scores) and all(score == 0 for score in scores))
            games += 1
            current_game = None
            current_position_count = 0

        with path.open("r", encoding="utf-8") as handle:
            found_metadata = False
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                record_type = record.get("record_type")
                if not found_metadata:
                    if record_type != "metadata":
                        raise ValueError(f"{path}:{line_number}: first record must be metadata")
                    for field, values in metadata_values.items():
                        values.add(_hashable_metadata_value(record.get(field)))
                    found_metadata = True
                    continue
                if record_type == "game":
                    finish_game()
                    current_game = record
                    current_position_count = 0
                    included_actors = set()
                    built_actors = set()
                    previous_action_by_actor = {}
                    continue
                if record_type != "position" or current_game is None:
                    raise ValueError(
                        f"{path}:{line_number}: position must follow its game record"
                    )

                actor = _integer(record.get("actor"), "position.actor", minimum=0)
                key = record.get("selected_action_key")
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{path}:{line_number}: selected action key is missing")
                root = key.split("|", 1)[0]
                phase = str(record.get("phase") or "unknown")
                action_counts[root] += 1
                phase_action_counts.setdefault(phase, Counter())[root] += 1
                included_actors.add(actor)
                positions += 1
                current_position_count += 1

                if root == "loan":
                    loans += 1
                    repeat_loans += int(previous_action_by_actor.get(actor) == "loan")
                if root in {"network", "double_network"} and actor not in built_actors:
                    network_before_first_build += 1
                if root == "build":
                    built_actors.add(actor)
                    build_industries[_industry_from_key(key)] += 1
                previous_action_by_actor[actor] = root
        if not found_metadata:
            raise ValueError(f"{path}: shard is empty or missing metadata")
        finish_game()

    return {
        "shards": [str(path.resolve()) for path in shard_paths],
        "games": games,
        "positions": positions,
        "all_player_trajectories": len(all_scores),
        "included_player_trajectories": len(included_scores),
        "scores": {
            "all_players": _score_summary(all_scores),
            "included_actors": _score_summary(included_scores),
            "both_zero_games": both_zero_games,
        },
        "outcomes": {
            "nonpositive_final_income_levels": sum(
                value <= 0 for value in final_income_levels
            ),
            "zero_final_money": sum(value == 0 for value in final_money),
        },
        "actions": {
            "overall": dict(sorted(action_counts.items())),
            "by_phase": {
                phase: dict(sorted(counts.items()))
                for phase, counts in sorted(phase_action_counts.items())
            },
            "build_industries": dict(sorted(build_industries.items())),
        },
        "lifecycle": {
            "loans": loans,
            "repeat_loans": repeat_loans,
            "network_before_first_build": network_before_first_build,
        },
        "metadata": {
            field: sorted(values, key=lambda value: (str(type(value)), str(value)))
            for field, values in metadata_values.items()
        },
    }


def _score_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "zero": 0,
            "at_least_50": 0,
            "at_least_100": 0,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "zero": sum(value == 0 for value in values),
        "at_least_50": sum(value >= 50 for value in values),
        "at_least_100": sum(value >= 100 for value in values),
    }


def _industry_from_key(key: str) -> str:
    for token in key.partition("|")[2].split(","):
        if token.startswith("i") and token[1:].isdigit():
            return token[1:]
    return "unknown"


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_integer(item, name) for item in value]


def _hashable_metadata_value(value: object) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return json.dumps(value, sort_keys=True)


def _write_report(path: Path, serialized: str) -> None:
    output = path.resolve()
    partial = Path(f"{output}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


if __name__ == "__main__":
    main()
