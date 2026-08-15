from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluate import (
    MatchResult,
    _splitmix64,
    build_parser as build_evaluation_parser,
    evaluate_checkpoints,
    summarize_gate,
)


@dataclass(frozen=True)
class EvaluationRoundPlan:
    worker_index: int
    round_offset: int
    rounds: int


@dataclass(frozen=True)
class EvaluationPartitionResult:
    plan: EvaluationRoundPlan
    report: dict[str, Any]


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_parallel(args)
    report_path = None
    if args.output:
        report_path = write_evaluation_report(args.output, report)
    displayed = (
        {
            "candidate": report["candidate"],
            "candidate_model_id": report["candidate_model_id"],
            "candidate_checkpoint_step": report["candidate_checkpoint_step"],
            "champion": report["champion"],
            "champion_model_id": report["champion_model_id"],
            "champion_checkpoint_step": report["champion_checkpoint_step"],
            "rounds": report["rounds"],
            "parallel_workers": report["parallel_workers"],
            "search_simulations": report["search_simulations"],
            "summary": report["summary"],
            "report_path": str(report_path) if report_path is not None else None,
        }
        if args.summary_only
        else report
    )
    print(json.dumps(displayed, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = build_evaluation_parser()
    parser.description = (
        "Run seat-rotated candidate-vs-champion evaluation groups in "
        "independent local processes."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def evaluate_parallel(args: argparse.Namespace) -> dict[str, Any]:
    plans = build_round_plans(args.rounds, args.workers)
    if args.workers == 1:
        report = evaluate_checkpoints(args)
        report["parallel_workers"] = 1
        return report

    context = multiprocessing.get_context("spawn")
    results: list[EvaluationPartitionResult] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_evaluate_partition, args, plan): plan for plan in plans
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "event": "parallel_evaluation_partition",
                            "worker": result.plan.worker_index,
                            "round_offset": result.plan.round_offset,
                            "rounds": result.plan.rounds,
                            "games": len(result.report["games"]),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return combine_evaluation_reports(args, tuple(results))


def build_round_plans(
    rounds: int,
    workers: int,
) -> tuple[EvaluationRoundPlan, ...]:
    if workers <= 0 or workers > 64:
        raise ValueError("workers must be between 1 and 64")
    if rounds < workers:
        raise ValueError("rounds must be greater than or equal to workers")
    rounds_per_worker, extra_rounds = divmod(rounds, workers)
    next_round = 0
    plans: list[EvaluationRoundPlan] = []
    for worker_index in range(workers):
        worker_rounds = rounds_per_worker + (
            1 if worker_index < extra_rounds else 0
        )
        plans.append(
            EvaluationRoundPlan(
                worker_index=worker_index,
                round_offset=next_round,
                rounds=worker_rounds,
            )
        )
        next_round += worker_rounds
    if next_round != rounds:
        raise RuntimeError("parallel evaluation plan does not cover every round")
    return tuple(plans)


def combine_evaluation_reports(
    args: argparse.Namespace,
    partition_results: tuple[EvaluationPartitionResult, ...],
) -> dict[str, Any]:
    if not partition_results:
        raise ValueError("parallel evaluation returned no partition reports")
    ordered = tuple(
        sorted(partition_results, key=lambda result: result.plan.round_offset)
    )
    reference = ordered[0].report
    invariant_keys = (
        "candidate",
        "champion",
        "candidate_source_kind",
        "champion_source_kind",
        "candidate_model_id",
        "champion_model_id",
        "candidate_checkpoint_step",
        "champion_checkpoint_step",
        "players",
        "feature_version",
        "agent_mode",
        "search_simulations",
        "exploration_constant",
        "search_determinizations",
        "inference_batch_size",
        "gate_confidence_unit",
    )
    matches: list[MatchResult] = []
    seen_games: set[tuple[int, int]] = set()
    for result in ordered:
        report = result.report
        for key in invariant_keys:
            if report.get(key) != reference.get(key):
                raise RuntimeError(
                    f"parallel evaluation partition changed {key}: "
                    f"{report.get(key)!r} != {reference.get(key)!r}"
                )
        expected_seed = args.seed + result.plan.round_offset
        if report.get("base_seed") != expected_seed:
            raise RuntimeError("parallel evaluation partition base seed changed")
        if report.get("rounds") != result.plan.rounds:
            raise RuntimeError("parallel evaluation partition round count changed")
        for game in report.get("games", []):
            local_round = int(game["round_index"])
            if not 0 <= local_round < result.plan.rounds:
                raise RuntimeError("partition returned an out-of-range round index")
            global_round = result.plan.round_offset + local_round
            candidate_seat = int(game["candidate_seat"])
            identity = (global_round, candidate_seat)
            if identity in seen_games:
                raise RuntimeError("parallel evaluation returned a duplicate game")
            seen_games.add(identity)
            game_seed = int(game["game_seed"])
            if game_seed != _splitmix64(args.seed + global_round):
                raise RuntimeError("parallel evaluation game seed changed")
            matches.append(
                MatchResult(
                    round_index=global_round,
                    game_seed=game_seed,
                    candidate_seat=candidate_seat,
                    actions=int(game["actions"]),
                    candidate_shared_win=float(game["candidate_shared_win"]),
                    candidate_score_delta=float(game["candidate_score_delta"]),
                    candidate_victory_point_margin=int(
                        game["candidate_victory_point_margin"]
                    ),
                    candidate_placement=int(game["candidate_placement"]),
                    candidate_is_official_winner=bool(
                        game["candidate_is_official_winner"]
                    ),
                    official_winners=tuple(game["official_winners"]),
                    victory_points=tuple(game["victory_points"]),
                )
            )

    expected_games = args.rounds * args.players
    if len(matches) != expected_games:
        raise RuntimeError(
            f"parallel evaluation returned {len(matches)} games, "
            f"expected {expected_games}"
        )
    matches.sort(key=lambda match: (match.round_index, match.candidate_seat))
    summary = summarize_gate(
        matches,
        args.promotion_margin,
        args.minimum_games,
    )
    combined = {
        key: value
        for key, value in reference.items()
        if key not in {"rounds", "base_seed", "summary", "games"}
    }
    combined.update(
        {
            "rounds": args.rounds,
            "base_seed": args.seed,
            "parallel_workers": args.workers,
            "summary": asdict(summary),
            "games": [asdict(match) for match in matches],
        }
    )
    return combined


def write_evaluation_report(path: Path, report: dict[str, Any]) -> Path:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(
            f"temporary evaluation report already exists: {temporary}"
        )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _evaluate_partition(
    args: argparse.Namespace,
    plan: EvaluationRoundPlan,
) -> EvaluationPartitionResult:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.rounds = plan.rounds
    worker_args.seed = args.seed + plan.round_offset
    report = evaluate_checkpoints(worker_args)
    return EvaluationPartitionResult(plan=plan, report=report)


if __name__ == "__main__":
    main()
