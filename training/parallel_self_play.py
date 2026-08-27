from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .model_self_play import (
    MASK_64,
    ModelSelfPlayConfig,
    ModelSelfPlaySummary,
    detect_engine_revision,
    export_model_self_play,
)
from .strategy_prior import STRATEGY_PRIOR_VERSION, SUPPORTED_STRATEGY_PRIOR_VERSIONS


@dataclass(frozen=True)
class ParallelSelfPlayConfig:
    output_prefix: Path
    checkpoint: Path | None = None
    inference_url: str | None = None
    inference_timeout_seconds: float = 120.0
    games: int = 4
    workers: int = 4
    game_index_offset: int = 0
    num_players: int = 2
    base_seed: int = 20_260_815
    simulations_per_decision: int = 800
    exploration_constant: float = 1.5
    search_determinizations: int = 4
    inference_batch_size: int = 32
    max_game_actions: int = 256
    device: str = "auto"
    engine_revision: str = "unknown"
    strategy_prior_strength: float = 0.0
    strategy_prior_version: str = STRATEGY_PRIOR_VERSION
    group_card_choices: bool = False
    selection_temperature: float = 1.0
    score_utility_weight: float = 0.0

    def validate(self) -> None:
        if self.workers <= 0 or self.workers > 64:
            raise ValueError("workers must be between 1 and 64")
        if self.games < self.workers:
            raise ValueError("games must be greater than or equal to workers")
        if self.output_prefix.name in ("", ".", ".."):
            raise ValueError("output_prefix must include a file-name prefix")
        if self.output_prefix.suffix == ".jsonl":
            raise ValueError("output_prefix must not end in .jsonl")
        _to_model_config(
            self,
            output=Path(f"{self.output_prefix}-000.jsonl"),
            games=1,
            game_index_offset=self.game_index_offset,
        ).validate()
        if self.game_index_offset + self.games - 1 > MASK_64:
            raise ValueError("parallel game index range exceeds unsigned 64-bit range")


@dataclass(frozen=True)
class SelfPlayShardPlan:
    shard_index: int
    output: Path
    game_index_offset: int
    games: int


@dataclass(frozen=True)
class SelfPlayShardResult:
    shard_index: int
    game_index_offset: int
    games: int
    summary: ModelSelfPlaySummary


@dataclass(frozen=True)
class ParallelSelfPlaySummary:
    games_written: int
    positions_written: int
    workers: int
    elapsed_seconds: float
    positions_per_second: float
    model_id: str
    checkpoint_step: int
    shards: tuple[SelfPlayShardResult, ...]


def main() -> None:
    args = build_parser().parse_args()
    config = ParallelSelfPlayConfig(
        output_prefix=Path(args.output_prefix),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        inference_url=args.inference_url,
        inference_timeout_seconds=args.inference_timeout_seconds,
        games=args.games,
        workers=args.workers,
        game_index_offset=args.game_index_offset,
        num_players=args.players,
        base_seed=args.seed,
        simulations_per_decision=args.simulations,
        exploration_constant=args.exploration,
        search_determinizations=args.search_determinizations,
        inference_batch_size=args.inference_batch_size,
        max_game_actions=args.max_game_actions,
        device=args.device,
        engine_revision=args.engine_revision or detect_engine_revision(),
        strategy_prior_strength=args.strategy_prior_strength,
        strategy_prior_version=args.strategy_prior_version,
        group_card_choices=args.group_card_choices,
        selection_temperature=args.selection_temperature,
        score_utility_weight=args.score_utility_weight,
    )
    summary = export_parallel_self_play(config)
    print(json.dumps(asdict(summary), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic self-play shards in independent local processes."
        )
    )
    parser.add_argument("--output-prefix", required=True)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint")
    model_source.add_argument("--inference-url")
    parser.add_argument("--inference-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--game-index-offset", type=int, default=0)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20_260_815)
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument(
        "--search-determinizations",
        "--value-determinizations",
        dest="search_determinizations",
        type=int,
        default=4,
    )
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-game-actions", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--engine-revision")
    parser.add_argument(
        "--strategy-prior-strength",
        type=float,
        default=0.0,
        help="Blend the human-strategy prior into root PUCT (0 disables it)",
    )
    parser.add_argument(
        "--strategy-prior-version",
        choices=SUPPORTED_STRATEGY_PRIOR_VERSIONS,
        default=STRATEGY_PRIOR_VERSION,
    )
    parser.add_argument(
        "--group-card-choices",
        action="store_true",
        help="Normalize search priors across card-equivalent action intents",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=1.0,
        help="Visit-count action-selection temperature (0 selects stable argmax)",
    )
    parser.add_argument(
        "--score-utility-weight",
        type=float,
        default=0.0,
        help="Blend secured-score progress into neural PUCT exploitation",
    )
    return parser


def export_parallel_self_play(
    config: ParallelSelfPlayConfig,
) -> ParallelSelfPlaySummary:
    config.validate()
    plans = build_shard_plans(config)
    _preflight_outputs(plans)

    started_at = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    results: list[SelfPlayShardResult] = []
    with ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_export_shard, config, plan): plan for plan in plans
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "event": "parallel_self_play_shard",
                            "shard": result.shard_index,
                            "games": result.games,
                            "positions": result.summary.positions_written,
                            "output": result.summary.output,
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

    elapsed_seconds = time.perf_counter() - started_at
    ordered = tuple(sorted(results, key=lambda result: result.shard_index))
    if len(ordered) != len(plans):
        raise RuntimeError("parallel self-play did not return every planned shard")
    model_ids = {result.summary.model_id for result in ordered}
    checkpoint_steps = {result.summary.checkpoint_step for result in ordered}
    if len(model_ids) != 1 or len(checkpoint_steps) != 1:
        raise RuntimeError("parallel self-play workers used different model identities")
    games_written = sum(result.summary.games_written for result in ordered)
    positions_written = sum(result.summary.positions_written for result in ordered)
    if games_written != config.games:
        raise RuntimeError(
            f"parallel self-play wrote {games_written} games, expected {config.games}"
        )
    return ParallelSelfPlaySummary(
        games_written=games_written,
        positions_written=positions_written,
        workers=config.workers,
        elapsed_seconds=elapsed_seconds,
        positions_per_second=(positions_written / elapsed_seconds),
        model_id=model_ids.pop(),
        checkpoint_step=checkpoint_steps.pop(),
        shards=ordered,
    )


def build_shard_plans(config: ParallelSelfPlayConfig) -> tuple[SelfPlayShardPlan, ...]:
    config.validate()
    games_per_worker, extra_games = divmod(config.games, config.workers)
    next_game_index = config.game_index_offset
    plans: list[SelfPlayShardPlan] = []
    width = max(3, len(str(config.workers - 1)))
    for shard_index in range(config.workers):
        games = games_per_worker + (1 if shard_index < extra_games else 0)
        plans.append(
            SelfPlayShardPlan(
                shard_index=shard_index,
                output=Path(
                    f"{config.output_prefix}-{shard_index:0{width}d}.jsonl"
                ),
                game_index_offset=next_game_index,
                games=games,
            )
        )
        next_game_index += games
    if next_game_index != config.game_index_offset + config.games:
        raise RuntimeError("parallel shard plan does not cover the requested games")
    return tuple(plans)


def _preflight_outputs(plans: tuple[SelfPlayShardPlan, ...]) -> None:
    conflicts: list[Path] = []
    for plan in plans:
        resolved = plan.output.resolve()
        partial = Path(f"{resolved}.partial")
        if resolved.exists():
            conflicts.append(resolved)
        if partial.exists():
            conflicts.append(partial)
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"parallel self-play output already exists: {joined}")


def _export_shard(
    config: ParallelSelfPlayConfig,
    plan: SelfPlayShardPlan,
) -> SelfPlayShardResult:
    model_config = _to_model_config(
        config,
        output=plan.output,
        games=plan.games,
        game_index_offset=plan.game_index_offset,
    )
    summary = export_model_self_play(
        model_config,
        on_game_complete=lambda game: print(
            json.dumps(
                {
                    "event": "parallel_self_play_game",
                    "shard": plan.shard_index,
                    "game_index": game["game_index"],
                    "positions": game["positions"],
                    "official_winners": game["official_winners"],
                    "victory_points": game["victory_points"],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        ),
    )
    return SelfPlayShardResult(
        shard_index=plan.shard_index,
        game_index_offset=plan.game_index_offset,
        games=plan.games,
        summary=summary,
    )


def _to_model_config(
    config: ParallelSelfPlayConfig,
    *,
    output: Path,
    games: int,
    game_index_offset: int,
) -> ModelSelfPlayConfig:
    return ModelSelfPlayConfig(
        output=output,
        checkpoint=config.checkpoint,
        inference_url=config.inference_url,
        inference_timeout_seconds=config.inference_timeout_seconds,
        games=games,
        game_index_offset=game_index_offset,
        num_players=config.num_players,
        base_seed=config.base_seed,
        simulations_per_decision=config.simulations_per_decision,
        exploration_constant=config.exploration_constant,
        search_determinizations=config.search_determinizations,
        inference_batch_size=config.inference_batch_size,
        max_game_actions=config.max_game_actions,
        device=config.device,
        engine_revision=config.engine_revision,
        strategy_prior_strength=config.strategy_prior_strength,
        strategy_prior_version=config.strategy_prior_version,
        group_card_choices=config.group_card_choices,
        selection_temperature=config.selection_temperature,
        score_utility_weight=config.score_utility_weight,
    )


if __name__ == "__main__":
    main()
