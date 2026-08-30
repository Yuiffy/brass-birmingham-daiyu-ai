"""Generate quality-filtered expert-iteration data locally.

This module is the orchestration layer for the fast training loop:

    human prior/checkpoint -> diverse batched self-play -> quality gates
    -> accepted trajectories -> learner/replay/validation shards

The Rust engine remains the source of truth for rules and search.  The Python
code only schedules independent workers, audits the resulting JSONL records,
and writes derived shards.  It deliberately has no browser or Playwright
dependency.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import itertools
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Sequence

from .demonstration_prior import (
    DEMONSTRATION_PRIOR_VERSION,
    DemonstrationPrior,
    load_demonstration_prior,
)
from .model_self_play import detect_engine_revision
from .parallel_self_play import (
    ParallelSelfPlayConfig,
    ParallelSelfPlaySummary,
    export_parallel_self_play,
)
from .strategy_prior import (
    RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
    SUPPORTED_STRATEGY_PRIOR_VERSIONS,
)


EXPERT_ITERATION_FORMAT = "fast_brass_expert_iteration_manifest"
EXPERT_ITERATION_FORMAT_VERSION = 1
QUALITY_FILTER_VERSION = "trajectory-quality-v1"


@dataclass(frozen=True)
class QualityGate:
    """Acceptance rules for one complete generated game.

    The defaults are intentionally conservative.  A game is useful training
    data only when it is complete, every player has a non-zero result, the
    ending economy is healthy, and at least one top-ranked player meets the
    declared score/margin threshold.
    """

    minimum_actor_vp: int = 80
    minimum_vp_margin: int = 0
    minimum_positions_per_actor: int = 1
    minimum_final_income: int = 1
    minimum_final_money: int = 1
    require_top_actor: bool = True
    reject_any_zero_score: bool = True
    reject_lifecycle_violations: bool = True
    require_complete_outcome: bool = True

    def validate(self) -> None:
        integer_fields = (
            ("minimum_actor_vp", self.minimum_actor_vp),
            ("minimum_vp_margin", self.minimum_vp_margin),
            ("minimum_positions_per_actor", self.minimum_positions_per_actor),
            ("minimum_final_income", self.minimum_final_income),
            ("minimum_final_money", self.minimum_final_money),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.minimum_actor_vp < 0:
            raise ValueError("minimum_actor_vp must be non-negative")
        if self.minimum_positions_per_actor <= 0:
            raise ValueError("minimum_positions_per_actor must be positive")
        if self.minimum_final_income < 0:
            raise ValueError("minimum_final_income must be non-negative")
        if self.minimum_final_money < 0:
            raise ValueError("minimum_final_money must be non-negative")
        for name in (
            "require_top_actor",
            "reject_any_zero_score",
            "reject_lifecycle_violations",
            "require_complete_outcome",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class GenerationRecipe:
    """One self-play policy mixture used for a batch of games."""

    recipe_index: int
    games: int
    strategy_prior_strength: float
    selection_temperature: float
    score_utility_weight: float
    final_vp_utility_weight: float
    demonstration_prior_strength: float = 0.0


@dataclass(frozen=True)
class GameQuality:
    """Auditable quality result for one generated game."""

    game_index: int
    game_seed: int | None
    positions: int
    player_count: int
    scores: tuple[int, ...]
    winners: tuple[int, ...]
    eligible_players: tuple[int, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    lifecycle: dict[str, int]
    final_income: tuple[int, ...]
    final_money: tuple[int, ...]

    @property
    def accepted_trajectories(self) -> int:
        return len(self.eligible_players) if self.accepted else 0


@dataclass(frozen=True)
class QualityFilterSummary:
    source: str
    output: str
    games_seen: int
    games_accepted: int
    games_rejected: int
    source_positions: int
    positions_written: int
    eligible_player_trajectories: int
    accepted_game_seeds: tuple[int, ...]
    rejection_reasons: dict[str, int]
    score_summary: dict[str, int | float | None]


@dataclass(frozen=True)
class ExpertDataProductionConfig:
    """Configuration for one train/validation data-production run."""

    output_dir: Path
    checkpoint: Path | None = None
    inference_url: str | None = None
    inference_timeout_seconds: float = 120.0
    train_games: int = 64
    validation_games: int = 16
    workers: int = 4
    num_players: int = 2
    base_seed: int = 20_260_830
    validation_base_seed: int = 30_260_830
    simulations_per_decision: int = 256
    exploration_constant: float = 1.5
    search_determinizations: int = 4
    inference_batch_size: int = 64
    max_game_actions: int = 256
    device: str = "auto"
    engine_revision: str = "unknown"
    strategy_prior_version: str = RESOURCE_AWARE_STRATEGY_PRIOR_VERSION
    strategy_prior_strengths: tuple[float, ...] = (0.65, 0.85)
    selection_temperatures: tuple[float, ...] = (0.35, 0.70)
    score_utility_weights: tuple[float, ...] = (0.0,)
    final_vp_utility_weights: tuple[float, ...] = (0.0,)
    demonstration_shards: tuple[Path, ...] = ()
    demonstration_prior_version: str = DEMONSTRATION_PRIOR_VERSION
    demonstration_prior_strengths: tuple[float, ...] = (0.0,)
    demonstration_prior_players: tuple[int, ...] | None = None
    group_card_choices: bool = True
    quality_gate: QualityGate = QualityGate()
    minimum_accepted_games: int = 1
    minimum_accepted_positions: int = 1
    maximum_recipes: int = 16

    def validate(self) -> None:
        if (self.checkpoint is None) == (self.inference_url is None):
            raise ValueError("exactly one of checkpoint or inference_url is required")
        if self.inference_url is not None and not self.inference_url.strip():
            raise ValueError("inference_url must not be empty")
        if not self.output_dir.name or self.output_dir.name in {".", ".."}:
            raise ValueError("output_dir must name a directory")
        for name, value in (
            ("train_games", self.train_games),
            ("validation_games", self.validation_games),
            ("workers", self.workers),
            ("num_players", self.num_players),
            ("simulations_per_decision", self.simulations_per_decision),
            ("search_determinizations", self.search_determinizations),
            ("inference_batch_size", self.inference_batch_size),
            ("max_game_actions", self.max_game_actions),
            ("minimum_accepted_games", self.minimum_accepted_games),
            ("minimum_accepted_positions", self.minimum_accepted_positions),
            ("maximum_recipes", self.maximum_recipes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.train_games <= 0:
            raise ValueError("train_games must be positive")
        if self.validation_games < 0:
            raise ValueError("validation_games must be non-negative")
        if not 1 <= self.workers <= 64:
            raise ValueError("workers must be between 1 and 64")
        if self.num_players not in (2, 3, 4):
            raise ValueError("num_players must be between 2 and 4")
        if not 1 <= self.simulations_per_decision <= 1_000_000:
            raise ValueError("simulations_per_decision must be between 1 and 1000000")
        if not math.isfinite(self.exploration_constant) or self.exploration_constant < 0:
            raise ValueError("exploration_constant must be finite and non-negative")
        if not 1 <= self.search_determinizations <= 64:
            raise ValueError("search_determinizations must be between 1 and 64")
        if not 1 <= self.inference_batch_size <= 256:
            raise ValueError("inference_batch_size must be between 1 and 256")
        if self.max_game_actions <= 0:
            raise ValueError("max_game_actions must be positive")
        if self.minimum_accepted_games <= 0:
            raise ValueError("minimum_accepted_games must be positive")
        if self.minimum_accepted_positions <= 0:
            raise ValueError("minimum_accepted_positions must be positive")
        if self.maximum_recipes <= 0:
            raise ValueError("maximum_recipes must be positive")
        if not isinstance(self.engine_revision, str) or not self.engine_revision.strip():
            raise ValueError("engine_revision must not be empty")
        if (
            isinstance(self.inference_timeout_seconds, bool)
            or not isinstance(self.inference_timeout_seconds, (int, float))
            or not math.isfinite(float(self.inference_timeout_seconds))
            or not 0.0 < float(self.inference_timeout_seconds) <= 3600.0
        ):
            raise ValueError(
                "inference_timeout_seconds must be between 0 and 3600"
            )
        if self.strategy_prior_version not in SUPPORTED_STRATEGY_PRIOR_VERSIONS:
            raise ValueError(
                f"unsupported strategy prior version {self.strategy_prior_version!r}"
            )
        if (
            not isinstance(self.demonstration_prior_version, str)
            or not self.demonstration_prior_version.strip()
        ):
            raise ValueError("demonstration_prior_version must not be empty")
        if not isinstance(self.demonstration_shards, (tuple, list)):
            raise ValueError("demonstration_shards must be a sequence of paths")
        for index, path in enumerate(self.demonstration_shards):
            if not isinstance(path, (str, Path)) or not str(path).strip():
                raise ValueError(f"demonstration_shards[{index}] must be a path")
        _validate_seed(self.base_seed, "base_seed")
        _validate_seed(self.validation_base_seed, "validation_base_seed")
        if self.base_seed == self.validation_base_seed and self.validation_games:
            raise ValueError("train and validation base seeds must differ")
        _validate_float_sequence(
            self.strategy_prior_strengths,
            "strategy_prior_strengths",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_float_sequence(
            self.selection_temperatures,
            "selection_temperatures",
            minimum=0.0,
            maximum=10.0,
        )
        _validate_float_sequence(
            self.score_utility_weights,
            "score_utility_weights",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_float_sequence(
            self.final_vp_utility_weights,
            "final_vp_utility_weights",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_float_sequence(
            self.demonstration_prior_strengths,
            "demonstration_prior_strengths",
            minimum=0.0,
            maximum=1.0,
        )
        if not self.demonstration_shards and any(
            float(value) > 0.0 for value in self.demonstration_prior_strengths
        ):
            raise ValueError(
                "demonstration_prior_strengths require demonstration_shards"
            )
        if self.demonstration_prior_players is not None:
            if not self.demonstration_prior_players:
                raise ValueError("demonstration_prior_players must not be empty")
            if len(set(self.demonstration_prior_players)) != len(
                self.demonstration_prior_players
            ):
                raise ValueError("demonstration_prior_players must be unique")
            for player in self.demonstration_prior_players:
                if (
                    isinstance(player, bool)
                    or not isinstance(player, int)
                    or not 0 <= player < self.num_players
                ):
                    raise ValueError(
                        "demonstration_prior_players must contain valid seat indices"
                    )
        self.quality_gate.validate()


@dataclass(frozen=True)
class ExpertDataProductionSummary:
    manifest: str
    train_shards: tuple[str, ...]
    validation_shards: tuple[str, ...]
    raw_train_shards: tuple[str, ...]
    raw_validation_shards: tuple[str, ...]
    train_games_requested: int
    validation_games_requested: int
    train_games_accepted: int
    validation_games_accepted: int
    train_positions_written: int
    validation_positions_written: int
    model_id: str
    checkpoint_step: int


@dataclass(frozen=True)
class _RawPosition:
    record: dict[str, Any]
    raw_line: bytes
    line_number: int


@dataclass(frozen=True)
class _RawGame:
    record: dict[str, Any]
    positions: tuple[_RawPosition, ...]
    line_number: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diverse local Rust/PyTorch self-play, reject unhealthy or "
            "low-quality trajectories, and write expert-iteration shards."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--inference-url")
    parser.add_argument("--inference-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-games", type=int, default=64)
    parser.add_argument("--validation-games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--seed", type=int, default=20_260_830)
    parser.add_argument("--validation-seed", type=int, default=30_260_830)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument("--search-determinizations", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--max-game-actions", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--engine-revision")
    parser.add_argument(
        "--strategy-prior-version",
        choices=SUPPORTED_STRATEGY_PRIOR_VERSIONS,
        default=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
    )
    parser.add_argument(
        "--strategy-prior-strengths",
        nargs="+",
        type=float,
        default=(0.65, 0.85),
        help="One or more prior strengths; recipes are generated from their product.",
    )
    parser.add_argument(
        "--selection-temperatures",
        nargs="+",
        type=float,
        default=(0.35, 0.70),
        help="One or more root selection temperatures for trajectory diversity.",
    )
    parser.add_argument(
        "--score-utility-weights",
        nargs="+",
        type=float,
        default=(0.0,),
    )
    parser.add_argument(
        "--final-vp-utility-weights",
        nargs="+",
        type=float,
        default=(0.0,),
    )
    parser.add_argument(
        "--demonstration-shards",
        nargs="+",
        default=(),
        help=(
            "Audited human replay JSONL shards used to guide generated self-play "
            "by context and action intent"
        ),
    )
    parser.add_argument(
        "--demonstration-prior-version",
        default=DEMONSTRATION_PRIOR_VERSION,
        help="Version tag recorded for the human demonstration prior",
    )
    parser.add_argument(
        "--demonstration-prior-strengths",
        nargs="+",
        type=float,
        default=(0.0,),
        help=(
            "One or more human-prior strengths; combined with the other recipe "
            "dimensions"
        ),
    )
    parser.add_argument(
        "--demonstration-prior-players",
        "--demonstration-prior-seats",
        dest="demonstration_prior_players",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Only guide these seat indices with the human prior; omit to guide "
            "all seats"
        ),
    )
    parser.add_argument("--no-group-card-choices", dest="group_card_choices", action="store_false")
    parser.set_defaults(group_card_choices=True)
    parser.add_argument("--minimum-actor-vp", type=int, default=80)
    parser.add_argument("--minimum-vp-margin", type=int, default=0)
    parser.add_argument("--minimum-positions-per-actor", type=int, default=1)
    parser.add_argument("--minimum-final-income", type=int, default=1)
    parser.add_argument("--minimum-final-money", type=int, default=1)
    parser.add_argument("--include-non-top-actors", action="store_true")
    parser.add_argument("--allow-zero-score", action="store_true")
    parser.add_argument("--allow-lifecycle-violations", action="store_true")
    parser.add_argument("--allow-incomplete-outcome", action="store_true")
    parser.add_argument("--minimum-accepted-games", type=int, default=1)
    parser.add_argument("--minimum-accepted-positions", type=int, default=1)
    parser.add_argument("--maximum-recipes", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExpertDataProductionConfig(
        output_dir=Path(args.output_dir),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        inference_url=args.inference_url,
        inference_timeout_seconds=args.inference_timeout_seconds,
        train_games=args.train_games,
        validation_games=args.validation_games,
        workers=args.workers,
        num_players=args.players,
        base_seed=args.seed,
        validation_base_seed=args.validation_seed,
        simulations_per_decision=args.simulations,
        exploration_constant=args.exploration,
        search_determinizations=args.search_determinizations,
        inference_batch_size=args.inference_batch_size,
        max_game_actions=args.max_game_actions,
        device=args.device,
        engine_revision=args.engine_revision or detect_engine_revision(),
        strategy_prior_version=args.strategy_prior_version,
        strategy_prior_strengths=tuple(args.strategy_prior_strengths),
        selection_temperatures=tuple(args.selection_temperatures),
        score_utility_weights=tuple(args.score_utility_weights),
        final_vp_utility_weights=tuple(args.final_vp_utility_weights),
        demonstration_shards=tuple(Path(path) for path in args.demonstration_shards),
        demonstration_prior_version=args.demonstration_prior_version,
        demonstration_prior_strengths=tuple(args.demonstration_prior_strengths),
        demonstration_prior_players=(
            tuple(args.demonstration_prior_players)
            if args.demonstration_prior_players is not None
            else None
        ),
        group_card_choices=args.group_card_choices,
        quality_gate=QualityGate(
            minimum_actor_vp=args.minimum_actor_vp,
            minimum_vp_margin=args.minimum_vp_margin,
            minimum_positions_per_actor=args.minimum_positions_per_actor,
            minimum_final_income=args.minimum_final_income,
            minimum_final_money=args.minimum_final_money,
            require_top_actor=not args.include_non_top_actors,
            reject_any_zero_score=not args.allow_zero_score,
            reject_lifecycle_violations=not args.allow_lifecycle_violations,
            require_complete_outcome=not args.allow_incomplete_outcome,
        ),
        minimum_accepted_games=args.minimum_accepted_games,
        minimum_accepted_positions=args.minimum_accepted_positions,
        maximum_recipes=args.maximum_recipes,
    )
    summary = produce_expert_data(config)
    print(json.dumps(asdict(summary), sort_keys=True))


def build_generation_recipes(
    *,
    games: int,
    strategy_prior_strengths: Sequence[float],
    selection_temperatures: Sequence[float],
    score_utility_weights: Sequence[float] = (0.0,),
    final_vp_utility_weights: Sequence[float] = (0.0,),
    demonstration_prior_strengths: Sequence[float] = (0.0,),
    maximum_recipes: int = 16,
) -> tuple[GenerationRecipe, ...]:
    """Build a deterministic Cartesian recipe plan and allocate game counts."""

    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("games must be positive")
    if maximum_recipes <= 0:
        raise ValueError("maximum_recipes must be positive")
    values = (
        tuple(strategy_prior_strengths),
        tuple(selection_temperatures),
        tuple(score_utility_weights),
        tuple(final_vp_utility_weights),
        tuple(demonstration_prior_strengths),
    )
    if any(not sequence for sequence in values):
        raise ValueError("every recipe dimension must contain at least one value")
    _validate_float_sequence(
        values[0], "strategy_prior_strengths", minimum=0.0, maximum=1.0
    )
    _validate_float_sequence(
        values[1], "selection_temperatures", minimum=0.0, maximum=10.0
    )
    _validate_float_sequence(
        values[2], "score_utility_weights", minimum=0.0, maximum=1.0
    )
    _validate_float_sequence(
        values[3], "final_vp_utility_weights", minimum=0.0, maximum=1.0
    )
    _validate_float_sequence(
        values[4],
        "demonstration_prior_strengths",
        minimum=0.0,
        maximum=1.0,
    )
    combinations = tuple(itertools.product(*values))
    if len(combinations) > maximum_recipes:
        raise ValueError(
            f"recipe product has {len(combinations)} combinations, "
            f"exceeding maximum_recipes={maximum_recipes}"
        )
    if games < len(combinations):
        raise ValueError(
            f"games={games} is too small for {len(combinations)} recipes; "
            "increase games or reduce recipe dimensions"
        )
    counts = _balanced_counts(games, len(combinations))
    recipes = []
    for index, (combination, count) in enumerate(zip(combinations, counts, strict=True)):
        strength, temperature, score_weight, final_vp_weight, demonstration_strength = combination
        recipes.append(
            GenerationRecipe(
                recipe_index=index,
                games=count,
                strategy_prior_strength=_finite_float(
                    strength, f"strategy_prior_strengths[{index}]"
                ),
                selection_temperature=_finite_float(
                    temperature, f"selection_temperatures[{index}]"
                ),
                score_utility_weight=_finite_float(
                    score_weight, f"score_utility_weights[{index}]"
                ),
                final_vp_utility_weight=_finite_float(
                    final_vp_weight, f"final_vp_utility_weights[{index}]"
                ),
                demonstration_prior_strength=_finite_float(
                    demonstration_strength,
                    f"demonstration_prior_strengths[{index}]",
                ),
            )
        )
    return tuple(recipes)


def produce_expert_data(
    config: ExpertDataProductionConfig,
    *,
    exporter: Callable[[ParallelSelfPlayConfig], ParallelSelfPlaySummary]
    | None = None,
) -> ExpertDataProductionSummary:
    """Run generation, auditing, filtering, and manifest publication.

    ``exporter`` is injectable so unit tests can exercise orchestration without
    loading the Rust extension or a GPU.  Production callers should leave it
    unset.
    """

    config.validate()
    exporter = exporter or export_parallel_self_play
    output_dir = config.output_dir.resolve()
    raw_dir = output_dir / "raw"
    train_filtered_dir = output_dir / "train"
    validation_filtered_dir = output_dir / "validation"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() or Path(f"{manifest_path}.partial").exists():
        raise FileExistsError(f"expert-iteration manifest already exists: {manifest_path}")
    for directory in (raw_dir, train_filtered_dir, validation_filtered_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_recipes = build_generation_recipes(
        games=config.train_games,
        strategy_prior_strengths=config.strategy_prior_strengths,
        selection_temperatures=config.selection_temperatures,
        score_utility_weights=config.score_utility_weights,
        final_vp_utility_weights=config.final_vp_utility_weights,
        demonstration_prior_strengths=config.demonstration_prior_strengths,
        maximum_recipes=config.maximum_recipes,
    )
    validation_recipes = _build_validation_recipes(config, train_recipes)

    demonstration_prior = (
        load_demonstration_prior(
            config.demonstration_shards,
            version=config.demonstration_prior_version,
        )
        if config.demonstration_shards
        else None
    )

    train_result = _produce_split(
        split="train",
        recipes=train_recipes,
        base_seed=config.base_seed,
        config=config,
        raw_dir=raw_dir,
        filtered_dir=train_filtered_dir,
        exporter=exporter,
        demonstration_prior=demonstration_prior,
    )
    validation_result = (
        _produce_split(
            split="validation",
            recipes=validation_recipes,
            base_seed=config.validation_base_seed,
            config=config,
            raw_dir=raw_dir,
            filtered_dir=validation_filtered_dir,
            exporter=exporter,
            demonstration_prior=demonstration_prior,
        )
        if config.validation_games > 0
        else _EmptySplitResult()
    )

    train_accepted_games = sum(item.games_accepted for item in train_result.filters)
    train_positions = sum(item.positions_written for item in train_result.filters)
    validation_accepted_games = sum(
        item.games_accepted for item in validation_result.filters
    )
    validation_positions = sum(
        item.positions_written for item in validation_result.filters
    )
    all_raw_seeds = _collect_game_seeds(
        train_result.raw_shards + validation_result.raw_shards
    )
    train_seeds = _collect_game_seeds(train_result.raw_shards)
    validation_seeds = _collect_game_seeds(validation_result.raw_shards)
    overlap = train_seeds & validation_seeds
    if overlap:
        raise RuntimeError(
            "generated train and validation game seeds overlap: "
            + ", ".join(str(seed) for seed in sorted(overlap)[:8])
        )

    model_ids = {
        summary.model_id
        for summary in train_result.self_play_summaries
        + validation_result.self_play_summaries
    }
    checkpoint_steps = {
        summary.checkpoint_step
        for summary in train_result.self_play_summaries
        + validation_result.self_play_summaries
    }
    if len(model_ids) != 1 or len(checkpoint_steps) != 1:
        raise RuntimeError(
            "expert data recipes used inconsistent model identities or checkpoint steps"
        )
    model_id = next(iter(model_ids))
    checkpoint_step = next(iter(checkpoint_steps))

    manifest = {
        "format": EXPERT_ITERATION_FORMAT,
        "format_version": EXPERT_ITERATION_FORMAT_VERSION,
        "quality_filter_version": QUALITY_FILTER_VERSION,
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "engine_revision": config.engine_revision,
        "model_id": model_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint": (
            str(config.checkpoint.resolve()) if config.checkpoint is not None else None
        ),
        "inference_url": config.inference_url,
        "train": _split_manifest(train_result, config.train_games),
        "validation": _split_manifest(validation_result, config.validation_games),
        "recipes": [asdict(recipe) for recipe in train_recipes],
        "validation_recipes": [asdict(recipe) for recipe in validation_recipes],
        "demonstration_prior": (
            demonstration_prior.metadata() if demonstration_prior is not None else None
        ),
        "quality_gate": asdict(config.quality_gate),
        "seed_partition": {
            "train_base_seed": config.base_seed,
            "validation_base_seed": config.validation_base_seed,
            "train_game_seeds": sorted(train_seeds),
            "validation_game_seeds": sorted(validation_seeds),
            "all_generated_game_seeds": sorted(all_raw_seeds),
            "disjoint": not overlap,
        },
        "training_inputs": {
            "teacher_shards": list(train_result.filtered_shards),
            "validation_shards": list(validation_result.filtered_shards),
            "replay_shards": [],
            "human_shards": [],
            "demonstration_shards": [
                str(Path(path).resolve()) for path in config.demonstration_shards
            ],
            "demonstration_prior_strengths": list(
                config.demonstration_prior_strengths
            ),
            "demonstration_prior_players": (
                list(config.demonstration_prior_players)
                if config.demonstration_prior_players is not None
                else None
            ),
            "suggested_replay_fraction": 0.25,
            "suggested_human_fraction": 0.05,
        },
        "status": {
            "train_games_accepted": train_accepted_games,
            "train_positions_written": train_positions,
            "validation_games_accepted": validation_accepted_games,
            "validation_positions_written": validation_positions,
            "minimum_accepted_games": config.minimum_accepted_games,
            "minimum_accepted_positions": config.minimum_accepted_positions,
            "train_quality_gate_passed": (
                train_accepted_games >= config.minimum_accepted_games
                and train_positions >= config.minimum_accepted_positions
            ),
        },
        "playwright_required": False,
    }
    _write_json_atomically(manifest_path, manifest)

    if train_accepted_games < config.minimum_accepted_games:
        raise RuntimeError(
            f"quality filter accepted {train_accepted_games} train games, "
            f"fewer than minimum {config.minimum_accepted_games}; "
            f"see {manifest_path}"
        )
    if train_positions < config.minimum_accepted_positions:
        raise RuntimeError(
            f"quality filter wrote {train_positions} train positions, "
            f"fewer than minimum {config.minimum_accepted_positions}; "
            f"see {manifest_path}"
        )

    return ExpertDataProductionSummary(
        manifest=str(manifest_path),
        train_shards=tuple(train_result.filtered_shards),
        validation_shards=tuple(validation_result.filtered_shards),
        raw_train_shards=tuple(train_result.raw_shards),
        raw_validation_shards=tuple(validation_result.raw_shards),
        train_games_requested=config.train_games,
        validation_games_requested=config.validation_games,
        train_games_accepted=train_accepted_games,
        validation_games_accepted=validation_accepted_games,
        train_positions_written=train_positions,
        validation_positions_written=validation_positions,
        model_id=model_id,
        checkpoint_step=checkpoint_step,
    )


@dataclass(frozen=True)
class _SplitResult:
    recipes: tuple[GenerationRecipe, ...]
    self_play_summaries: tuple[ParallelSelfPlaySummary, ...]
    raw_shards: tuple[str, ...]
    filtered_shards: tuple[str, ...]
    filters: tuple[QualityFilterSummary, ...]


@dataclass(frozen=True)
class _EmptySplitResult:
    recipes: tuple[GenerationRecipe, ...] = ()
    self_play_summaries: tuple[ParallelSelfPlaySummary, ...] = ()
    raw_shards: tuple[str, ...] = ()
    filtered_shards: tuple[str, ...] = ()
    filters: tuple[QualityFilterSummary, ...] = ()


def _produce_split(
    *,
    split: str,
    recipes: tuple[GenerationRecipe, ...],
    base_seed: int,
    config: ExpertDataProductionConfig,
    raw_dir: Path,
    filtered_dir: Path,
    exporter: Callable[[ParallelSelfPlayConfig], ParallelSelfPlaySummary],
    demonstration_prior: DemonstrationPrior | None,
) -> _SplitResult:
    next_game_index = 0
    self_play_summaries: list[ParallelSelfPlaySummary] = []
    raw_shards: list[str] = []
    filtered_shards: list[str] = []
    filters: list[QualityFilterSummary] = []
    for recipe in recipes:
        if recipe.games <= 0:
            continue
        raw_prefix = raw_dir / f"{split}-recipe-{recipe.recipe_index:03d}"
        worker_count = min(config.workers, recipe.games)
        self_play_config = ParallelSelfPlayConfig(
            output_prefix=raw_prefix,
            checkpoint=config.checkpoint,
            inference_url=config.inference_url,
            inference_timeout_seconds=config.inference_timeout_seconds,
            games=recipe.games,
            workers=worker_count,
            game_index_offset=next_game_index,
            num_players=config.num_players,
            base_seed=base_seed,
            simulations_per_decision=config.simulations_per_decision,
            exploration_constant=config.exploration_constant,
            search_determinizations=config.search_determinizations,
            inference_batch_size=config.inference_batch_size,
            max_game_actions=config.max_game_actions,
            device=config.device,
            engine_revision=config.engine_revision,
            strategy_prior_strength=recipe.strategy_prior_strength,
            strategy_prior_version=config.strategy_prior_version,
            group_card_choices=config.group_card_choices,
            selection_temperature=recipe.selection_temperature,
            score_utility_weight=recipe.score_utility_weight,
            final_vp_utility_weight=recipe.final_vp_utility_weight,
            demonstration_prior=demonstration_prior,
            demonstration_prior_strength=recipe.demonstration_prior_strength,
            demonstration_prior_players=config.demonstration_prior_players,
        )
        summary = exporter(self_play_config)
        if summary.games_written != recipe.games:
            raise RuntimeError(
                f"self-play recipe {recipe.recipe_index} wrote "
                f"{summary.games_written} games, expected {recipe.games}"
            )
        if summary.model_id is None or not str(summary.model_id).strip():
            raise RuntimeError(
                f"self-play recipe {recipe.recipe_index} returned no model identity"
            )
        self_play_summaries.append(summary)
        shard_outputs = tuple(
            str(Path(shard.summary.output).resolve()) for shard in summary.shards
        )
        if not shard_outputs:
            raise RuntimeError(f"self-play recipe {recipe.recipe_index} returned no shards")
        raw_shards.extend(shard_outputs)
        for shard_index, raw_shard in enumerate(shard_outputs):
            filtered_path = (
                filtered_dir
                / f"{split}-recipe-{recipe.recipe_index:03d}-{shard_index:03d}.jsonl"
            ).resolve()
            result = filter_quality_shard(
                Path(raw_shard), filtered_path, config.quality_gate
            )
            filters.append(result)
            # Keep metadata-only outputs on disk for auditability, but do not
            # advertise them as training inputs: the source-mixture sampler
            # requires every listed shard to contribute at least one position.
            if result.positions_written > 0:
                filtered_shards.append(str(filtered_path))
        next_game_index += recipe.games
    expected_games = sum(recipe.games for recipe in recipes)
    if next_game_index != expected_games:
        raise RuntimeError(
            f"{split} recipe plan wrote {next_game_index} game indices, expected {expected_games}"
        )
    return _SplitResult(
        recipes=recipes,
        self_play_summaries=tuple(self_play_summaries),
        raw_shards=tuple(raw_shards),
        filtered_shards=tuple(filtered_shards),
        filters=tuple(filters),
    )


def _build_validation_recipes(
    config: ExpertDataProductionConfig,
    train_recipes: tuple[GenerationRecipe, ...],
) -> tuple[GenerationRecipe, ...]:
    if config.validation_games <= 0:
        return ()
    if config.validation_games >= len(train_recipes):
        counts = _balanced_counts(config.validation_games, len(train_recipes))
        return tuple(
            GenerationRecipe(
                recipe_index=recipe.recipe_index,
                games=count,
                strategy_prior_strength=recipe.strategy_prior_strength,
                selection_temperature=recipe.selection_temperature,
                score_utility_weight=recipe.score_utility_weight,
                final_vp_utility_weight=recipe.final_vp_utility_weight,
                demonstration_prior_strength=recipe.demonstration_prior_strength,
            )
            for recipe, count in zip(train_recipes, counts, strict=True)
        )
    # A small validation set still samples every recipe dimension in a stable
    # round-robin order; no validation seed is reused for training.
    active = train_recipes[: config.validation_games]
    counts = _balanced_counts(config.validation_games, len(active))
    return tuple(
        GenerationRecipe(
            recipe_index=recipe.recipe_index,
            games=count,
            strategy_prior_strength=recipe.strategy_prior_strength,
            selection_temperature=recipe.selection_temperature,
            score_utility_weight=recipe.score_utility_weight,
            final_vp_utility_weight=recipe.final_vp_utility_weight,
            demonstration_prior_strength=recipe.demonstration_prior_strength,
        )
        for recipe, count in zip(active, counts, strict=True)
    )


def filter_quality_shards(
    shard_paths: Sequence[str | Path],
    output_prefix: str | Path,
    gate: QualityGate,
) -> tuple[QualityFilterSummary, ...]:
    """Filter several raw shards, preserving one output shard per input."""

    if not shard_paths:
        raise ValueError("at least one shard is required")
    gate.validate()
    prefix = Path(output_prefix)
    if not prefix.name or prefix.suffix == ".jsonl":
        raise ValueError("output_prefix must be a file-name prefix without .jsonl")
    width = max(3, len(str(len(shard_paths) - 1)))
    outputs = tuple(
        (prefix.parent / f"{prefix.name}-{index:0{width}d}.jsonl").resolve()
        for index in range(len(shard_paths))
    )
    conflicts = [
        candidate
        for output in outputs
        for candidate in (output, Path(f"{output}.partial"))
        if candidate.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "quality-filter output already exists: "
            + ", ".join(str(path) for path in conflicts)
        )
    return tuple(
        filter_quality_shard(Path(source), output, gate)
        for source, output in zip(shard_paths, outputs, strict=True)
    )


def filter_quality_shard(
    source: Path,
    output: Path,
    gate: QualityGate,
) -> QualityFilterSummary:
    """Write only complete, healthy, high-quality actor trajectories."""

    gate.validate()
    source = source.resolve()
    output = output.resolve()
    partial = Path(f"{output}.partial")
    if not source.is_file():
        raise FileNotFoundError(f"self-play source does not exist: {source}")
    if output.exists() or partial.exists():
        raise FileExistsError(f"quality-filter output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    games_seen = games_accepted = games_rejected = 0
    source_positions = positions_written = eligible_trajectories = 0
    accepted_seeds: list[int] = []
    rejection_reasons: Counter[str] = Counter()
    accepted_scores: list[int] = []

    try:
        with source.open("rb") as input_handle, partial.open("xb") as output_handle:
            metadata, games = _read_shard_stream(input_handle, source)
            derived_metadata = dict(metadata)
            derived_metadata.update(
                {
                    "derived_dataset": "expert_iteration_quality_filter",
                    "derived_from": str(source),
                    "quality_filter_version": QUALITY_FILTER_VERSION,
                    "quality_gate": asdict(gate),
                    "source_type": "expert_iteration",
                    "value_target_usable": True,
                }
            )
            _write_json_line(output_handle, derived_metadata)

            for raw_game in games:
                games_seen += 1
                source_positions += len(raw_game.positions)
                quality = assess_game_quality(raw_game.record, raw_game.positions, gate)
                for reason in quality.rejection_reasons:
                    rejection_reasons[reason] += 1
                if not quality.accepted:
                    games_rejected += 1
                    continue
                games_accepted += 1
                eligible = set(quality.eligible_players)
                selected_positions = [
                    item
                    for item in raw_game.positions
                    if int(item.record["actor"]) in eligible
                ]
                if not selected_positions:
                    games_rejected += 1
                    rejection_reasons["eligible_actor_has_no_positions"] += 1
                    games_accepted -= 1
                    continue
                game_record = dict(raw_game.record)
                game_record["source_positions"] = game_record.get(
                    "positions", len(raw_game.positions)
                )
                game_record["positions"] = len(selected_positions)
                game_record["quality_eligible_players"] = list(quality.eligible_players)
                game_record["quality_metrics"] = asdict(quality)
                _write_json_line(output_handle, game_record)
                for item in selected_positions:
                    output_handle.write(
                        item.raw_line if item.raw_line.endswith(b"\n") else item.raw_line + b"\n"
                    )
                positions_written += len(selected_positions)
                eligible_trajectories += len(quality.eligible_players)
                if quality.game_seed is not None:
                    accepted_seeds.append(quality.game_seed)
                accepted_scores.extend(
                    quality.scores[index] for index in quality.eligible_players
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return QualityFilterSummary(
        source=str(source),
        output=str(output),
        games_seen=games_seen,
        games_accepted=games_accepted,
        games_rejected=games_rejected,
        source_positions=source_positions,
        positions_written=positions_written,
        eligible_player_trajectories=eligible_trajectories,
        accepted_game_seeds=tuple(sorted(set(accepted_seeds))),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        score_summary=_score_summary(accepted_scores),
    )


def assess_game_quality(
    game: dict[str, Any],
    positions: Sequence[_RawPosition] | Sequence[dict[str, Any]],
    gate: QualityGate,
) -> GameQuality:
    """Assess one game without writing anything.

    This public function is useful for dashboards and unit tests.  Structural
    failures become explicit rejection reasons; malformed JSON or a position
    that cannot be associated with a game remains a hard input error in the
    stream reader.
    """

    gate.validate()
    game_index = _optional_nonnegative_int(game.get("game_index"))
    if game_index is None:
        reasons: list[str] = ["missing_game_index"]
        game_index = -1
    else:
        reasons = []
    game_seed = _optional_nonnegative_int(game.get("game_seed"))
    if game_seed is None:
        reasons.append("missing_game_seed")
    declared_positions = _optional_nonnegative_int(game.get("positions"))
    if declared_positions is None:
        reasons.append("missing_positions_count")
        declared_positions = len(positions)
    if declared_positions != len(positions):
        reasons.append("incomplete_trajectory")

    scores, valid_scores = _parse_integer_vector(game.get("victory_points"))
    income, valid_income = _parse_integer_vector(game.get("income_levels"))
    money, valid_money = _parse_integer_vector(game.get("money"))
    if not scores:
        reasons.append("missing_victory_points")
    elif not valid_scores or any(score < 0 for score in scores):
        reasons.append("invalid_victory_points")
    if income and not valid_income:
        reasons.append("invalid_income_levels")
    if money and not valid_money:
        reasons.append("invalid_money")
    player_count = len(scores)
    if player_count and (len(income) != player_count or len(money) != player_count):
        reasons.append("outcome_vector_length_mismatch")
    if gate.require_complete_outcome:
        # ``official_winners`` is a variable-length list (ties may contain
        # several seats); the remaining terminal vectors have one entry per
        # player.
        winners_value = game.get("official_winners")
        if not isinstance(winners_value, list) or not winners_value:
            reasons.append("missing_outcome_official_winners")
        for field in (
            "shared_win_values",
            "placements",
            "finish_order",
            "victory_point_margins",
        ):
            value = game.get(field)
            if not isinstance(value, list) or len(value) != player_count:
                reasons.append(f"missing_outcome_{field}")

    actors = [_position_actor(item) for item in positions]
    if any(
        _position_index(item) != expected_index
        for expected_index, item in enumerate(positions)
    ):
        reasons.append("noncontiguous_position_index")
    if any(actor is None for actor in actors):
        reasons.append("invalid_position_actor")
    valid_actors = [actor for actor in actors if actor is not None]
    if player_count and any(actor >= player_count for actor in valid_actors):
        reasons.append("position_actor_out_of_range")

    lifecycle = _lifecycle_counts(positions)
    if gate.reject_lifecycle_violations:
        if lifecycle["network_before_first_build"]:
            reasons.append("network_before_first_build")
        if lifecycle["repeat_loans"]:
            reasons.append("repeat_loan")

    if scores and gate.reject_any_zero_score and any(score == 0 for score in scores):
        reasons.append("zero_score")
    if income and any(value < gate.minimum_final_income for value in income):
        reasons.append("unhealthy_final_income")
    if money and any(value < gate.minimum_final_money for value in money):
        reasons.append("unhealthy_final_money")

    winners = _winner_indices(game, scores)
    if scores and not winners:
        reasons.append("missing_winner")
    if scores:
        raw_winners = game.get("official_winners")
        if isinstance(raw_winners, list) and any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= player_count
            for value in raw_winners
        ):
            reasons.append("invalid_winner_index")

    positions_by_actor = Counter(valid_actors)
    eligible: list[int] = []
    if scores:
        for actor, score in enumerate(scores):
            opponents = [other for index, other in enumerate(scores) if index != actor]
            margin = score - max(opponents, default=score)
            if score < gate.minimum_actor_vp:
                continue
            if margin < gate.minimum_vp_margin:
                continue
            if gate.require_top_actor and actor not in winners:
                continue
            if positions_by_actor.get(actor, 0) < gate.minimum_positions_per_actor:
                continue
            eligible.append(actor)
    if not eligible:
        if scores and all(score < gate.minimum_actor_vp for score in scores):
            reasons.append("actor_score_below_threshold")
        if scores and all(
            score - max(
                (other for index, other in enumerate(scores) if index != actor),
                default=score,
            )
            < gate.minimum_vp_margin
            for actor, score in enumerate(scores)
        ):
            reasons.append("actor_margin_below_threshold")
        if gate.require_top_actor and scores and not any(
            actor in winners
            and scores[actor] >= gate.minimum_actor_vp
            and scores[actor]
            - max(
                (other for index, other in enumerate(scores) if index != actor),
                default=scores[actor],
            )
            >= gate.minimum_vp_margin
            and positions_by_actor.get(actor, 0) >= gate.minimum_positions_per_actor
            for actor in range(len(scores))
        ):
            reasons.append("no_top_actor_above_threshold")
        reasons.append("no_eligible_actor")

    # Preserve stable reason order while avoiding duplicate diagnostics when a
    # malformed game violates several vector checks at once.
    unique_reasons = tuple(dict.fromkeys(reasons))
    accepted = not unique_reasons and bool(eligible)
    return GameQuality(
        game_index=game_index,
        game_seed=game_seed,
        positions=declared_positions,
        player_count=player_count,
        scores=tuple(scores),
        winners=tuple(winners),
        eligible_players=tuple(eligible) if accepted else (),
        accepted=accepted,
        rejection_reasons=unique_reasons,
        lifecycle=dict(lifecycle),
        final_income=tuple(income),
        final_money=tuple(money),
    )


def _read_shard_stream(
    handle: BinaryIO,
    source: Path,
) -> tuple[dict[str, Any], Iterator[_RawGame]]:
    """Read metadata eagerly and return a lazy game iterator."""

    line_number = 0
    metadata: dict[str, Any] | None = None
    while True:
        raw_line = handle.readline()
        if not raw_line:
            break
        line_number += 1
        if not raw_line.strip():
            continue
        record = _decode_record(raw_line, source, line_number)
        if record.get("record_type") != "metadata":
            raise ValueError(f"{source}:{line_number}: first record must be metadata")
        metadata = record
        break
    if metadata is None:
        raise ValueError(f"{source}: source shard has no metadata")

    # Materialize records one game at a time.  The generator closes over the
    # open handle owned by filter_quality_shard and starts immediately after
    # the metadata line consumed above.
    def iterator() -> Iterator[_RawGame]:
        pending_game: dict[str, Any] | None = None
        pending_line_number = 0
        pending_positions: list[_RawPosition] = []
        current_line_number = line_number
        for raw_line in handle:
            current_line_number += 1
            if not raw_line.strip():
                continue
            record = _decode_record(raw_line, source, current_line_number)
            record_type = record.get("record_type")
            if record_type == "game":
                if pending_game is not None:
                    yield _RawGame(
                        record=pending_game,
                        positions=tuple(pending_positions),
                        line_number=pending_line_number,
                    )
                pending_game = record
                pending_line_number = current_line_number
                pending_positions = []
                continue
            if record_type == "position":
                if pending_game is None:
                    raise ValueError(
                        f"{source}:{current_line_number}: position precedes its game"
                    )
                if record.get("game_index") != pending_game.get("game_index"):
                    raise ValueError(
                        f"{source}:{current_line_number}: position game index changed"
                    )
                pending_positions.append(
                    _RawPosition(
                        record=record,
                        raw_line=raw_line,
                        line_number=current_line_number,
                    )
                )
                continue
            if record_type == "metadata":
                raise ValueError(f"{source}:{current_line_number}: duplicate metadata")
            raise ValueError(
                f"{source}:{current_line_number}: unknown record type {record_type!r}"
            )
        if pending_game is not None:
            yield _RawGame(
                record=pending_game,
                positions=tuple(pending_positions),
                line_number=pending_line_number,
            )

    return metadata, iterator()


def _decode_record(line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return record


def _write_json_line(handle: BinaryIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    handle.write(b"\n")


def _position_actor(position: _RawPosition | dict[str, Any]) -> int | None:
    record = position.record if isinstance(position, _RawPosition) else position
    value = record.get("actor")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _position_index(position: _RawPosition | dict[str, Any]) -> int | None:
    record = position.record if isinstance(position, _RawPosition) else position
    value = record.get("position_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _lifecycle_counts(
    positions: Sequence[_RawPosition] | Sequence[dict[str, Any]],
) -> dict[str, int]:
    built: set[int] = set()
    previous_action: dict[int, str] = {}
    counts = {"network_before_first_build": 0, "repeat_loans": 0}
    for item in positions:
        record = item.record if isinstance(item, _RawPosition) else item
        actor = _position_actor(record)
        key = record.get("selected_action_key")
        if actor is None or not isinstance(key, str):
            continue
        root = key.split("|", 1)[0]
        if root in {"network", "double_network"} and actor not in built:
            counts["network_before_first_build"] += 1
        if root == "loan" and previous_action.get(actor) == "loan":
            counts["repeat_loans"] += 1
        if root == "build":
            built.add(actor)
        previous_action[actor] = root
    return counts


def _winner_indices(game: dict[str, Any], scores: Sequence[int]) -> tuple[int, ...]:
    raw = game.get("official_winners")
    if isinstance(raw, list):
        values = [
            int(value)
            for value in raw
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        if values:
            return tuple(sorted(set(values)))
    if not scores:
        return ()
    maximum = max(scores)
    return tuple(index for index, score in enumerate(scores) if score == maximum)


def _parse_integer_vector(value: object) -> tuple[list[int], bool]:
    if not isinstance(value, list):
        return [], False
    valid = all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    return [int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)], valid


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _score_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "at_least_50": 0,
            "at_least_100": 0,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "at_least_50": sum(value >= 50 for value in values),
        "at_least_100": sum(value >= 100 for value in values),
    }


def _collect_game_seeds(shards: Iterable[str]) -> set[int]:
    seeds: set[int] = set()
    for raw_path in shards:
        path = Path(raw_path)
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = _decode_record(raw_line, path, line_number)
                if record.get("record_type") != "game":
                    continue
                seed = _optional_nonnegative_int(record.get("game_seed"))
                if seed is not None:
                    if seed in seeds:
                        raise RuntimeError(f"duplicate generated game seed: {seed}")
                    seeds.add(seed)
    return seeds


def _split_manifest(result: _SplitResult | _EmptySplitResult, requested: int) -> dict[str, Any]:
    return {
        "games_requested": requested,
        "raw_shards": list(result.raw_shards),
        "filtered_shards": list(result.filtered_shards),
        "self_play": [
            {
                "games_written": summary.games_written,
                "positions_written": summary.positions_written,
                "model_id": summary.model_id,
                "checkpoint_step": summary.checkpoint_step,
                "workers": summary.workers,
                "elapsed_seconds": summary.elapsed_seconds,
                "positions_per_second": summary.positions_per_second,
            }
            for summary in result.self_play_summaries
        ],
        "quality_filters": [asdict(item) for item in result.filters],
        "games_accepted": sum(item.games_accepted for item in result.filters),
        "positions_written": sum(item.positions_written for item in result.filters),
    }


def _balanced_counts(total: int, buckets: int) -> tuple[int, ...]:
    if buckets <= 0 or total < buckets:
        raise ValueError("total must be at least the number of buckets")
    base, extra = divmod(total, buckets)
    return tuple(base + (1 if index < extra else 0) for index in range(buckets))


def _validate_seed(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_float_sequence(
    values: Sequence[float],
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    for index, value in enumerate(values):
        parsed = _finite_float(value, f"{name}[{index}]")
        if not minimum <= parsed <= maximum:
            raise ValueError(
                f"{name}[{index}] must be between {minimum:g} and {maximum:g}"
            )


def _write_json_atomically(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"output already exists: {path}")
    try:
        with partial.open("xb") as handle:
            _write_json_line(handle, record)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
