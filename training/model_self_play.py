from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .evaluator import PolicyValueEvaluator, PolicyValuePredictionLike
from .neural_search import BATCHED_NEURAL_PUCT_METHOD, run_batched_neural_puct
from .schema import SELF_PLAY_FORMAT, SELF_PLAY_FORMAT_VERSION, FeatureSchema

GAME_SEED_STREAM = 0x6761_6D65_5F73_6565
SEARCH_SEED_STREAM = 0x7365_6172_6368_5F73
SELECTION_SEED_STREAM = 0x7365_6C65_6374_5F73
MASK_64 = (1 << 64) - 1
ROOT_PUCT_METHOD = "determinized_root_puct_policy_random_rollout"


@dataclass(frozen=True)
class ModelSelfPlayConfig:
    output: Path
    checkpoint: Path | None = None
    inference_url: str | None = None
    inference_timeout_seconds: float = 120.0
    games: int = 1
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

    def validate(self) -> None:
        if (self.checkpoint is None) == (self.inference_url is None):
            raise ValueError("exactly one of checkpoint or inference_url is required")
        if self.inference_url is not None and not self.inference_url.strip():
            raise ValueError("inference_url must not be empty")
        if (
            not math.isfinite(self.inference_timeout_seconds)
            or self.inference_timeout_seconds <= 0.0
            or self.inference_timeout_seconds > 3600.0
        ):
            raise ValueError(
                "inference_timeout_seconds must be between 0 and 3600"
            )
        if self.games <= 0:
            raise ValueError("games must be positive")
        if not 0 <= self.game_index_offset <= MASK_64:
            raise ValueError(
                "game_index_offset must fit in an unsigned 64-bit integer"
            )
        if self.game_index_offset + self.games - 1 > MASK_64:
            raise ValueError(
                "game index range must fit in an unsigned 64-bit integer"
            )
        if self.num_players not in (2, 3, 4):
            raise ValueError("num_players must be between 2 and 4")
        if not 1 <= self.simulations_per_decision <= 1_000_000:
            raise ValueError("simulations_per_decision must be between 1 and 1000000")
        if (
            not math.isfinite(self.exploration_constant)
            or self.exploration_constant < 0.0
        ):
            raise ValueError("exploration_constant must be finite and non-negative")
        if not 1 <= self.search_determinizations <= 64:
            raise ValueError("search_determinizations must be between 1 and 64")
        if not 1 <= self.inference_batch_size <= 256:
            raise ValueError("inference_batch_size must be between 1 and 256")
        if self.max_game_actions <= 0:
            raise ValueError("max_game_actions must be positive")
        if not 0 <= self.base_seed <= MASK_64:
            raise ValueError("base_seed must fit in an unsigned 64-bit integer")
        if not self.engine_revision.strip():
            raise ValueError("engine_revision must not be empty")


@dataclass(frozen=True)
class ModelSelfPlaySummary:
    output: str
    games_written: int
    positions_written: int
    model_id: str
    checkpoint_step: int


def main() -> None:
    args = build_parser().parse_args()
    config = ModelSelfPlayConfig(
        output=Path(args.output),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        inference_url=args.inference_url,
        inference_timeout_seconds=args.inference_timeout_seconds,
        games=args.games,
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
    )
    summary = export_model_self_play(
        config,
        on_game_complete=lambda game: print(
            json.dumps(
                {
                    "event": "model_self_play_game",
                    "game": game["game_index"] + 1,
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
    print(json.dumps(asdict(summary), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Fast Brass self-play JSONL using checkpoint priors and Rust "
            "root PUCT search."
        )
    )
    parser.add_argument("--output", required=True)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint")
    model_source.add_argument("--inference-url")
    parser.add_argument("--inference-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--games", type=int, default=1)
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
    return parser


def export_model_self_play(
    config: ModelSelfPlayConfig,
    *,
    engine_module: Any | None = None,
    evaluator: PolicyValueEvaluator | None = None,
    on_game_complete: Callable[[dict], None] | None = None,
) -> ModelSelfPlaySummary:
    config.validate()
    output = config.output.resolve()
    partial = Path(f"{output}.partial")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing shard {output}")
    if partial.exists():
        raise FileExistsError(
            f"incomplete shard already exists at {partial}; inspect or remove it first"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    if engine_module is None:
        try:
            import fast_brass as engine_module
        except ImportError as error:
            raise RuntimeError(
                "fast_brass Python extension is required; build/install the abi3 wheel first"
            ) from error
    probe = engine_module.BrassRLGame(config.num_players, config.base_seed)
    raw_schema = probe.get_training_feature_schema()
    schema = FeatureSchema.from_schema_dict(raw_schema)
    if evaluator is None:
        if config.inference_url is not None:
            from .remote_inference import RemoteInferenceEvaluator

            evaluator = RemoteInferenceEvaluator(
                config.inference_url,
                raw_schema,
                timeout_seconds=config.inference_timeout_seconds,
            )
        else:
            from .inference import CheckpointEvaluator

            if config.checkpoint is None:
                raise RuntimeError("checkpoint model source is missing")
            evaluator = CheckpointEvaluator(config.checkpoint, config.device)
    evaluator.schema.assert_compatible(schema, "model-guided self-play engine")
    header = _build_header(config, raw_schema, evaluator)

    games_written = 0
    positions_written = 0
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        _write_json_line(handle, header)
        for local_game_index in range(config.games):
            game_index = config.game_index_offset + local_game_index
            game_record, positions = generate_model_self_play_game(
                config,
                engine_module,
                evaluator,
                schema,
                game_index,
            )
            _write_json_line(handle, game_record)
            for position in positions:
                _write_json_line(handle, position)
            games_written += 1
            positions_written += len(positions)
            if on_game_complete is not None:
                on_game_complete(game_record)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(output)
    return ModelSelfPlaySummary(
        output=str(output),
        games_written=games_written,
        positions_written=positions_written,
        model_id=evaluator.model_id,
        checkpoint_step=evaluator.checkpoint_step,
    )


def generate_model_self_play_game(
    config: ModelSelfPlayConfig,
    engine_module: Any,
    evaluator: PolicyValueEvaluator,
    schema: FeatureSchema,
    game_index: int,
) -> tuple[dict, list[dict]]:
    game_seed = _derive_stream_seed(config.base_seed, GAME_SEED_STREAM, game_index)
    game = engine_module.BrassRLGame(config.num_players, game_seed)
    positions: list[dict] = []

    while not game.is_done():
        while game.current_decision_mode() == "shortfall":
            _resolve_shortfall(game)
        if game.is_done():
            break
        if len(positions) >= config.max_game_actions:
            raise RuntimeError(
                f"model self-play game {game_index} exceeded "
                f"{config.max_game_actions} decisions"
            )

        position_index = len(positions)
        state_record = game.get_training_state()
        legal_record = game.get_legal_actions()
        actor = int(state_record["decision_player"])
        if int(legal_record["decision_player"]) != actor:
            raise RuntimeError("state and legal-action decision players disagree")
        prediction = evaluator.predict(state_record, legal_record, schema)
        search_seed = _derive_stream_seed(
            game_seed, SEARCH_SEED_STREAM, position_index
        )
        selection_seed = _derive_stream_seed(
            game_seed, SELECTION_SEED_STREAM, position_index
        )
        if config.num_players == 2:
            report = run_batched_neural_puct(
                game,
                evaluator,
                schema,
                prediction,
                simulations=config.simulations_per_decision,
                search_seed=search_seed,
                exploration_constant=config.exploration_constant,
                determinizations=config.search_determinizations,
                inference_batch_size=config.inference_batch_size,
            )
            expected_method = BATCHED_NEURAL_PUCT_METHOD
        else:
            report = game.search_legal_actions_with_policy(
                list(prediction.policy_probabilities),
                prediction.model_id,
                prediction.shared_win_rate,
                prediction.victory_point_margin,
                config.simulations_per_decision,
                search_seed,
                config.exploration_constant,
            )
            expected_method = ROOT_PUCT_METHOD
        _validate_search_report(report, prediction, legal_record, config, expected_method)
        action_targets = _build_action_targets(legal_record, report)
        selected_action_index = _sample_action_index(action_targets, selection_seed)
        selected_action_key = action_targets[selected_action_index]["key"]
        observation = game.get_observation(actor)

        positions.append(
            {
                "record_type": "position",
                "format_version": SELF_PLAY_FORMAT_VERSION,
                "feature_version": schema.version,
                "game_index": game_index,
                "position_index": position_index,
                "game_seed": game_seed,
                "search_seed": search_seed,
                "selection_seed": selection_seed,
                "actor": actor,
                "phase": _phase_name(observation),
                "turn_count": int(observation.get("turn_count", 0)),
                "round_in_phase": int(observation.get("round_in_phase", 0)),
                "actions_remaining_in_turn": int(
                    observation.get("actions_remaining", 0)
                ),
                "forced_action_slots_before": int(
                    state_record.get("forced_advances", 0)
                )
                + int(legal_record.get("forced_advances", 0))
                + int(report.get("forced_advances", 0)),
                "state_features": state_record["features"],
                "legal_actions": action_targets,
                "selected_action_index": selected_action_index,
                "selected_action_key": selected_action_key,
                "search_method": report["method"],
                "model_id": prediction.model_id,
                "checkpoint_step": prediction.checkpoint_step,
                "root_model_shared_win_rate": prediction.shared_win_rate,
                "root_model_victory_point_margin": prediction.victory_point_margin,
                "search_determinizations": (
                    config.search_determinizations
                    if config.num_players == 2
                    else None
                ),
                "inference_batch_size": (
                    config.inference_batch_size if config.num_players == 2 else None
                ),
                "max_search_depth": report.get("max_search_depth"),
                "neural_leaf_evaluations": report.get("neural_leaf_evaluations"),
                "inference_batches": report.get("inference_batches"),
                "value_target": {"shared_win": 0.0, "victory_point_margin": 0},
            }
        )
        game.step_legal_action(selected_action_index)

    outcome = game.get_outcome()
    for position in positions:
        actor = position["actor"]
        position["value_target"] = {
            "shared_win": float(outcome["shared_win_values"][actor]),
            "victory_point_margin": int(outcome["victory_point_margins"][actor]),
        }
    game_record = {
        "record_type": "game",
        "format_version": SELF_PLAY_FORMAT_VERSION,
        "game_index": game_index,
        "game_seed": game_seed,
        "positions": len(positions),
        "official_winners": [int(value) for value in outcome["official_winners"]],
        "shared_win_values": [
            float(value) for value in outcome["shared_win_values"]
        ],
        "placements": [int(value) for value in outcome["placements"]],
        "finish_order": [int(value) for value in outcome["finish_order"]],
        "victory_points": [int(value) for value in outcome["victory_points"]],
        "victory_point_margins": [
            int(value) for value in outcome["victory_point_margins"]
        ],
        "income_levels": [int(value) for value in outcome["income_levels"]],
        "money": [int(value) for value in outcome["money"]],
    }
    return game_record, positions


def _build_header(
    config: ModelSelfPlayConfig,
    raw_schema: dict,
    evaluator: PolicyValueEvaluator,
) -> dict:
    return {
        "record_type": "metadata",
        "format": SELF_PLAY_FORMAT,
        "format_version": SELF_PLAY_FORMAT_VERSION,
        "crate_version": _installed_crate_version(),
        "engine_revision": config.engine_revision,
        "feature_schema": raw_schema,
        "games": config.games,
        "game_index_offset": config.game_index_offset,
        "num_players": config.num_players,
        "base_seed": config.base_seed,
        "simulations_per_decision": config.simulations_per_decision,
        "exploration_constant": config.exploration_constant,
        "search_determinizations": (
            config.search_determinizations if config.num_players == 2 else None
        ),
        "inference_batch_size": (
            config.inference_batch_size if config.num_players == 2 else None
        ),
        "max_game_actions": config.max_game_actions,
        "search_method": (
            BATCHED_NEURAL_PUCT_METHOD if config.num_players == 2 else ROOT_PUCT_METHOD
        ),
        "model_id": evaluator.model_id,
        "checkpoint_step": evaluator.checkpoint_step,
        "model_source": (
            "remote_inference" if config.inference_url is not None else "checkpoint"
        ),
        "checkpoint_name": (
            config.checkpoint.name if config.checkpoint is not None else None
        ),
        "inference_url": config.inference_url,
        "action_selection": "deterministic_sample_from_root_visit_counts",
        "shortfall_resolution": "ascending_liquidation_value_then_location",
    }


def _validate_search_report(
    report: dict,
    prediction: PolicyValuePredictionLike,
    legal_record: dict,
    config: ModelSelfPlayConfig,
    expected_method: str,
) -> None:
    actions = legal_record.get("actions")
    searched_actions = report.get("actions")
    if report.get("method") != expected_method:
        raise RuntimeError(f"unexpected model search method {report.get('method')!r}")
    if report.get("model_id") != prediction.model_id:
        raise RuntimeError("search report model ID mismatch")
    if report.get("completed_simulations") != config.simulations_per_decision:
        raise RuntimeError("search did not complete all requested simulations")
    if not isinstance(actions, list) or not isinstance(searched_actions, list):
        raise RuntimeError("legal actions or search actions are missing")
    if len(actions) != len(searched_actions):
        raise RuntimeError("search did not return the complete stable action list")
    expected_keys = tuple(action.get("key") for action in actions)
    actual_keys = tuple(action.get("key") for action in searched_actions)
    if expected_keys != prediction.action_keys or actual_keys != expected_keys:
        raise RuntimeError("model, engine, and search action-key order disagree")
    probability_mass = sum(prediction.policy_probabilities)
    if not math.isfinite(probability_mass) or probability_mass <= 0.0:
        raise RuntimeError("model policy probabilities have invalid total mass")
    for index, (searched, probability) in enumerate(
        zip(searched_actions, prediction.policy_probabilities)
    ):
        if searched.get("index") != index:
            raise RuntimeError("search action indices are not stable and contiguous")
        actual_probability = searched.get("policy_probability")
        normalized_probability = probability / probability_mass
        if not isinstance(actual_probability, (int, float)) or not math.isclose(
            float(actual_probability), normalized_probability, abs_tol=1e-12
        ):
            raise RuntimeError("search report changed a model policy probability")


def _build_action_targets(legal_record: dict, report: dict) -> list[dict]:
    legal_actions = legal_record["actions"]
    searched_actions = report["actions"]
    completed_simulations = int(report["completed_simulations"])
    if completed_simulations <= 0:
        raise RuntimeError("cannot build targets from zero completed simulations")
    targets: list[dict] = []
    visit_sum = 0
    for index, (legal, searched) in enumerate(zip(legal_actions, searched_actions)):
        if legal.get("index") != index or searched.get("index") != index:
            raise RuntimeError("action indices are not stable and contiguous")
        if legal.get("key") != searched.get("key"):
            raise RuntimeError("legal and searched action keys disagree")
        visits = int(searched["visits"])
        if visits < 0:
            raise RuntimeError("search returned a negative visit count")
        visit_sum += visits
        targets.append(
            {
                "index": index,
                "key": legal["key"],
                "feature_indices": legal["feature_indices"],
                "visits": visits,
                "policy_target": visits / completed_simulations,
                "model_policy_probability": searched["policy_probability"],
                "value_source": searched.get("value_source"),
                "value_sample_count": searched.get("value_sample_count"),
                "estimated_shared_win_rate": searched[
                    "estimated_shared_win_rate"
                ],
                "average_victory_point_margin": searched[
                    "average_victory_point_margin"
                ],
            }
        )
    if visit_sum != completed_simulations:
        raise RuntimeError(
            f"stable action targets contain {visit_sum} visits, expected "
            f"{completed_simulations}"
        )
    return targets


def _sample_action_index(targets: list[dict], selection_seed: int) -> int:
    total_visits = sum(int(target["visits"]) for target in targets)
    if total_visits <= 0:
        raise RuntimeError("cannot sample an action from zero visits")
    draw = _splitmix64(selection_seed) % total_visits
    for target in targets:
        visits = int(target["visits"])
        if draw < visits:
            return int(target["index"])
        draw -= visits
    raise RuntimeError("visit sampling failed to select an action")


def _resolve_shortfall(game: Any) -> None:
    actor = game.current_decision_player()
    observation = game.get_observation(actor)
    shortfall = observation.get("shortfall_state")
    if not isinstance(shortfall, dict):
        raise RuntimeError("shortfall mode has no shortfall state")
    tiles = sorted(
        shortfall.get("removable_tiles", []),
        key=lambda tile: (int(tile[1]), int(tile[0])),
    )
    game.step_composite_action(
        {"shortfall_tile_order": [int(tile[0]) for tile in tiles]}
    )


def _phase_name(observation: dict) -> str:
    global_features = observation.get("global_features")
    if not isinstance(global_features, list) or len(global_features) < 3:
        return "unknown"
    phase_index = max(range(3), key=lambda index: float(global_features[index]))
    return ("canal", "railroad", "game_end")[phase_index]


def detect_engine_revision() -> str:
    configured = os.environ.get("FAST_BRASS_ENGINE_REVISION", "").strip()
    if configured:
        return configured
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return f"fast_brass-{_installed_crate_version()}-unknown"
    return f"{revision}-dirty" if dirty else revision


def _installed_crate_version() -> str:
    try:
        return importlib.metadata.version("fast_brass")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _derive_stream_seed(base_seed: int, stream: int, index: int) -> int:
    mixed = (base_seed ^ stream ^ ((index * 0x9E37_79B9_7F4A_7C15) & MASK_64)) & MASK_64
    return _splitmix64(mixed)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E37_79B9_7F4A_7C15) & MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK_64
    value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & MASK_64
    return (value ^ (value >> 31)) & MASK_64


def _write_json_line(handle: Any, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


if __name__ == "__main__":
    main()
