from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .evaluator import PolicyValueEvaluator
from .neural_search import BATCHED_NEURAL_PUCT_METHOD, run_batched_neural_puct
from .schema import FeatureSchema
from .strategy_prior import (
    STRATEGY_PRIOR_VERSION,
    SUPPORTED_STRATEGY_PRIOR_VERSIONS,
    blend_policy_with_strategy,
    build_strategy_prior,
)

ROOT_PUCT_METHOD = "determinized_root_puct_policy_random_rollout"
SEARCH_SEED_STREAM = 0x6576_616C_5F73_6561

T_CRITICAL_95 = (
    math.inf,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)

STANDARD_NORMAL_975 = 1.959963984540054


@dataclass(frozen=True)
class MatchResult:
    round_index: int
    game_seed: int
    candidate_seat: int
    actions: int
    candidate_shared_win: float
    candidate_score_delta: float
    candidate_victory_point_margin: int
    candidate_placement: int
    candidate_is_official_winner: bool
    official_winners: tuple[int, ...]
    victory_points: tuple[int, ...]


@dataclass(frozen=True)
class GateSummary:
    games: int
    independent_seed_groups: int
    candidate_mean_shared_win: float
    candidate_mean_score_delta: float
    score_delta_standard_error: float | None
    score_delta_lower_95: float | None
    candidate_mean_victory_point_margin: float
    candidate_first_place_rate: float
    promotion_margin: float
    minimum_games: int
    promote: bool


class EvaluationPolicy:
    def __init__(
        self,
        evaluator: PolicyValueEvaluator,
        schema: FeatureSchema,
        source: str,
        source_kind: str,
        strategy_prior_strength: float = 0.0,
        strategy_prior_version: str = STRATEGY_PRIOR_VERSION,
        score_utility_weight: float = 0.0,
        final_vp_utility_weight: float = 0.0,
    ) -> None:
        evaluator.schema.assert_compatible(schema, source)
        if (
            not math.isfinite(score_utility_weight)
            or not 0.0 <= score_utility_weight <= 1.0
        ):
            raise ValueError("score_utility_weight must be between 0 and 1")
        if (
            not math.isfinite(final_vp_utility_weight)
            or not 0.0 <= final_vp_utility_weight <= 1.0
        ):
            raise ValueError("final_vp_utility_weight must be between 0 and 1")
        self.evaluator = evaluator
        self.schema = schema
        self.source = source
        self.source_kind = source_kind
        self.model_id = evaluator.model_id
        self.checkpoint_step = evaluator.checkpoint_step
        self.strategy_prior_strength = strategy_prior_strength
        self.strategy_prior_version = strategy_prior_version
        self.score_utility_weight = score_utility_weight
        self.final_vp_utility_weight = final_vp_utility_weight

    def select_action(
        self, observation: dict, state_record: dict, legal_record: dict
    ) -> int:
        prediction = self.evaluator.predict(state_record, legal_record, self.schema)
        expected_keys = tuple(
            action.get("key") for action in legal_record.get("actions", [])
        )
        if prediction.action_keys != expected_keys:
            raise RuntimeError("model changed the stable legal-action order")
        probabilities = self._root_policy_probabilities(
            observation, legal_record, prediction.policy_probabilities
        )
        return max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )

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
        prediction = self.evaluator.predict(
            state_record, legal_record, self.schema
        )
        root_policy_probabilities = self._root_policy_probabilities(
            observation, legal_record, prediction.policy_probabilities
        )
        report = run_batched_neural_puct(
            game,
            self.evaluator,
            self.schema,
            prediction,
            simulations=simulations,
            search_seed=search_seed,
            exploration_constant=exploration_constant,
            determinizations=search_determinizations,
            inference_batch_size=inference_batch_size,
            root_policy_probabilities=root_policy_probabilities,
            score_utility_weight=self.score_utility_weight,
            final_vp_utility_weight=self.final_vp_utility_weight,
        )
        expected_method = BATCHED_NEURAL_PUCT_METHOD
        return _select_search_action(
            report,
            prediction.action_keys,
            prediction.model_id,
            expected_method,
            simulations,
        )

    def _root_policy_probabilities(
        self,
        observation: dict,
        legal_record: dict,
        model_probabilities: tuple[float, ...],
    ) -> tuple[float, ...]:
        if self.strategy_prior_strength <= 0.0:
            return model_probabilities
        prior = build_strategy_prior(
            observation,
            legal_record,
            version=self.strategy_prior_version,
        )
        return blend_policy_with_strategy(
            model_probabilities,
            prior.probabilities,
            self.strategy_prior_strength,
            guarded_actions=prior.guarded_actions,
        )


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_checkpoints(args)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run seat-rotated candidate-vs-champion Fast Brass evaluation games."
    )
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument(
        "--candidate", metavar="CHECKPOINT", help="Candidate checkpoint path"
    )
    candidate_source.add_argument(
        "--candidate-inference-url",
        help="Candidate HTTP inference service",
    )
    champion_source = parser.add_mutually_exclusive_group(required=True)
    champion_source.add_argument(
        "--champion", metavar="CHECKPOINT", help="Champion checkpoint path"
    )
    champion_source.add_argument(
        "--champion-inference-url",
        help="Champion HTTP inference service",
    )
    parser.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Seed groups; each group rotates the candidate through every seat",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--minimum-games", type=int, default=40)
    parser.add_argument("--promotion-margin", type=float, default=0.0)
    parser.add_argument(
        "--search-simulations",
        type=int,
        default=0,
        help="Use deployed PUCT agents when positive; zero keeps the greedy-policy gate",
    )
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument(
        "--search-determinizations",
        "--value-determinizations",
        dest="search_determinizations",
        type=int,
        default=4,
    )
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument(
        "--candidate-strategy-prior-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate-strategy-prior-version",
        choices=SUPPORTED_STRATEGY_PRIOR_VERSIONS,
        default=STRATEGY_PRIOR_VERSION,
    )
    parser.add_argument(
        "--champion-strategy-prior-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--champion-strategy-prior-version",
        choices=SUPPORTED_STRATEGY_PRIOR_VERSIONS,
        default=STRATEGY_PRIOR_VERSION,
    )
    parser.add_argument(
        "--candidate-score-utility-weight",
        type=float,
        default=0.0,
        help="Blend secured-score quality into candidate neural PUCT exploitation",
    )
    parser.add_argument(
        "--champion-score-utility-weight",
        type=float,
        default=0.0,
        help="Blend secured-score quality into champion neural PUCT exploitation",
    )
    parser.add_argument(
        "--candidate-final-vp-utility-weight",
        type=float,
        default=0.0,
        help=(
            "Add a bounded opponent-relative final-VP tie-breaker to candidate "
            "neural PUCT when shared-win values are uncertain"
        ),
    )
    parser.add_argument(
        "--champion-final-vp-utility-weight",
        type=float,
        default=0.0,
        help=(
            "Add a bounded opponent-relative final-VP tie-breaker to champion "
            "neural PUCT when shared-win values are uncertain"
        ),
    )
    return parser


def evaluate_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    candidate_checkpoint = getattr(args, "candidate", None)
    candidate_inference_url = getattr(args, "candidate_inference_url", None)
    champion_checkpoint = getattr(args, "champion", None)
    champion_inference_url = getattr(args, "champion_inference_url", None)
    _validate_model_source(
        "candidate", candidate_checkpoint, candidate_inference_url
    )
    _validate_model_source(
        "champion", champion_checkpoint, champion_inference_url
    )
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    if args.max_actions <= 0:
        raise ValueError("max-actions must be positive")
    if args.minimum_games <= 1:
        raise ValueError("minimum-games must be at least two")
    if not math.isfinite(args.promotion_margin):
        raise ValueError("promotion-margin must be finite")
    if not 0 <= args.search_simulations <= 1_000_000:
        raise ValueError("search-simulations must be between 0 and 1000000")
    if not math.isfinite(args.exploration) or args.exploration < 0.0:
        raise ValueError("exploration must be finite and non-negative")
    if not 1 <= args.search_determinizations <= 64:
        raise ValueError("search-determinizations must be between 1 and 64")
    if not 1 <= args.inference_batch_size <= 256:
        raise ValueError("inference-batch-size must be between 1 and 256")
    for name in (
        "candidate_strategy_prior_strength",
        "champion_strategy_prior_strength",
        "candidate_score_utility_weight",
        "champion_score_utility_weight",
        "candidate_final_vp_utility_weight",
        "champion_final_vp_utility_weight",
    ):
        strength = float(getattr(args, name))
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(
                f"{name.replace('_', '-')} must be between 0 and 1"
            )
    inference_timeout = float(
        getattr(args, "inference_timeout_seconds", 120.0)
    )
    if (
        not math.isfinite(inference_timeout)
        or inference_timeout <= 0.0
        or inference_timeout > 3600.0
    ):
        raise ValueError("inference-timeout-seconds must be between 0 and 3600")

    try:
        import fast_brass
    except ImportError as error:
        raise RuntimeError(
            "fast_brass Python extension is required; build/install the abi3 wheel first"
        ) from error

    probe = fast_brass.BrassRLGame(args.players, args.seed)
    raw_schema = probe.get_training_feature_schema()
    schema = FeatureSchema.from_schema_dict(raw_schema)
    candidate = _build_evaluation_policy(
        checkpoint=candidate_checkpoint,
        inference_url=candidate_inference_url,
        raw_schema=raw_schema,
        schema=schema,
        device=args.device,
        timeout_seconds=inference_timeout,
        strategy_prior_strength=args.candidate_strategy_prior_strength,
        strategy_prior_version=args.candidate_strategy_prior_version,
        score_utility_weight=args.candidate_score_utility_weight,
        final_vp_utility_weight=args.candidate_final_vp_utility_weight,
    )
    champion = _build_evaluation_policy(
        checkpoint=champion_checkpoint,
        inference_url=champion_inference_url,
        raw_schema=raw_schema,
        schema=schema,
        device=args.device,
        timeout_seconds=inference_timeout,
        strategy_prior_strength=args.champion_strategy_prior_strength,
        strategy_prior_version=args.champion_strategy_prior_version,
        score_utility_weight=args.champion_score_utility_weight,
        final_vp_utility_weight=args.champion_final_vp_utility_weight,
    )

    results: list[MatchResult] = []
    for round_index in range(args.rounds):
        game_seed = _splitmix64(args.seed + round_index)
        for candidate_seat in range(args.players):
            policies = [champion] * args.players
            policies[candidate_seat] = candidate
            result = play_match(
                fast_brass,
                policies,
                candidate_seat,
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
                        "event": "evaluation_game",
                        "game": len(results),
                        "candidate_seat": candidate_seat,
                        "candidate_score_delta": result.candidate_score_delta,
                        "candidate_vp_margin": result.candidate_victory_point_margin,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    summary = summarize_gate(results, args.promotion_margin, args.minimum_games)
    return {
        "candidate": candidate.source,
        "champion": champion.source,
        "candidate_source_kind": candidate.source_kind,
        "champion_source_kind": champion.source_kind,
        "candidate_model_id": candidate.model_id,
        "champion_model_id": champion.model_id,
        "candidate_checkpoint_step": candidate.checkpoint_step,
        "champion_checkpoint_step": champion.checkpoint_step,
        "candidate_strategy_prior_strength": candidate.strategy_prior_strength,
        "candidate_strategy_prior_version": candidate.strategy_prior_version,
        "candidate_score_utility_weight": candidate.score_utility_weight,
        "candidate_final_vp_utility_weight": candidate.final_vp_utility_weight,
        "champion_strategy_prior_strength": champion.strategy_prior_strength,
        "champion_strategy_prior_version": champion.strategy_prior_version,
        "champion_score_utility_weight": champion.score_utility_weight,
        "champion_final_vp_utility_weight": champion.final_vp_utility_weight,
        "players": args.players,
        "rounds": args.rounds,
        "base_seed": args.seed,
        "feature_version": schema.version,
        "agent_mode": (
            "puct_search" if args.search_simulations > 0 else "greedy_policy"
        ),
        "search_simulations": args.search_simulations,
        "exploration_constant": args.exploration,
        "search_determinizations": (
            args.search_determinizations if args.search_simulations > 0 else None
        ),
        "inference_batch_size": (
            args.inference_batch_size if args.search_simulations > 0 else None
        ),
        "gate_confidence_unit": "seat_rotated_seed_group",
        "summary": asdict(summary),
        "games": [asdict(result) for result in results],
    }


def play_match(
    engine_module: Any,
    policies: Sequence[EvaluationPolicy],
    candidate_seat: int,
    round_index: int,
    game_seed: int,
    max_actions: int,
    search_simulations: int = 0,
    exploration_constant: float = 1.5,
    search_determinizations: int = 4,
    inference_batch_size: int = 32,
) -> MatchResult:
    game = engine_module.BrassRLGame(len(policies), game_seed)
    actions = 0
    while not game.is_done():
        while game.current_decision_mode() == "shortfall":
            _resolve_shortfall(game)
        if game.is_done():
            break
        actor = game.current_decision_player()
        observation = game.get_observation(actor)
        state_record = game.get_training_state()
        legal_record = game.get_legal_actions()
        if (
            state_record["decision_player"] != actor
            or legal_record["decision_player"] != actor
        ):
            raise RuntimeError("rule engine returned inconsistent decision players")
        if search_simulations > 0:
            search_seed = _decision_seed(game_seed, SEARCH_SEED_STREAM, actions)
            action_index = policies[actor].select_action_with_search(
                game,
                observation,
                state_record,
                legal_record,
                len(policies),
                search_simulations,
                exploration_constant,
                search_determinizations,
                inference_batch_size,
                search_seed,
            )
        else:
            action_index = policies[actor].select_action(
                observation, state_record, legal_record
            )
        game.step_legal_action(action_index)
        actions += 1
        if actions > max_actions:
            raise RuntimeError(f"evaluation game exceeded {max_actions} actions")

    outcome = game.get_outcome()
    shared_win_values = [float(value) for value in outcome["shared_win_values"]]
    candidate_shared_win = shared_win_values[candidate_seat]
    score_delta = multiplayer_score_delta(candidate_shared_win, len(policies))
    return MatchResult(
        round_index=round_index,
        game_seed=game_seed,
        candidate_seat=candidate_seat,
        actions=actions,
        candidate_shared_win=candidate_shared_win,
        candidate_score_delta=score_delta,
        candidate_victory_point_margin=int(
            outcome["victory_point_margins"][candidate_seat]
        ),
        candidate_placement=int(outcome["placements"][candidate_seat]),
        candidate_is_official_winner=candidate_seat in outcome["official_winners"],
        official_winners=tuple(int(value) for value in outcome["official_winners"]),
        victory_points=tuple(int(value) for value in outcome["victory_points"]),
    )


def multiplayer_score_delta(candidate_shared_win: float, num_players: int) -> float:
    if num_players < 2:
        raise ValueError("num_players must be at least two")
    if not 0.0 <= candidate_shared_win <= 1.0:
        raise ValueError("candidate_shared_win must be in [0, 1]")
    average_champion_shared_win = (1.0 - candidate_shared_win) / (num_players - 1)
    return candidate_shared_win - average_champion_shared_win


def _select_search_action(
    report: dict,
    expected_action_keys: tuple[str, ...],
    expected_model_id: str,
    expected_method: str,
    expected_simulations: int,
) -> int:
    if report.get("method") != expected_method:
        raise RuntimeError(f"unexpected search method {report.get('method')!r}")
    if report.get("model_id") != expected_model_id:
        raise RuntimeError("search report model ID mismatch")
    if report.get("completed_simulations") != expected_simulations:
        raise RuntimeError("search did not complete the requested simulations")
    actions = report.get("actions")
    if not isinstance(actions, list) or len(actions) != len(expected_action_keys):
        raise RuntimeError("search report does not contain the stable action list")
    visit_sum = 0
    for index, (row, expected_key) in enumerate(zip(actions, expected_action_keys)):
        if (
            not isinstance(row, dict)
            or row.get("index") != index
            or row.get("key") != expected_key
        ):
            raise RuntimeError("search report action order changed")
        visits = row.get("visits")
        if not isinstance(visits, int) or isinstance(visits, bool) or visits < 0:
            raise RuntimeError("search report contains an invalid visit count")
        visit_sum += visits
    if visit_sum != expected_simulations:
        raise RuntimeError("search action visits do not sum to completed simulations")

    def ranking_key(index: int) -> tuple[float, float, float, int]:
        row = actions[index]
        value = row.get("estimated_shared_win_rate")
        prior = row.get("policy_probability")
        return (
            float(row["visits"]),
            float(value) if isinstance(value, (int, float)) else -math.inf,
            float(prior) if isinstance(prior, (int, float)) else -math.inf,
            -index,
        )

    return max(range(len(actions)), key=ranking_key)


def _decision_seed(game_seed: int, stream: int, decision_index: int) -> int:
    mixed = (
        game_seed
        ^ stream
        ^ ((decision_index + 1) * 0x9E37_79B9_7F4A_7C15)
    ) & ((1 << 64) - 1)
    return _splitmix64(mixed)


def summarize_gate(
    results: Sequence[MatchResult], promotion_margin: float, minimum_games: int
) -> GateSummary:
    if not results:
        raise ValueError("at least one match result is required")
    deltas = [result.candidate_score_delta for result in results]
    grouped_deltas: dict[tuple[int, int], list[float]] = {}
    for result in results:
        grouped_deltas.setdefault(
            (result.round_index, result.game_seed), []
        ).append(result.candidate_score_delta)
    seed_group_means = [
        statistics.fmean(group) for group in grouped_deltas.values()
    ]
    mean_delta = statistics.fmean(seed_group_means)
    if len(seed_group_means) >= 2:
        standard_error = statistics.stdev(seed_group_means) / math.sqrt(
            len(seed_group_means)
        )
        degrees_of_freedom = len(seed_group_means) - 1
        critical = _student_t_critical_95(degrees_of_freedom)
        lower_bound = mean_delta - critical * standard_error
    else:
        standard_error = None
        lower_bound = None
    promote = (
        len(results) >= minimum_games
        and lower_bound is not None
        and lower_bound > promotion_margin
    )
    return GateSummary(
        games=len(results),
        independent_seed_groups=len(seed_group_means),
        candidate_mean_shared_win=statistics.fmean(
            result.candidate_shared_win for result in results
        ),
        candidate_mean_score_delta=mean_delta,
        score_delta_standard_error=standard_error,
        score_delta_lower_95=lower_bound,
        candidate_mean_victory_point_margin=statistics.fmean(
            result.candidate_victory_point_margin for result in results
        ),
        candidate_first_place_rate=statistics.fmean(
            float(result.candidate_is_official_winner) for result in results
        ),
        promotion_margin=promotion_margin,
        minimum_games=minimum_games,
        promote=promote,
    )


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


def _validate_model_source(
    name: str, checkpoint: str | None, inference_url: str | None
) -> None:
    if (checkpoint is None) == (inference_url is None):
        raise ValueError(
            f"exactly one of --{name} or --{name}-inference-url is required"
        )


def _build_evaluation_policy(
    *,
    checkpoint: str | None,
    inference_url: str | None,
    raw_schema: dict,
    schema: FeatureSchema,
    device: str,
    timeout_seconds: float,
        strategy_prior_strength: float = 0.0,
        strategy_prior_version: str = STRATEGY_PRIOR_VERSION,
        score_utility_weight: float = 0.0,
        final_vp_utility_weight: float = 0.0,
) -> EvaluationPolicy:
    if inference_url is not None:
        from .remote_inference import RemoteInferenceEvaluator

        evaluator: PolicyValueEvaluator = RemoteInferenceEvaluator(
            inference_url,
            raw_schema,
            timeout_seconds=timeout_seconds,
        )
        source = evaluator.base_url
        source_kind = "remote_inference"
    else:
        if checkpoint is None:
            raise RuntimeError("checkpoint model source is missing")
        from .inference import CheckpointEvaluator

        evaluator = CheckpointEvaluator(checkpoint, device)
        source = str(Path(checkpoint).resolve())
        source_kind = "checkpoint"
    return EvaluationPolicy(
        evaluator,
        schema,
        source,
        source_kind,
        strategy_prior_strength,
        strategy_prior_version,
        score_utility_weight,
        final_vp_utility_weight,
    )


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom < len(T_CRITICAL_95):
        return T_CRITICAL_95[degrees_of_freedom]

    # Cornish-Fisher expansion for the two-sided 95% Student-t quantile.
    # Starting above df=30 keeps the error below the precision needed by the gate.
    z = STANDARD_NORMAL_975
    df = float(degrees_of_freedom)
    return (
        z + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
        + (
            79.0 * z**9
            + 776.0 * z**7
            + 1482.0 * z**5
            - 1920.0 * z**3
            - 945.0 * z
        )
        / (92160.0 * df**4)
    )


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


if __name__ == "__main__":
    main()
