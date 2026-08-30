from __future__ import annotations

import math
from typing import Any, Sequence

from .evaluator import PolicyValueEvaluator, PolicyValuePredictionLike
from .policy_normalization import rebalance_card_choice_groups
from .schema import FeatureSchema

BATCHED_NEURAL_PUCT_METHOD = "determinized_batched_neural_puct"


def run_batched_neural_puct(
    game: Any,
    evaluator: PolicyValueEvaluator,
    schema: FeatureSchema,
    root_prediction: PolicyValuePredictionLike,
    *,
    simulations: int,
    search_seed: int,
    exploration_constant: float,
    determinizations: int,
    inference_batch_size: int,
    root_policy_probabilities: Sequence[float] | None = None,
    group_leaf_card_choices: bool = False,
    score_utility_weight: float = 0.0,
    final_vp_utility_weight: float = 0.0,
) -> dict:
    if not 1 <= inference_batch_size <= 256:
        raise ValueError("inference_batch_size must be between 1 and 256")
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
    policy_probabilities = tuple(
        root_prediction.policy_probabilities
        if root_policy_probabilities is None
        else root_policy_probabilities
    )
    action_keys = getattr(root_prediction, "action_keys", None)
    if action_keys is not None and len(policy_probabilities) != len(action_keys):
        raise ValueError("root policy probabilities must cover every legal action")
    search = game.start_batched_neural_search(
        list(policy_probabilities),
        root_prediction.model_id,
        root_prediction.shared_win_rate,
        root_prediction.victory_point_margin,
        simulations,
        search_seed,
        exploration_constant,
        determinizations,
        score_utility_weight,
        root_prediction.actor_victory_points,
        group_leaf_card_choices,
        final_vp_utility_weight,
    )

    while not search.is_complete():
        raw_batch = search.next_inference_batch(inference_batch_size)
        positions = raw_batch.get("positions") if isinstance(raw_batch, dict) else None
        if not isinstance(positions, list):
            raise RuntimeError("neural search returned an invalid leaf batch")
        if not positions:
            if raw_batch.get("is_complete") is True and search.is_complete():
                break
            raise RuntimeError("neural search returned an empty incomplete leaf batch")
        request_ids: list[int] = []
        states: list[dict] = []
        legal_actions: list[dict] = []
        for batch_index, position in enumerate(positions):
            if not isinstance(position, dict):
                raise RuntimeError(f"neural leaf {batch_index} is not an object")
            request_id = position.get("request_id")
            state = position.get("state")
            actions = position.get("legal_actions")
            if (
                not isinstance(request_id, int)
                or isinstance(request_id, bool)
                or request_id < 0
            ):
                raise RuntimeError(f"neural leaf {batch_index} has an invalid request ID")
            if not isinstance(state, dict) or not isinstance(actions, dict):
                raise RuntimeError(f"neural leaf {batch_index} omitted model features")
            request_ids.append(request_id)
            states.append(state)
            legal_actions.append(actions)

        predictions = evaluator.predict_batch(states, legal_actions, schema)
        if len(predictions) != len(positions):
            raise RuntimeError("batched model predictions do not match neural leaves")
        if any(prediction.model_id != root_prediction.model_id for prediction in predictions):
            raise RuntimeError("root and leaf model IDs disagree")
        leaf_probabilities = [
            (
                rebalance_card_choice_groups(
                    prediction.action_keys,
                    prediction.policy_probabilities,
                )
                if group_leaf_card_choices
                else prediction.policy_probabilities
            )
            for prediction in predictions
        ]
        search.submit_inference_batch(
            request_ids,
            [list(prediction.action_keys) for prediction in predictions],
            [list(probabilities) for probabilities in leaf_probabilities],
            [prediction.shared_win_rate for prediction in predictions],
            [prediction.victory_point_margin for prediction in predictions],
            root_prediction.model_id,
            [prediction.actor_victory_points for prediction in predictions],
        )

    report = search.finish()
    if report.get("method") != BATCHED_NEURAL_PUCT_METHOD:
        raise RuntimeError(f"unexpected neural search method {report.get('method')!r}")
    if report.get("model_id") != root_prediction.model_id:
        raise RuntimeError("neural search report model ID mismatch")
    if report.get("completed_simulations") != simulations:
        raise RuntimeError("neural search did not complete all requested simulations")
    max_depth = report.get("max_search_depth")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth <= 0:
        raise RuntimeError("neural search report omitted a positive search depth")
    return report
