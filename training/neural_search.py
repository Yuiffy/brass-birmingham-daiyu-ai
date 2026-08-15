from __future__ import annotations

from typing import Any

from .evaluator import PolicyValueEvaluator, PolicyValuePredictionLike
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
) -> dict:
    if not 1 <= inference_batch_size <= 256:
        raise ValueError("inference_batch_size must be between 1 and 256")
    search = game.start_batched_neural_search(
        list(root_prediction.policy_probabilities),
        root_prediction.model_id,
        root_prediction.shared_win_rate,
        root_prediction.victory_point_margin,
        simulations,
        search_seed,
        exploration_constant,
        determinizations,
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
        search.submit_inference_batch(
            request_ids,
            [list(prediction.action_keys) for prediction in predictions],
            [list(prediction.policy_probabilities) for prediction in predictions],
            [prediction.shared_win_rate for prediction in predictions],
            [prediction.victory_point_margin for prediction in predictions],
            root_prediction.model_id,
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
