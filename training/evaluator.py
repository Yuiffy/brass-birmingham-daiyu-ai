from __future__ import annotations

from typing import Protocol

from .schema import FeatureSchema


class PolicyValuePredictionLike(Protocol):
    model_id: str
    checkpoint_step: int
    action_keys: tuple[str, ...]
    policy_probabilities: tuple[float, ...]
    shared_win_rate: float
    victory_point_margin: float


class PolicyValueEvaluator(Protocol):
    schema: FeatureSchema
    model_id: str
    checkpoint_step: int

    def predict(
        self,
        state_record: dict,
        legal_record: dict,
        request_schema: FeatureSchema | None = None,
    ) -> PolicyValuePredictionLike: ...

    def predict_batch(
        self,
        state_records: list[dict],
        legal_records: list[dict],
        request_schema: FeatureSchema | None = None,
    ) -> tuple[PolicyValuePredictionLike, ...]: ...
