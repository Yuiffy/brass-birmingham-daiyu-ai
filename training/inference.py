from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .data import FeatureSchema, TrainingBatch
from .model import segmented_policy_probabilities


@dataclass(frozen=True)
class PolicyValuePrediction:
    model_id: str
    checkpoint_step: int
    action_keys: tuple[str, ...]
    policy_probabilities: tuple[float, ...]
    shared_win_rate: float
    victory_point_margin: float

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "checkpoint_step": self.checkpoint_step,
            "action_keys": list(self.action_keys),
            "policy_probabilities": list(self.policy_probabilities),
            "shared_win_rate": self.shared_win_rate,
            "victory_point_margin": self.victory_point_margin,
        }


@dataclass(frozen=True)
class BatchValuePrediction:
    model_id: str
    checkpoint_step: int
    shared_win_rates: tuple[float, ...]
    victory_point_margins: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "checkpoint_step": self.checkpoint_step,
            "shared_win_rates": list(self.shared_win_rates),
            "victory_point_margins": list(self.victory_point_margins),
        }


@dataclass(frozen=True)
class RootActionValuePrediction:
    model_id: str
    checkpoint_step: int
    action_keys: tuple[str, ...]
    shared_win_rates: tuple[float, ...]
    victory_point_margins: tuple[float, ...]
    shared_win_standard_errors: tuple[float | None, ...]
    sample_counts: tuple[int, ...]


class CheckpointEvaluator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(checkpoint_path).resolve()
        self.device = _resolve_device(device)
        self.model, self.payload = load_model_checkpoint(
            self.path,
            expected_schema=None,
            map_location=self.device,
        )
        metadata = self.payload["metadata"]
        self.schema = _schema_from_checkpoint_metadata(metadata)
        if self.model.config.state_dim != self.schema.state_dim:
            raise ValueError("checkpoint state_dim disagrees with its model config")
        if self.model.config.action_dim != self.schema.action_dim:
            raise ValueError("checkpoint action_dim disagrees with its model config")
        self.model_id = f"sha256:{_sha256_file(self.path)}"
        self.checkpoint_step = int(metadata.get("training", {}).get("global_step", 0))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        state_record: dict,
        legal_record: dict,
        request_schema: FeatureSchema | None = None,
    ) -> PolicyValuePrediction:
        return self.predict_batch(
            [state_record], [legal_record], request_schema=request_schema
        )[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        state_records: list[dict],
        legal_records: list[dict],
        request_schema: FeatureSchema | None = None,
    ) -> tuple[PolicyValuePrediction, ...]:
        if request_schema is not None:
            self.schema.assert_compatible(request_schema, "inference request")
        batch = build_inference_batch_many(
            state_records, legal_records, self.schema
        ).to(self.device)
        output = self.model(batch)
        flat_probabilities = segmented_policy_probabilities(
            output.policy_logits, batch.action_counts
        )
        shared_win_rates = torch.sigmoid(output.shared_win_logits).cpu()
        victory_point_margins = (
            output.victory_point_margin_normalized
            * self.model.config.vp_margin_scale
        ).cpu()
        flat_probability_values = tuple(
            float(value) for value in flat_probabilities.cpu()
        )
        predictions: list[PolicyValuePrediction] = []
        action_offset = 0
        for batch_index, action_count in enumerate(batch.action_counts.cpu().tolist()):
            next_offset = action_offset + action_count
            probability_values = flat_probability_values[action_offset:next_offset]
            shared_win_rate = float(shared_win_rates[batch_index])
            victory_point_margin = float(victory_point_margins[batch_index])
            if not math.isclose(sum(probability_values), 1.0, abs_tol=1e-5):
                raise RuntimeError(
                    f"model policy probabilities for batch row {batch_index} "
                    "do not sum to one"
                )
            if (
                not math.isfinite(shared_win_rate)
                or not 0.0 <= shared_win_rate <= 1.0
            ):
                raise RuntimeError(
                    f"model produced an invalid shared-win estimate for batch row {batch_index}"
                )
            if not math.isfinite(victory_point_margin):
                raise RuntimeError(
                    f"model produced an invalid VP-margin estimate for batch row {batch_index}"
                )
            predictions.append(
                PolicyValuePrediction(
                    model_id=self.model_id,
                    checkpoint_step=self.checkpoint_step,
                    action_keys=batch.action_keys[batch_index],
                    policy_probabilities=probability_values,
                    shared_win_rate=shared_win_rate,
                    victory_point_margin=victory_point_margin,
                )
            )
            action_offset = next_offset
        if action_offset != len(flat_probability_values):
            raise RuntimeError("batched policy action counts do not cover model output")
        return tuple(predictions)

    def select_action(self, state_record: dict, legal_record: dict) -> int:
        prediction = self.predict(state_record, legal_record)
        return max(
            range(len(prediction.policy_probabilities)),
            key=prediction.policy_probabilities.__getitem__,
        )

    @torch.inference_mode()
    def predict_values(
        self,
        state_records: list[dict],
        request_schema: FeatureSchema | None = None,
    ) -> BatchValuePrediction:
        if request_schema is not None:
            self.schema.assert_compatible(request_schema, "value inference request")
        states = build_value_state_batch(state_records, self.schema).to(self.device)
        shared_win_logits, normalized_margins = self.model.forward_values(states)
        shared_win_rates = tuple(
            float(value) for value in torch.sigmoid(shared_win_logits).cpu()
        )
        victory_point_margins = tuple(
            float(value)
            for value in (
                normalized_margins * self.model.config.vp_margin_scale
            ).cpu()
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in shared_win_rates
        ):
            raise RuntimeError("model produced an invalid batched shared-win estimate")
        if any(not math.isfinite(value) for value in victory_point_margins):
            raise RuntimeError("model produced an invalid batched VP-margin estimate")
        return BatchValuePrediction(
            model_id=self.model_id,
            checkpoint_step=self.checkpoint_step,
            shared_win_rates=shared_win_rates,
            victory_point_margins=victory_point_margins,
        )

    def predict_successor_action_values(
        self,
        successor_batch: dict,
        request_schema: FeatureSchema | None = None,
    ) -> RootActionValuePrediction:
        states = successor_batch.get("states")
        if not isinstance(states, list):
            raise ValueError("successor batch states must be an array")
        if states:
            prediction = self.predict_values(states, request_schema)
        else:
            if request_schema is not None:
                self.schema.assert_compatible(
                    request_schema, "terminal successor value request"
                )
            prediction = BatchValuePrediction(
                model_id=self.model_id,
                checkpoint_step=self.checkpoint_step,
                shared_win_rates=(),
                victory_point_margins=(),
            )
        return aggregate_two_player_successor_values(successor_batch, prediction)


def build_inference_batch(
    state_record: dict,
    legal_record: dict,
    schema: FeatureSchema,
) -> TrainingBatch:
    return build_inference_batch_many([state_record], [legal_record], schema)


def build_inference_batch_many(
    state_records: list[dict],
    legal_records: list[dict],
    schema: FeatureSchema,
) -> TrainingBatch:
    if not isinstance(state_records, list) or not state_records:
        raise ValueError("batched inference requires at least one state")
    if not isinstance(legal_records, list) or len(legal_records) != len(state_records):
        raise ValueError("batched inference states and legal actions must have equal lengths")

    states: list[list[float]] = []
    flat_features: list[int] = []
    offsets = [0]
    action_batch_indices: list[int] = []
    action_counts: list[int] = []
    batched_action_keys: list[tuple[str, ...]] = []
    for batch_index, (state_record, legal_record) in enumerate(
        zip(state_records, legal_records, strict=True)
    ):
        states.append(
            _validated_state_features(
                state_record, schema, f"inference state {batch_index}"
            )
        )
        if not isinstance(legal_record, dict):
            raise ValueError(f"inference legal actions {batch_index} must be an object")
        if legal_record.get("feature_version") != schema.version:
            raise ValueError(
                f"inference legal actions {batch_index} feature version mismatch"
            )
        actions = legal_record.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"inference position {batch_index} has no legal actions")
        action_counts.append(len(actions))
        action_keys: list[str] = []
        seen_keys: set[str] = set()
        for expected_index, action in enumerate(actions):
            if not isinstance(action, dict) or action.get("index") != expected_index:
                raise ValueError(
                    f"inference actions for batch row {batch_index} are not in stable index order"
                )
            key = action.get("key")
            if not isinstance(key, str) or not key or key in seen_keys:
                raise ValueError(
                    f"inference action keys for batch row {batch_index} "
                    "must be non-empty and unique"
                )
            seen_keys.add(key)
            indices = action.get("feature_indices")
            if not isinstance(indices, list) or not indices:
                raise ValueError(
                    f"inference action {expected_index} in batch row {batch_index} "
                    "has no sparse features"
                )
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= schema.action_dim
                for index in indices
            ):
                raise ValueError(
                    f"inference action {expected_index} in batch row {batch_index} "
                    "contains an invalid sparse feature"
                )
            if any(left >= right for left, right in zip(indices, indices[1:])):
                raise ValueError(
                    f"inference action {expected_index} in batch row {batch_index} "
                    "features must be sorted and unique"
                )
            flat_features.extend(indices)
            offsets.append(len(flat_features))
            action_batch_indices.append(batch_index)
            action_keys.append(key)
        batched_action_keys.append(tuple(action_keys))

    total_actions = sum(action_counts)
    return TrainingBatch(
        states=torch.tensor(states, dtype=torch.float32),
        action_feature_indices=torch.tensor(flat_features, dtype=torch.long),
        action_feature_offsets=torch.tensor(offsets, dtype=torch.long),
        action_batch_indices=torch.tensor(action_batch_indices, dtype=torch.long),
        action_counts=torch.tensor(action_counts, dtype=torch.long),
        policy_targets=torch.zeros(total_actions, dtype=torch.float32),
        shared_win_targets=torch.zeros(len(states), dtype=torch.float32),
        victory_point_margin_targets=torch.zeros(len(states), dtype=torch.float32),
        action_keys=tuple(batched_action_keys),
        record_ids=tuple((0, batch_index) for batch_index in range(len(states))),
    )


def build_value_state_batch(
    state_records: list[dict], schema: FeatureSchema
) -> torch.Tensor:
    if not isinstance(state_records, list) or not state_records:
        raise ValueError("value inference requires a non-empty states list")
    rows = [
        _validated_state_features(record, schema, f"value state {index}")
        for index, record in enumerate(state_records)
    ]
    return torch.tensor(rows, dtype=torch.float32)


def aggregate_two_player_successor_values(
    successor_batch: dict,
    prediction: BatchValuePrediction,
) -> RootActionValuePrediction:
    if not isinstance(successor_batch, dict):
        raise ValueError("successor batch must be an object")
    if successor_batch.get("num_players") != 2:
        raise ValueError("scalar successor values require exactly two players")
    root_player = successor_batch.get("root_player")
    if not isinstance(root_player, int) or isinstance(root_player, bool) or root_player not in (0, 1):
        raise ValueError("successor batch root_player is invalid")
    action_keys = successor_batch.get("action_keys")
    if (
        not isinstance(action_keys, list)
        or not action_keys
        or any(not isinstance(key, str) or not key for key in action_keys)
        or len(set(action_keys)) != len(action_keys)
    ):
        raise ValueError("successor batch action_keys are invalid")
    determinizations = successor_batch.get("determinizations_per_action")
    if (
        not isinstance(determinizations, int)
        or isinstance(determinizations, bool)
        or determinizations <= 0
    ):
        raise ValueError("successor batch determinization count is invalid")
    states = successor_batch.get("states")
    samples = successor_batch.get("samples")
    if not isinstance(states, list) or not isinstance(samples, list):
        raise ValueError("successor batch states and samples must be arrays")
    if len(prediction.shared_win_rates) != len(states) or len(
        prediction.victory_point_margins
    ) != len(states):
        raise ValueError("batched predictions do not match successor states")
    if len(samples) != len(action_keys) * determinizations:
        raise ValueError("successor batch does not contain every action sample")

    state_observers: list[int] = []
    for expected_index, state in enumerate(states):
        if not isinstance(state, dict) or state.get("index") != expected_index:
            raise ValueError("successor states are not in stable index order")
        observer = state.get("observer_idx")
        if not isinstance(observer, int) or isinstance(observer, bool) or observer not in (0, 1):
            raise ValueError("successor state observer is invalid")
        state_observers.append(observer)

    win_values: list[list[float]] = [[] for _ in action_keys]
    margin_values: list[list[float]] = [[] for _ in action_keys]
    referenced_states: set[int] = set()
    for flat_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("successor sample must be an object")
        action_index = flat_index // determinizations
        sample_index = flat_index % determinizations
        if (
            sample.get("action_index") != action_index
            or sample.get("sample_index") != sample_index
            or sample.get("action_key") != action_keys[action_index]
        ):
            raise ValueError("successor samples are not in stable action order")
        evaluation_player = sample.get("evaluation_player")
        if (
            not isinstance(evaluation_player, int)
            or isinstance(evaluation_player, bool)
            or evaluation_player not in (0, 1)
        ):
            raise ValueError("successor sample evaluation player is invalid")
        state_index = sample.get("state_index")
        if state_index is None:
            root_win = _finite_number(
                sample.get("terminal_root_shared_win_rate"),
                "terminal root shared-win value",
            )
            root_margin = _finite_number(
                sample.get("terminal_root_victory_point_margin"),
                "terminal root VP-margin value",
            )
            if not 0.0 <= root_win <= 1.0:
                raise ValueError("terminal root shared-win value must be in [0, 1]")
        else:
            if (
                not isinstance(state_index, int)
                or isinstance(state_index, bool)
                or state_index < 0
                or state_index >= len(states)
                or state_index in referenced_states
            ):
                raise ValueError("successor sample state index is invalid or duplicated")
            referenced_states.add(state_index)
            if state_observers[state_index] != evaluation_player:
                raise ValueError("successor state observer and sample player disagree")
            raw_win = prediction.shared_win_rates[state_index]
            raw_margin = prediction.victory_point_margins[state_index]
            if evaluation_player == root_player:
                root_win, root_margin = raw_win, raw_margin
            else:
                root_win, root_margin = 1.0 - raw_win, -raw_margin
        win_values[action_index].append(root_win)
        margin_values[action_index].append(root_margin)

    if referenced_states != set(range(len(states))):
        raise ValueError("successor batch contains unreferenced value states")
    shared_win_rates = tuple(sum(values) / len(values) for values in win_values)
    victory_point_margins = tuple(sum(values) / len(values) for values in margin_values)
    standard_errors = tuple(_sample_standard_error(values) for values in win_values)
    sample_counts = tuple(len(values) for values in win_values)
    return RootActionValuePrediction(
        model_id=prediction.model_id,
        checkpoint_step=prediction.checkpoint_step,
        action_keys=tuple(action_keys),
        shared_win_rates=shared_win_rates,
        victory_point_margins=victory_point_margins,
        shared_win_standard_errors=standard_errors,
        sample_counts=sample_counts,
    )


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _sample_standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    corrected_sum = sum((value - mean) ** 2 for value in values)
    sample_variance = corrected_sum / (len(values) - 1)
    return math.sqrt(sample_variance / len(values))


def _validated_state_features(
    state_record: dict, schema: FeatureSchema, context: str
) -> list[float]:
    if not isinstance(state_record, dict):
        raise ValueError(f"{context} must be an object")
    if state_record.get("feature_version") != schema.version:
        raise ValueError(f"{context} feature version mismatch")
    state_features = state_record.get("features")
    if not isinstance(state_features, list) or len(state_features) != schema.state_dim:
        raise ValueError(f"invalid {context} features")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in state_features
    ):
        raise ValueError(f"{context} contains a non-finite numeric value")
    return [float(value) for value in state_features]


def _schema_from_checkpoint_metadata(metadata: dict) -> FeatureSchema:
    version = metadata.get("feature_version")
    state_dim = metadata.get("state_dim")
    action_dim = metadata.get("action_dim")
    signature = metadata.get("feature_schema_signature")
    if not isinstance(version, int) or version <= 0:
        raise ValueError("checkpoint feature_version is invalid")
    if not isinstance(state_dim, int) or state_dim <= 0:
        raise ValueError("checkpoint state_dim is invalid")
    if not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError("checkpoint action_dim is invalid")
    if not isinstance(signature, str) or not signature:
        raise ValueError("checkpoint feature schema signature is missing")
    return FeatureSchema(version, state_dim, action_dim, signature)


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, str) and value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
