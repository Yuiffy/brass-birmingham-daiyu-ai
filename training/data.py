from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .schema import SELF_PLAY_FORMAT, SELF_PLAY_FORMAT_VERSION, FeatureSchema


POLICY_TARGET_FIELD = "policy_target"
SEARCH_POLICY_TARGET_FIELD = "search_policy_target"
POLICY_TARGET_EXPONENT_DEFAULT = 1.0
SUPPORTED_POLICY_TARGET_FIELDS = (
    POLICY_TARGET_FIELD,
    SEARCH_POLICY_TARGET_FIELD,
)

ACTOR_VP_TARGET_ABSOLUTE = "absolute_final_vp"
ACTOR_VP_TARGET_REMAINING = "remaining_vp"
ACTOR_VP_TARGET_PHASE_REMAINING = "phase_conditioned_remaining_vp"
ACTOR_VP_RAILROAD_SCALE_DEFAULT = 100.0
# The v1 global block starts with Canal, Railroad, and GameEnd one-hot values.
ACTOR_VP_PHASE_RAILROAD_FEATURE_INDEX = 1
SUPPORTED_ACTOR_VP_TARGET_MODES = (
    ACTOR_VP_TARGET_ABSOLUTE,
    ACTOR_VP_TARGET_REMAINING,
    ACTOR_VP_TARGET_PHASE_REMAINING,
)
ACTOR_VP_TARGET_REMAINING_MODES = (
    ACTOR_VP_TARGET_REMAINING,
    ACTOR_VP_TARGET_PHASE_REMAINING,
)


@dataclass(frozen=True)
class PositionLocator:
    path: Path
    offset: int
    line_number: int


@dataclass
class PositionExample:
    state: torch.Tensor
    action_features: tuple[torch.Tensor, ...]
    action_keys: tuple[str, ...]
    policy_target: torch.Tensor
    shared_win_target: torch.Tensor
    victory_point_margin_target: torch.Tensor
    actor_victory_points_target: torch.Tensor
    game_index: int
    position_index: int
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


@dataclass
class TrainingBatch:
    states: torch.Tensor
    action_feature_indices: torch.Tensor
    action_feature_offsets: torch.Tensor
    action_batch_indices: torch.Tensor
    action_counts: torch.Tensor
    policy_targets: torch.Tensor
    shared_win_targets: torch.Tensor
    victory_point_margin_targets: torch.Tensor
    actor_victory_points_targets: torch.Tensor
    action_keys: tuple[tuple[str, ...], ...]
    record_ids: tuple[tuple[int, int], ...]
    # Optional so inference-only batches remain source-compatible with older callers.
    policy_loss_weights: torch.Tensor | None = None
    value_loss_weights: torch.Tensor | None = None

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "TrainingBatch":
        return TrainingBatch(
            states=self.states.to(device, non_blocking=non_blocking),
            action_feature_indices=self.action_feature_indices.to(
                device, non_blocking=non_blocking
            ),
            action_feature_offsets=self.action_feature_offsets.to(
                device, non_blocking=non_blocking
            ),
            action_batch_indices=self.action_batch_indices.to(
                device, non_blocking=non_blocking
            ),
            action_counts=self.action_counts.to(device, non_blocking=non_blocking),
            policy_targets=self.policy_targets.to(device, non_blocking=non_blocking),
            shared_win_targets=self.shared_win_targets.to(
                device, non_blocking=non_blocking
            ),
            victory_point_margin_targets=self.victory_point_margin_targets.to(
                device, non_blocking=non_blocking
            ),
            actor_victory_points_targets=self.actor_victory_points_targets.to(
                device, non_blocking=non_blocking
            ),
            action_keys=self.action_keys,
            record_ids=self.record_ids,
            policy_loss_weights=(
                self.policy_loss_weights.to(device, non_blocking=non_blocking)
                if self.policy_loss_weights is not None
                else None
            ),
            value_loss_weights=(
                self.value_loss_weights.to(device, non_blocking=non_blocking)
                if self.value_loss_weights is not None
                else None
            ),
        )


class SelfPlayDataset(Dataset[PositionExample]):
    """Offset-indexed JSONL dataset with strict schema and record validation."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        expected_schema: FeatureSchema | None = None,
        allow_mixed_engine_revisions: bool = False,
        policy_target_field: str = POLICY_TARGET_FIELD,
        search_policy_target_mix: float = 0.0,
        policy_target_exponent: float = POLICY_TARGET_EXPONENT_DEFAULT,
        actor_vp_target_mode: str = ACTOR_VP_TARGET_ABSOLUTE,
        sampling_weights: Mapping[str | Path, float] | None = None,
    ) -> None:
        if not shard_paths:
            raise ValueError("at least one self-play shard is required")
        if policy_target_field not in SUPPORTED_POLICY_TARGET_FIELDS:
            raise ValueError(
                "unsupported policy target field "
                f"{policy_target_field!r}; expected one of "
                f"{SUPPORTED_POLICY_TARGET_FIELDS}"
            )
        if (
            not math.isfinite(search_policy_target_mix)
            or not 0.0 <= search_policy_target_mix <= 1.0
        ):
            raise ValueError("search_policy_target_mix must be between 0 and 1")
        if (
            policy_target_field == SEARCH_POLICY_TARGET_FIELD
            and search_policy_target_mix != 0.0
        ):
            raise ValueError(
                "search_policy_target_mix requires policy_target_field='policy_target'"
            )
        policy_target_exponent = validate_policy_target_exponent(
            policy_target_exponent
        )
        if actor_vp_target_mode not in SUPPORTED_ACTOR_VP_TARGET_MODES:
            raise ValueError(
                "unsupported actor_vp_target_mode "
                f"{actor_vp_target_mode!r}; expected one of "
                f"{SUPPORTED_ACTOR_VP_TARGET_MODES}"
            )
        self.shard_paths = tuple(Path(path).resolve() for path in shard_paths)
        self.schema = expected_schema
        self.metadata: list[dict] = []
        self._shard_value_target_usable: dict[Path, bool] = {}
        self.engine_revisions: set[str] = set()
        self.game_seeds: set[int] = set()
        self.game_victory_points: dict[tuple[Path, int], tuple[int, ...]] = {}
        self.allow_mixed_engine_revisions = allow_mixed_engine_revisions
        self.policy_target_field = policy_target_field
        self.search_policy_target_mix = search_policy_target_mix
        self.policy_target_exponent = policy_target_exponent
        self.actor_vp_target_mode = actor_vp_target_mode
        self._positions: list[PositionLocator] = []
        self._handles: dict[Path, BinaryIO] = {}
        self._sampling_weights: torch.Tensor | None = None
        self.sampling_metadata: dict[str, object] = {}

        for path in self.shard_paths:
            self._scan_shard(path)
        if self.schema is None:
            raise ValueError("no metadata record found in self-play shards")
        if not self._positions:
            raise ValueError("self-play shards contain no position records")
        if sampling_weights is not None:
            self.configure_sampling_weights(sampling_weights)

    def _scan_shard(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"self-play shard does not exist: {path}")
        found_metadata = False
        with path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                record = _decode_json_line(line, path, line_number)
                record_type = record.get("record_type")
                if not found_metadata:
                    if record_type != "metadata":
                        raise ValueError(
                            f"{path}:{line_number}: first record must be metadata"
                        )
                    schema = FeatureSchema.from_metadata(record)
                    if self.schema is None:
                        self.schema = schema
                    else:
                        self.schema.assert_compatible(schema, str(path))
                    engine_revision = record.get("engine_revision")
                    if (
                        not isinstance(engine_revision, str)
                        or not engine_revision.strip()
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: metadata.engine_revision is required"
                        )
                    self.engine_revisions.add(engine_revision)
                    if (
                        len(self.engine_revisions) > 1
                        and not self.allow_mixed_engine_revisions
                    ):
                        raise ValueError(
                            "self-play shards use different engine revisions; pass "
                            "allow_mixed_engine_revisions=True only after auditing compatibility"
                        )
                    self.metadata.append(record)
                    raw_value_usable = record.get("value_target_usable")
                    # Human replays must explicitly opt in to terminal value
                    # labels.  A legacy human shard without the marker is
                    # treated as policy-only rather than trusted silently.
                    if record.get("source_type") == "human_replay":
                        self._shard_value_target_usable[path] = raw_value_usable is True
                    else:
                        self._shard_value_target_usable[path] = (
                            raw_value_usable is not False
                        )
                    found_metadata = True
                elif record_type == "position":
                    self._positions.append(PositionLocator(path, offset, line_number))
                elif record_type == "game":
                    game_index = _non_negative_int(
                        record.get("game_index"), f"{path}:{line_number}: game_index"
                    )
                    raw_victory_points = record.get("victory_points")
                    if not isinstance(raw_victory_points, list) or not raw_victory_points:
                        raise ValueError(
                            f"{path}:{line_number}: game.victory_points is required"
                        )
                    victory_points = tuple(
                        _non_negative_int(
                            value, f"{path}:{line_number}: victory_points"
                        )
                        for value in raw_victory_points
                    )
                    game_key = (path, game_index)
                    if game_key in self.game_victory_points:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate game index {game_index}"
                        )
                    self.game_victory_points[game_key] = victory_points
                    game_seed = record.get("game_seed")
                    if game_seed is not None:
                        self.game_seeds.add(
                            _non_negative_int(
                                game_seed, f"{path}:{line_number}: game_seed"
                            )
                        )
                elif record_type == "metadata":
                    raise ValueError(f"{path}:{line_number}: duplicate metadata record")
                else:
                    raise ValueError(
                        f"{path}:{line_number}: unknown record_type {record_type!r}"
                    )
        if not found_metadata:
            raise ValueError(f"{path}: shard is empty or missing metadata")

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def sampling_weights(self) -> torch.Tensor | None:
        """Per-position weights for an optional replacement sampler.

        The default is ``None`` so existing callers keep ordinary shuffled
        epochs. Training can opt into a deterministic source mixture by
        configuring one weight for every shard.
        """
        return self._sampling_weights

    @property
    def shard_position_counts(self) -> dict[Path, int]:
        """Return the number of indexed positions contributed by each shard."""
        counts = {path: 0 for path in self.shard_paths}
        for locator in self._positions:
            counts[locator.path] = counts.get(locator.path, 0) + 1
        return counts

    def configure_sampling_weights(
        self,
        weights: Mapping[str | Path, float],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Configure source-mixture weights without changing decoded targets."""
        normalized: dict[Path, float] = {}
        for raw_path, raw_weight in weights.items():
            path = Path(raw_path).resolve()
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or float(raw_weight) < 0.0
            ):
                raise ValueError(
                    f"sampling weight for {path} must be finite and non-negative"
                )
            if path in normalized:
                raise ValueError(f"duplicate sampling-weight path: {path}")
            normalized[path] = float(raw_weight)

        expected = set(self.shard_paths)
        actual = set(normalized)
        missing = expected - actual
        unknown = actual - expected
        if missing or unknown:
            details = []
            if missing:
                details.append(
                    "missing " + ", ".join(str(path) for path in sorted(missing))
                )
            if unknown:
                details.append(
                    "unknown " + ", ".join(str(path) for path in sorted(unknown))
                )
            raise ValueError(
                "sampling weights must cover exactly the dataset shards ("
                + "; ".join(details)
                + ")"
            )

        position_weights = [normalized[locator.path] for locator in self._positions]
        if not any(weight > 0.0 for weight in position_weights):
            raise ValueError(
                "sampling weights must contain at least one positive position"
            )
        self._sampling_weights = torch.tensor(position_weights, dtype=torch.double)
        self.sampling_metadata = dict(metadata or {})
        self.sampling_metadata["enabled"] = True

    def __getitem__(self, index: int) -> PositionExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        locator = self._positions[index]
        handle = self._handle_for(locator.path)
        handle.seek(locator.offset)
        line = handle.readline()
        record = _decode_json_line(line, locator.path, locator.line_number)
        return self._decode_position(record, locator)

    def _handle_for(self, path: Path) -> BinaryIO:
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = path.open("rb")
            self._handles[path] = handle
        return handle

    def _decode_position(
        self, record: dict, locator: PositionLocator
    ) -> PositionExample:
        assert self.schema is not None
        context = f"{locator.path}:{locator.line_number}"
        if record.get("record_type") != "position":
            raise ValueError(f"{context}: indexed record is not a position")
        if record.get("format_version") != SELF_PLAY_FORMAT_VERSION:
            raise ValueError(f"{context}: position format version mismatch")
        if record.get("feature_version") != self.schema.version:
            raise ValueError(f"{context}: position feature version mismatch")

        state_values = record.get("state_features")
        if (
            not isinstance(state_values, list)
            or len(state_values) != self.schema.state_dim
        ):
            actual = len(state_values) if isinstance(state_values, list) else None
            raise ValueError(
                f"{context}: state feature length {actual}, expected {self.schema.state_dim}"
            )
        state = torch.tensor(state_values, dtype=torch.float32)
        if not torch.isfinite(state).all().item():
            raise ValueError(f"{context}: state contains non-finite values")

        raw_actions = record.get("legal_actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError(f"{context}: legal_actions must be a non-empty list")
        action_features: list[torch.Tensor] = []
        action_keys: list[str] = []
        visits: list[int] = []
        serialized_targets: list[float] = []
        serialized_search_targets: list[float] = []
        for expected_index, action in enumerate(raw_actions):
            if not isinstance(action, dict):
                raise ValueError(
                    f"{context}: action {expected_index} must be an object"
                )
            if action.get("index") != expected_index:
                raise ValueError(
                    f"{context}: action indices are not stable and contiguous"
                )
            key = action.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context}: action {expected_index} has no key")
            raw_indices = action.get("feature_indices")
            if not isinstance(raw_indices, list) or not raw_indices:
                raise ValueError(
                    f"{context}: action {expected_index} has no sparse features"
                )
            feature_indices = [
                _non_negative_int(value, f"{context}: action feature")
                for value in raw_indices
            ]
            if any(
                left >= right
                for left, right in zip(feature_indices, feature_indices[1:])
            ):
                raise ValueError(
                    f"{context}: action {expected_index} features must be sorted and unique"
                )
            if feature_indices[-1] >= self.schema.action_dim:
                raise ValueError(
                    f"{context}: action {expected_index} feature exceeds action_dim"
                )
            action_features.append(torch.tensor(feature_indices, dtype=torch.long))
            action_keys.append(key)
            visits.append(_non_negative_int(action.get("visits"), f"{context}: visits"))
            serialized_target = action.get("policy_target")
            if (
                isinstance(serialized_target, bool)
                or not isinstance(serialized_target, (int, float))
                or not math.isfinite(float(serialized_target))
            ):
                raise ValueError(
                    f"{context}: action {expected_index} has invalid policy target"
                )
            serialized_targets.append(float(serialized_target))
            serialized_search_target = action.get(
                "search_policy_target", serialized_target
            )
            if (
                isinstance(serialized_search_target, bool)
                or not isinstance(serialized_search_target, (int, float))
                or not math.isfinite(float(serialized_search_target))
            ):
                raise ValueError(
                    f"{context}: action {expected_index} has invalid search policy target"
                )
            serialized_search_targets.append(float(serialized_search_target))

        visit_sum = sum(visits)
        if visit_sum <= 0:
            raise ValueError(f"{context}: position has zero search visits")
        policy_values = [visit / visit_sum for visit in visits]
        if any(
            abs(expected - serialized) > 1e-5
            for expected, serialized in zip(policy_values, serialized_search_targets)
        ):
            raise ValueError(
                f"{context}: serialized search policy targets disagree with visits"
            )
        selected_targets = (
            serialized_targets
            if self.policy_target_field == POLICY_TARGET_FIELD
            else serialized_search_targets
        )
        if (
            self.policy_target_field == POLICY_TARGET_FIELD
            and self.search_policy_target_mix > 0.0
        ):
            raw_mix = self.search_policy_target_mix
            selected_targets = [
                (1.0 - raw_mix) * guided + raw_mix * raw
                for guided, raw in zip(
                    serialized_targets, serialized_search_targets, strict=True
                )
            ]
        behavior_values = normalize_policy_targets(
            selected_targets,
            exponent=self.policy_target_exponent,
            context=f"{context}: {self.policy_target_field}",
        )

        selected_action_index = _non_negative_int(
            record.get("selected_action_index"), f"{context}: selected_action_index"
        )
        if selected_action_index >= len(raw_actions):
            raise ValueError(f"{context}: selected action index is out of range")
        if visits[selected_action_index] <= 0:
            raise ValueError(f"{context}: selected action has zero visits")
        if record.get("selected_action_key") != action_keys[selected_action_index]:
            raise ValueError(f"{context}: selected action key does not match its index")

        value_target = record.get("value_target")
        if not isinstance(value_target, dict):
            raise ValueError(f"{context}: value_target must be an object")
        shared_win = value_target.get("shared_win")
        vp_margin = value_target.get("victory_point_margin")
        if (
            isinstance(shared_win, bool)
            or not isinstance(shared_win, (int, float))
            or not math.isfinite(float(shared_win))
        ):
            raise ValueError(f"{context}: invalid shared-win target")
        if not 0.0 <= float(shared_win) <= 1.0:
            raise ValueError(f"{context}: shared-win target must be in [0, 1]")
        if not isinstance(vp_margin, int) or isinstance(vp_margin, bool):
            raise ValueError(f"{context}: VP-margin target must be an integer")

        game_index = _non_negative_int(
            record.get("game_index"), f"{context}: game_index"
        )
        actor = _non_negative_int(record.get("actor"), f"{context}: actor")
        victory_points = self.game_victory_points.get((locator.path, game_index))
        if victory_points is None:
            raise ValueError(
                f"{context}: position has no preceding game victory-point record"
            )
        if actor >= len(victory_points):
            raise ValueError(f"{context}: actor is outside game victory-point records")

        actor_victory_points_target = float(victory_points[actor])
        if self.actor_vp_target_mode in ACTOR_VP_TARGET_REMAINING_MODES:
            actor_victory_points_target = max(
                0.0,
                actor_victory_points_target
                - _current_observer_victory_points(state_values, self.schema),
            )

        raw_policy_loss_weight = record.get("policy_loss_weight", 1.0)
        if (
            isinstance(raw_policy_loss_weight, bool)
            or not isinstance(raw_policy_loss_weight, (int, float))
            or not math.isfinite(float(raw_policy_loss_weight))
            or float(raw_policy_loss_weight) < 0.0
        ):
            raise ValueError(f"{context}: policy_loss_weight must be finite and non-negative")
        default_value_loss_weight = 1.0 if self._shard_value_target_usable.get(locator.path, True) else 0.0
        raw_value_loss_weight = record.get("value_loss_weight", default_value_loss_weight)
        if (
            isinstance(raw_value_loss_weight, bool)
            or not isinstance(raw_value_loss_weight, (int, float))
            or not math.isfinite(float(raw_value_loss_weight))
            or float(raw_value_loss_weight) < 0.0
        ):
            raise ValueError(f"{context}: value_loss_weight must be finite and non-negative")

        return PositionExample(
            state=state,
            action_features=tuple(action_features),
            action_keys=tuple(action_keys),
            policy_target=torch.tensor(behavior_values, dtype=torch.float32),
            shared_win_target=torch.tensor(float(shared_win), dtype=torch.float32),
            victory_point_margin_target=torch.tensor(
                float(vp_margin), dtype=torch.float32
            ),
            actor_victory_points_target=torch.tensor(
                actor_victory_points_target, dtype=torch.float32
            ),
            game_index=game_index,
            position_index=_non_negative_int(
                record.get("position_index"), f"{context}: position_index"
            ),
            policy_loss_weight=float(raw_policy_loss_weight),
            value_loss_weight=float(raw_value_loss_weight),
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def collate_positions(examples: Iterable[PositionExample]) -> TrainingBatch:
    examples = tuple(examples)
    if not examples:
        raise ValueError("cannot collate an empty batch")
    states = torch.stack([example.state for example in examples])
    flat_features: list[int] = []
    offsets = [0]
    action_batch_indices: list[int] = []
    action_counts: list[int] = []
    policy_targets: list[torch.Tensor] = []
    for batch_index, example in enumerate(examples):
        action_counts.append(len(example.action_features))
        policy_targets.append(example.policy_target)
        for features in example.action_features:
            flat_features.extend(features.tolist())
            offsets.append(len(flat_features))
            action_batch_indices.append(batch_index)

    return TrainingBatch(
        states=states,
        action_feature_indices=torch.tensor(flat_features, dtype=torch.long),
        action_feature_offsets=torch.tensor(offsets, dtype=torch.long),
        action_batch_indices=torch.tensor(action_batch_indices, dtype=torch.long),
        action_counts=torch.tensor(action_counts, dtype=torch.long),
        policy_targets=torch.cat(policy_targets),
        shared_win_targets=torch.stack(
            [example.shared_win_target for example in examples]
        ),
        victory_point_margin_targets=torch.stack(
            [example.victory_point_margin_target for example in examples]
        ),
        actor_victory_points_targets=torch.stack(
            [example.actor_victory_points_target for example in examples]
        ),
        action_keys=tuple(example.action_keys for example in examples),
        record_ids=tuple(
            (example.game_index, example.position_index) for example in examples
        ),
        policy_loss_weights=torch.tensor(
            [example.policy_loss_weight for example in examples],
            dtype=torch.float32,
        ),
        value_loss_weights=torch.tensor(
            [example.value_loss_weight for example in examples],
            dtype=torch.float32,
        ),
    )


def _decode_json_line(line: bytes, path: Path, line_number: int) -> dict:
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return record


def validate_policy_target_exponent(value: float) -> float:
    """Return a finite, strictly positive policy-target exponent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("policy_target_exponent must be finite and positive")
    exponent = float(value)
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("policy_target_exponent must be finite and positive")
    return exponent


def normalize_policy_targets(
    values: Sequence[float],
    *,
    exponent: float = POLICY_TARGET_EXPONENT_DEFAULT,
    context: str = "policy targets",
) -> list[float]:
    """Normalize non-negative targets and optionally sharpen their distribution."""
    exponent = validate_policy_target_exponent(exponent)
    if not values:
        raise ValueError(f"{context} must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError(f"{context} must be finite and non-negative")
    mass = math.fsum(float(value) for value in values)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{context} have no mass")
    normalized = [float(value) / mass for value in values]
    if exponent == 1.0:
        return normalized

    sharpened = [math.pow(value, exponent) for value in normalized]
    sharpened_mass = math.fsum(sharpened)
    if not math.isfinite(sharpened_mass) or sharpened_mass <= 0.0:
        raise ValueError(f"{context} have no mass after exponentiation")
    return [value / sharpened_mass for value in sharpened]


def actor_vp_target_scales(
    states: torch.Tensor,
    *,
    target_mode: str,
    actor_vp_scale: float,
    actor_vp_railroad_scale: float = ACTOR_VP_RAILROAD_SCALE_DEFAULT,
) -> torch.Tensor:
    """Return the per-position scale used by the actor VP target head."""
    if target_mode not in SUPPORTED_ACTOR_VP_TARGET_MODES:
        raise ValueError(
            "actor_vp_target_mode must be one of "
            f"{SUPPORTED_ACTOR_VP_TARGET_MODES}"
        )
    for name, value in (
        ("actor_vp_scale", actor_vp_scale),
        ("actor_vp_railroad_scale", actor_vp_railroad_scale),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if states.ndim != 2:
        raise ValueError("states must be a two-dimensional tensor")

    scales = torch.full(
        (states.shape[0],),
        float(actor_vp_scale),
        dtype=states.dtype,
        device=states.device,
    )
    if target_mode != ACTOR_VP_TARGET_PHASE_REMAINING:
        return scales
    if states.shape[1] <= ACTOR_VP_PHASE_RAILROAD_FEATURE_INDEX:
        raise ValueError(
            "phase-conditioned actor VP target requires the global phase features"
        )
    railroad_scales = torch.full_like(scales, float(actor_vp_railroad_scale))
    is_railroad = states[:, ACTOR_VP_PHASE_RAILROAD_FEATURE_INDEX] >= 0.5
    return torch.where(is_railroad, railroad_scales, scales)


def _current_observer_victory_points(
    state_values: list[object], schema: FeatureSchema
) -> float:
    raw_schema = schema.to_schema_dict()
    players_block = next(
        (
            block
            for block in raw_schema.get("state_blocks", [])
            if isinstance(block, dict) and block.get("name") == "players"
        ),
        None,
    )
    max_players = raw_schema.get("max_players")
    if not isinstance(players_block, dict) or not isinstance(max_players, int):
        raise ValueError(
            "remaining_vp target requires a players state block and max_players"
        )
    offset = players_block.get("offset")
    size = players_block.get("size")
    if (
        not isinstance(offset, int)
        or not isinstance(size, int)
        or max_players <= 0
        or size % max_players != 0
        or size // max_players <= 7
    ):
        raise ValueError("players state block cannot locate observer victory points")
    index = offset + 7
    if index < 0 or index >= len(state_values):
        raise ValueError("observer victory-point feature is outside the state vector")
    value = state_values[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("observer victory-point feature must be numeric")
    value = float(value) * 100.0
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("observer victory-point feature must be finite and non-negative")
    return value
def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
