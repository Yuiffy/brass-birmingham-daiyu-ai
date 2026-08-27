from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .schema import SELF_PLAY_FORMAT, SELF_PLAY_FORMAT_VERSION, FeatureSchema


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
        )


class SelfPlayDataset(Dataset[PositionExample]):
    """Offset-indexed JSONL dataset with strict schema and record validation."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        expected_schema: FeatureSchema | None = None,
        allow_mixed_engine_revisions: bool = False,
    ) -> None:
        if not shard_paths:
            raise ValueError("at least one self-play shard is required")
        self.shard_paths = tuple(Path(path).resolve() for path in shard_paths)
        self.schema = expected_schema
        self.metadata: list[dict] = []
        self.engine_revisions: set[str] = set()
        self.game_seeds: set[int] = set()
        self.game_victory_points: dict[tuple[Path, int], tuple[int, ...]] = {}
        self.allow_mixed_engine_revisions = allow_mixed_engine_revisions
        self._positions: list[PositionLocator] = []
        self._handles: dict[Path, BinaryIO] = {}

        for path in self.shard_paths:
            self._scan_shard(path)
        if self.schema is None:
            raise ValueError("no metadata record found in self-play shards")
        if not self._positions:
            raise ValueError("self-play shards contain no position records")

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
        if any(value < 0.0 for value in serialized_targets):
            raise ValueError(f"{context}: policy targets must be non-negative")
        behavior_mass = sum(serialized_targets)
        if not math.isfinite(behavior_mass) or behavior_mass <= 0.0:
            raise ValueError(f"{context}: policy targets have no mass")
        behavior_values = [value / behavior_mass for value in serialized_targets]

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
                float(victory_points[actor]), dtype=torch.float32
            ),
            game_index=game_index,
            position_index=_non_negative_int(
                record.get("position_index"), f"{context}: position_index"
            ),
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
    )


def _decode_json_line(line: bytes, path: Path, line_number: int) -> dict:
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return record
def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
