from __future__ import annotations

import json
from dataclasses import dataclass

SELF_PLAY_FORMAT = "fast_brass_self_play_jsonl"
SELF_PLAY_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FeatureSchema:
    version: int
    state_dim: int
    action_dim: int
    signature: str

    @classmethod
    def from_metadata(cls, metadata: dict) -> "FeatureSchema":
        if metadata.get("format") != SELF_PLAY_FORMAT:
            raise ValueError(
                f"unsupported self-play format: {metadata.get('format')!r}"
            )
        if metadata.get("format_version") != SELF_PLAY_FORMAT_VERSION:
            raise ValueError(
                "unsupported self-play format version: "
                f"{metadata.get('format_version')!r}"
            )
        raw_schema = metadata.get("feature_schema")
        if not isinstance(raw_schema, dict):
            raise ValueError("metadata.feature_schema must be an object")
        return cls.from_schema_dict(raw_schema)

    @classmethod
    def from_schema_dict(cls, raw_schema: dict) -> "FeatureSchema":
        version = _positive_int(raw_schema.get("version"), "feature_schema.version")
        state_dim = _positive_int(
            raw_schema.get("state_dim"), "feature_schema.state_dim"
        )
        action_dim = _positive_int(
            raw_schema.get("action_dim"), "feature_schema.action_dim"
        )
        signature = json.dumps(raw_schema, sort_keys=True, separators=(",", ":"))
        return cls(
            version=version,
            state_dim=state_dim,
            action_dim=action_dim,
            signature=signature,
        )

    def assert_compatible(self, other: "FeatureSchema", context: str) -> None:
        if self != other:
            raise ValueError(
                f"feature schema mismatch for {context}: expected version/state/action "
                f"{self.version}/{self.state_dim}/{self.action_dim}, got "
                f"{other.version}/{other.state_dim}/{other.action_dim}"
            )

    def to_schema_dict(self) -> dict:
        return json.loads(self.signature)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
