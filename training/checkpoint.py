from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .data import FeatureSchema
from .model import BrassPolicyValueNet, ModelConfig

CHECKPOINT_FORMAT = "fast_brass_policy_value"
CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: BrassPolicyValueNet,
    feature_schema: FeatureSchema,
    optimizer: torch.optim.Optimizer | None = None,
    training_metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model.config.state_dim != feature_schema.state_dim:
        raise ValueError("model state_dim does not match feature schema")
    if model.config.action_dim != feature_schema.action_dim:
        raise ValueError("model action_dim does not match feature schema")

    payload = {
        "metadata": {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "feature_version": feature_schema.version,
            "state_dim": feature_schema.state_dim,
            "action_dim": feature_schema.action_dim,
            "feature_schema_signature": feature_schema.signature,
            "model_config": model.config.to_dict(),
            "training": training_metadata or {},
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_model_checkpoint(
    path: str | Path,
    expected_schema: FeatureSchema | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[BrassPolicyValueNet, dict]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata is missing")
    _validate_metadata(metadata, expected_schema)
    config_values = metadata.get("model_config")
    if not isinstance(config_values, dict):
        raise ValueError("checkpoint model_config is missing")
    config = ModelConfig.from_dict(config_values)
    model = BrassPolicyValueNet(config)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model_state_dict is missing")
    model.load_state_dict(model_state, strict=True)
    return model, payload


def _validate_metadata(metadata: dict, expected_schema: FeatureSchema | None) -> None:
    if metadata.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format: {metadata.get('checkpoint_format')!r}"
        )
    if metadata.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format version: "
            f"{metadata.get('checkpoint_format_version')!r}"
        )
    if expected_schema is None:
        return
    expected = (
        expected_schema.version,
        expected_schema.state_dim,
        expected_schema.action_dim,
        expected_schema.signature,
    )
    actual = (
        metadata.get("feature_version"),
        metadata.get("state_dim"),
        metadata.get("action_dim"),
        metadata.get("feature_schema_signature"),
    )
    if actual != expected:
        raise ValueError(
            "checkpoint feature schema mismatch: expected version/state/action "
            f"{expected_schema.version}/{expected_schema.state_dim}/{expected_schema.action_dim}, "
            f"got {actual[0]}/{actual[1]}/{actual[2]}"
        )
