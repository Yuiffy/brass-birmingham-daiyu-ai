from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_model_checkpoint, save_checkpoint
from .model import fuse_policy_and_value_models
from .schema import FeatureSchema


def main() -> None:
    args = build_parser().parse_args()
    report = fuse_checkpoints(
        policy_path=Path(args.policy),
        value_path=Path(args.value),
        output_path=Path(args.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse one checkpoint's policy tower with another checkpoint's "
            "independent value tower."
        )
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--output", required=True)
    return parser


def fuse_checkpoints(
    *,
    policy_path: Path,
    value_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    value_path = value_path.resolve()
    output_path = output_path.resolve()
    policy_model, policy_payload = load_model_checkpoint(policy_path)
    value_model, value_payload = load_model_checkpoint(value_path)

    policy_metadata = policy_payload["metadata"]
    value_metadata = value_payload["metadata"]
    policy_schema = _feature_schema(policy_metadata)
    value_schema = _feature_schema(value_metadata)
    policy_schema.assert_compatible(value_schema, "value checkpoint")

    fused = fuse_policy_and_value_models(policy_model, value_model)
    policy_training = policy_metadata.get("training", {})
    value_training = value_metadata.get("training", {})
    policy_step = int(policy_training.get("global_step", 0))
    value_step = int(value_training.get("global_step", 0))
    policy_sha256 = _sha256_file(policy_path)
    value_sha256 = _sha256_file(value_path)
    save_checkpoint(
        output_path,
        fused,
        policy_schema,
        training_metadata={
            "global_step": policy_step,
            "model_kind": "independent_policy_value_fusion",
            "policy_source": {
                "checkpoint": str(policy_path),
                "checkpoint_step": policy_step,
                "sha256": policy_sha256,
            },
            "value_source": {
                "checkpoint": str(value_path),
                "checkpoint_step": value_step,
                "sha256": value_sha256,
            },
        },
    )
    return {
        "output": str(output_path),
        "policy_source": str(policy_path),
        "policy_checkpoint_step": policy_step,
        "policy_sha256": policy_sha256,
        "value_source": str(value_path),
        "value_checkpoint_step": value_step,
        "value_sha256": value_sha256,
        "model_parameters": sum(parameter.numel() for parameter in fused.parameters()),
        "model_config": fused.config.to_dict(),
    }


def _feature_schema(metadata: dict[str, Any]) -> FeatureSchema:
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
        raise ValueError("checkpoint feature_schema_signature is invalid")
    return FeatureSchema(version, state_dim, action_dim, signature)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
