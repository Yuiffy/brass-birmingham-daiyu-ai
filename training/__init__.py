"""Policy/value training tools for Fast Brass self-play shards."""

from .schema import FeatureSchema

__all__ = [
    "BrassPolicyValueNet",
    "FeatureSchema",
    "ModelConfig",
    "SelfPlayDataset",
    "TrainingBatch",
    "collate_positions",
    "compute_losses",
    "load_model_checkpoint",
    "save_checkpoint",
]


def __getattr__(name: str):
    if name in {"SelfPlayDataset", "TrainingBatch", "collate_positions"}:
        from . import data

        return getattr(data, name)
    if name in {"load_model_checkpoint", "save_checkpoint"}:
        from . import checkpoint

        return getattr(checkpoint, name)
    if name in {"BrassPolicyValueNet", "ModelConfig", "compute_losses"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
