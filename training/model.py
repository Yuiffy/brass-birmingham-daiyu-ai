from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .data import (
    ACTOR_VP_RAILROAD_SCALE_DEFAULT,
    ACTOR_VP_TARGET_ABSOLUTE,
    SUPPORTED_ACTOR_VP_TARGET_MODES,
    TrainingBatch,
    actor_vp_target_scales,
)


FINAL_VP_QUALITY_REFERENCE_VP = 100.0
FINAL_VP_QUALITY_WEIGHT_MAX = 2.0
# Human demonstrations are intentionally applied as a residual policy signal.
# Keeping the residual bounded prevents a small, distribution-specific replay
# shard from overwhelming the established policy on out-of-distribution states.
POLICY_ADAPTER_MAX_LOGIT_DELTA = 1.0


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    state_hidden_dim: int = 256
    action_embedding_dim: int = 128
    trunk_dim: int = 256
    vp_margin_scale: float = 100.0
    actor_vp_scale: float = 140.0
    actor_vp_railroad_scale: float = ACTOR_VP_RAILROAD_SCALE_DEFAULT
    policy_adapter_dim: int = 0
    separate_value_encoder: bool = False
    actor_vp_target_mode: str = ACTOR_VP_TARGET_ABSOLUTE

    def validate(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "state_hidden_dim",
            "action_embedding_dim",
            "trunk_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.trunk_dim < 2:
            raise ValueError("trunk_dim must be at least two")
        if self.vp_margin_scale <= 0.0:
            raise ValueError("vp_margin_scale must be positive")
        if self.actor_vp_scale <= 0.0:
            raise ValueError("actor_vp_scale must be positive")
        if self.actor_vp_railroad_scale <= 0.0:
            raise ValueError("actor_vp_railroad_scale must be positive")
        if self.actor_vp_target_mode not in SUPPORTED_ACTOR_VP_TARGET_MODES:
            raise ValueError(
                "actor_vp_target_mode must be one of "
                f"{SUPPORTED_ACTOR_VP_TARGET_MODES}"
            )
        if self.policy_adapter_dim < 0:
            raise ValueError("policy_adapter_dim must be non-negative")
        if not isinstance(self.separate_value_encoder, bool):
            raise ValueError("separate_value_encoder must be a boolean")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        config = cls(**values)
        config.validate()
        return config


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    shared_win_logits: torch.Tensor
    victory_point_margin_normalized: torch.Tensor
    actor_victory_points_normalized: torch.Tensor


@dataclass
class LossOutput:
    total: torch.Tensor
    policy_cross_entropy: torch.Tensor
    policy_kl: torch.Tensor
    policy_top1_accuracy: torch.Tensor
    shared_win_bce: torch.Tensor
    victory_point_margin_huber: torch.Tensor
    actor_victory_points_huber: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "policy_cross_entropy": float(self.policy_cross_entropy.detach().cpu()),
            "policy_kl": float(self.policy_kl.detach().cpu()),
            "policy_top1_accuracy": float(self.policy_top1_accuracy.detach().cpu()),
            "shared_win_bce": float(self.shared_win_bce.detach().cpu()),
            "victory_point_margin_huber": float(
                self.victory_point_margin_huber.detach().cpu()
            ),
            "actor_victory_points_huber": float(
                self.actor_victory_points_huber.detach().cpu()
            ),
        }


class BrassPolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden_dim),
            nn.Linear(config.state_hidden_dim, config.trunk_dim),
            nn.GELU(),
        )
        self.action_embedding = nn.EmbeddingBag(
            config.action_dim,
            config.action_embedding_dim,
            mode="mean",
            include_last_offset=True,
        )
        self.action_normalization = nn.LayerNorm(config.action_embedding_dim)
        self.policy_scorer = nn.Sequential(
            nn.Linear(config.trunk_dim + config.action_embedding_dim, config.trunk_dim),
            nn.GELU(),
            nn.Linear(config.trunk_dim, 1),
        )
        self.shared_win_head = nn.Sequential(
            nn.Linear(config.trunk_dim, config.trunk_dim // 2),
            nn.GELU(),
            nn.Linear(config.trunk_dim // 2, 1),
        )
        self.victory_point_margin_head = nn.Sequential(
            nn.Linear(config.trunk_dim, config.trunk_dim // 2),
            nn.GELU(),
            nn.Linear(config.trunk_dim // 2, 1),
        )
        self.actor_victory_points_head = nn.Sequential(
            nn.Linear(config.trunk_dim, config.trunk_dim // 2),
            nn.GELU(),
            nn.Linear(config.trunk_dim // 2, 1),
        )
        if config.policy_adapter_dim > 0:
            adapter_dim = config.policy_adapter_dim
            self.policy_adapter_state_encoder = nn.Sequential(
                nn.Linear(config.state_dim, adapter_dim),
                nn.GELU(),
                nn.LayerNorm(adapter_dim),
                nn.Linear(adapter_dim, adapter_dim),
                nn.GELU(),
            )
            self.policy_adapter_action_embedding = nn.EmbeddingBag(
                config.action_dim,
                adapter_dim,
                mode="mean",
                include_last_offset=True,
            )
            self.policy_adapter_action_normalization = nn.LayerNorm(adapter_dim)
            self.policy_adapter_scorer = nn.Sequential(
                nn.Linear(adapter_dim * 2, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, 1),
            )
            output = self.policy_adapter_scorer[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        if config.separate_value_encoder:
            self.value_state_encoder = nn.Sequential(
                nn.Linear(config.state_dim, config.state_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.state_hidden_dim),
                nn.Linear(config.state_hidden_dim, config.trunk_dim),
                nn.GELU(),
            )

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        state_context = self.state_encoder(batch.states)
        action_context = self.action_embedding(
            batch.action_feature_indices,
            batch.action_feature_offsets,
        )
        action_context = self.action_normalization(action_context)
        owning_state_context = state_context.index_select(0, batch.action_batch_indices)
        policy_logits = self.policy_scorer(
            torch.cat((owning_state_context, action_context), dim=-1)
        ).squeeze(-1)
        if self.config.policy_adapter_dim > 0:
            adapter_state_context = self.policy_adapter_state_encoder(batch.states)
            adapter_action_context = self.policy_adapter_action_embedding(
                batch.action_feature_indices,
                batch.action_feature_offsets,
            )
            adapter_action_context = self.policy_adapter_action_normalization(
                adapter_action_context
            )
            owning_adapter_state_context = adapter_state_context.index_select(
                0, batch.action_batch_indices
            )
            adapter_logits = self.policy_adapter_scorer(
                torch.cat(
                    (owning_adapter_state_context, adapter_action_context), dim=-1
                )
            ).squeeze(-1)
            policy_logits = policy_logits + POLICY_ADAPTER_MAX_LOGIT_DELTA * torch.tanh(
                adapter_logits / POLICY_ADAPTER_MAX_LOGIT_DELTA
            )
        (
            shared_win_logits,
            victory_point_margin_normalized,
            actor_victory_points_normalized,
        ) = self.forward_value_heads(
            self._value_state_context(batch.states, state_context)
        )
        return ModelOutput(
            policy_logits=policy_logits,
            shared_win_logits=shared_win_logits,
            victory_point_margin_normalized=victory_point_margin_normalized,
            actor_victory_points_normalized=actor_victory_points_normalized,
        )

    def forward_values(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate value heads without constructing policy action tensors."""
        if states.ndim != 2 or states.shape[1] != self.config.state_dim:
            raise ValueError(
                f"states must have shape [batch, {self.config.state_dim}]"
            )
        return self.forward_value_heads(self._value_state_context(states))

    def _value_state_context(
        self,
        states: torch.Tensor,
        shared_state_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.separate_value_encoder:
            return self.value_state_encoder(states)
        if shared_state_context is not None:
            return shared_state_context
        return self.state_encoder(states)

    def forward_value_heads(
        self, state_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.shared_win_head(state_context).squeeze(-1),
            self.victory_point_margin_head(state_context).squeeze(-1),
            self.actor_victory_points_head(state_context).squeeze(-1),
        )

    def initialize_actor_victory_points_head(self, seed: int = 20_260_828) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        hidden = self.actor_victory_points_head[0]
        output = self.actor_victory_points_head[2]
        assert isinstance(hidden, nn.Linear) and isinstance(output, nn.Linear)
        nn.init.xavier_uniform_(hidden.weight, generator=generator)
        nn.init.zeros_(hidden.bias)
        nn.init.xavier_uniform_(output.weight, generator=generator)
        with torch.no_grad():
            output.weight.mul_(0.01)
            output.bias.fill_(30.0 / self.config.actor_vp_scale)


def expand_with_policy_adapter(
    model: BrassPolicyValueNet,
    policy_adapter_dim: int,
) -> BrassPolicyValueNet:
    if policy_adapter_dim <= 0:
        raise ValueError("policy_adapter_dim must be positive")
    if model.config.policy_adapter_dim > 0:
        if model.config.policy_adapter_dim != policy_adapter_dim:
            raise ValueError(
                "checkpoint policy adapter dimension does not match the requested dimension"
            )
        return model

    expanded = BrassPolicyValueNet(
        replace(model.config, policy_adapter_dim=policy_adapter_dim)
    )
    incompatible = expanded.load_state_dict(model.state_dict(), strict=False)
    expected_missing = {
        key
        for key in expanded.state_dict()
        if key.startswith("policy_adapter_")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError("policy adapter expansion changed base model parameters")
    if incompatible.unexpected_keys:
        raise RuntimeError("policy adapter expansion found unexpected parameters")
    return expanded


def with_actor_vp_target_mode(
    model: BrassPolicyValueNet,
    actor_vp_target_mode: str,
) -> BrassPolicyValueNet:
    if actor_vp_target_mode not in SUPPORTED_ACTOR_VP_TARGET_MODES:
        raise ValueError(
            "actor_vp_target_mode must be one of "
            f"{SUPPORTED_ACTOR_VP_TARGET_MODES}"
        )
    if model.config.actor_vp_target_mode == actor_vp_target_mode:
        return model
    converted = BrassPolicyValueNet(
        replace(model.config, actor_vp_target_mode=actor_vp_target_mode)
    )
    converted.load_state_dict(model.state_dict(), strict=True)
    return converted


def fuse_policy_and_value_models(
    policy_model: BrassPolicyValueNet,
    value_model: BrassPolicyValueNet,
) -> BrassPolicyValueNet:
    policy_config = policy_model.config
    value_config = value_model.config
    compatible_dimensions = (
        "state_dim",
        "action_dim",
        "state_hidden_dim",
        "trunk_dim",
    )
    mismatches = [
        name
        for name in compatible_dimensions
        if getattr(policy_config, name) != getattr(value_config, name)
    ]
    if mismatches:
        raise ValueError(
            "policy and value model dimensions differ: " + ", ".join(mismatches)
        )

    fused = BrassPolicyValueNet(
        replace(
            policy_config,
            vp_margin_scale=value_config.vp_margin_scale,
            actor_vp_scale=value_config.actor_vp_scale,
            actor_vp_railroad_scale=value_config.actor_vp_railroad_scale,
            actor_vp_target_mode=value_config.actor_vp_target_mode,
            separate_value_encoder=True,
        )
    )
    incompatible = fused.load_state_dict(policy_model.state_dict(), strict=False)
    expected_missing = {
        key
        for key in fused.state_dict()
        if key.startswith("value_state_encoder.")
        and key not in policy_model.state_dict()
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError("policy model is incompatible with the fused network")
    if incompatible.unexpected_keys:
        raise RuntimeError("policy model contains unexpected fused-network parameters")

    source_value_encoder = (
        value_model.value_state_encoder
        if value_config.separate_value_encoder
        else value_model.state_encoder
    )
    fused.value_state_encoder.load_state_dict(source_value_encoder.state_dict())
    fused.shared_win_head.load_state_dict(value_model.shared_win_head.state_dict())
    fused.victory_point_margin_head.load_state_dict(
        value_model.victory_point_margin_head.state_dict()
    )
    fused.actor_victory_points_head.load_state_dict(
        value_model.actor_victory_points_head.state_dict()
    )
    return fused


def compute_losses(
    output: ModelOutput,
    batch: TrainingBatch,
    vp_margin_scale: float,
    actor_vp_scale: float = 140.0,
    actor_vp_railroad_scale: float = ACTOR_VP_RAILROAD_SCALE_DEFAULT,
    actor_vp_target_mode: str = ACTOR_VP_TARGET_ABSOLUTE,
    shared_win_weight: float = 1.0,
    victory_point_margin_weight: float = 0.25,
    actor_victory_points_weight: float = 0.25,
    final_vp_quality_weight: float = 0.0,
) -> LossOutput:
    if vp_margin_scale <= 0.0:
        raise ValueError("vp_margin_scale must be positive")
    if actor_vp_scale <= 0.0:
        raise ValueError("actor_vp_scale must be positive")
    if (
        not math.isfinite(final_vp_quality_weight)
        or not 0.0 <= final_vp_quality_weight <= FINAL_VP_QUALITY_WEIGHT_MAX
    ):
        raise ValueError(
            "final_vp_quality_weight must be finite and between 0 and "
            f"{FINAL_VP_QUALITY_WEIGHT_MAX}"
        )
    if (
        final_vp_quality_weight > 0.0
        and actor_vp_target_mode != ACTOR_VP_TARGET_ABSOLUTE
    ):
        raise ValueError(
            "final_vp_quality_weight requires absolute_final_vp targets"
        )
    if output.actor_victory_points_normalized.shape != batch.actor_victory_points_targets.shape:
        raise ValueError("actor victory-point output and targets must have equal shapes")
    position_weights = final_vp_quality_position_weights(
        batch.actor_victory_points_targets,
        final_vp_quality_weight,
    )
    policy_position_weights = _policy_position_weights(batch, position_weights)
    value_position_weights = _value_position_weights(batch, position_weights)
    policy_cross_entropy, policy_kl, policy_top1_accuracy = _segmented_policy_metrics(
        output.policy_logits,
        batch.policy_targets,
        batch.action_counts,
        policy_position_weights,
    )
    shared_win_bce = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output.shared_win_logits,
            batch.shared_win_targets,
            reduction="none",
        ),
        value_position_weights,
    )
    normalized_margin_target = batch.victory_point_margin_targets / vp_margin_scale
    victory_point_margin_huber = _weighted_mean(
        F.smooth_l1_loss(
            output.victory_point_margin_normalized,
            normalized_margin_target,
            reduction="none",
        ),
        value_position_weights,
    )
    normalized_actor_vp_target = batch.actor_victory_points_targets / actor_vp_target_scales(
        batch.states,
        target_mode=actor_vp_target_mode,
        actor_vp_scale=actor_vp_scale,
        actor_vp_railroad_scale=actor_vp_railroad_scale,
    )
    actor_victory_points_huber = _weighted_mean(
        F.smooth_l1_loss(
            output.actor_victory_points_normalized,
            normalized_actor_vp_target,
            reduction="none",
        ),
        value_position_weights,
    )
    total = (
        policy_cross_entropy
        + shared_win_weight * shared_win_bce
        + victory_point_margin_weight * victory_point_margin_huber
        + actor_victory_points_weight * actor_victory_points_huber
    )
    return LossOutput(
        total=total,
        policy_cross_entropy=policy_cross_entropy,
        policy_kl=policy_kl,
        policy_top1_accuracy=policy_top1_accuracy,
        shared_win_bce=shared_win_bce,
        victory_point_margin_huber=victory_point_margin_huber,
        actor_victory_points_huber=actor_victory_points_huber,
    )


def final_vp_quality_position_weights(
    actor_victory_points_targets: torch.Tensor,
    quality_weight: float = 0.0,
) -> torch.Tensor:
    """Return normalized position weights that mildly favor high-VP trajectories."""
    if actor_victory_points_targets.ndim != 1:
        raise ValueError("actor victory-point targets must be one-dimensional")
    if (
        not math.isfinite(quality_weight)
        or not 0.0 <= quality_weight <= FINAL_VP_QUALITY_WEIGHT_MAX
    ):
        raise ValueError(
            "quality_weight must be finite and between 0 and "
            f"{FINAL_VP_QUALITY_WEIGHT_MAX}"
        )
    if (
        not torch.isfinite(actor_victory_points_targets).all().item()
        or (actor_victory_points_targets < 0.0).any().item()
    ):
        raise ValueError("actor victory-point targets must be finite and non-negative")
    if quality_weight == 0.0:
        return torch.ones_like(actor_victory_points_targets)
    quality = (
        actor_victory_points_targets / FINAL_VP_QUALITY_REFERENCE_VP
    ).clamp(0.0, 1.0)
    weights = 1.0 + quality_weight * quality
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).tiny)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError("loss values and position weights must have equal shapes")
    return (values * weights).sum() / weights.sum().clamp_min(
        torch.finfo(values.dtype).tiny
    )


def _policy_position_weights(
    batch: TrainingBatch, quality_weights: torch.Tensor
) -> torch.Tensor:
    """Combine optional per-position expert weights with quality weighting.

    Expert weighting is deliberately applied only to the policy metrics. Value
    heads use their own optional mask, so a policy-only demonstration cannot
    multiply a contaminated terminal value target into the loss.
    """
    explicit = batch.policy_loss_weights
    if explicit is None:
        return quality_weights
    if explicit.ndim != 1 or explicit.shape != quality_weights.shape:
        raise ValueError("policy loss weights must match the position batch")
    explicit = explicit.to(device=quality_weights.device, dtype=quality_weights.dtype)
    if (
        not torch.isfinite(explicit).all().item()
        or (explicit < 0.0).any().item()
        or not (explicit > 0.0).any().item()
    ):
        raise ValueError("policy loss weights must be finite, non-negative, and non-zero")
    return quality_weights * explicit


def _value_position_weights(
    batch: TrainingBatch, quality_weights: torch.Tensor
) -> torch.Tensor:
    """Combine trajectory-quality weights with optional value-target masks."""
    explicit = batch.value_loss_weights
    if explicit is None:
        return quality_weights
    if explicit.ndim != 1 or explicit.shape != quality_weights.shape:
        raise ValueError("value loss weights must match the position batch")
    explicit = explicit.to(device=quality_weights.device, dtype=quality_weights.dtype)
    if (
        not torch.isfinite(explicit).all().item()
        or (explicit < 0.0).any().item()
    ):
        raise ValueError("value loss weights must be finite and non-negative")
    return quality_weights * explicit


def segmented_policy_probabilities(
    policy_logits: torch.Tensor, action_counts: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.empty_like(policy_logits)
    start = 0
    for count in action_counts.detach().cpu().tolist():
        end = start + count
        probabilities[start:end] = F.softmax(policy_logits[start:end], dim=0)
        start = end
    if start != policy_logits.numel():
        raise ValueError("action counts do not cover policy logits")
    return probabilities


def _segmented_policy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_counts: torch.Tensor,
    position_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if position_weights is None:
        position_weights = torch.ones(
            action_counts.shape[0], dtype=logits.dtype, device=logits.device
        )
    if position_weights.ndim != 1 or position_weights.shape[0] != action_counts.shape[0]:
        raise ValueError("policy position weights must match action-count positions")
    cross_entropies = []
    kls = []
    top1 = []
    segment_weights = []
    start = 0
    for position_index, count in enumerate(action_counts.detach().cpu().tolist()):
        if count <= 0:
            raise ValueError("each position must contain at least one legal action")
        end = start + count
        segment_logits = logits[start:end]
        segment_targets = targets[start:end]
        target_sum = segment_targets.sum()
        if not torch.isclose(
            target_sum,
            torch.ones((), device=target_sum.device, dtype=target_sum.dtype),
            atol=1e-5,
        ):
            raise ValueError("policy target segment does not sum to one")
        log_probabilities = F.log_softmax(segment_logits, dim=0)
        cross_entropy = -(segment_targets * log_probabilities).sum()
        target_entropy = -(
            segment_targets
            * torch.log(
                segment_targets.clamp_min(torch.finfo(segment_targets.dtype).tiny)
            )
        ).sum()
        cross_entropies.append(cross_entropy)
        kls.append((cross_entropy - target_entropy).clamp_min(0.0))
        top1.append(
            (segment_logits.argmax() == segment_targets.argmax()).to(dtype=logits.dtype)
        )
        segment_weights.append(position_weights[position_index])
        start = end
    if start != logits.numel() or targets.numel() != logits.numel():
        raise ValueError("action counts do not cover policy logits and targets")
    weights = torch.stack(segment_weights)
    return (
        _weighted_mean(torch.stack(cross_entropies), weights),
        _weighted_mean(torch.stack(kls), weights),
        _weighted_mean(torch.stack(top1), weights),
    )
