from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import TrainingBatch


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    state_hidden_dim: int = 256
    action_embedding_dim: int = 128
    trunk_dim: int = 256
    vp_margin_scale: float = 100.0

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


@dataclass
class LossOutput:
    total: torch.Tensor
    policy_cross_entropy: torch.Tensor
    policy_kl: torch.Tensor
    policy_top1_accuracy: torch.Tensor
    shared_win_bce: torch.Tensor
    victory_point_margin_huber: torch.Tensor

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
        shared_win_logits, victory_point_margin_normalized = self.forward_value_heads(
            state_context
        )
        return ModelOutput(
            policy_logits=policy_logits,
            shared_win_logits=shared_win_logits,
            victory_point_margin_normalized=victory_point_margin_normalized,
        )

    def forward_values(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate value heads without constructing policy action tensors."""
        if states.ndim != 2 or states.shape[1] != self.config.state_dim:
            raise ValueError(
                f"states must have shape [batch, {self.config.state_dim}]"
            )
        return self.forward_value_heads(self.state_encoder(states))

    def forward_value_heads(
        self, state_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.shared_win_head(state_context).squeeze(-1),
            self.victory_point_margin_head(state_context).squeeze(-1),
        )


def compute_losses(
    output: ModelOutput,
    batch: TrainingBatch,
    vp_margin_scale: float,
    shared_win_weight: float = 1.0,
    victory_point_margin_weight: float = 0.25,
) -> LossOutput:
    if vp_margin_scale <= 0.0:
        raise ValueError("vp_margin_scale must be positive")
    policy_cross_entropy, policy_kl, policy_top1_accuracy = _segmented_policy_metrics(
        output.policy_logits,
        batch.policy_targets,
        batch.action_counts,
    )
    shared_win_bce = F.binary_cross_entropy_with_logits(
        output.shared_win_logits,
        batch.shared_win_targets,
    )
    normalized_margin_target = batch.victory_point_margin_targets / vp_margin_scale
    victory_point_margin_huber = F.smooth_l1_loss(
        output.victory_point_margin_normalized,
        normalized_margin_target,
    )
    total = (
        policy_cross_entropy
        + shared_win_weight * shared_win_bce
        + victory_point_margin_weight * victory_point_margin_huber
    )
    return LossOutput(
        total=total,
        policy_cross_entropy=policy_cross_entropy,
        policy_kl=policy_kl,
        policy_top1_accuracy=policy_top1_accuracy,
        shared_win_bce=shared_win_bce,
        victory_point_margin_huber=victory_point_margin_huber,
    )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cross_entropies = []
    kls = []
    top1 = []
    start = 0
    for count in action_counts.detach().cpu().tolist():
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
        start = end
    if start != logits.numel() or targets.numel() != logits.numel():
        raise ValueError("action counts do not cover policy logits and targets")
    return (
        torch.stack(cross_entropies).mean(),
        torch.stack(kls).mean(),
        torch.stack(top1).mean(),
    )
