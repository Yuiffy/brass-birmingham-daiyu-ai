from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence


CARD_CHOICE_GROUPING_VERSION = "card-invariant-intent-v1-max"
_CARD_TOKEN = re.compile(r"c\d+\Z")


def card_invariant_action_intent(action_key: str) -> str:
    """Return an action identity with hand-index choices removed."""
    root, separator, raw_choices = action_key.partition("|")
    if not separator:
        return action_key
    choices = [
        choice
        for choice in raw_choices.split(",")
        if choice and _CARD_TOKEN.fullmatch(choice) is None
    ]
    return f"{root}|{','.join(choices)}"


def rebalance_card_choice_groups(
    action_keys: Sequence[str], probabilities: Sequence[float]
) -> tuple[float, ...]:
    """Remove policy-mass inflation caused only by enumerated discard choices.

    Each card-invariant intent receives mass according to its best full-action
    probability. The original model probabilities remain the conditional
    distribution over card choices within that intent.
    """
    if len(action_keys) != len(probabilities):
        raise ValueError("action keys and probabilities must have equal lengths")
    if not action_keys:
        raise ValueError("cannot rebalance an empty policy")

    normalized = _normalize(probabilities, "policy")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(action_keys):
        if not isinstance(key, str) or not key:
            raise ValueError("action keys must be non-empty strings")
        grouped_indices[card_invariant_action_intent(key)].append(index)

    group_weights = {
        intent: max(normalized[index] for index in indices)
        for intent, indices in grouped_indices.items()
    }
    total_group_weight = sum(group_weights.values())
    if total_group_weight <= 0.0:
        raise ValueError("card-invariant policy groups have no probability mass")

    result = [0.0] * len(normalized)
    for intent, indices in grouped_indices.items():
        group_probability = group_weights[intent] / total_group_weight
        conditional_total = sum(normalized[index] for index in indices)
        if conditional_total <= 0.0:
            continue
        for index in indices:
            result[index] = (
                group_probability * normalized[index] / conditional_total
            )
    return _normalize(result, "card-invariant policy")


def _normalize(values: Sequence[float], label: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError(f"{label} probabilities must be finite and non-negative")
    total = sum(converted)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} probabilities must have positive mass")
    return tuple(value / total for value in converted)
