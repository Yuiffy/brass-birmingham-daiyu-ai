"""A small, auditable behavior prior learned from confirmed human actions.

This is intentionally a prior over *action intents*, not a replacement policy
network.  It is useful for bootstrapping self-play from a tiny human replay:
the neural checkpoint still supplies state-specific probabilities and PUCT
still verifies successors, while this module nudges root mass toward intents
the human actually used in the same phase/round context.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .policy_normalization import card_invariant_action_intent


DEMONSTRATION_PRIOR_VERSION = "human-demonstration-v1-context-intent"
_INDUSTRY_TOKEN = re.compile(r"i\d+\Z")
_MAX_ROUND_BUCKET = 4
_EPSILON = 1.0e-12


@dataclass(frozen=True)
class DemonstrationPrior:
    """Immutable counts and probability lookup for a human demonstration set."""

    version: str
    source_paths: tuple[str, ...]
    positions: int
    context_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    phase_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    global_counts: tuple[tuple[str, int], ...]

    def probabilities(
        self,
        observation: Mapping[str, Any],
        legal_record: Mapping[str, Any],
    ) -> tuple[float, ...]:
        """Return a smoothed legal-action distribution for the current context."""

        actions = legal_record.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("demonstration prior requires non-empty legal actions")
        context = _context_key_from_observation(observation)
        phase = context.split("|", 1)[0]
        context_counts = _counter_from_pairs(dict(self.context_counts).get(context, ()))
        phase_counts = _counter_from_pairs(dict(self.phase_counts).get(phase, ()))
        global_counts = Counter(dict(self.global_counts))
        context_family_counts = _family_counts(context_counts)
        phase_family_counts = _family_counts(phase_counts)
        global_family_counts = _family_counts(global_counts)

        # A tiny additive floor preserves legal alternatives and prevents a
        # four-dozen-position replay from becoming a hard action mask.  Context
        # evidence dominates phase evidence, which dominates the global mix.
        scores: list[float] = []
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("demonstration prior encountered a malformed action")
            key = action.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("demonstration prior encountered an empty action key")
            signature = action_intent_signature(key)
            family = action_family(key)
            context_signature = context_counts.get(signature, 0)
            phase_signature = phase_counts.get(signature, 0)
            global_signature = global_counts.get(signature, 0)
            context_family = context_family_counts.get(family, 0)
            phase_family = phase_family_counts.get(family, 0)
            global_family = global_family_counts.get(family, 0)
            score = (
                1.0
                + 4.0 * context_signature
                + 1.0 * context_family
                + 1.5 * phase_signature
                + 0.5 * phase_family
                + 0.25 * global_signature
                + 0.10 * global_family
            )
            scores.append(score)
        return _normalize(scores)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sources": list(self.source_paths),
            "positions": self.positions,
            "context_count_keys": len(self.context_counts),
        }


def blend_policy_with_demonstration(
    model_probabilities: Sequence[float],
    demonstration_probabilities: Sequence[float],
    strength: float,
    *,
    guarded_actions: Sequence[bool] | None = None,
) -> tuple[float, ...]:
    """Blend a state-specific policy with a human-intent distribution.

    The geometric mixture is preferable to an additive mixture here: a small
    demonstration shard can express a useful relative preference without
    erasing the checkpoint's state-dependent mass on unfamiliar actions.  The
    result is always normalized and remains defined when one input contains a
    zero-probability action.
    """

    if len(model_probabilities) != len(demonstration_probabilities):
        raise ValueError(
            "model and demonstration probabilities must have equal lengths"
        )
    if not model_probabilities:
        raise ValueError("cannot blend an empty action distribution")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 <= float(strength) <= 1.0
    ):
        raise ValueError("demonstration prior strength must be between 0 and 1")
    if guarded_actions is not None and len(guarded_actions) != len(model_probabilities):
        raise ValueError("guarded action mask must match the policy length")
    model = _normalize(model_probabilities)
    demonstration = _normalize(demonstration_probabilities)
    if strength == 0.0:
        return model
    blended = tuple(
        0.0
        if guarded_actions is not None and guarded_actions[index]
        else math.exp(
            (1.0 - strength) * math.log(max(model_value, _EPSILON))
            + strength * math.log(max(demonstration_value, _EPSILON))
        )
        for index, (model_value, demonstration_value) in enumerate(zip(
            model, demonstration, strict=True
        ))
    )
    return _normalize(blended)


def load_demonstration_prior(
    shard_paths: Iterable[str | Path],
    *,
    version: str = DEMONSTRATION_PRIOR_VERSION,
) -> DemonstrationPrior:
    """Load policy-only positions from one or more JSONL demonstration shards."""

    paths = tuple(Path(path).resolve() for path in shard_paths)
    if not paths:
        raise ValueError("at least one demonstration shard is required")
    if not version.strip():
        raise ValueError("demonstration prior version must not be empty")

    context_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    phase_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    positions = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"demonstration shard does not exist: {path}")
        found_metadata = False
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                record_type = record.get("record_type")
                if not found_metadata:
                    if record_type != "metadata":
                        raise ValueError(
                            f"{path}:{line_number}: first record must be metadata"
                        )
                    found_metadata = True
                    continue
                if record_type != "position":
                    if record_type == "metadata":
                        raise ValueError(f"{path}:{line_number}: duplicate metadata")
                    # Game records carry no action label and are safe to skip.
                    if record_type == "game":
                        continue
                    raise ValueError(
                        f"{path}:{line_number}: unknown record type {record_type!r}"
                    )
                key = record.get("selected_action_key")
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"{path}:{line_number}: selected_action_key is required"
                    )
                phase = _phase_from_record(record)
                context = _context_key_from_record(record, phase)
                signature = action_intent_signature(key)
                context_counts[context][signature] += 1
                phase_counts[phase][signature] += 1
                global_counts[signature] += 1
                positions += 1
        if not found_metadata:
            raise ValueError(f"{path}: shard is empty or missing metadata")
    if positions <= 0:
        raise ValueError("demonstration shards contain no position records")

    return DemonstrationPrior(
        version=version,
        source_paths=tuple(str(path) for path in paths),
        positions=positions,
        context_counts=_freeze_nested_counts(context_counts),
        phase_counts=_freeze_nested_counts(phase_counts),
        global_counts=tuple(sorted(global_counts.items())),
    )


def action_family(action_key: str) -> str:
    """Return the root action family used for fallback counts."""

    return action_key.partition("|")[0]


def action_intent_signature(action_key: str) -> str:
    """Generalize a semantic key while retaining useful industry intent.

    Card slots and exact map/resource choices are removed.  Build/develop
    industry tokens are retained in sorted order; other actions fall back to
    their root family.  This makes a human ``build cotton`` example useful for
    a different legal card/location without copying an illegal action.
    """

    root = action_family(action_key)
    _separator, _bar, raw_choices = action_key.partition("|")
    industries = sorted(
        token.strip()
        for token in raw_choices.split(",")
        if _INDUSTRY_TOKEN.fullmatch(token.strip())
    )
    if industries and root in {"build", "develop", "develop_double", "sell"}:
        return f"{root}:{','.join(industries)}"
    # card_invariant_action_intent remains useful for future families where a
    # non-card semantic token is stable; normalize it to a root fallback here.
    _ = card_invariant_action_intent(action_key)
    return root


def _phase_from_record(record: Mapping[str, Any]) -> str:
    value = record.get("phase")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "unknown"


def _context_key_from_record(record: Mapping[str, Any], phase: str) -> str:
    round_value = _as_nonnegative_int(record.get("round_in_phase"), 0)
    actions_value = _as_nonnegative_int(
        record.get("actions_remaining_in_turn"), 0
    )
    return _context_key(phase, round_value, actions_value)


def _context_key_from_observation(observation: Mapping[str, Any]) -> str:
    phase = _phase_from_observation(observation)
    round_value = _as_nonnegative_int(observation.get("round_in_phase"), 0)
    actions_value = _as_nonnegative_int(observation.get("actions_remaining"), 0)
    return _context_key(phase, round_value, actions_value)


def _context_key(phase: str, round_value: int, actions_value: int) -> str:
    return f"{phase}|r{min(round_value, _MAX_ROUND_BUCKET)}|a{min(actions_value, 2)}"


def _phase_from_observation(observation: Mapping[str, Any]) -> str:
    global_features = observation.get("global_features")
    if isinstance(global_features, list) and len(global_features) >= 3:
        try:
            index = max(range(3), key=lambda item: float(global_features[item]))
        except (TypeError, ValueError):
            index = 0
        return ("canal", "railroad", "game_end")[index]
    value = observation.get("phase")
    return value.strip().lower() if isinstance(value, str) and value.strip() else "unknown"


def _freeze_nested_counts(
    counts: Mapping[str, Counter[str]],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    return tuple(
        (context, tuple(sorted(counter.items())))
        for context, counter in sorted(counts.items())
    )


def _counter_from_pairs(values: Iterable[tuple[str, int]]) -> Counter[str]:
    return Counter({str(key): int(value) for key, value in values})


def _family_counts(counts: Mapping[str, int]) -> Counter[str]:
    result: Counter[str] = Counter()
    for intent, count in counts.items():
        result[str(intent).partition(":")[0]] += int(count)
    return result


def _as_nonnegative_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    if not values or any(
        not math.isfinite(float(value)) or float(value) < 0 for value in values
    ):
        raise ValueError("demonstration probabilities must be finite and non-negative")
    total = sum(float(value) for value in values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("demonstration probabilities have no mass")
    return tuple(float(value) / total for value in values)


__all__ = [
    "DEMONSTRATION_PRIOR_VERSION",
    "DemonstrationPrior",
    "action_family",
    "action_intent_signature",
    "blend_policy_with_demonstration",
    "load_demonstration_prior",
]
