from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .policy_normalization import card_invariant_action_intent


LEGACY_STRATEGY_PRIOR_VERSION = "human-strategy-v1"
CAPPED_STRATEGY_PRIOR_VERSION = "human-strategy-v2-capped-canal-guard"
GROUPED_STRATEGY_PRIOR_VERSION = "human-strategy-v2-grouped-canal-guard"
STRATEGY_PRIOR_VERSION = "human-strategy-v3-balanced-industry-canal-guard"
LIFECYCLE_STRATEGY_PRIOR_VERSION = "human-strategy-v4-industry-lifecycle"
MAP_AWARE_STRATEGY_PRIOR_VERSION = "human-strategy-v5-map-aware-lifecycle"
MAP_AWARE_NETWORK_SCORING_VERSION = "network-route-v1"
SUPPORTED_STRATEGY_PRIOR_VERSIONS = (
    LEGACY_STRATEGY_PRIOR_VERSION,
    CAPPED_STRATEGY_PRIOR_VERSION,
    GROUPED_STRATEGY_PRIOR_VERSION,
    STRATEGY_PRIOR_VERSION,
    LIFECYCLE_STRATEGY_PRIOR_VERSION,
    MAP_AWARE_STRATEGY_PRIOR_VERSION,
)
CANAL_NETWORK_PRODUCTIVE_CAP = 0.18
CANAL_NETWORK_FALLBACK_CAP = 0.35
LIFECYCLE_CANAL_NETWORK_CAP = 0.12

# These references provide the rules and strategic principles used by the prior.
# They are not replay datasets and are kept in the shard metadata for auditability.
STRATEGY_REFERENCE_SOURCES = (
    "https://rulespal.com/brass-birmingham/rulebook",
    "https://boardgamegeek.com/thread/3100883/brass-strategy-guide-to-score-5vpaction-and-win-th",
    "https://github.com/npow/brass-birmingham",
)
HUMAN_REFERENCE_SCORE_SAMPLES = (102, 126, 140, 144, 155, 157, 160, 168, 178)
HUMAN_REFERENCE_TARGET_RANGE = (100, 140)

_EPSILON = 1e-8
_MAX_LOG_SCORE = 4.0
_INDUSTRY_COAL = 0
_INDUSTRY_IRON = 1
_INDUSTRY_BEER = 2
_INDUSTRY_GOODS = 3
_INDUSTRY_POTTERY = 4
_INDUSTRY_COTTON = 5
_SELLABLE_INDUSTRIES = (_INDUSTRY_COTTON, _INDUSTRY_POTTERY, _INDUSTRY_GOODS)
_NUM_PLAYERS = 4
_NUM_MAP_LOCATIONS = 27
_OBSERVATION_LOCATION_SCALE = 54.0
_TOWN_BUILD_LOCATION_RANGES = (
    (0, 2),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10),
    (10, 12),
    (12, 15),
    (15, 17),
    (17, 19),
    (19, 22),
    (22, 25),
    (25, 28),
    (28, 30),
    (30, 32),
    (32, 34),
    (34, 36),
    (36, 40),
    (40, 42),
    (42, 45),
    (45, 47),
)
_TRADE_POST_LOCATION_BY_SLOT = (22, 23, 23, 24, 24, 25, 25, 26, 26)
_MERCHANT_INDUSTRIES = {
    0: frozenset(_SELLABLE_INDUSTRIES),
    1: frozenset((_INDUSTRY_COTTON,)),
    2: frozenset((_INDUSTRY_GOODS,)),
    3: frozenset((_INDUSTRY_POTTERY,)),
    4: frozenset(),
}
_BEER_NEEDED = {
    _INDUSTRY_GOODS: (1, 1, 0, 1, 2, 1, 0, 1),
    _INDUSTRY_POTTERY: (1, 1, 2, 1, 2),
    _INDUSTRY_COTTON: (1, 1, 1, 1),
}
_ROAD_VP = {
    _INDUSTRY_COAL: (2, 1, 1, 1),
    _INDUSTRY_IRON: (1, 1, 1, 1),
    _INDUSTRY_BEER: (2, 2, 2, 2),
    _INDUSTRY_GOODS: (2, 1, 0, 1, 2, 1, 0, 1),
    _INDUSTRY_POTTERY: (1, 1, 1, 1, 1),
    _INDUSTRY_COTTON: (1, 2, 1, 1),
}


@dataclass(frozen=True)
class StrategyPrior:
    version: str
    phase: str
    probabilities: tuple[float, ...]
    scores: tuple[float, ...]
    action_families: tuple[str, ...]
    family_scores: tuple[float, ...]
    guarded_actions: tuple[bool, ...]


def build_strategy_prior(
    observation: dict[str, Any],
    legal_record: dict[str, Any],
    *,
    version: str = STRATEGY_PRIOR_VERSION,
) -> StrategyPrior:
    actions = legal_record.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("strategy prior requires a non-empty legal action list")

    phase = _phase_name(observation)
    context = _build_context(observation, phase)
    action_families = tuple(_action_family(action) for action in actions)
    context["has_sell_action"] = any(
        family == "sell" for family in action_families
    )
    scores = tuple(
        (
            _score_map_aware_action(action, context)
            if version == MAP_AWARE_STRATEGY_PRIOR_VERSION
            else (
                _score_lifecycle_action(action, context)
                if version == LIFECYCLE_STRATEGY_PRIOR_VERSION
                else _score_action(action, context)
            )
        )
        for action in actions
    )
    has_build_action = any(family == "build" for family in action_families)
    has_productive_industry_action = any(
        family in {"build", "sell"} for family in action_families
    )
    guard_network = (
        phase == "canal"
        and not context["own_buildings"]
        and has_build_action
    )
    if version == LEGACY_STRATEGY_PRIOR_VERSION:
        probabilities = _softmax(scores)
        family_scores = scores
        guarded_actions = tuple(False for _ in actions)
    elif version in {CAPPED_STRATEGY_PRIOR_VERSION, STRATEGY_PRIOR_VERSION}:
        guarded_actions = tuple(
            guard_network and family == "network"
            for family in action_families
        )
        probabilities = _apply_guard(_softmax(scores), guarded_actions)
        if phase == "canal" and not guard_network:
            probabilities = _cap_family_probability(
                probabilities,
                action_families,
                family="network",
                maximum_mass=(
                    CANAL_NETWORK_PRODUCTIVE_CAP
                    if has_productive_industry_action
                    else CANAL_NETWORK_FALLBACK_CAP
                ),
            )
        if version == STRATEGY_PRIOR_VERSION:
            probabilities = _balance_build_industries(
                probabilities,
                scores,
                actions,
            )
        family_scores = scores
    elif version == GROUPED_STRATEGY_PRIOR_VERSION:
        guarded_actions = tuple(
            guard_network and family == "network"
            for family in action_families
        )
        probabilities, family_scores = _grouped_softmax(
            scores,
            action_families,
            guarded_actions,
        )
    elif version in {
        LIFECYCLE_STRATEGY_PRIOR_VERSION,
        MAP_AWARE_STRATEGY_PRIOR_VERSION,
    }:
        guarded_actions = tuple(
            guard_network and family == "network"
            for family in action_families
        )
        probabilities = _intent_grouped_softmax(
            scores,
            actions,
            guarded_actions,
        )
        if phase == "canal" and (
            context["has_sell_action"]
            or context["unflipped_sellable_count"] == 0
        ):
            probabilities = _cap_family_probability(
                probabilities,
                action_families,
                family="network",
                maximum_mass=LIFECYCLE_CANAL_NETWORK_CAP,
            )
        probabilities = _balance_build_industries(
            probabilities,
            scores,
            actions,
        )
        family_maxima = {
            family: max(
                score
                for score, candidate_family in zip(
                    scores, action_families, strict=True
                )
                if candidate_family == family
            )
            for family in set(action_families)
        }
        family_scores = tuple(
            family_maxima[family] for family in action_families
        )
    else:
        raise ValueError(f"unsupported strategy prior version {version!r}")
    return StrategyPrior(
        version=version,
        phase=phase,
        probabilities=probabilities,
        scores=scores,
        action_families=action_families,
        family_scores=family_scores,
        guarded_actions=guarded_actions,
    )


def blend_policy_with_strategy(
    model_probabilities: Sequence[float],
    strategy_probabilities: Sequence[float],
    strength: float,
    *,
    guarded_actions: Sequence[bool] | None = None,
) -> tuple[float, ...]:
    if len(model_probabilities) != len(strategy_probabilities):
        raise ValueError("model and strategy probabilities must have equal lengths")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strategy prior strength must be between 0 and 1")
    if not model_probabilities:
        raise ValueError("cannot blend an empty action distribution")
    if (
        guarded_actions is not None
        and len(guarded_actions) != len(model_probabilities)
    ):
        raise ValueError("guarded action mask must match the policy length")

    model = _normalize(model_probabilities, "model")
    strategy = _normalize(strategy_probabilities, "strategy")
    if strength == 0.0:
        return model

    blended = tuple(
        0.0
        if guarded_actions is not None and guarded_actions[index]
        else math.exp(
            (1.0 - strength) * math.log(max(model_value, _EPSILON))
            + strength * math.log(max(strategy_value, _EPSILON))
        )
        for index, (model_value, strategy_value) in enumerate(
            zip(model, strategy, strict=True)
        )
    )
    return _normalize(blended, "blended")


def _build_context(observation: dict[str, Any], phase: str) -> dict[str, Any]:
    actor = _as_int(observation.get("decision_player"), 0)
    players = observation.get("players_public")
    player = players[actor] if isinstance(players, list) and actor < len(players) else []
    money = _as_float(player[3], 0.0) * 100.0 if len(player) > 3 else 0.0
    income_level = _as_float(player[4], 0.0) * 100.0 if len(player) > 4 else 0.0
    income = _as_float(player[5], 0.0) * 30.0 if len(player) > 5 else 0.0

    industry_mats = observation.get("industry_mats")
    own_mat = (
        industry_mats[actor]
        if isinstance(industry_mats, list) and actor < len(industry_mats)
        else []
    )
    next_levels = tuple(
        _decode_next_level(own_mat, industry_index) for industry_index in range(6)
    )

    buildings = observation.get("buildings")
    all_buildings = []
    own_buildings = []
    open_build_slots: list[list[frozenset[int]]] = [[] for _ in range(20)]
    if isinstance(buildings, list):
        for location, row in enumerate(buildings):
            if not isinstance(row, list) or len(row) < 1 + 4 + 6 + 3:
                continue
            if _as_float(row[0], 0.0) <= 0.0:
                town = _town_for_build_location(location)
                allowed_offset = 1 + _NUM_PLAYERS + 6 + 3
                if town is not None and town < 20 and len(row) >= allowed_offset + 6:
                    open_build_slots[town].append(
                        frozenset(
                            industry
                            for industry in range(6)
                            if _as_float(row[allowed_offset + industry], 0.0) > 0.5
                        )
                    )
                continue
            owner = next(
                (
                    player_index
                    for player_index in range(_NUM_PLAYERS)
                    if 1 + player_index < len(row)
                    and _as_float(row[1 + player_index], 0.0) > 0.5
                ),
                -1,
            )
            industry = next(
                (
                    index
                    for index in range(6)
                    if 1 + 4 + index < len(row)
                    and _as_float(row[1 + 4 + index], 0.0) > 0.5
                ),
                -1,
            )
            level_offset = 1 + 4 + 6
            flipped_offset = level_offset + 2
            resource = _as_float(row[level_offset + 1], 0.0)
            building = {
                "location": location,
                "town": _town_for_build_location(location),
                "owner": owner,
                "industry": industry,
                "level": _decode_level(row[level_offset]),
                "flipped": _as_float(row[flipped_offset], 0.0) > 0.5,
                "resource": resource,
                "resource_units": max(0, int(round(resource * 5.0))),
            }
            all_buildings.append(building)
            if owner == actor:
                own_buildings.append(building)

    unflipped_sellable_count = sum(
        not building["flipped"]
        and building["industry"]
        in (_INDUSTRY_COTTON, _INDUSTRY_POTTERY, _INDUSTRY_GOODS)
        for building in own_buildings
    )
    unflipped_beer_count = sum(
        not building["flipped"]
        and building["industry"] == _INDUSTRY_BEER
        and building["resource"] > 0.0
        for building in own_buildings
    )
    industry_building_counts = tuple(
        sum(building["industry"] == industry for building in own_buildings)
        for industry in range(6)
    )
    road_context = _build_road_context(observation, actor, phase, own_buildings)
    self_hand_counts = observation.get("self_hand_counts")

    return {
        "phase": phase,
        "money": money,
        "income_level": income_level,
        "income": income,
        "next_levels": next_levels,
        "all_buildings": all_buildings,
        "own_buildings": own_buildings,
        "unflipped_sellable_count": unflipped_sellable_count,
        "unflipped_beer_count": unflipped_beer_count,
        "industry_building_counts": industry_building_counts,
        "merchants": _build_merchant_context(observation),
        "open_build_slots": tuple(tuple(slots) for slots in open_build_slots),
        "self_hand_counts": (
            tuple(_as_float(value, 0.0) for value in self_hand_counts)
            if isinstance(self_hand_counts, list)
            else ()
        ),
        **road_context,
    }


def _score_action(action: dict[str, Any], context: dict[str, Any]) -> float:
    root = _action_root(action)
    phase = context["phase"]
    score = 0.0

    if root == "pass":
        return -2.0
    if root == "scout":
        return -3.0
    if root == "loan":
        score = -0.55
        if context["money"] < 8.0 and context["income_level"] > -7.0:
            score += 0.75
        if context["money"] < 4.0 and context["income_level"] > -7.0:
            score += 0.55
        if context["income_level"] < 0.0:
            score += 0.08 * context["income_level"]
        if context["income_level"] <= -7.0:
            score -= 1.5
        return score

    if root in ("build", "build_building"):
        industry = _action_industry(action)
        location = _as_int(action.get("build_location"), -1)
        score = 0.15
        if phase == "canal":
            score += 0.2
        if industry == _INDUSTRY_BEER:
            score += 0.85
            if location in (47, 48):
                score += 0.65
        elif industry in (_INDUSTRY_COTTON, _INDUSTRY_POTTERY):
            score += 0.55
        elif industry == _INDUSTRY_GOODS:
            score += 0.25
        else:
            score += 0.08

        next_level = _next_level(context, industry)
        if next_level >= 1:
            # Level II+ buildings can score in both eras and are the most useful
            # way to turn a Canal action into durable VP.
            score += 0.55
        if phase == "railroad" and next_level == 0:
            score -= 0.75
        return score

    if root in ("sell",):
        score = 0.95
        if phase == "canal":
            score += 0.35
        targets = action.get("sell_targets")
        if isinstance(targets, list) and targets:
            valuable_targets = 0
            for location in targets:
                building = _building_at(context, _as_int(location, -1))
                if building is None or building["flipped"]:
                    continue
                valuable_targets += int(
                    building["industry"]
                    in (_INDUSTRY_COTTON, _INDUSTRY_POTTERY, _INDUSTRY_GOODS)
                )
            score += min(0.6, 0.2 * valuable_targets)
        if any(
            isinstance(choice, dict) and choice.get("kind") == "beer_source"
            for choice in action.get("choices", [])
            if isinstance(action.get("choices"), list)
        ):
            score += 0.2
        return score

    if root in ("develop", "develop_double"):
        score = -0.35
        if phase == "canal":
            score += 0.05
        industries = _action_industries(action)
        score += sum(
            0.15
            for industry in industries
            if industry in (_INDUSTRY_COTTON, _INDUSTRY_POTTERY, _INDUSTRY_GOODS)
        )
        score += sum(
            0.08
            for industry in industries
            if _next_level(context, industry) <= 1
        )
        if root == "develop_double":
            score -= 0.35
        if not context["own_buildings"]:
            score -= 0.55
        return score

    if root in ("network", "double_network"):
        # The low Canal-era prior addresses the observed failure mode where the
        # model builds many links without ever activating industries. Once the
        # player has an unflipped tile, a Canal link becomes purposeful because
        # it can deliver coal/iron and turn that tile into scored VP.
        if phase == "canal":
            has_unflipped = any(
                not building["flipped"] for building in context["own_buildings"]
            )
            score = 0.35 if has_unflipped else -1.0
        else:
            score = 0.35
        if root == "double_network":
            score += 0.1 if context["money"] >= 15.0 else -0.2
        return score

    return 0.0


def _score_lifecycle_action(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    root = _action_root(action)
    phase = context["phase"]
    has_sell_action = bool(context["has_sell_action"])
    unflipped_sellable = int(context["unflipped_sellable_count"])

    if root == "pass":
        return -4.0
    if root == "scout":
        return -1.75
    if root == "loan":
        if context["money"] < 8.0 and context["income_level"] > -4.0:
            return 0.75
        if context["money"] < 15.0 and context["income_level"] >= 0.0:
            return 0.25
        return -1.25

    if root in ("sell",):
        targets = action.get("sell_targets")
        target_count = len(targets) if isinstance(targets, list) else 0
        return 2.6 + min(1.0, 0.35 * target_count)

    if root in ("build", "build_building"):
        industry = _action_industry(action)
        location = _as_int(action.get("build_location"), -1)
        counts = context["industry_building_counts"]
        existing_count = counts[industry] if 0 <= industry < len(counts) else 0

        if industry in (_INDUSTRY_COTTON, _INDUSTRY_POTTERY, _INDUSTRY_GOODS):
            score = 1.55 - 0.35 * max(0, unflipped_sellable - 1)
            score -= 0.12 * existing_count
            if has_sell_action:
                score -= 1.1
        elif industry == _INDUSTRY_BEER:
            beer_supply = int(context["unflipped_beer_count"])
            if unflipped_sellable == 0:
                score = 0.15 if beer_supply == 0 else -1.0
            elif beer_supply < unflipped_sellable:
                score = 0.8
            else:
                score = -0.65
            if location in (47, 48) and beer_supply == 0:
                score += 0.2
        elif industry == _INDUSTRY_IRON:
            score = 1.0 - 0.15 * existing_count
        elif industry == _INDUSTRY_COAL:
            score = 0.9 - 0.15 * existing_count
        else:
            score = 0.4

        next_level = _next_level(context, industry)
        if next_level >= 1:
            score += 0.4
        if phase == "railroad" and next_level == 0:
            score -= 0.8
        return score

    if root in ("network", "double_network"):
        if has_sell_action:
            return -0.8
        if unflipped_sellable > 0:
            score = 1.35
        elif context["own_buildings"]:
            score = 0.1
        else:
            score = -1.2
        if root == "double_network":
            score += 0.25 if context["money"] >= 15.0 else -0.35
        return score

    if root in ("develop", "develop_double"):
        if has_sell_action:
            return -0.9
        industries = _action_industries(action)
        score = 0.4
        score += sum(
            0.25
            for industry in industries
            if industry in (_INDUSTRY_COTTON, _INDUSTRY_POTTERY, _INDUSTRY_GOODS)
        )
        score += sum(
            0.12 for industry in industries if _next_level(context, industry) <= 1
        )
        if root == "develop_double":
            score += 0.2
        if unflipped_sellable > 0:
            score -= 0.45
        return score

    return 0.0


def _score_map_aware_action(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    score = _score_lifecycle_action(action, context)
    if _action_family(action) != "network":
        return score

    candidate_roads = frozenset(_action_road_indices(action))
    road_locations = context.get("road_locations")
    if not candidate_roads or not isinstance(road_locations, tuple):
        return score
    if any(
        road < 0 or road >= len(road_locations) or not road_locations[road]
        for road in candidate_roads
    ):
        return score

    sellable_buildings = [
        building
        for building in context["own_buildings"]
        if not building["flipped"]
        and building["industry"] in _SELLABLE_INDUSTRIES
        and building["town"] is not None
    ]
    newly_ready = 0
    newly_connected = 0
    distance_progress = 0.0
    stalled_routes = 0
    relevant_merchant_locations: set[int] = set()
    for building in sellable_buildings:
        merchant_locations = _merchant_locations_for_industry(
            context, building["industry"]
        )
        relevant_merchant_locations.update(merchant_locations)
        if not merchant_locations:
            continue
        before_distance = _minimum_new_roads_to_targets(
            context, building["town"], merchant_locations, frozenset()
        )
        after_distance = _minimum_new_roads_to_targets(
            context, building["town"], merchant_locations, candidate_roads
        )
        before_connected = before_distance == 0
        after_connected = after_distance == 0
        before_ready = before_connected and _has_sale_beer(
            context, building, frozenset()
        )
        after_ready = after_connected and _has_sale_beer(
            context, building, candidate_roads
        )
        newly_ready += int(after_ready and not before_ready)
        newly_connected += int(after_connected and not before_connected)
        if before_distance is not None and after_distance is not None:
            distance_progress += max(0, before_distance - after_distance)
            stalled_routes += int(after_distance >= before_distance)

    candidate_locations = {
        location
        for road in candidate_roads
        for location in road_locations[road]
    }
    direct_relevant_merchants = len(
        candidate_locations.intersection(relevant_merchant_locations)
    )
    new_network_locations = candidate_locations.difference(
        context.get("own_network_locations", frozenset())
    )
    road_vp = _candidate_road_vp(context, candidate_roads)
    build_access = _new_industry_build_access(context, new_network_locations)

    score += 2.8 * newly_ready
    score += 1.1 * newly_connected
    score += 0.8 * distance_progress
    score += 0.35 * direct_relevant_merchants
    score += 0.18 * road_vp
    score += min(0.45, 0.12 * build_access)

    if sellable_buildings and not (
        newly_ready or newly_connected or distance_progress > 0.0
    ):
        score -= 1.6
    elif sellable_buildings:
        score -= 0.15 * stalled_routes
    return score


def _action_root(action: dict[str, Any]) -> str:
    key_root = str(action.get("key", "")).split("|", 1)[0]
    named_root = str(action.get("root_action_name") or "")
    root = key_root or named_root
    if root in {
        "build",
        "network",
        "double_network",
        "develop",
        "develop_double",
        "sell",
        "loan",
        "scout",
        "pass",
    }:
        return root
    return {
        "build_building": "build",
        "build_railroad": "network",
        "build_double_railroad": "double_network",
    }.get(named_root, root)


def _action_family(action: dict[str, Any]) -> str:
    root = _action_root(action)
    if root == "double_network":
        return "network"
    if root == "develop_double":
        return "develop"
    return root


def _action_industry(action: dict[str, Any]) -> int:
    value = action.get("selected_industry")
    if value is not None:
        return _as_int(value, -1)
    return _first_prefixed_value(action.get("key", ""), "i")


def _action_industries(action: dict[str, Any]) -> tuple[int, ...]:
    values = []
    for field in ("selected_industry", "selected_second_industry"):
        value = action.get(field)
        if value is not None:
            values.append(_as_int(value, -1))
    if values:
        return tuple(value for value in values if 0 <= value < 6)
    key = str(action.get("key", ""))
    return tuple(
        value
        for value in (_first_prefixed_value(key, "i"),)
        if 0 <= value < 6
    )


def _action_road_indices(action: dict[str, Any]) -> tuple[int, ...]:
    values = []
    for field in ("road", "second_road"):
        value = action.get(field)
        if value is not None:
            values.append(_as_int(value, -1))
    if values:
        return tuple(value for value in values if value >= 0)
    return tuple(
        int(token[1:])
        for token in str(action.get("key", "")).partition("|")[2].split(",")
        if token.startswith("r") and token[1:].isdigit()
    )


def _first_prefixed_value(key: object, prefix: str) -> int:
    for token in str(key).replace("|", ",").split(","):
        if token.startswith(prefix) and token[len(prefix) :].isdigit():
            return int(token[len(prefix) :])
    return -1


def _next_level(context: dict[str, Any], industry: int) -> int:
    levels = context["next_levels"]
    return levels[industry] if 0 <= industry < len(levels) else 0


def _building_at(context: dict[str, Any], location: int) -> dict[str, Any] | None:
    for building in context["own_buildings"]:
        if building["location"] == location:
            return building
    return None


def _town_for_build_location(location: int) -> int | None:
    if location == 47:
        return 20
    if location == 48:
        return 21
    for town, (start, end) in enumerate(_TOWN_BUILD_LOCATION_RANGES):
        if start <= location < end:
            return town
    return None


def _build_road_context(
    observation: dict[str, Any],
    actor: int,
    phase: str,
    own_buildings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = observation.get("roads")
    if not isinstance(rows, list) or not rows:
        return {
            "road_locations": None,
            "built_roads": frozenset(),
            "phase_legal_roads": frozenset(),
            "own_network_locations": frozenset(
                building["town"]
                for building in own_buildings
                if building["town"] is not None
            ),
        }

    road_locations: list[tuple[int, ...]] = []
    built_roads: set[int] = set()
    phase_legal_roads: set[int] = set()
    own_network_locations = {
        building["town"]
        for building in own_buildings
        if building["town"] is not None
    }
    owner_offset = 1 + actor
    canal_offset = 1 + _NUM_PLAYERS
    rail_offset = canal_offset + 1
    location_offset = rail_offset + 1
    for road, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < location_offset + 3:
            road_locations.append(())
            continue
        locations = tuple(
            location
            for location in (
                _decode_map_location(value)
                for value in row[location_offset : location_offset + 3]
            )
            if location is not None
        )
        road_locations.append(locations)
        if _as_float(row[0], 0.0) > 0.5:
            built_roads.add(road)
            if owner_offset < len(row) and _as_float(row[owner_offset], 0.0) > 0.5:
                own_network_locations.update(locations)
        phase_flag_offset = canal_offset if phase == "canal" else rail_offset
        if _as_float(row[phase_flag_offset], 0.0) > 0.5:
            phase_legal_roads.add(road)
    return {
        "road_locations": tuple(road_locations),
        "built_roads": frozenset(built_roads),
        "phase_legal_roads": frozenset(phase_legal_roads),
        "own_network_locations": frozenset(own_network_locations),
    }


def _build_merchant_context(observation: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    slots = observation.get("trade_post_slots")
    beer = observation.get("trade_post_beer")
    if not isinstance(slots, list):
        return ()
    merchants = []
    for slot, merchant_type_value in enumerate(slots):
        if slot >= len(_TRADE_POST_LOCATION_BY_SLOT):
            break
        merchant_type = _as_int(merchant_type_value, -1)
        industries = _MERCHANT_INDUSTRIES.get(merchant_type, frozenset())
        if not industries:
            continue
        merchants.append(
            {
                "slot": slot,
                "location": _TRADE_POST_LOCATION_BY_SLOT[slot],
                "industries": industries,
                "has_beer": (
                    isinstance(beer, list)
                    and slot < len(beer)
                    and _as_float(beer[slot], 0.0) > 0.5
                ),
            }
        )
    return tuple(merchants)


def _decode_map_location(value: object) -> int | None:
    normalized = _as_float(value, -1.0)
    if normalized < 0.0:
        return None
    location = int(round(normalized * _OBSERVATION_LOCATION_SCALE))
    return location if 0 <= location < _NUM_MAP_LOCATIONS else None


def _merchant_locations_for_industry(
    context: dict[str, Any], industry: int
) -> frozenset[int]:
    return frozenset(
        merchant["location"]
        for merchant in context.get("merchants", ())
        if industry in merchant["industries"]
    )


def _minimum_new_roads_to_targets(
    context: dict[str, Any],
    start: int,
    targets: frozenset[int],
    candidate_roads: frozenset[int],
) -> int | None:
    if start in targets:
        return 0
    road_locations = context.get("road_locations")
    if not isinstance(road_locations, tuple):
        return None
    built_roads = context.get("built_roads", frozenset())
    usable_roads = context.get("phase_legal_roads", frozenset()).union(built_roads)
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(_NUM_MAP_LOCATIONS)
    ]
    for road in usable_roads:
        if road < 0 or road >= len(road_locations):
            continue
        locations = road_locations[road]
        cost = 0 if road in built_roads or road in candidate_roads else 1
        for source in locations:
            for destination in locations:
                if source != destination:
                    adjacency[source].append((destination, cost))

    distances = [math.inf] * _NUM_MAP_LOCATIONS
    distances[start] = 0
    pending = [(0, start)]
    while pending:
        distance, location = heapq.heappop(pending)
        if distance != distances[location]:
            continue
        if location in targets:
            return int(distance)
        for destination, cost in adjacency[location]:
            next_distance = distance + cost
            if next_distance < distances[destination]:
                distances[destination] = next_distance
                heapq.heappush(pending, (next_distance, destination))
    return None


def _has_sale_beer(
    context: dict[str, Any],
    building: dict[str, Any],
    candidate_roads: frozenset[int],
) -> bool:
    needed_by_level = _BEER_NEEDED.get(building["industry"], ())
    level = int(building["level"])
    beer_needed = needed_by_level[level] if 0 <= level < len(needed_by_level) else 1
    if beer_needed <= 0:
        return True

    available = sum(
        other["resource_units"]
        for other in context["own_buildings"]
        if other["industry"] == _INDUSTRY_BEER
        and not other["flipped"]
        and other["resource_units"] > 0
    )
    building_town = building["town"]
    for other in context.get("all_buildings", ()):
        if (
            other["owner"] < 0
            or other["owner"] == building["owner"]
            or other["industry"] != _INDUSTRY_BEER
            or other["flipped"]
            or other["resource_units"] <= 0
            or other["town"] is None
        ):
            continue
        distance = _minimum_new_roads_to_targets(
            context,
            building_town,
            frozenset((other["town"],)),
            candidate_roads,
        )
        if distance == 0:
            available += other["resource_units"]
    for merchant in context.get("merchants", ()):
        if not merchant["has_beer"] or building["industry"] not in merchant["industries"]:
            continue
        distance = _minimum_new_roads_to_targets(
            context,
            building_town,
            frozenset((merchant["location"],)),
            candidate_roads,
        )
        if distance == 0:
            available += 1
    return available >= beer_needed


def _candidate_road_vp(
    context: dict[str, Any], candidate_roads: frozenset[int]
) -> int:
    road_locations = context["road_locations"]
    vp = 0
    for road in candidate_roads:
        endpoint_locations = set(road_locations[road])
        for building in context.get("all_buildings", ()):
            if not building["flipped"] or building["town"] not in endpoint_locations:
                continue
            values = _ROAD_VP.get(building["industry"], ())
            level = int(building["level"])
            if 0 <= level < len(values):
                vp += values[level]
    return vp


def _new_industry_build_access(
    context: dict[str, Any], new_network_locations: set[int]
) -> int:
    hand = context.get("self_hand_counts")
    open_slots = context.get("open_build_slots")
    if not isinstance(hand, tuple) or not isinstance(open_slots, tuple):
        return 0
    access = 0
    for town in new_network_locations:
        if town < 0 or town >= len(open_slots):
            continue
        for industries in open_slots[town]:
            if any(_hand_can_build_industry(hand, industry) for industry in industries):
                access += 1
    return access


def _hand_can_build_industry(hand: tuple[float, ...], industry: int) -> bool:
    exact = 20 + industry
    if exact < len(hand) and hand[exact] > 0.0:
        return True
    if (
        industry in (_INDUSTRY_GOODS, _INDUSTRY_COTTON)
        and len(hand) > 26
        and hand[26] > 0.0
    ):
        return True
    return len(hand) > 28 and hand[28] > 0.0


def _phase_name(observation: dict[str, Any]) -> str:
    features = observation.get("global_features")
    if isinstance(features, list) and len(features) >= 3:
        index = max(range(3), key=lambda item: _as_float(features[item], 0.0))
        return ("canal", "railroad", "game_end")[index]
    return "canal"


def _decode_next_level(row: object, industry: int) -> int:
    if not isinstance(row, list) or industry < 0 or industry * 3 >= len(row):
        return 0
    # Observation industry levels use (level_index + 1) / 8.
    return max(0, min(7, int(round(_as_float(row[industry * 3], 0.125) * 8.0 - 1.0))))


def _decode_level(value: object) -> int:
    return max(0, min(7, int(round(_as_float(value, 0.125) * 8.0 - 1.0))))


def _softmax(scores: Sequence[float]) -> tuple[float, ...]:
    maximum = max(scores)
    exponentials = tuple(math.exp(max(-_MAX_LOG_SCORE, min(_MAX_LOG_SCORE, score - maximum))) for score in scores)
    return _normalize(exponentials, "strategy")


def _grouped_softmax(
    scores: Sequence[float],
    action_families: Sequence[str],
    guarded_actions: Sequence[bool],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not (len(scores) == len(action_families) == len(guarded_actions)):
        raise ValueError("grouped strategy inputs must have equal lengths")

    all_family_indices: dict[str, list[int]] = {}
    family_indices: dict[str, list[int]] = {}
    for index, family in enumerate(action_families):
        all_family_indices.setdefault(family, []).append(index)
        if guarded_actions[index]:
            continue
        family_indices.setdefault(family, []).append(index)
    if not family_indices:
        raise ValueError("strategy guard removed every legal action")

    family_names = tuple(family_indices)
    score_by_family = {
        family: max(scores[index] for index in indices)
        for family, indices in all_family_indices.items()
    }
    family_probabilities = _softmax(
        tuple(score_by_family[family] for family in family_names)
    )
    probability_by_family = dict(zip(family_names, family_probabilities, strict=True))

    probabilities = [0.0] * len(scores)
    family_scores = [score_by_family[family] for family in action_families]
    for family, indices in family_indices.items():
        conditional = _softmax(tuple(scores[index] for index in indices))
        for index, conditional_probability in zip(indices, conditional, strict=True):
            probabilities[index] = (
                probability_by_family[family] * conditional_probability
            )
            family_scores[index] = score_by_family[family]

    return (
        _normalize(probabilities, "grouped strategy"),
        tuple(family_scores),
    )


def _intent_grouped_softmax(
    scores: Sequence[float],
    actions: Sequence[dict[str, Any]],
    guarded_actions: Sequence[bool],
) -> tuple[float, ...]:
    if not (len(scores) == len(actions) == len(guarded_actions)):
        raise ValueError("intent-grouped strategy inputs must have equal lengths")

    grouped_indices: dict[str, list[int]] = {}
    for index, action in enumerate(actions):
        if guarded_actions[index]:
            continue
        key = str(action.get("key") or "")
        grouped_indices.setdefault(card_invariant_action_intent(key), []).append(index)
    if not grouped_indices:
        raise ValueError("strategy guard removed every legal action")

    group_names = tuple(grouped_indices)
    group_probabilities = _softmax(
        tuple(
            max(scores[index] for index in grouped_indices[group])
            for group in group_names
        )
    )
    probabilities = [0.0] * len(scores)
    for group, group_probability in zip(
        group_names, group_probabilities, strict=True
    ):
        indices = grouped_indices[group]
        conditional = _softmax(tuple(scores[index] for index in indices))
        for index, conditional_probability in zip(indices, conditional, strict=True):
            probabilities[index] = group_probability * conditional_probability
    return _normalize(probabilities, "intent-grouped strategy")


def _apply_guard(
    probabilities: Sequence[float], guarded_actions: Sequence[bool]
) -> tuple[float, ...]:
    return _normalize(
        tuple(
            0.0 if guarded else probability
            for probability, guarded in zip(
                probabilities, guarded_actions, strict=True
            )
        ),
        "guarded strategy",
    )


def _cap_family_probability(
    probabilities: Sequence[float],
    action_families: Sequence[str],
    *,
    family: str,
    maximum_mass: float,
) -> tuple[float, ...]:
    normalized = _normalize(probabilities, "uncapped strategy")
    family_mass = sum(
        probability
        for probability, action_family in zip(
            normalized, action_families, strict=True
        )
        if action_family == family
    )
    if family_mass <= maximum_mass:
        return normalized
    other_mass = 1.0 - family_mass
    if other_mass <= 0.0:
        return normalized

    family_scale = maximum_mass / family_mass
    other_scale = (1.0 - maximum_mass) / other_mass
    return _normalize(
        tuple(
            probability
            * (family_scale if action_family == family else other_scale)
            for probability, action_family in zip(
                normalized, action_families, strict=True
            )
        ),
        "capped strategy",
    )


def _balance_build_industries(
    probabilities: Sequence[float],
    scores: Sequence[float],
    actions: Sequence[dict[str, Any]],
) -> tuple[float, ...]:
    normalized = _normalize(probabilities, "unbalanced build strategy")
    build_indices = [
        index
        for index, action in enumerate(actions)
        if _action_family(action) == "build"
    ]
    if len(build_indices) < 2:
        return normalized

    industry_indices: dict[int, list[int]] = {}
    for index in build_indices:
        industry_indices.setdefault(_action_industry(actions[index]), []).append(index)
    if len(industry_indices) < 2:
        return normalized

    build_mass = sum(normalized[index] for index in build_indices)
    industry_names = tuple(industry_indices)
    industry_probabilities = _softmax(
        tuple(
            max(scores[index] for index in industry_indices[industry])
            for industry in industry_names
        )
    )
    result = list(normalized)
    for industry, industry_probability in zip(
        industry_names, industry_probabilities, strict=True
    ):
        indices = industry_indices[industry]
        conditional = _softmax(tuple(scores[index] for index in indices))
        for index, conditional_probability in zip(indices, conditional, strict=True):
            result[index] = (
                build_mass * industry_probability * conditional_probability
            )
    return _normalize(result, "balanced build strategy")


def _normalize(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not values or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError(f"{label} probabilities must be finite and non-negative")
    total = sum(float(value) for value in values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} probabilities must have positive mass")
    return tuple(float(value) / total for value in values)


def _as_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
