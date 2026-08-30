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
RESOURCE_AWARE_STRATEGY_PRIOR_VERSION = "human-strategy-v6-resource-aware"
ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION = "human-strategy-v7-action-efficiency-route"
CONSERVATIVE_ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION = (
    "human-strategy-v8-conservative-action-efficiency"
)
MAP_AWARE_NETWORK_SCORING_VERSION = "network-route-v1"
SUPPORTED_STRATEGY_PRIOR_VERSIONS = (
    LEGACY_STRATEGY_PRIOR_VERSION,
    CAPPED_STRATEGY_PRIOR_VERSION,
    GROUPED_STRATEGY_PRIOR_VERSION,
    STRATEGY_PRIOR_VERSION,
    LIFECYCLE_STRATEGY_PRIOR_VERSION,
    MAP_AWARE_STRATEGY_PRIOR_VERSION,
    RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
    ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
    CONSERVATIVE_ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
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
    "https://eriktwice.com/en/2021/01/15/brass-birmingham-understanding-the-industries/",
    "https://eriktwice.com/en/2020/11/06/brass-birmingham-beginner-mistakes/",
    "https://steamcommunity.com/sharedfiles/filedetails/?id=2539095235",
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
_MARKET_COAL_MAX = 14
_MARKET_IRON_MAX = 10
_MARKET_SOURCE = 49
_COAL_PRICE_TABLE = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)
_IRON_PRICE_TABLE = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5)

# These compact tables mirror the engine's industry mat. V6 only needs the
# resource demand and base money cost of the next tile to price an action.
_BUILDING_COAL_DEMAND = (
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (0, 0, 0, 0),
    (1, 0, 2, 0, 1, 0, 1, 0),
    (0, 1, 0, 0, 2),
    (0, 1, 1, 1),
)
_BUILDING_IRON_DEMAND = (
    (0, 0, 1, 1),
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (0, 1, 0, 1, 2, 1, 1, 2),
    (1, 0, 0, 1, 0),
    (0, 0, 1, 1),
)
_BUILDING_MONEY_COST = (
    (5, 7, 8, 10),
    (5, 7, 9, 12),
    (5, 7, 9, 9),
    (8, 10, 12, 8, 16, 20, 16, 20),
    (17, 0, 22, 0, 24),
    (12, 14, 16, 18),
)
_BUILDING_VP = (
    (1, 2, 3, 4),
    (3, 5, 7, 9),
    (4, 5, 7, 10),
    (3, 5, 4, 3, 8, 7, 9, 11),
    (10, 1, 11, 1, 20),
    (5, 5, 9, 12),
)
_BUILDING_ROAD_VP = (
    (2, 1, 1, 1),
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (2, 1, 0, 1, 2, 1, 0, 1),
    (1, 1, 1, 1, 1),
    (1, 2, 1, 1),
)
_BUILDING_RESOURCE_OUTPUT = (
    (2, 3, 4, 5),
    (4, 4, 5, 6),
    (1, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0),
)
_BUILDING_INCOME = (
    (4, 7, 6, 5),
    (3, 3, 2, 1),
    (4, 5, 5, 5),
    (5, 0, 4, 6, 2, 6, 4, 1),
    (5, 1, 5, 1, 5),
    (5, 4, 3, 2),
)


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
    context["actions"] = tuple(actions)
    context["action_families"] = action_families
    context["has_sell_action"] = any(
        family == "sell" for family in action_families
    )
    scores = tuple(
        (
            _score_map_aware_action(action, context)
            if version == MAP_AWARE_STRATEGY_PRIOR_VERSION
            else (
                _score_resource_aware_action(action, context)
                if version == RESOURCE_AWARE_STRATEGY_PRIOR_VERSION
                else (
                    _score_action_efficiency_action(action, context)
                    if version == ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION
                    else (
                        _score_conservative_action_efficiency_action(action, context)
                        if version
                        == CONSERVATIVE_ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION
                        else (
                            _score_lifecycle_action(action, context)
                            if version == LIFECYCLE_STRATEGY_PRIOR_VERSION
                            else _score_action(action, context)
                        )
                    )
                )
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
        RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
        CONSERVATIVE_ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
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
    market_coal = _market_remaining(observation, 10, _MARKET_COAL_MAX)
    market_iron = _market_remaining(observation, 11, _MARKET_IRON_MAX)

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
    unflipped_beer_units = sum(
        building["resource_units"]
        for building in own_buildings
        if not building["flipped"] and building["industry"] == _INDUSTRY_BEER
    )
    unflipped_coal_units = sum(
        building["resource_units"]
        for building in own_buildings
        if not building["flipped"] and building["industry"] == _INDUSTRY_COAL
    )
    unflipped_iron_units = sum(
        building["resource_units"]
        for building in own_buildings
        if not building["flipped"] and building["industry"] == _INDUSTRY_IRON
    )
    sellable_beer_demand = sum(
        _beer_needed_for_building(building)
        for building in own_buildings
        if not building["flipped"]
        and building["industry"] in _SELLABLE_INDUSTRIES
    )
    all_unflipped_beer_units = sum(
        building["resource_units"]
        for building in all_buildings
        if not building["flipped"] and building["industry"] == _INDUSTRY_BEER
    )
    all_unflipped_coal_units = sum(
        building["resource_units"]
        for building in all_buildings
        if not building["flipped"] and building["industry"] == _INDUSTRY_COAL
    )
    all_unflipped_iron_units = sum(
        building["resource_units"]
        for building in all_buildings
        if not building["flipped"] and building["industry"] == _INDUSTRY_IRON
    )
    global_sellable_beer_demand = sum(
        _beer_needed_for_building(building)
        for building in all_buildings
        if not building["flipped"] and building["industry"] in _SELLABLE_INDUSTRIES
    )
    industry_remaining = tuple(
        max(
            0,
            int(
                round(
                    _as_float(
                        own_mat[industry * 3 + 1]
                        if industry * 3 + 1 < len(own_mat)
                        else 0.0,
                        0.0,
                    )
                    * 3.0
                )
            ),
        )
        for industry in range(6)
    )
    industry_building_counts = tuple(
        sum(building["industry"] == industry for building in own_buildings)
        for industry in range(6)
    )
    road_context = _build_road_context(observation, actor, phase, own_buildings)
    self_hand_counts = observation.get("self_hand_counts")
    merchants = _build_merchant_context(observation)
    merchant_beer_units = sum(
        1 for merchant in merchants if merchant.get("has_beer")
    )
    global_features = observation.get("global_features")
    actions_remaining = _as_int(observation.get("actions_remaining"), -1)
    if actions_remaining < 0 and isinstance(global_features, list) and len(global_features) > 7:
        actions_remaining = int(round(_as_float(global_features[7], 1.0) * 2.0))
    round_in_phase = _as_int(observation.get("round_in_phase"), -1)
    if round_in_phase < 0 and isinstance(global_features, list) and len(global_features) > 6:
        round_in_phase = int(round(_as_float(global_features[6], 0.0) * 16.0))

    return {
        "phase": phase,
        "money": money,
        "income_level": income_level,
        "income": income,
        "market_coal": market_coal,
        "market_iron": market_iron,
        "next_levels": next_levels,
        "all_buildings": all_buildings,
        "own_buildings": own_buildings,
        "unflipped_sellable_count": unflipped_sellable_count,
        "unflipped_beer_count": unflipped_beer_count,
        "unflipped_beer_units": unflipped_beer_units,
        "unflipped_coal_units": unflipped_coal_units,
        "unflipped_iron_units": unflipped_iron_units,
        "sellable_beer_demand": sellable_beer_demand,
        "all_unflipped_beer_units": all_unflipped_beer_units,
        "all_unflipped_coal_units": all_unflipped_coal_units,
        "all_unflipped_iron_units": all_unflipped_iron_units,
        "global_sellable_beer_demand": global_sellable_beer_demand,
        "merchant_beer_units": merchant_beer_units,
        "industry_remaining": industry_remaining,
        "actions_remaining": max(0, actions_remaining),
        "round_in_phase": max(0, round_in_phase),
        "industry_building_counts": industry_building_counts,
        "merchants": merchants,
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


def _score_resource_aware_action(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    root = _action_root(action)

    if root == "pass":
        return -4.0
    if root == "scout":
        return _score_resource_aware_scout(action, context)
    if root == "loan":
        return _score_resource_aware_loan(action, context)
    if root in ("build", "build_building"):
        return _score_resource_aware_build(action, context)
    if root == "sell":
        return _score_resource_aware_sell(action, context)
    if root in ("develop", "develop_double"):
        return _score_resource_aware_develop(action, context)
    if root in ("network", "double_network"):
        return _score_resource_aware_network(action, context)
    return _score_lifecycle_action(action, context)


def _score_resource_aware_build(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    industry = _action_industry(action)
    location = _as_int(action.get("build_location"), -1)
    next_level = _next_level(context, industry)
    score = _score_lifecycle_action(action, context)
    score -= 0.10 * _action_market_cost(action, context)
    score += _card_alignment_score(action, industry, location)

    if industry in (_INDUSTRY_GOODS, _INDUSTRY_COTTON):
        if context["phase"] == "canal" and next_level >= 1:
            score += 0.55
        if context["phase"] == "canal":
            score += 0.18
        if _merchant_locations_for_industry(context, industry):
            score += 0.12
    elif industry == _INDUSTRY_POTTERY:
        if context["phase"] == "canal" and next_level >= 1:
            score += 0.35
    elif industry == _INDUSTRY_BEER:
        demand = int(context["sellable_beer_demand"])
        supply = int(context["unflipped_beer_units"])
        if demand > supply:
            score += min(0.9, 0.30 * (demand - supply))
        else:
            score -= min(1.2, 0.25 * (supply - demand + 1))
        if demand == 0:
            score -= 0.35
    elif industry == _INDUSTRY_COAL:
        score += _coal_production_score(context)

    if context["phase"] == "railroad" and next_level == 0:
        score -= 0.25
    return score


def _score_resource_aware_sell(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    score = _score_lifecycle_action(action, context)
    targets = action.get("sell_targets")
    if not isinstance(targets, list) or not targets:
        return score

    beer_demand = _action_sell_beer_demand(action, context)
    beer_sources = sum(
        1
        for choice in action.get("choices", ())
        if isinstance(choice, dict) and choice.get("kind") == "beer_source"
    )
    score += min(0.75, 0.20 * beer_demand)
    if beer_demand > 0 and beer_sources >= beer_demand:
        score += 0.12
    for raw_location in targets:
        building = _building_at(context, _as_int(raw_location, -1))
        if building is None:
            continue
        if building["industry"] in (_INDUSTRY_GOODS, _INDUSTRY_COTTON):
            score += 0.12
        if building["level"] >= 1:
            score += 0.10
    return score


def _score_resource_aware_develop(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    industries = _action_industries(action)
    if not industries:
        return _score_lifecycle_action(action, context)

    utilities = tuple(
        _develop_resource_utility(context, industry) for industry in industries
    )
    # Keep the V5 lifecycle scale as the anchor. Summing absolute utilities
    # makes a double develop dominate all productive actions in the opening.
    score = _score_lifecycle_action(action, context)
    score += 0.25 * sum(max(0.0, utility - 0.5) for utility in utilities)
    score -= 0.10 * _action_market_cost(action, context)
    if context["has_sell_action"]:
        score -= 0.65
    if context["phase"] == "canal" and any(
        industry in (_INDUSTRY_GOODS, _INDUSTRY_COTTON)
        and _next_level(context, industry) <= 1
        for industry in industries
    ):
        score += 0.20
    if any(
        industry == _INDUSTRY_COAL
        and context["unflipped_coal_units"] >= 3
        for industry in industries
    ):
        score -= 0.45
    if any(
        industry == _INDUSTRY_IRON
        and context["unflipped_iron_units"] >= 3
        for industry in industries
    ):
        score -= 0.25

    if _action_root(action) == "develop_double":
        score += 0.10 if len(utilities) == 2 and min(utilities) >= 0.8 else -0.25
    return score


def _score_resource_aware_network(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    score = _score_map_aware_action(action, context)
    market_cost = _action_market_cost(action, context)
    score -= 0.10 * market_cost

    candidate_roads = frozenset(_action_road_indices(action))
    road_locations = context.get("road_locations")
    if not candidate_roads or not isinstance(road_locations, tuple):
        return score
    if any(
        road < 0 or road >= len(road_locations) or not road_locations[road]
        for road in candidate_roads
    ):
        return score

    candidate_locations = {
        location
        for road in candidate_roads
        for location in road_locations[road]
    }
    new_network_locations = candidate_locations.difference(
        context.get("own_network_locations", frozenset())
    )
    road_vp = _candidate_road_vp(context, candidate_roads)
    build_access = _new_industry_build_access(context, new_network_locations)
    frontier = _network_frontier_value(context, candidate_locations, candidate_roads)
    has_merchant_endpoint = any(location >= 22 for location in candidate_locations)

    if context["phase"] == "railroad":
        score += min(0.75, 0.12 * road_vp)
        score += min(0.45, 0.08 * frontier)
        score += min(0.35, 0.10 * build_access)
        if _action_root(action) == "double_network":
            has_action_beer = any(
                isinstance(choice, dict)
                and choice.get("kind") == "action_beer_source"
                for choice in action.get("choices", ())
            )
            cash_buffer = context["money"] - 15.0 - market_cost
            score += 0.22 if has_action_beer else -0.45
            if cash_buffer >= 3.0:
                score += 0.45
            elif cash_buffer < 0.0:
                score -= 0.75
            if road_vp == 0 and build_access == 0 and not has_merchant_endpoint:
                score -= 0.65
    return score


def _score_resource_aware_loan(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    score = _score_lifecycle_action(action, context)
    best = _best_future_action(context)
    if best is not None:
        best_value, best_cost = best
        cash_buffer = context["money"] - best_cost
        if best_value >= 1.0 and cash_buffer < 0.0:
            score += 0.75
        elif best_value >= 1.0 and cash_buffer < 3.0:
            score += 0.45
        elif context["money"] > max(20.0, best_cost * 2.5):
            score -= 0.55
        if best_value < 0.8:
            score -= 0.30
    if context["has_sell_action"]:
        score -= 0.75
    if context["income_level"] <= -4.0:
        score -= 0.55
    return score


def _score_resource_aware_scout(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    card_types = _action_card_types(action)
    if len(card_types) < 3:
        return _score_lifecycle_action(action, context)
    usefulness = tuple(
        _card_usefulness_resource_aware(context, card_type)
        for card_type in card_types
    )
    dead_count = sum(value < 0.6 for value in usefulness)
    score = -1.0 + 0.85 * dead_count - 0.18 * sum(usefulness)
    if dead_count >= 2 and any(
        family == "build" for family in context.get("action_families", ())
    ):
        score += 0.25
    if any(card_type in (27, 28) for card_type in card_types):
        score -= 0.75
    return score


def _score_action_efficiency_action(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    """Add small marginal-value signals on top of the V6 resource prior.

    The web references consistently describe Brass as an action-efficiency
    game. These terms stay deliberately bounded so the model/search policy can
    still override a heuristic when the board state disagrees with the broad
    route signal.
    """
    root = _action_root(action)
    score = _score_resource_aware_action(action, context)
    if root in ("build", "build_building"):
        industry = _action_industry(action)
        location = _as_int(action.get("build_location"), -1)
        return score + _v7_build_efficiency(action, context) + _v7_route_focus(
            industry, context
        ) + _v7_build_location_value(industry, location, context) + _v7_resource_scarcity(
            industry, context
        )
    if root == "sell":
        return score + _v7_sell_efficiency(action, context)
    if root in ("develop", "develop_double"):
        return score + _v7_develop_efficiency(action, context)
    if root in ("network", "double_network"):
        return score + _v7_network_efficiency(action, context)
    return score


def _score_conservative_action_efficiency_action(
    action: dict[str, Any], context: dict[str, Any]
) -> float:
    """Apply the V7 signal as a small residual on top of the V6 score."""
    resource_score = _score_resource_aware_action(action, context)
    action_efficiency_score = _score_action_efficiency_action(action, context)
    return resource_score + 0.25 * (action_efficiency_score - resource_score)


def _v7_build_efficiency(action: dict[str, Any], context: dict[str, Any]) -> float:
    industry = _action_industry(action)
    level = _next_level(context, industry)
    tile = _v7_tile_stats(industry, level)
    if tile is None:
        return 0.0

    cash_cost = _action_cash_cost(action, context)
    gross_value = (
        0.08 * tile[4]
        + 0.025 * max(0, tile[7])
        + 0.045 * tile[5]
        + 0.035 * tile[6]
    )
    if industry in _SELLABLE_INDUSTRIES:
        gross_value += 0.025 * tile[3]
    if context["phase"] == "railroad" and not tile[8]:
        gross_value -= 0.08
    efficiency = gross_value - 0.018 * cash_cost
    return max(-0.22, min(0.32, efficiency))


def _v7_sell_efficiency(action: dict[str, Any], context: dict[str, Any]) -> float:
    targets = action.get("sell_targets")
    if not isinstance(targets, list) or not targets:
        return 0.0
    value = 0.0
    for raw_location in targets:
        building = _building_at(context, _as_int(raw_location, -1))
        if building is None:
            continue
        tile = _v7_tile_stats(building["industry"], building["level"])
        if tile is None:
            continue
        value += 0.07 * tile[4] + 0.02 * max(0, tile[7]) + 0.035 * tile[5]
    value += min(0.10, 0.05 * max(0, len(targets) - 1))
    return min(0.45, value)


def _v7_develop_efficiency(action: dict[str, Any], context: dict[str, Any]) -> float:
    industries = _action_industries(action)
    if not industries:
        return 0.0
    value = 0.0
    for industry in industries:
        level = _next_level(context, industry)
        remaining = _as_int(
            context.get("industry_remaining", ())[industry]
            if industry < len(context.get("industry_remaining", ()))
            else 0,
            0,
        )
        current = _v7_tile_stats(industry, level)
        next_tile = _v7_tile_stats(industry, level + 1)
        if current is None or next_tile is None:
            value -= 0.08
            continue
        if remaining <= 1:
            gain = (
                0.08 * max(0, next_tile[4] - current[4])
                + 0.02 * max(0, next_tile[7] - current[7])
                + 0.03 * max(0, next_tile[5] - current[5])
            )
            value += 0.12 + min(0.20, gain)
        else:
            # One develop action that does not unlock a better tile has a
            # meaningful opportunity cost in a short game.
            value -= 0.08
        value += 0.04 * _v7_route_focus(industry, context)
    return max(-0.25, min(0.45, value))


def _v7_network_efficiency(action: dict[str, Any], context: dict[str, Any]) -> float:
    roads = frozenset(_action_road_indices(action))
    road_locations = context.get("road_locations")
    if not roads or not isinstance(road_locations, tuple):
        return -0.05
    if any(road < 0 or road >= len(road_locations) for road in roads):
        return -0.05
    candidate_locations = {
        location for road in roads for location in road_locations[road]
    }
    new_locations = candidate_locations.difference(
        context.get("own_network_locations", frozenset())
    )
    road_vp = _candidate_road_vp(context, roads)
    frontier = _network_frontier_value(context, candidate_locations, roads)
    build_access = _new_industry_build_access(context, new_locations)
    useful = road_vp + frontier + build_access
    value = 0.08 if len(roads) >= 2 else 0.02
    value += min(0.18, 0.025 * useful)
    if useful == 0 and not any(location >= 22 for location in candidate_locations):
        value -= 0.12
    if _action_root(action) == "double_network" and context["phase"] == "railroad":
        has_action_beer = any(
            isinstance(choice, dict)
            and choice.get("kind") == "action_beer_source"
            for choice in action.get("choices", ())
        )
        value += 0.06 if has_action_beer else -0.08
    return max(-0.25, min(0.30, value))


def _v7_route_focus(industry: int, context: dict[str, Any]) -> float:
    if not 0 <= industry < 6:
        return 0.0
    counts = context.get("industry_building_counts", ())
    own_count = _as_int(counts[industry], 0) if industry < len(counts) else 0
    hand = context.get("self_hand_counts", ())
    exact_cards = (
        _as_float(hand[20 + industry], 0.0)
        if 20 + industry < len(hand)
        else 0.0
    )
    dual_cards = _as_float(hand[26], 0.0) if len(hand) > 26 else 0.0
    values = []
    for candidate in range(6):
        candidate_count = (
            _as_int(counts[candidate], 0) if candidate < len(counts) else 0
        )
        candidate_hand = (
            _as_float(hand[20 + candidate], 0.0)
            if 20 + candidate < len(hand)
            else 0.0
        )
        candidate_value = 0.18 * candidate_count + 0.10 * min(2.0, candidate_hand)
        if candidate in _SELLABLE_INDUSTRIES:
            candidate_value += 0.10
        if candidate in (_INDUSTRY_GOODS, _INDUSTRY_COTTON) and dual_cards > 0.0:
            candidate_value += 0.05
        values.append(candidate_value)
    best = max(values, default=0.0)
    current = values[industry]
    if best <= 0.0 or best - current <= 0.10:
        return 0.0
    if current >= best - 0.10:
        return 0.14
    if industry in _SELLABLE_INDUSTRIES and best - current >= 0.35:
        return -0.12
    return 0.0


def _v7_build_location_value(
    industry: int, location: int, context: dict[str, Any]
) -> float:
    town = _town_for_build_location(location)
    if town is None:
        return 0.0
    value = 0.0
    if town in context.get("own_network_locations", frozenset()):
        value += 0.08
    if town in _merchant_locations_for_industry(context, industry):
        value += 0.10
    if any(
        building["industry"] == industry and building["town"] == town
        for building in context.get("own_buildings", ())
    ):
        value += 0.06
    return value


def _v7_resource_scarcity(industry: int, context: dict[str, Any]) -> float:
    if industry == _INDUSTRY_BEER:
        demand = int(context.get("global_sellable_beer_demand", 0))
        supply = int(context.get("all_unflipped_beer_units", 0)) + int(
            context.get("merchant_beer_units", 0)
        )
        deficit = demand - supply
        if deficit > 0:
            return min(0.28, 0.07 * deficit)
        if supply > demand + 3:
            return -0.10
        return 0.0

    resource = "coal" if industry == _INDUSTRY_COAL else "iron"
    if industry not in (_INDUSTRY_COAL, _INDUSTRY_IRON):
        return 0.0
    near_term_demand = _v7_near_term_resource_demand(context, resource)
    market = int(context["market_coal"] if resource == "coal" else context["market_iron"])
    public_units = int(
        context[
            "all_unflipped_coal_units"
            if resource == "coal"
            else "all_unflipped_iron_units"
        ]
    )
    value = 0.0
    if near_term_demand > 0 and public_units + market <= near_term_demand:
        value += 0.10
    if market <= (4 if resource == "coal" else 3) and near_term_demand > 0:
        value += 0.06
    return value


def _v7_near_term_resource_demand(context: dict[str, Any], resource: str) -> int:
    seen_intents: set[str] = set()
    demand = 0
    for action in context.get("actions", ()):
        root = _action_root(action)
        if root not in {"build", "build_building", "network", "double_network", "develop", "develop_double"}:
            continue
        intent = card_invariant_action_intent(str(action.get("key") or ""))
        if intent in seen_intents:
            continue
        seen_intents.add(intent)
        demand += min(2, _action_resource_demand(action, context, resource))
    return min(4, demand)


def _v7_tile_stats(industry: int, level: int) -> tuple[int, ...] | None:
    if not 0 <= industry < 6:
        return None
    if not 0 <= level < len(_BUILDING_MONEY_COST[industry]):
        return None
    return (
        _BUILDING_MONEY_COST[industry][level],
        _BUILDING_COAL_DEMAND[industry][level],
        _BUILDING_IRON_DEMAND[industry][level],
        _BEER_NEEDED.get(industry, (0,) * len(_BUILDING_MONEY_COST[industry]))[
            level
        ],
        _BUILDING_VP[industry][level],
        _BUILDING_ROAD_VP[industry][level],
        _BUILDING_RESOURCE_OUTPUT[industry][level],
        _BUILDING_INCOME[industry][level],
        int(not (industry in (_INDUSTRY_COAL, _INDUSTRY_IRON, _INDUSTRY_BEER, _INDUSTRY_GOODS, _INDUSTRY_COTTON) and level == 0)),
    )


def _coal_production_score(context: dict[str, Any]) -> float:
    score = 0.0
    coal_units = int(context["unflipped_coal_units"])
    if not any(
        family == "network" for family in context.get("action_families", ())
    ):
        score -= 0.65
    if coal_units >= 2:
        score -= min(0.9, 0.28 * (coal_units - 1))
    if context["phase"] == "railroad" and context["market_coal"] <= 4:
        score += 0.25
    return score


def _develop_resource_utility(context: dict[str, Any], industry: int) -> float:
    level = _next_level(context, industry)
    if industry in (_INDUSTRY_GOODS, _INDUSTRY_COTTON):
        score = 1.05
        if context["phase"] == "canal":
            score += 0.35
    elif industry == _INDUSTRY_POTTERY:
        score = 0.95
    elif industry == _INDUSTRY_BEER:
        score = 0.45 if context["sellable_beer_demand"] > 0 else 0.05
    elif industry in (_INDUSTRY_COAL, _INDUSTRY_IRON):
        score = 0.55
    else:
        score = 0.2
    if level >= 1:
        score += 0.25
    if level >= 6:
        score -= 0.5
    return score


def _card_alignment_score(action: dict[str, Any], industry: int, location: int) -> float:
    card_types = _action_card_types(action)
    if not card_types:
        return 0.0
    town = _town_for_build_location(location)
    best = 0.0
    for card_type in card_types:
        if 0 <= card_type < 20 and town == card_type:
            best = max(best, 0.25)
        elif card_type == 20 + industry:
            best = max(best, 0.25)
        elif card_type == 26:
            best = max(best, 0.16 if industry in (_INDUSTRY_GOODS, _INDUSTRY_COTTON) else 0.08)
        elif card_type in (27, 28):
            best = max(best, 0.08)
    return best


def _action_card_types(action: dict[str, Any]) -> tuple[int, ...]:
    values = action.get("discard_card_types")
    if not isinstance(values, list):
        return ()
    return tuple(_as_int(value, -1) for value in values if _as_int(value, -1) >= 0)


def _card_usefulness_resource_aware(context: dict[str, Any], card_type: int) -> float:
    if card_type in (27, 28):
        return 5.0
    if 0 <= card_type < 20:
        open_slots = context.get("open_build_slots", ())
        if card_type >= len(open_slots):
            return 0.2
        values = [
            _industry_future_value(context, industry)
            for industries in open_slots[card_type]
            for industry in industries
        ]
        return max(values, default=0.2)
    if 20 <= card_type <= 25:
        return _industry_future_value(context, card_type - 20)
    if card_type == 26:
        return max(
            _industry_future_value(context, _INDUSTRY_GOODS),
            _industry_future_value(context, _INDUSTRY_COTTON),
        )
    return 0.2


def _industry_future_value(context: dict[str, Any], industry: int) -> float:
    if industry not in range(6):
        return 0.2
    value = 0.8
    if industry in (_INDUSTRY_GOODS, _INDUSTRY_COTTON):
        value += 0.45
    if industry == _INDUSTRY_BEER and context["sellable_beer_demand"] > 0:
        value += 0.25
    if _next_level(context, industry) >= 1:
        value += 0.35
    return value


def _best_future_action(context: dict[str, Any]) -> tuple[float, float] | None:
    candidates = []
    for action in context.get("actions", ()):
        if _action_root(action) not in {"build", "network", "develop", "develop_double"}:
            continue
        candidates.append(
            (
                _score_lifecycle_action(action, context),
                _action_cash_cost(action, context),
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))


def _action_cash_cost(action: dict[str, Any], context: dict[str, Any]) -> float:
    root = _action_root(action)
    if root in ("build", "build_building"):
        industry = _action_industry(action)
        level = _next_level(context, industry)
        base = (
            _BUILDING_MONEY_COST[industry][level]
            if 0 <= industry < len(_BUILDING_MONEY_COST)
            and 0 <= level < len(_BUILDING_MONEY_COST[industry])
            else 0.0
        )
        return base + _action_market_cost(action, context)
    if root in ("network", "double_network"):
        if root == "double_network":
            base = 15.0
        else:
            base = 5.0 if context["phase"] == "railroad" else 3.0
        return base + _action_market_cost(action, context)
    if root in ("develop", "develop_double"):
        return _action_market_cost(action, context)
    return 0.0


def _action_market_cost(action: dict[str, Any], context: dict[str, Any]) -> float:
    coal_units = _action_market_units(action, context, "coal")
    iron_units = _action_market_units(action, context, "iron")
    return float(
        _market_cost(
            int(context["market_coal"]), coal_units, _COAL_PRICE_TABLE, 8
        )
        + _market_cost(
            int(context["market_iron"]), iron_units, _IRON_PRICE_TABLE, 6
        )
    )


def _action_market_units(
    action: dict[str, Any], context: dict[str, Any], resource: str
) -> int:
    root = _action_root(action)
    kind = f"{resource}_source"
    choices = tuple(
        choice
        for choice in action.get("choices", ())
        if isinstance(choice, dict) and choice.get("kind") == kind
    )
    market_choices = sum(
        _as_int(choice.get("value"), -1) == _MARKET_SOURCE for choice in choices
    )
    if root in ("network", "double_network"):
        return market_choices

    demand = _action_resource_demand(action, context, resource)
    if demand <= 0 or market_choices == 0:
        return 0
    capacity = 0
    buildings = {
        building["location"]: building
        for building in context.get("all_buildings", ())
    }
    expected_industry = (
        _INDUSTRY_COAL if resource == "coal" else _INDUSTRY_IRON
    )
    seen_sources: set[int] = set()
    for choice in choices:
        source = _as_int(choice.get("value"), -1)
        if source in seen_sources or source == _MARKET_SOURCE:
            continue
        seen_sources.add(source)
        building = buildings.get(source)
        if (
            building is not None
            and building["industry"] == expected_industry
            and not building["flipped"]
        ):
            capacity += building["resource_units"]
    return max(0, demand - capacity)


def _action_resource_demand(
    action: dict[str, Any], context: dict[str, Any], resource: str
) -> int:
    root = _action_root(action)
    if resource == "coal" and root in ("network", "double_network"):
        return 2 if root == "double_network" else 1
    if resource == "iron" and root in ("develop", "develop_double"):
        industries = _action_industries(action)
        return len(industries) if root == "develop_double" else 1
    if root not in ("build", "build_building"):
        return 0
    industry = _action_industry(action)
    level = _next_level(context, industry)
    table = (
        _BUILDING_COAL_DEMAND
        if resource == "coal"
        else _BUILDING_IRON_DEMAND
    )
    if not 0 <= industry < len(table) or not 0 <= level < len(table[industry]):
        return 0
    return table[industry][level]


def _market_cost(
    remaining: int, units: int, price_table: Sequence[int], empty_price: int
) -> int:
    total = 0
    remaining = max(0, remaining)
    for _ in range(max(0, units)):
        if remaining > 0:
            index = max(0, len(price_table) - remaining)
            total += price_table[min(index, len(price_table) - 1)]
            remaining -= 1
        else:
            total += empty_price
    return total


def _action_sell_beer_demand(action: dict[str, Any], context: dict[str, Any]) -> int:
    targets = action.get("sell_targets")
    if not isinstance(targets, list):
        return 0
    return sum(
        _beer_needed_for_building(building)
        for raw_location in targets
        for building in (_building_at(context, _as_int(raw_location, -1)),)
        if building is not None
    )


def _beer_needed_for_building(building: dict[str, Any]) -> int:
    needed_by_level = _BEER_NEEDED.get(building.get("industry"), ())
    level = _as_int(building.get("level"), 0)
    if 0 <= level < len(needed_by_level):
        return needed_by_level[level]
    return 1


def _network_frontier_value(
    context: dict[str, Any],
    candidate_locations: set[int],
    candidate_roads: frozenset[int],
) -> int:
    road_locations = context.get("road_locations")
    if not isinstance(road_locations, tuple):
        return 0
    built_roads = context.get("built_roads", frozenset())
    return sum(
        1
        for road, locations in enumerate(road_locations)
        if road not in built_roads
        and road not in candidate_roads
        and candidate_locations.intersection(locations)
    )


def _market_remaining(
    observation: dict[str, Any], index: int, maximum: int
) -> int:
    features = observation.get("global_features")
    if not isinstance(features, list) or index >= len(features):
        return maximum
    normalized = _as_float(features[index], 1.0)
    return max(0, min(maximum, int(round(normalized * maximum))))


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
