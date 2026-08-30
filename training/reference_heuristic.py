"""Deterministic port of the open-source npow/brass-birmingham heuristic.

The upstream browser AI is a stateless evaluator over complete action plans. This
module applies the same scalar scoring rules to Fast Brass' legal composite
actions so the two agents can be compared under one rules engine. It is a
reference opponent, not human replay data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .strategy_prior import (
    _action_family,
    _action_industries,
    _action_industry,
    _action_road_indices,
    _action_root,
    _as_float,
    _as_int,
    _build_context,
    _phase_name,
)

REFERENCE_COMMIT = "2b1da2d41036f2afaafcab320b2175ba3fd9f877"
REFERENCE_SOURCE = f"npow/brass-birmingham@{REFERENCE_COMMIT}"
REFERENCE_POLICY_VERSION = "npow-stateless-heuristic-v1"

VP_WEIGHT = 1.0
MONEY_WEIGHT = 0.12
BASE_INCOME_WEIGHT = 0.35
FLEX_WEIGHT = 0.8

_NUM_PLAYERS = 4
_NUM_BUILD_LOCATIONS = 49
_NUM_MAP_LOCATIONS = 27
_NUM_TRADE_POST_LOCATIONS = 5
_MAX_HAND_MASK_DIM = 64
_CANAL_LINK_COST = 3
_RAIL_LINK_COST = 5
_LOAN_AMOUNT = 30
_LOAN_INCOME_PENALTY = 3
_MIN_INCOME = -10
_TOTAL_CARDS_BY_PLAYERS = (0, 0, 43, 57, 67)
_COAL_PRICE_TABLE = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)
_IRON_PRICE_TABLE = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5)

_COAL = 0
_IRON = 1
_BEER = 2
_GOODS = 3
_POTTERY = 4
_COTTON = 5
_SELLABLE_INDUSTRIES = frozenset((_GOODS, _POTTERY, _COTTON))
_RESOURCE_INDUSTRIES = frozenset((_COAL, _IRON))


@dataclass(frozen=True)
class TileData:
    cost: int
    coal: int
    iron: int
    beer: int
    vp: int
    road_vp: int
    resource_cubes: int
    income: int
    rail_era: bool
    can_develop: bool


# Indexed by the Fast Brass industry index, then by zero-based level.
_TILES: tuple[tuple[TileData, ...], ...] = (
    (
        TileData(5, 0, 0, 0, 1, 2, 2, 4, False, True),
        TileData(7, 0, 0, 0, 2, 1, 3, 7, True, True),
        TileData(8, 0, 1, 0, 3, 1, 4, 6, True, True),
        TileData(10, 0, 1, 0, 4, 1, 5, 5, True, True),
    ),
    (
        TileData(5, 1, 0, 0, 3, 1, 4, 3, False, True),
        TileData(7, 1, 0, 0, 5, 1, 4, 3, True, True),
        TileData(9, 1, 0, 0, 7, 1, 5, 2, True, True),
        TileData(12, 1, 0, 0, 9, 1, 6, 1, True, True),
    ),
    (
        TileData(5, 0, 1, 0, 4, 2, 1, 4, False, True),
        TileData(7, 0, 1, 0, 5, 2, 1, 5, True, True),
        TileData(9, 0, 1, 0, 7, 2, 1, 5, True, True),
        TileData(9, 0, 1, 0, 10, 2, 2, 5, True, True),
    ),
    (
        TileData(8, 1, 0, 1, 3, 2, 0, 5, False, True),
        TileData(10, 0, 1, 1, 5, 1, 0, 0, True, True),
        TileData(12, 2, 0, 0, 4, 0, 0, 4, True, True),
        TileData(8, 0, 1, 1, 3, 1, 0, 6, True, True),
        TileData(16, 1, 0, 2, 8, 2, 0, 2, True, True),
        TileData(20, 0, 0, 1, 7, 1, 0, 6, True, True),
        TileData(16, 1, 1, 0, 9, 0, 0, 4, True, True),
        TileData(20, 0, 2, 1, 11, 1, 0, 1, True, True),
    ),
    (
        TileData(17, 0, 1, 1, 10, 1, 0, 5, True, False),
        TileData(0, 1, 0, 1, 1, 1, 0, 1, True, True),
        TileData(22, 2, 0, 2, 11, 1, 0, 5, True, False),
        TileData(0, 1, 0, 1, 1, 1, 0, 1, True, True),
        TileData(24, 2, 0, 2, 20, 1, 0, 5, True, True),
    ),
    (
        TileData(12, 0, 0, 1, 5, 1, 0, 5, False, True),
        TileData(14, 1, 0, 1, 5, 2, 0, 4, True, True),
        TileData(16, 1, 1, 1, 9, 1, 0, 3, True, True),
        TileData(18, 1, 1, 1, 12, 1, 0, 2, True, True),
    ),
)

_MERCHANT_INDUSTRIES = {
    0: frozenset((_GOODS, _POTTERY, _COTTON)),
    1: frozenset((_COTTON,)),
    2: frozenset((_GOODS,)),
    3: frozenset((_POTTERY,)),
}
_MERCHANT_LOCATION_BY_SLOT = (22, 23, 23, 24, 24, 25, 25, 26, 26)
_MERCHANT_BONUS_BY_SLOT = {
    0: ("vp", 4),
    1: ("income", 2),
    2: ("income", 2),
    3: ("develop", 1),
    4: ("develop", 1),
    5: ("money", 5),
    6: ("money", 5),
    7: ("vp", 3),
    8: ("vp", 3),
}

_ROOT_PRIORITY = {
    "build": 6,
    "network": 5,
    "develop": 4,
    "sell": 3,
    "loan": 2,
    "scout": 1,
    "pass": 0,
}
_NEGATIVE_INFINITY = -1.0e30


class ReferenceHeuristicPolicy:
    """The deterministic core of the upstream stateless heuristic.

    The upstream normal difficulty adds unseeded Math.random noise. That noise
    is intentionally omitted here so every match is reproducible. The upstream
    denial bonus is also omitted because opponent hands are private in Fast
    Brass observations.
    """

    source = REFERENCE_SOURCE
    source_kind = "open_source_heuristic"
    model_id = REFERENCE_SOURCE
    checkpoint_step = 0
    strategy_prior_strength = 0.0
    strategy_prior_version = REFERENCE_POLICY_VERSION

    def select_action(
        self, observation: dict[str, Any], state_record: dict[str, Any], legal_record: dict[str, Any]
    ) -> int:
        del state_record
        scores = self.score_actions(observation, legal_record)
        if not scores:
            raise RuntimeError("reference heuristic received no legal actions")
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def select_action_with_search(
        self,
        game: Any,
        observation: dict[str, Any],
        state_record: dict[str, Any],
        legal_record: dict[str, Any],
        num_players: int,
        simulations: int,
        exploration_constant: float,
        search_determinizations: int,
        inference_batch_size: int,
        search_seed: int,
    ) -> int:
        """Allow this policy to be used in the shared match runner.

        The upstream policy has no search layer, so search arguments are kept
        only for interface compatibility and intentionally ignored.
        """
        del (
            game,
            num_players,
            simulations,
            exploration_constant,
            search_determinizations,
            inference_batch_size,
            search_seed,
        )
        return self.select_action(observation, state_record, legal_record)

    def score_actions(
        self, observation: dict[str, Any], legal_record: dict[str, Any]
    ) -> tuple[float, ...]:
        actions = legal_record.get("actions")
        if not isinstance(actions, list) or not actions:
            return ()
        context = _make_context(observation, actions)
        scores = tuple(_score_action(action, context) for action in actions)
        return tuple(score + _tie_break(action, index) for index, (action, score) in enumerate(zip(actions, scores)))


def _make_context(
    observation: dict[str, Any], actions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    phase = _phase_name(observation)
    context = dict(_build_context(observation, phase))
    context["phase"] = phase
    context["rounds_remaining"] = _estimate_rounds_remaining(observation)
    context["income_level_external"] = _income_level_from_space(
        int(round(context.get("income_level", 0.0)))
    )
    context["market_coal"] = _market_remaining(observation, 10, 14)
    context["market_iron"] = _market_remaining(observation, 11, 10)
    context["own_built_roads"] = _own_built_roads(observation)
    context["remaining_tiles"] = _remaining_tiles(observation)
    context["actions"] = tuple(actions)
    context["has_sell_action"] = any(
        _action_family(action) == "sell" for action in actions
    )
    context["card_types_by_index"] = _card_types_by_index(actions)
    return context


def _estimate_rounds_remaining(observation: dict[str, Any]) -> float:
    players = observation.get("players_public")
    num_players = len(players) if isinstance(players, list) else 2
    total_cards = _TOTAL_CARDS_BY_PLAYERS[min(max(num_players, 2), 4)]
    hands = observation.get("hand_sizes")
    hand_count = (
        sum(int(round(_as_float(value, 0.0) * _MAX_HAND_MASK_DIM)) for value in hands)
        if isinstance(hands, list)
        else 0
    )
    discards = observation.get("discard_counts")
    discard_count = (
        sum(int(round(_as_float(value, 0.0))) for value in discards)
        if isinstance(discards, list)
        else 0
    )
    deck_size = max(0, total_cards - hand_count - discard_count)
    return max(1.0, min(10.0, deck_size / max(1, num_players)))


def _market_remaining(observation: dict[str, Any], index: int, maximum: int) -> int:
    features = observation.get("global_features")
    if not isinstance(features, list) or index >= len(features):
        return maximum
    return max(0, min(maximum, int(round(_as_float(features[index], 1.0) * maximum))))


def _income_level_from_space(space: int) -> int:
    if space <= 10:
        return space - 10
    if space <= 30:
        return math.ceil((space - 10) / 2)
    if space <= 60:
        return 10 + math.ceil((space - 30) / 3)
    if space <= 96:
        return 20 + math.ceil((space - 60) / 4)
    return 30


def _own_built_roads(observation: dict[str, Any]) -> frozenset[int]:
    actor = _as_int(observation.get("decision_player"), 0)
    rows = observation.get("roads")
    if not isinstance(rows, list):
        return frozenset()
    owner_offset = 1 + actor
    return frozenset(
        road
        for road, row in enumerate(rows)
        if isinstance(row, list)
        and len(row) > owner_offset
        and _as_float(row[0], 0.0) > 0.5
        and _as_float(row[owner_offset], 0.0) > 0.5
    )


def _remaining_tiles(observation: dict[str, Any]) -> tuple[tuple[int, bool], ...]:
    actor = _as_int(observation.get("decision_player"), 0)
    mats = observation.get("industry_mats")
    row = mats[actor] if isinstance(mats, list) and actor < len(mats) else []
    result = []
    for industry in range(6):
        offset = industry * 3
        has_tiles = (
            offset + 2 < len(row) and _as_float(row[offset + 2], 0.0) > 0.5
        )
        result.append((industry, has_tiles))
    return tuple(result)


def _card_types_by_index(actions: Sequence[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for action in actions:
        if _action_root(action) != "scout":
            continue
        values = [
            _as_int(choice.get("value"), -1)
            for choice in action.get("choices", [])
            if isinstance(choice, dict) and choice.get("kind") == "card"
        ]
        types = action.get("discard_card_types")
        if not isinstance(types, list):
            continue
        for index, card_type in zip(values, types):
            if index >= 0:
                result[index] = _as_int(card_type, -1)
    return result


def _score_action(action: dict[str, Any], context: dict[str, Any]) -> float:
    root = _action_root(action)
    if root == "build":
        return _score_build(action, context)
    if root == "network":
        return _score_network(action, context)
    if root == "develop":
        return _score_develop(action, context)
    if root == "sell":
        return _score_sell(action, context)
    if root == "loan":
        return _score_loan(action, context)
    if root == "scout":
        return _score_scout(action, context)
    if root == "pass":
        return -0.5
    return _NEGATIVE_INFINITY


def _score_build(action: dict[str, Any], context: dict[str, Any]) -> float:
    industry = _action_industry(action)
    location = _as_int(action.get("build_location"), -1)
    level = _next_level(context, industry)
    tile = _tile_at(industry, level)
    if tile is None or location < 0:
        return _NEGATIVE_INFINITY

    flip_probability = _estimate_flip_probability(context, industry, location)
    link_self_value = (
        tile.road_vp * flip_probability * 0.5
        if _owns_link_touching(context, location)
        else 0.0
    )
    resource_self_sufficiency = (
        0.15 * tile.resource_cubes if industry in _RESOURCE_INDUSTRIES else 0.0
    )
    network_expansion = 0.1 * _count_new_neighbor_roads(context, location)
    return _vp_equivalent(
        context,
        vp=tile.vp * flip_probability + link_self_value,
        income=tile.income * flip_probability,
        money=-_build_cost(action, context, tile),
    ) + resource_self_sufficiency + network_expansion


def _score_network(action: dict[str, Any], context: dict[str, Any]) -> float:
    if action.get("second_road") is not None or action.get("action_type") == "build_double_railroad":
        # The upstream AI only asks GameLogic for single-link network targets.
        return _NEGATIVE_INFINITY
    roads = _action_road_indices(action)
    if len(roads) != 1:
        return _NEGATIVE_INFINITY
    road_locations = context.get("road_locations")
    if not isinstance(road_locations, tuple) or not 0 <= roads[0] < len(road_locations):
        return _NEGATIVE_INFINITY
    candidate_locations = set(road_locations[roads[0]])
    hand = context.get("self_hand_counts", ())
    own_network = context.get("own_network_locations", frozenset())
    access_gain = sum(
        _as_float(hand[location], 0.0)
        for location in candidate_locations
        if 0 <= location < 20
        and location < len(hand)
        and location not in own_network
    )
    access_gain += sum(
        0.25 * _as_float(hand[index], 0.0)
        for index in range(20, min(27, len(hand)))
    )
    merchant_gain = 1.5 if any(location >= 22 for location in candidate_locations) else 0.0
    has_industry = any(building["location"] < 47 for building in context.get("own_buildings", ()))
    links_built = len(context.get("own_built_roads", ()))
    over_networking_penalty = 1.0 if not has_industry and links_built >= 1 else 0.0
    exploration_bonus = max(0.0, 1.6 - links_built * 0.3)
    return (
        _vp_equivalent(
            context,
            vp=access_gain + merchant_gain,
            money=-_network_cost(action, context),
        )
        + exploration_bonus
        - over_networking_penalty
    )


def _score_develop(action: dict[str, Any], context: dict[str, Any]) -> float:
    developable = [
        industry
        for industry, has_tiles in context.get("remaining_tiles", ())
        if has_tiles and _tile_at(industry, _next_level(context, industry)) is not None
        and _tile_at(industry, _next_level(context, industry)).can_develop
    ]
    if not developable:
        return _NEGATIVE_INFINITY
    ranked = sorted(
        developable,
        key=lambda industry: (
            -_develop_utility(context, industry),
            industry,
        ),
    )
    first = ranked[0]
    double_available = any(
        _action_family(candidate) == "develop"
        and (
            _action_root(candidate) == "develop_double"
            or candidate.get("action_type") == "develop_double"
            or candidate.get("selected_second_industry") is not None
        )
        for candidate in context.get("actions", ())
    )
    second = ranked[1] if double_available and len(ranked) > 1 else None
    industries = _action_industries(action)
    is_double = action.get("selected_second_industry") is not None or action.get("action_type") == "develop_double"
    expected = (first, second) if second is not None and is_double else (first,)
    if tuple(industries) != tuple(value for value in expected if value is not None):
        return _NEGATIVE_INFINITY
    iron_demand = len(expected)
    return _vp_equivalent(
        context,
        vp=sum(_develop_utility(context, industry) for industry in expected if industry is not None),
        money=-_resource_cost(action, context, "iron_source", iron_demand),
    )


def _score_sell(action: dict[str, Any], context: dict[str, Any]) -> float:
    targets = action.get("sell_targets")
    if not isinstance(targets, list) or not targets:
        return _NEGATIVE_INFINITY
    total_vp = 0.0
    total_income = 0.0
    total_bonus = 0.0
    for raw_location in targets:
        location = _as_int(raw_location, -1)
        building = _building_at(context, location)
        if building is None or building["flipped"]:
            return _NEGATIVE_INFINITY
        tile = _tile_at(building["industry"], building["level"])
        if tile is None or building["industry"] not in _SELLABLE_INDUSTRIES:
            return _NEGATIVE_INFINITY
        total_vp += tile.vp
        total_income += tile.income
        total_bonus += _best_merchant_beer_bonus(context, building)
    return _vp_equivalent(context, vp=total_vp, income=total_income) + total_bonus


def _score_loan(action: dict[str, Any], context: dict[str, Any]) -> float:
    cheapest = math.inf
    for candidate in context.get("actions", ()):
        root = _action_root(candidate)
        if root == "build":
            level = _next_level(context, _action_industry(candidate))
            tile = _tile_at(_action_industry(candidate), level)
            if tile is not None:
                cheapest = min(cheapest, _build_cost(candidate, context, tile))
        elif root == "network" and candidate.get("second_road") is None:
            cheapest = min(cheapest, _network_cost(candidate, context))
    money = context.get("money", 0.0)
    income_level = context.get("income_level_external", 0)
    cash_crunch_bonus = 4.0 if money < cheapest else 0.0
    already_flush_penalty = 2.5 if money > cheapest * 2.5 else 0.0
    income_floor_penalty = 2.5 if income_level <= _MIN_INCOME + 5 else 0.0
    return (
        _vp_equivalent(
            context,
            money=_LOAN_AMOUNT * 0.25,
            income=-_LOAN_INCOME_PENALTY,
        )
        + cash_crunch_bonus
        - already_flush_penalty
        - income_floor_penalty
    )


def _score_scout(action: dict[str, Any], context: dict[str, Any]) -> float:
    values = [
        _as_int(choice.get("value"), -1)
        for choice in action.get("choices", [])
        if isinstance(choice, dict) and choice.get("kind") == "card"
    ]
    if len(values) != 3:
        return _NEGATIVE_INFINITY
    card_types = context.get("card_types_by_index", {})
    usefulness = tuple(
        _card_usefulness(context, card_types.get(index, -1)) for index in values
    )
    dead_count = sum(value <= 0.0 for value in usefulness)
    score = _vp_equivalent(
        context,
        flex=dead_count * 1.2 - (3 - dead_count) * 0.6,
    )
    desired = _desired_scout_indices(context)
    if desired is not None and tuple(sorted(values)) == desired:
        score += 1.0e-4
    return score


def _vp_equivalent(
    context: dict[str, Any], *, vp: float = 0.0, income: float = 0.0, money: float = 0.0, flex: float = 0.0
) -> float:
    income_weight = BASE_INCOME_WEIGHT * (context["rounds_remaining"] / 5.0)
    return vp * VP_WEIGHT + income * income_weight + money * MONEY_WEIGHT + flex * FLEX_WEIGHT


def _next_level(context: dict[str, Any], industry: int) -> int:
    levels = context.get("next_levels", ())
    return _as_int(levels[industry], 0) if 0 <= industry < len(levels) else 0


def _tile_at(industry: int, level: int) -> TileData | None:
    if not 0 <= industry < len(_TILES):
        return None
    tiles = _TILES[industry]
    if not 0 <= level < len(tiles):
        return None
    return tiles[level]


def _estimate_flip_probability(context: dict[str, Any], industry: int, location: int) -> float:
    if industry in (_COAL, _IRON):
        base = 0.7
    elif industry == _BEER:
        base = 0.5
    else:
        base = 0.6
        merchants = _merchant_locations_for_industry(context, industry)
        if any(_globally_reachable(context, _location_for_building(location), merchant) for merchant in merchants):
            base += 0.2
        if context["rounds_remaining"] < 2:
            base -= 0.3
    return max(0.1, min(1.0, base))


def _location_for_building(location: int) -> int:
    if location == 47:
        return 20
    if location == 48:
        return 21
    ranges = (
        (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
        (12, 15), (15, 17), (17, 19), (19, 22), (22, 25), (25, 28),
        (28, 30), (30, 32), (32, 34), (34, 36), (36, 40), (40, 42),
        (42, 45), (45, 47),
    )
    for town, (start, end) in enumerate(ranges):
        if start <= location < end:
            return town
    return -1


def _road_endpoints(context: dict[str, Any], road: int) -> tuple[int, ...]:
    roads = context.get("road_locations")
    if not isinstance(roads, tuple) or not 0 <= road < len(roads):
        return ()
    return tuple(roads[road])


def _globally_reachable(context: dict[str, Any], start: int, target: int) -> bool:
    if start < 0 or target < 0:
        return False
    if start == target:
        return True
    roads = context.get("road_locations")
    built = context.get("built_roads", frozenset())
    if not isinstance(roads, tuple):
        return False
    adjacency = [[] for _ in range(_NUM_MAP_LOCATIONS)]
    for road in built:
        endpoints = _road_endpoints(context, road)
        for source in endpoints:
            for destination in endpoints:
                if source != destination and 0 <= source < _NUM_MAP_LOCATIONS and 0 <= destination < _NUM_MAP_LOCATIONS:
                    adjacency[source].append(destination)
    pending = [start]
    visited = {start}
    while pending:
        current = pending.pop()
        for destination in adjacency[current] if 0 <= current < len(adjacency) else ():
            if destination == target:
                return True
            if destination not in visited:
                visited.add(destination)
                pending.append(destination)
    return False


def _owns_link_touching(context: dict[str, Any], location: int) -> bool:
    town = _location_for_building(location)
    return any(town in _road_endpoints(context, road) for road in context.get("own_built_roads", ()))


def _count_new_neighbor_roads(context: dict[str, Any], location: int) -> int:
    town = _location_for_building(location)
    built = context.get("built_roads", frozenset())
    roads = context.get("road_locations", ())
    return sum(1 for road, endpoints in enumerate(roads) if road not in built and town in endpoints)


def _merchant_locations_for_industry(context: dict[str, Any], industry: int) -> tuple[int, ...]:
    return tuple(
        merchant["location"]
        for merchant in context.get("merchants", ())
        if industry in merchant.get("industries", ())
    )


def _build_cost(action: dict[str, Any], context: dict[str, Any], tile: TileData) -> int:
    return tile.cost + _resource_cost(action, context, "coal_source", tile.coal) + _resource_cost(action, context, "iron_source", tile.iron)


def _network_cost(action: dict[str, Any], context: dict[str, Any]) -> int:
    base = _RAIL_LINK_COST if context["phase"] == "railroad" else _CANAL_LINK_COST
    coal = 1 if context["phase"] == "railroad" else 0
    return base + _resource_cost(action, context, "coal_source", coal)


def _resource_cost(
    action: dict[str, Any], context: dict[str, Any], choice_kind: str, demand: int
) -> int:
    if demand <= 0:
        return 0
    remaining = demand
    market_units = 0
    buildings = {
        building["location"]: building
        for building in context.get("all_buildings", ())
    }
    for choice in action.get("choices", ()):
        if not isinstance(choice, dict) or choice.get("kind") != choice_kind:
            continue
        source = _as_int(choice.get("value"), -1)
        if source == _NUM_BUILD_LOCATIONS:
            market_units += remaining
            remaining = 0
            break
        building = buildings.get(source)
        if building is not None and not building["flipped"]:
            capacity = max(0, int(building.get("resource_units", 0)))
            remaining = max(0, remaining - capacity)
    if choice_kind == "coal_source":
        market_remaining = int(context.get("market_coal", 14))
        table = _COAL_PRICE_TABLE
        empty_price = 8
        maximum = 14
    else:
        market_remaining = int(context.get("market_iron", 10))
        table = _IRON_PRICE_TABLE
        empty_price = 6
        maximum = 10
    total = 0
    for _ in range(market_units):
        if market_remaining > 0:
            total += table[min(len(table) - 1, maximum - market_remaining)]
            market_remaining -= 1
        else:
            total += empty_price
    return total


def _building_at(context: dict[str, Any], location: int) -> dict[str, Any] | None:
    for building in context.get("own_buildings", ()):
        if building["location"] == location:
            return building
    return None


def _best_merchant_beer_bonus(context: dict[str, Any], building: dict[str, Any]) -> float:
    best = 0.0
    for merchant in context.get("merchants", ()):
        if not merchant.get("has_beer") or building["industry"] not in merchant.get("industries", ()):
            continue
        if not _globally_reachable(context, building["town"], merchant["location"]):
            continue
        slot = _as_int(merchant.get("slot"), -1)
        bonus_type, amount = _MERCHANT_BONUS_BY_SLOT.get(slot, ("", 0))
        if bonus_type == "vp":
            value = float(amount) * VP_WEIGHT
        elif bonus_type == "money":
            value = float(amount) * MONEY_WEIGHT
        elif bonus_type == "income":
            value = float(amount) * BASE_INCOME_WEIGHT * (context["rounds_remaining"] / 5.0)
        elif bonus_type == "develop":
            value = float(amount) * 0.5
        else:
            value = 0.0
        best = max(best, value)
    return best


def _develop_utility(context: dict[str, Any], industry: int) -> float:
    tile = _tile_at(industry, _next_level(context, industry))
    if tile is None:
        return -math.inf
    urgency = 2.0 if not tile.rail_era else 0.5
    return urgency + 0.3 * (_next_level(context, industry) + 1)


def _card_usefulness(context: dict[str, Any], card_type: int) -> float:
    if card_type in (27, 28):
        return 5.0
    remaining = dict(context.get("remaining_tiles", ()))
    if 0 <= card_type < 20:
        open_slots = context.get("open_build_slots", ())
        if card_type >= len(open_slots):
            return 0.0
        return 2.0 if any(
            any(remaining.get(industry, False) for industry in slot)
            for slot in open_slots[card_type]
        ) else 0.0
    if 20 <= card_type <= 25:
        return 1.5 if remaining.get(card_type - 20, False) else 0.0
    if card_type == 26:
        return 1.5 if remaining.get(_GOODS, False) or remaining.get(_COTTON, False) else 0.0
    return 0.5


def _desired_scout_indices(context: dict[str, Any]) -> tuple[int, ...] | None:
    card_types = context.get("card_types_by_index", {})
    if not card_types:
        return None
    ranked = sorted(
        card_types,
        key=lambda index: (_card_usefulness(context, card_types[index]), index),
    )
    return tuple(sorted(ranked[:3])) if len(ranked) >= 3 else None


def _tie_break(action: dict[str, Any], index: int) -> float:
    root = _action_root(action)
    priority = _ROOT_PRIORITY.get(root, -1)
    score = priority * 1.0e-6
    if root != "scout":
        card_types = action.get("discard_card_types")
        if isinstance(card_types, list) and card_types:
            score += 1.0e-4 if _as_int(card_types[0], -1) not in (27, 28) else -1.0e-4
        selected_card = action.get("selected_card")
        if selected_card is not None:
            score -= _as_int(selected_card, index) * 1.0e-8
    return score
