//! A common VP scale for completed investments, conversion work and liquidity.

use super::*;
use crate::core::types::BuildingTypeData;
use crate::game::search::potential_era_victory_points;
use crate::market::merchants::slot_to_trade_post;

#[derive(Default)]
struct Value {
    secured: f64,
    carryover: f64,
    products: f64,
    resources: f64,
    income: f64,
    cash: f64,
    network: f64,
}

fn remaining_actions(runner: &GameRunner, actor: usize) -> f64 {
    if runner.is_game_finished() {
        return 0.0;
    }
    let state = &runner.framework.board.state;
    state.players[actor].hand.cards.len() as f64
        + state.deck.cards.len() as f64 / state.players.len() as f64
}

fn era_multiplier(runner: &GameRunner, tile: &BuildingTypeData) -> f64 {
    if runner.game_phase == GamePhase::Canal && !tile.removed_after_phase1 {
        2.0
    } else {
        1.0
    }
}

fn income_horizon(runner: &GameRunner, actor: usize) -> f64 {
    let rounds = remaining_actions(runner, actor) / 2.0;
    let next_era = if runner.game_phase == GamePhase::Canal {
        3.0
    } else {
        0.0
    };
    // Future income becomes less scarce as new industries come online.
    (rounds + next_era).min(5.0)
}

fn income_gain(runner: &GameRunner, actor: usize, steps: i8) -> f64 {
    let player = &runner.framework.board.state.players[actor];
    let level = (i16::from(player.income_level) + i16::from(steps)).clamp(0, 99) as u8;
    f64::from(player.get_income_amount(level) - player.get_income_amount(player.income_level))
}

fn money_value(money: f64, actions: f64) -> f64 {
    // Cash is a means of buying the remaining actions; it has no terminal VP.
    let usable = money.min(actions * 14.0 + 5.0);
    18.0 * (1.0 + usable.max(0.0) / 25.0).ln()
}

fn value(runner: &GameRunner, actor: usize, position: &PositionSnapshot) -> Value {
    let state = &runner.framework.board.state;
    let player = &state.players[actor];
    let mut result = Value {
        secured: f64::from(player.victory_points),
        ..Value::default()
    };
    if runner.is_game_finished() {
        return result;
    }
    result.secured += f64::from(potential_era_victory_points(runner)[actor]);
    let actions = remaining_actions(runner, actor);
    let horizon = income_horizon(runner, actor);
    let income = f64::from(player.get_income_amount(player.income_level));
    let mut pending_income = Vec::<(f64, f64)>::new();
    let mut flip_probabilities = [0.0; N_BL];
    result.cash = money_value(f64::from(player.money), actions);
    let reserve = (-income).max(0.0);
    result.cash -= (reserve - f64::from(player.money)).max(0.0) * 2.0;
    let settlement_reserve = reserve * (actions / 2.0).clamp(1.0, 3.0);
    result.cash -= (settlement_reserve - f64::from(player.money)).max(0.0) * 0.65;
    let credit = ((income + 10.0) / 3.0).floor().clamp(0.0, 3.0) * 30.0;
    let runway = reserve * (actions / 2.0).min(4.0) + actions.min(2.0) * 6.0;
    result.cash -= (runway - f64::from(player.money) - credit).max(0.0) * 0.9;

    let distances = [
        IndustryType::Cotton,
        IndustryType::Goods,
        IndustryType::Pottery,
    ]
    .map(|industry| route_distance_map(runner, &merchant_target_mask(state, industry)));
    let merchant_distances: Vec<_> = state
        .trade_post_slots
        .iter()
        .enumerate()
        .map(|(slot, merchant)| {
            let mut targets = [false; TOTAL_TOWNS];
            if merchant.is_some() && slot < 9 {
                targets[slot_to_trade_post(slot).to_location_name().as_usize()] = true;
            }
            route_distance_map(runner, &targets)
        })
        .collect();
    let mut beer = [0u8; N_BL];
    let mut merchant_beer = state.trade_post_beer.clone();
    let mut stocks = [0.0; 3];
    for location in 0..N_BL {
        if let Some(building) = state.bl_to_building.get(&location) {
            if !building.flipped && !is_saleable_industry(building.industry) {
                stocks[building.industry.as_usize()] += f64::from(building.resource_amt);
                if building.industry == IndustryType::Beer {
                    beer[location] = building.resource_amt;
                }
            }
        }
    }

    // Allocate shared barrels once, preferring products with the best return.
    let mut products = state.player_building_mask[actor]
        .ones()
        .filter(|loc| {
            state
                .bl_to_building
                .get(loc)
                .is_some_and(|b| !b.flipped && is_saleable_industry(b.industry))
        })
        .collect::<Vec<_>>();
    products.sort_by(|a, b| {
        let quality = |loc: &usize| {
            let building = &state.bl_to_building[loc];
            let tile = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
            f64::from(tile.vp_on_flip) * era_multiplier(runner, tile)
                / (1.0 + f64::from(tile.beer_needed))
        };
        quality(b).total_cmp(&quality(a)).then_with(|| a.cmp(b))
    });
    for location in products {
        let building = &state.bl_to_building[&location];
        let tile = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        let town = LocationName::from_bl_idx(location);
        let industry_index = match building.industry {
            IndustryType::Cotton => 0,
            IndustryType::Goods => 1,
            _ => 2,
        };
        let distance = distances[industry_index][town.as_usize()];
        if distance == usize::MAX {
            continue;
        }
        let old_beer = beer;
        let old_merchant_beer = merchant_beer.clone();
        let mut needed = tile.beer_needed;
        let mut external = false;
        // A sale can use at most its chosen merchant's barrel.
        if needed > 0 {
            for slot in merchant_beer.clone().ones() {
                let matches = state.trade_post_slots[slot]
                    .as_ref()
                    .is_some_and(|m| m.industries.contains(building.industry.as_usize()));
                if matches && merchant_distances[slot][town.as_usize()] == distance {
                    merchant_beer.remove(slot);
                    needed -= 1;
                    external = true;
                    break;
                }
            }
        }
        for own in [true, false] {
            for source in 0..N_BL {
                if needed == 0 || beer[source] == 0 {
                    continue;
                }
                let brewery = &state.bl_to_building[&source];
                if (brewery.owner.as_usize() == actor) != own {
                    continue;
                }
                if !own
                    && !state
                        .connectivity
                        .are_towns_connected(town, LocationName::from_bl_idx(source))
                {
                    continue;
                }
                let used = needed.min(beer[source]);
                beer[source] -= used;
                needed -= used;
                external |= !own;
            }
        }
        let barrels_per_build = if runner.game_phase == GamePhase::Canal {
            1.0
        } else {
            2.0
        };
        let brewery_actions = (f64::from(needed) / barrels_per_build).ceil();
        let work = 1.0 + distance as f64 + brewery_actions;
        if actions + 0.001 < work {
            beer = old_beer;
            merchant_beer = old_merchant_beer;
            continue;
        }
        let probability = 0.78_f64.powf(work) * if external { 0.90 } else { 1.0 };
        let vp = f64::from(tile.vp_on_flip) * era_multiplier(runner, tile);
        let route_cost = distance as f64
            * if runner.game_phase == GamePhase::Canal {
                3.0
            } else {
                6.0
            };
        let conversion_cost = 0.35 * (route_cost + brewery_actions * 9.0);
        result.products += (probability * vp - conversion_cost).max(0.0);
        flip_probabilities[location] = probability;
        pending_income.push((work * 0.5, probability * f64::from(tile.income)));
    }

    for location in state.player_building_mask[actor].ones() {
        let Some(building) = state.bl_to_building.get(&location) else {
            continue;
        };
        let tile = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        if building.flipped {
            result.carryover += f64::from(tile.vp_on_flip) * (era_multiplier(runner, tile) - 1.0);
        } else if !is_saleable_industry(building.industry) {
            let town = LocationName::from_bl_idx(location);
            let connected_towns = (0..TOTAL_TOWNS)
                .filter(|other| {
                    state
                        .connectivity
                        .are_towns_connected(town, LocationName::from_usize(*other))
                })
                .count() as f64;
            let demand = match building.industry {
                IndustryType::Beer => {
                    f64::from(position.sellable_beer_demand)
                        + if runner.game_phase == GamePhase::Railroad {
                            actions * 0.30
                        } else {
                            actions * 0.12
                        }
                }
                IndustryType::Coal => {
                    actions
                        * state.players.len() as f64
                        * if runner.game_phase == GamePhase::Railroad {
                            0.65
                        } else {
                            0.30
                        }
                }
                _ => actions * state.players.len() as f64 * 0.25,
            };
            let supply = if building.industry == IndustryType::Beer {
                f64::from(position.unflipped_beer_units)
            } else {
                stocks[building.industry.as_usize()]
            };
            let mut probability = (demand / supply.max(1.0)).clamp(0.0, 0.90);
            if building.industry == IndustryType::Coal {
                probability *= (connected_towns / 6.0).clamp(0.25, 1.0);
            } else if building.industry == IndustryType::Beer && position.sellable_beer_demand == 0
            {
                probability *= (connected_towns / 6.0).clamp(0.35, 1.0);
            }
            result.resources +=
                probability * f64::from(tile.vp_on_flip) * era_multiplier(runner, tile);
            flip_probabilities[location] = probability;
            pending_income.push((1.0, probability * f64::from(tile.income)));
        }
    }
    // All investments advance one shared, nonlinear income track. Valuing each
    // independently made a loan appear to increase every idle tile's income.
    for round in 0..horizon.ceil() as usize {
        let steps: f64 = pending_income
            .iter()
            .filter(|(delay, _)| *delay <= round as f64)
            .map(|(_, steps)| *steps)
            .sum();
        let level = (f64::from(player.income_level) + steps).clamp(0.0, 100.0);
        let lower = player.get_income_amount(level.floor() as u8);
        let upper = player.get_income_amount(level.ceil() as u8);
        let projected = f64::from(lower) + f64::from(upper - lower) * level.fract();
        result.income += projected * (horizon - round as f64).min(1.0) * 0.48;
    }
    result.income += f64::from(player.income_level) * 0.025;
    result.network = position.available_build_sites.min(12) as f64 * 0.25
        + position.connected_trade_posts.min(3) as f64 * 0.3;
    for road in state.player_road_mask[actor].ones() {
        for town in LINK_LOCATIONS[road].locations.ones() {
            for location in LocationName::from_usize(town).to_bl_set().ones() {
                let Some(building) = state.bl_to_building.get(&location).filter(|b| !b.flipped)
                else {
                    continue;
                };
                let tile = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
                let probability = if building.owner.as_usize() == actor {
                    flip_probabilities[location]
                } else {
                    (actions / 6.0).min(0.65)
                };
                result.network += f64::from(tile.road_vp) * probability;
            }
        }
    }
    result
}

fn development_value(runner: &GameRunner, after: &GameRunner, intent: &ActionIntent) -> f64 {
    if !matches!(
        intent.action_type,
        ActionType::Develop | ActionType::DevelopDouble
    ) {
        return 0.0;
    }
    let actor = runner.framework.current_player;
    let actions = remaining_actions(after, actor);
    if actions < 2.0 {
        return -5.0;
    }
    let mut seen = [false; 6];
    let mut gain = 0.0;
    for industry in [intent.selected_industry, intent.selected_second_industry]
        .into_iter()
        .flatten()
    {
        if seen[industry.as_usize()] {
            continue;
        }
        seen[industry.as_usize()] = true;
        let old = runner.framework.board.state.players[actor]
            .industry_mat
            .get_tile_for_industry(industry);
        let new = after.framework.board.state.players[actor]
            .industry_mat
            .get_tile_for_industry(industry);
        let (Some(old), Some(new)) = (old, new) else {
            continue;
        };
        let quality = |tile: &BuildingTypeData| {
            f64::from(tile.vp_on_flip) * era_multiplier(runner, tile)
                + income_gain(runner, actor, tile.income) * 0.7
                - f64::from(tile.money_cost) * 0.25
                - f64::from(tile.beer_needed) * 2.0
        };
        let known_cards = runner.framework.board.state.players[actor]
            .hand
            .cards
            .len()
            .saturating_sub(1);
        let buildable = runner.game_phase == after.game_phase
            && after
                .framework
                .board
                .get_valid_build_options(actor)
                .iter()
                .any(|option| {
                    option.industry_type == industry && option.card_used_idx < known_cards
                });
        if buildable {
            gain += (quality(new) - quality(old)).max(0.0) * (actions / 6.0).min(1.5);
        }
    }
    gain
}

pub(super) fn known_continuation(
    before: &GameRunner,
    after: &GameRunner,
    intent: &ActionIntent,
) -> GameRunner {
    let mut known = after.clone();
    let actor = before.framework.current_player;
    if before.game_phase != after.game_phase {
        return known;
    }
    let discarded = usize::from(intent.selected_card_idx.is_some())
        + intent.scout_additional_discard_indices.len();
    let gained = if intent.action_type == ActionType::Scout {
        2
    } else {
        0
    };
    let known_count = before.framework.board.state.players[actor]
        .hand
        .cards
        .len()
        .saturating_sub(discarded)
        + gained;
    let hand = &mut known.framework.board.state.players[actor].hand.cards;
    if hand.len() > known_count {
        // Keep the public remaining-card budget, but do not plan using a draw.
        let unknown = hand.split_off(known_count);
        known.framework.board.state.deck.cards.extend(unknown);
    }
    known
}

pub(super) fn score_action(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
    card_values: &[f64],
) -> (RuleScoreBreakdown, f64) {
    let actor = runner.framework.current_player;
    let old = value(runner, actor, before);
    let new = value(after_runner, actor, after);
    let immediate_vp = new.secured - old.secured;
    let potential_vp = new.carryover - old.carryover;
    let industry = new.products - old.products;
    let resources = new.resources - old.resources;
    let income = new.income - old.income;
    let cash = new.cash - old.cash;
    let network = new.network - old.network;
    let opponents = runner.framework.board.state.players.len() - 1;
    let competitive_pressure = -(0..opponents + 1)
        .filter(|p| *p != actor)
        .map(|p| {
            let secured = f64::from(effect.player_victory_points_delta[p]);
            let potential = if after_runner.is_game_finished() {
                -f64::from(potential_era_victory_points(runner)[p])
            } else {
                f64::from(effect.player_potential_era_victory_points_delta[p])
            };
            secured + potential
        })
        .sum::<f64>()
        * 0.35
        / opponents as f64;
    let action_bias = development_value(runner, after_runner, intent)
        + if intent.action_type == ActionType::Pass {
            -2.0
        } else {
            0.0
        };
    let card_value = 0.35 * card_opportunity_score(runner, intent, card_values);
    let total = immediate_vp
        + potential_vp
        + industry
        + resources
        + income
        + cash
        + network
        + competitive_pressure
        + action_bias
        + card_value;
    (
        RuleScoreBreakdown {
            immediate_vp,
            potential_vp,
            industry,
            resources,
            income,
            cash,
            network,
            competitive_pressure,
            action_bias,
            card_value,
            safety: 0.0,
            tempo: 0.0,
            lookahead: 0.0,
            total,
        },
        total,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::building::BuiltBuilding;
    use crate::core::player::PlayerId;
    use crate::core::types::{Card, Era, IndustryLevel, IndustrySet};
    use crate::market::merchants::{MerchantTile, MerchantTileType};

    fn add_building(
        runner: &mut GameRunner,
        location: usize,
        industry: IndustryType,
        level: IndustryLevel,
        flipped: bool,
    ) {
        let actor = runner.framework.current_player;
        let state = &mut runner.framework.board.state;
        let mut building =
            BuiltBuilding::build(industry, level, location as u8, PlayerId::from_usize(actor));
        building.flipped = flipped;
        state.bl_to_building.insert(location, building);
        state.build_locations_occupied.insert(location);
        state.player_building_mask[actor].insert(location);
        match industry {
            IndustryType::Coal => state.coal_locations.insert(location),
            IndustryType::Iron => state.iron_locations.insert(location),
            IndustryType::Beer => state.beer_locations.insert(location),
            _ => (),
        }
    }

    fn position_value(runner: &GameRunner) -> Value {
        let actor = runner.framework.current_player;
        value(runner, actor, &snapshot(runner, actor))
    }

    fn product_position() -> GameRunner {
        let mut runner = GameRunner::new(2, Some(7301));
        runner.game_phase = GamePhase::Railroad;
        let actor = runner.framework.current_player;
        let state = &mut runner.framework.board.state;
        state.era = Era::Railroad;
        state.deck.cards.clear();
        state.trade_post_slots.fill(None);
        state.trade_post_slots[1] = Some(MerchantTile::from_type(MerchantTileType::Goods));
        state.trade_post_beer.clear();
        state.place_link(actor, 31);
        add_building(
            &mut runner,
            36,
            IndustryType::Goods,
            IndustryLevel::II,
            false,
        );
        runner
    }

    #[test]
    fn surviving_canal_tiles_score_in_both_eras_without_terminal_double_counting() {
        let mut runner = GameRunner::new(2, Some(7302));
        add_building(
            &mut runner,
            36,
            IndustryType::Cotton,
            IndustryLevel::II,
            true,
        );
        let canal = position_value(&runner);
        assert_eq!(canal.secured, 5.0);
        assert_eq!(canal.carryover, 5.0);
        runner.game_phase = GamePhase::Railroad;
        runner.end_era();
        let rail = position_value(&runner);
        assert_eq!(rail.secured, 10.0);
        assert_eq!(rail.carryover, 0.0);
        runner.game_phase = GamePhase::GameEnd;
        runner.end_era();
        let terminal = position_value(&runner);
        assert_eq!(terminal.secured, 10.0);
        assert_eq!(
            terminal.carryover
                + terminal.products
                + terminal.resources
                + terminal.cash
                + terminal.income,
            0.0
        );
    }

    #[test]
    fn level_one_has_no_second_era_value() {
        let mut runner = GameRunner::new(2, Some(7303));
        add_building(
            &mut runner,
            36,
            IndustryType::Cotton,
            IndustryLevel::I,
            true,
        );
        assert_eq!(position_value(&runner).carryover, 0.0);
    }

    #[test]
    fn product_needs_a_compatible_merchant_and_enough_action_cards() {
        let mut runner = product_position();
        let actor = runner.framework.current_player;
        add_building(
            &mut runner,
            47,
            IndustryType::Beer,
            IndustryLevel::II,
            false,
        );
        assert!(position_value(&runner).products > 0.0);
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .clear();
        assert_eq!(position_value(&runner).products, 0.0);
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .push(Card::new(CardType::WildIndustry));
        runner.framework.board.state.trade_post_slots.fill(None);
        assert_eq!(position_value(&runner).products, 0.0);
    }

    #[test]
    fn one_merchant_barrel_cannot_fund_two_products() {
        let mut runner = product_position();
        let actor = runner.framework.current_player;
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .truncate(1);
        runner.framework.board.state.trade_post_beer.insert(1);
        let single = position_value(&runner).products;
        assert!(single > 0.0);
        add_building(
            &mut runner,
            37,
            IndustryType::Goods,
            IndustryLevel::II,
            false,
        );
        assert_eq!(position_value(&runner).products, single);
        runner.framework.board.state.trade_post_slots[2] =
            Some(MerchantTile::from_type(MerchantTileType::Goods));
        runner.framework.board.state.trade_post_beer.insert(2);
        assert!((position_value(&runner).products - single * 2.0).abs() < 1e-9);
    }

    #[test]
    fn multiple_zero_beer_products_share_one_sell_action() {
        let mut runner = product_position();
        let actor = runner.framework.current_player;
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .truncate(1);
        runner
            .framework
            .board
            .state
            .bl_to_building
            .get_mut(&36)
            .unwrap()
            .level = IndustryLevel::III;
        let single = position_value(&runner).products;
        add_building(
            &mut runner,
            37,
            IndustryType::Goods,
            IndustryLevel::III,
            false,
        );
        assert!((position_value(&runner).products - 2.0 * single).abs() < 1e-9);
    }

    #[test]
    fn impossible_product_does_not_reserve_another_products_barrel() {
        let mut runner = product_position();
        let actor = runner.framework.current_player;
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .truncate(1);
        runner.framework.board.state.trade_post_beer.insert(1);
        let single = position_value(&runner).products;
        runner
            .framework
            .board
            .state
            .bl_to_building
            .get_mut(&36)
            .unwrap()
            .level = IndustryLevel::V;
        add_building(
            &mut runner,
            37,
            IndustryType::Goods,
            IndustryLevel::II,
            false,
        );
        assert_eq!(position_value(&runner).products, single);
    }

    #[test]
    fn isolated_coal_does_not_assume_global_consumption() {
        let mut runner = GameRunner::new(3, Some(7305));
        runner.framework.board.state.remaining_market_coal = MAX_MARKET_COAL;
        add_building(
            &mut runner,
            29,
            IndustryType::Coal,
            IndustryLevel::II,
            false,
        );
        let isolated = position_value(&runner).resources;
        let actor = runner.framework.current_player;
        for road in 0..LINK_LOCATIONS.len() {
            runner.framework.board.state.place_link(actor, road);
        }
        assert!(position_value(&runner).resources > isolated);
    }

    #[test]
    fn brewery_value_depends_on_real_product_demand() {
        let mut runner = GameRunner::new(3, Some(7306));
        add_building(
            &mut runner,
            47,
            IndustryType::Beer,
            IndustryLevel::II,
            false,
        );
        let idle = position_value(&runner).resources;
        add_building(
            &mut runner,
            36,
            IndustryType::Goods,
            IndustryLevel::V,
            false,
        );
        assert!(position_value(&runner).resources > idle);
    }

    #[test]
    fn a_planned_merchant_connection_can_supply_its_sale_barrel() {
        let mut runner = product_position();
        let actor = runner.framework.current_player;
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .truncate(2);
        runner.framework.board.state.trade_post_beer.insert(1);
        runner.framework.board.state.built_roads.clear();
        runner.framework.board.state.player_road_mask[actor].clear();
        runner.framework.board.state.connectivity = crate::board::connectivity::Connectivity::new();
        assert!(
            position_value(&runner).products > 0.0,
            "one route and one sell need no brewery"
        );
        runner.framework.board.state.trade_post_beer.clear();
        assert_eq!(
            position_value(&runner).products,
            0.0,
            "two cards cannot also build a brewery"
        );
    }

    #[test]
    fn borrowing_does_not_increase_the_value_of_idle_resource_income() {
        let mut runner = GameRunner::new(4, Some(7401));
        add_building(
            &mut runner,
            20,
            IndustryType::Coal,
            IndustryLevel::II,
            false,
        );
        add_building(
            &mut runner,
            11,
            IndustryType::Coal,
            IndustryLevel::II,
            false,
        );
        add_building(
            &mut runner,
            47,
            IndustryType::Beer,
            IndustryLevel::II,
            false,
        );
        let actor = runner.framework.current_player;
        let before = position_value(&runner);
        runner.framework.board.state.players[actor].decrease_income_level(3);
        runner.framework.board.state.players[actor].gain_money(30);
        let after = position_value(&runner);
        assert_eq!(before.resources, after.resources);
        assert!(after.income < before.income);
    }

    #[test]
    fn developing_into_debt_cannot_create_an_income_gain() {
        let mut runner = GameRunner::new(4, Some(7402));
        for location in [20, 11, 29] {
            add_building(
                &mut runner,
                location,
                IndustryType::Coal,
                IndustryLevel::II,
                false,
            );
        }
        let actor = runner.framework.current_player;
        let mut previous = f64::NEG_INFINITY;
        for space in 0..=100 {
            runner.framework.board.state.players[actor].income_level = space;
            let current = position_value(&runner).income;
            assert!(current >= previous, "income space {space}");
            previous = current;
        }
    }

    #[test]
    fn cash_has_diminishing_returns_and_no_vp_after_game_end() {
        assert!(
            money_value(30.0, 10.0) - money_value(0.0, 10.0)
                > money_value(90.0, 10.0) - money_value(60.0, 10.0)
        );
        let mut runner = product_position();
        runner.game_phase = GamePhase::GameEnd;
        runner.framework.board.state.players[runner.framework.current_player].money = 300;
        assert_eq!(position_value(&runner).cash, 0.0);
    }

    #[test]
    fn lookahead_stops_when_a_turn_advances_back_to_the_same_player() {
        let mut runner = GameRunner::new(2, Some(7403));
        let actor = runner.framework.current_player;
        runner.framework.board.state.turn_order = vec![actor, 1 - actor];
        runner.framework.board.state.players[1 - actor]
            .hand
            .cards
            .clear();
        runner.framework.board.state.deck.cards.clear();
        runner.framework.board.state.players[actor]
            .hand
            .cards
            .truncate(3);
        runner.personal_turns_taken.fill(1);
        runner.actions_remaining_in_turn = 1;
        runner.turn_started = true;
        let ranked = rank_rule_actions(&runner, &RuleDecisionConfig::default()).unwrap();
        assert!(ranked.iter().all(|a| a.breakdown.lookahead == 0.0));
    }

    #[test]
    fn changed_hidden_cards_do_not_change_scores_or_recommendations() {
        let mut runner = GameRunner::new(3, Some(7304));
        let actor = runner.framework.current_player;
        runner.personal_turns_taken.fill(1);
        runner.start_turn();
        runner.framework.board.state.players[actor].money = 40;
        runner.framework.board.state.players[actor].hand.cards = vec![
            Card::new(CardType::Industry(
                IndustrySet::new_from_industry_types(&[IndustryType::Cotton])
            ));
            8
        ];
        let mut shuffled = runner.clone();
        shuffled.framework.board.state.deck.cards.reverse();
        let opponent = (actor + 1) % 3;
        std::mem::swap(
            &mut shuffled.framework.board.state.deck.cards[0],
            &mut shuffled.framework.board.state.players[opponent].hand.cards[0],
        );
        for depth in [1, 2] {
            let config = RuleDecisionConfig {
                economic_evaluation: true,
                lookahead_depth: depth,
                lookahead_branching: 3,
                ..RuleDecisionConfig::legacy()
            };
            let summarize = |position: &GameRunner| {
                rank_rule_actions(position, &config)
                    .unwrap()
                    .into_iter()
                    .map(|a| (a.action.key(), a.score))
                    .collect::<Vec<_>>()
            };
            assert_eq!(summarize(&runner), summarize(&shuffled));
        }
    }
}
