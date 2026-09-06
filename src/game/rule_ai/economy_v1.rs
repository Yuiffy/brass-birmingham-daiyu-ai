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
    result.income = income * horizon * 0.48 + f64::from(player.income_level) * 0.025;
    result.cash = money_value(f64::from(player.money), actions);
    let reserve = (-income).max(0.0);
    result.cash -= (reserve - f64::from(player.money)).max(0.0) * 2.0;
    let settlement_reserve = reserve * (actions / 2.0).clamp(1.0, 3.0);
    result.cash -= (settlement_reserve - f64::from(player.money)).max(0.0) * 0.65;

    let distances = [
        IndustryType::Cotton,
        IndustryType::Goods,
        IndustryType::Pottery,
    ]
    .map(|industry| route_distance_map(runner, &merchant_target_mask(state, industry)));
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
                if matches
                    && state
                        .connectivity
                        .are_towns_connected(town, slot_to_trade_post(slot).to_location_name())
                {
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
        let future_income =
            income_gain(runner, actor, tile.income) * (horizon - work * 0.5).max(0.0) * 0.40;
        let route_cost = distance as f64
            * if runner.game_phase == GamePhase::Canal {
                3.0
            } else {
                6.0
            };
        let conversion_cost = 0.35 * (route_cost + brewery_actions * 9.0);
        result.products += (probability * (vp + future_income) - conversion_cost).max(0.0);
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
            if connected_towns <= 1.0
                && (building.industry == IndustryType::Coal
                    || (building.industry == IndustryType::Beer
                        && position.sellable_beer_demand == 0))
            {
                // A disconnected resource must first acquire a customer or route.
                probability *= 0.78;
            }
            result.resources += probability
                * (f64::from(tile.vp_on_flip) * era_multiplier(runner, tile)
                    + income_gain(runner, actor, tile.income) * (horizon - 1.0).max(0.0) * 0.4);
        }
    }
    result.network = position.available_build_sites.min(12) as f64 * 0.25
        + position.connected_trade_posts.min(3) as f64 * 0.3;
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
