//! Frozen teacher value network + human/card guidance on native legal transitions.
//! The checkpoint is shared with teacher-trained-v2; rules and hidden information
//! belong to the Rust game. JS self-play scores are not native-game benchmarks.
use std::collections::{HashMap, VecDeque};

use once_cell::sync::Lazy;
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;

use crate::core::building::BuiltBuilding;
use crate::core::locations::LocationName;
use crate::core::static_data::{INDUSTRY_MAT, LINK_LOCATIONS, MAX_LEVELS_PER_INDUSTRY};
use crate::core::types::{ActionType, BitSetWrapper, CardType, Era, IndustryType};
use crate::game::hidden_information::determinize_hidden_information;
use crate::game::rule_ai::{rank_rule_actions, RuleActionScore, RuleDecisionConfig};
use crate::game::runner::GameRunner;
use crate::game::search::{
    immediate_effect_between, RootActionEstimate, RootSearchReport, SampleRandomContinuation,
};
use crate::game::training::advance_successor_to_decision;
use crate::market::merchants::slot_to_trade_post;

pub const METHOD: &str = "teacher_trained_v2_native_guided";
pub const VALUE_SOURCE: &str = "teacher_value_human_card_native";
pub const MODEL_ID: &str = "human-teacher-20261009-epoch105-native-v1";
// The exact alphabetic industry order in brass-value-v2, not Rust enum order.
const TYPES: [IndustryType; 6] = [
    IndustryType::Beer,
    IndustryType::Coal,
    IndustryType::Cotton,
    IndustryType::Iron,
    IndustryType::Goods,
    IndustryType::Pottery,
];

#[derive(Deserialize)]
struct Layer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ValueModel {
    input_dim: usize,
    feature_version: String,
    scale: f64,
    layers: Vec<Layer>,
}
static MODEL: Lazy<ValueModel> = Lazy::new(|| {
    let m: ValueModel = serde_json::from_str(include_str!("models/teacher-value.json"))
        .expect("embedded teacher checkpoint");
    assert_eq!(m.feature_version, "brass-value-v2");
    assert_eq!(m.input_dim, 125);
    let mut dimension = m.input_dim;
    for l in &m.layers {
        assert_eq!(l.weights.len(), dimension);
        assert!(l
            .weights
            .iter()
            .all(|w| w.len() == l.bias.len() && w.iter().all(|v| v.is_finite())));
        assert!(l.bias.iter().all(|v| v.is_finite()));
        assert!(l.activation == "relu" || l.activation == "linear");
        dimension = l.bias.len();
    }
    assert_eq!(dimension, 1);
    assert_eq!(m.scale, 100.0);
    m
});

fn forward(input: &[f64]) -> f64 {
    assert_eq!(input.len(), MODEL.input_dim);
    let mut x = input.to_vec();
    for l in &MODEL.layers {
        let mut y = l.bias.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
        for (v, weights) in x.iter().zip(&l.weights) {
            for (out, w) in y.iter_mut().zip(weights) {
                *out += v * f64::from(*w);
            }
        }
        if l.activation == "relu" {
            y.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        x = y;
    }
    x[0] * MODEL.scale
}

fn tile(b: &BuiltBuilding) -> &'static crate::core::types::BuildingTypeData {
    &INDUSTRY_MAT[b.industry.as_usize()][b.level.as_usize()]
}
fn saleable(t: IndustryType) -> bool {
    matches!(
        t,
        IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
    )
}
fn remaining_tiles(
    r: &GameRunner,
    p: usize,
    t: IndustryType,
) -> Vec<(usize, &'static crate::core::types::BuildingTypeData)> {
    let mat = &r.framework.board.state.players[p].industry_mat;
    let (level, remaining) = mat.get_progress(t);
    (level.as_usize()..=MAX_LEVELS_PER_INDUSTRY[t.as_usize()].as_usize())
        .flat_map(|i| {
            let data = &INDUSTRY_MAT[t.as_usize()][i];
            let count = if i == level.as_usize() {
                remaining
            } else {
                data.num_tiles
            };
            (0..count).map(move |_| (i + 1, data))
        })
        .collect()
}

/// Semantic projection of native state into the checkpoint's 125 features.
/// The value head uses aggregate industries and link geography, so no fragile
/// JS slot-index mapping or foreign action vectors are needed.
pub fn value_features(r: &GameRunner, actor: usize) -> Vec<f64> {
    let s = &r.framework.board.state;
    let buildings = (0..49)
        .filter_map(|loc| s.bl_to_building.get(&loc))
        .collect::<Vec<_>>();
    let rail = s.era == Era::Railroad;
    let per_turn = if r.personal_turns_taken[r.framework.current_player] == 0 {
        1.0
    } else {
        2.0
    };
    let roles: Vec<Vec<f64>> = (0..4)
        .map(|p| {
            let mut row = vec![0.0; 49];
            let Some(pl) = s.players.get(p) else {
                return row;
            };
            let roads = s.player_road_mask[p].count_ones() as f64;
            row[..9].copy_from_slice(&[
                f64::from(pl.money) / 100.0,
                f64::from(pl.income_level) / 100.0,
                f64::from(pl.victory_points) / 100.0,
                pl.hand.cards.len() as f64 / 8.0,
                f64::from(pl.spent_this_turn) / 50.0,
                (14.0 - if rail { 0.0 } else { roads }) / 14.0,
                (14.0 - if rail { roads } else { 0.0 }) / 14.0,
                f64::from(
                    pl.hand
                        .cards
                        .iter()
                        .any(|c| c.card_type == CardType::WildLocation),
                ),
                f64::from(
                    pl.hand
                        .cards
                        .iter()
                        .any(|c| c.card_type == CardType::WildIndustry),
                ),
            ]);
            for (i, t) in TYPES.iter().enumerate() {
                let total: usize = (0..=MAX_LEVELS_PER_INDUSTRY[t.as_usize()].as_usize())
                    .map(|l| INDUSTRY_MAT[t.as_usize()][l].num_tiles as usize)
                    .sum();
                row[9 + i] = (total - remaining_tiles(r, p, *t).len()) as f64 / 12.0;
            }
            for b in buildings.iter().filter(|b| b.owner.as_usize() == p) {
                let i = TYPES.iter().position(|t| *t == b.industry).unwrap();
                row[15
                    + i * 4
                    + usize::from(b.level.as_usize() >= 1) * 2
                    + usize::from(b.flipped)] += 0.25;
                row[39 + usize::from(!b.flipped)] += f64::from(tile(b).vp_on_flip) / 100.0;
                row[41 + i] += f64::from(b.resource_amt) / 10.0;
            }
            row[47 + usize::from(rail)] = roads / 14.0;
            row
        })
        .collect();
    let extras: Vec<Vec<f64>> = (0..4)
        .map(|p| {
            let mut row = vec![0.0; 8];
            let Some(mask) = s.player_road_mask.get(p) else {
                return row;
            };
            for road in mask.ones() {
                let locations = &LINK_LOCATIONS[road].locations;
                row[0] += locations.ones().filter(|l| *l >= 22).count() as f64 * 2.0;
                for b in buildings.iter().filter(|b| {
                    locations.contains(LocationName::from_bl_idx(b.loc as usize).as_usize())
                }) {
                    let icon = f64::from(tile(b).road_vp);
                    row[usize::from(!b.flipped)] += icon;
                    if b.owner.as_usize() == p {
                        row[4 + usize::from(!b.flipped)] += icon;
                    }
                }
            }
            for b in buildings.iter().filter(|b| b.owner.as_usize() == p) {
                let vp = f64::from(tile(b).vp_on_flip);
                if b.level.as_usize() >= 1 {
                    row[2 + usize::from(!b.flipped)] += vp;
                }
                if !b.flipped && saleable(b.industry) {
                    row[6] += vp;
                    if b.level.as_usize() >= 1 {
                        row[7] += vp;
                    }
                }
            }
            row.iter_mut().for_each(|v| *v /= 100.0);
            row
        })
        .collect();
    let opponents = (s.players.len() - 1) as f64;
    let mean = |rows: &Vec<Vec<f64>>| -> Vec<f64> {
        (0..rows[0].len())
            .map(|i| {
                rows.iter()
                    .enumerate()
                    .filter(|(p, _)| *p != actor)
                    .map(|(_, row)| row[i])
                    .sum::<f64>()
                    / opponents
            })
            .collect()
    };
    // JS market encoders divide by the number of price entries (15/11).
    let mut v = vec![
        f64::from(rail),
        (r.round_in_phase + 1) as f64 / 16.0,
        s.players.len() as f64 / 4.0,
        s.deck.cards.len() as f64 / 64.0,
        (per_turn - f64::from(r.actions_remaining_in_turn)) / 2.0,
        per_turn / 2.0,
        f64::from(r.is_game_finished()),
        f64::from(s.remaining_market_coal) / 15.0,
        f64::from(s.remaining_market_iron) / 11.0,
    ];
    v.extend(&roles[actor]);
    v.extend(mean(&roles));
    v.extend([
        f64::from(r.framework.current_player == actor),
        s.players
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != actor)
            .map(|(_, p)| f64::from(p.victory_points) / 100.0)
            .fold(0.0, f64::max),
    ]);
    v.extend(&extras[actor]);
    v.extend(mean(&extras));
    v
}

pub fn learned_value(r: &GameRunner, actor: usize) -> f64 {
    if r.is_game_finished() {
        f64::from(r.framework.board.state.players[actor].victory_points)
    } else {
        forward(&value_features(r, actor))
    }
}
fn actions_left(r: &GameRunner, p: usize) -> f64 {
    let s = &r.framework.board.state;
    s.players[p].hand.cards.len() as f64 + s.deck.cards.len() as f64 / s.players.len() as f64
}

fn distances(r: &GameRunner, start: usize) -> [usize; 27] {
    let s = &r.framework.board.state;
    let mut d = [usize::MAX; 27];
    d[start] = 0;
    let mut q = VecDeque::from([start]);
    while let Some(city) = q.pop_front() {
        for (i, link) in LINK_LOCATIONS
            .iter()
            .enumerate()
            .filter(|(_, l)| l.locations.contains(city))
        {
            if !(if s.era == Era::Canal {
                link.can_build_canal
            } else {
                link.can_build_rail
            }) {
                continue;
            }
            let cost = usize::from(!s.built_roads.contains(i));
            for next in link.locations.ones() {
                if d[city] + cost < d[next] {
                    d[next] = d[city] + cost;
                    if cost == 0 {
                        q.push_front(next)
                    } else {
                        q.push_back(next)
                    }
                }
            }
        }
    }
    d
}

// Port of human_strategy.js, using native connectivity and tile data.
pub fn human_value(r: &GameRunner, actor: usize) -> f64 {
    let s = &r.framework.board.state;
    if r.is_game_finished() {
        return f64::from(s.players[actor].victory_points)
            - 0.15
                * s.players
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| *p != actor)
                    .map(|(_, p)| f64::from(p.victory_points))
                    .fold(0.0, f64::max);
    }
    let buildings = (0..49)
        .filter_map(|loc| s.bl_to_building.get(&loc))
        .collect::<Vec<_>>();
    let mut stock = [0.0; 6];
    for b in &buildings {
        stock[b.industry.as_usize()] += f64::from(b.resource_amt);
    }
    let mut probabilities = HashMap::new();
    for b in &buildings {
        let left = actions_left(r, b.owner.as_usize());
        let city = LocationName::from_bl_idx(b.loc as usize);
        let data = tile(b);
        let probability = if b.flipped {
            1.0
        } else if left < 1.0 {
            0.0
        } else if saleable(b.industry) {
            let d = distances(r, city.as_usize());
            let merchants = s
                .trade_post_slots
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.as_ref()
                        .is_some_and(|m| m.industries.contains(b.industry.as_usize()))
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let distance = merchants
                .iter()
                .map(|i| d[slot_to_trade_post(*i).to_location_name().as_usize()])
                .min()
                .unwrap_or(usize::MAX);
            let own_beer = buildings
                .iter()
                .filter(|source| {
                    source.industry == IndustryType::Beer
                        && (source.owner == b.owner
                            || s.connectivity.are_towns_connected(
                                city,
                                LocationName::from_bl_idx(source.loc as usize),
                            ))
                })
                .map(|source| f64::from(source.resource_amt))
                .sum::<f64>();
            let merchant_beer = merchants
                .iter()
                .filter(|i| {
                    s.trade_post_beer.contains(**i)
                        && d[slot_to_trade_post(**i).to_location_name().as_usize()] == 0
                })
                .count() as f64;
            let missing = (f64::from(data.beer_needed) - own_beer - merchant_beer).max(0.0);
            if distance == usize::MAX || left < (distance as f64 + 1.0 + (missing / 2.0).ceil()) {
                0.0
            } else {
                ((match distance {
                    0 => 0.94,
                    1 => 0.70,
                    2 => 0.40,
                    _ => 0.12,
                }) - missing * 0.15)
                    .clamp(0.0, 1.0)
            }
        } else if b.industry == IndustryType::Beer {
            let demand = buildings
                .iter()
                .filter(|t| t.owner == b.owner && !t.flipped && saleable(t.industry))
                .map(|t| f64::from(tile(t).beer_needed))
                .sum::<f64>();
            let rails = if s.era == Era::Railroad {
                (left / 2.0)
                    .min((14.0 - s.player_road_mask[b.owner.as_usize()].count_ones() as f64) / 2.0)
            } else {
                0.0
            };
            ((demand + rails + left * 0.12)
                / (f64::from(b.resource_amt).max(1.0) + stock[b.industry.as_usize()] * 0.3))
                .clamp(0.05, 0.94)
        } else {
            let iron = b.industry == IndustryType::Iron;
            let demand = if iron {
                (8.0 - f64::from(s.remaining_market_iron)).max(0.0)
            } else {
                (10.0 - f64::from(s.remaining_market_coal)).max(0.0)
            };
            let connected = if iron || s.is_location_connected_to_trade_post(city) {
                1.0
            } else {
                0.55
            };
            (connected * (left * if iron { 0.7 } else { 0.8 } + demand)
                / (stock[b.industry.as_usize()] + f64::from(b.resource_amt)).max(2.0))
            .clamp(0.03, 0.92)
        };
        probabilities.insert(b.loc, probability);
    }
    let values = s
        .players
        .iter()
        .enumerate()
        .map(|(p, pl)| {
            let left = actions_left(r, p);
            let canal = s.era == Era::Canal;
            let industry = buildings
                .iter()
                .filter(|b| b.owner.as_usize() == p)
                .map(|b| {
                    f64::from(tile(b).vp_on_flip)
                        * probabilities[&b.loc]
                        * if canal && b.level.as_usize() >= 1 {
                            1.9
                        } else {
                            1.0
                        }
                })
                .sum::<f64>();
            let links = s.player_road_mask[p]
                .ones()
                .map(|i| {
                    let loc = &LINK_LOCATIONS[i].locations;
                    loc.ones().filter(|c| *c >= 22).count() as f64 * 2.0
                        + buildings
                            .iter()
                            .filter(|b| {
                                loc.contains(LocationName::from_bl_idx(b.loc as usize).as_usize())
                            })
                            .map(|b| f64::from(tile(b).road_vp) * probabilities[&b.loc])
                            .sum::<f64>()
                })
                .sum::<f64>();
            let rounds = ((left - 1.0) / 2.0).max(0.0);
            let income = f64::from(pl.get_income_amount(pl.income_level));
            let money = f64::from(pl.money);
            let usable = money.min(left * 15.0);
            let cash = (usable.min(30.0) * 0.30 + (usable - 30.0).max(0.0) * 0.05)
                * (left / 3.0).clamp(0.0, 1.0);
            let income_value = (income * (rounds + if canal { 8.0 } else { 0.0 }))
                .min((left * 12.0 + if canal { 40.0 } else { 0.0 } - money).max(0.0))
                * 0.16;
            let insolvency =
                (-income * rounds - money - 30.0 * ((income + 10.0) / 3.0).floor().max(0.0))
                    .max(0.0)
                    * 0.5;
            let readiness = if left < 3.0 {
                0.0
            } else {
                let mut product = 0.0_f64;
                let mut support = 0.0_f64;
                for t in TYPES {
                    if saleable(t)
                        && !s
                            .trade_post_slots
                            .iter()
                            .flatten()
                            .any(|m| m.industries.contains(t.as_usize()))
                    {
                        continue;
                    }
                    let remaining = remaining_tiles(r, p, t);
                    let mut best = 0.0_f64;
                    let access = pl.hand.cards.iter().any(|c| match &c.card_type {
                        CardType::WildIndustry | CardType::WildLocation => true,
                        CardType::Industry(types) => types.contains(t.as_usize()),
                        CardType::Location(town) => LocationName::from_usize(town.as_usize())
                            .to_bl_set()
                            .ones()
                            .any(|i| {
                                crate::core::static_data::BUILD_LOCATION_MASK[i]
                                    .contains(t.as_usize())
                            }),
                    });
                    for (i, (_, data)) in remaining.iter().take(6).enumerate() {
                        if !data.can_build_in_era(s.era) {
                            continue;
                        }
                        if remaining[..i].iter().any(|(_, d)| !d.can_develop) {
                            break;
                        }
                        let develop = (i as f64 / 2.0).ceil();
                        let cycle = 1.0
                            + develop
                            + if saleable(t) { 0.5 } else { 0.0 }
                            + if data.money_cost >= 14 { 0.5 } else { 0.0 };
                        if cycle >= left {
                            continue;
                        }
                        let count = ((left - develop) / 2.5).max(0.0);
                        let reward = remaining[i..]
                            .iter()
                            .filter(|(_, d)| d.can_build_in_era(s.era))
                            .take(3)
                            .enumerate()
                            .map(|(j, (level, d))| {
                                f64::from(d.vp_on_flip)
                                    * if canal && *level >= 2 { 2.0 } else { 1.0 }
                                    * (count - j as f64).clamp(0.0, 1.0)
                                    * 0.75_f64.powi(j as i32)
                            })
                            .sum::<f64>();
                        best = best.max(
                            reward / cycle
                                * 0.8_f64.powf(develop)
                                * if access { 1.0 } else { 0.65 },
                        );
                    }
                    if saleable(t) {
                        product = product.max(best)
                    } else {
                        support = support.max(best)
                    }
                }
                product * 2.2 + support * 0.5
            };
            f64::from(pl.victory_points) + industry + links + cash + income_value - insolvency
                + readiness
        })
        .collect::<Vec<_>>();
    values[actor]
        - 0.15
            * values
                .iter()
                .enumerate()
                .filter(|(p, _)| *p != actor)
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max)
}

fn evaluate(r: &GameRunner, actor: usize) -> f64 {
    if r.is_game_finished() {
        human_value(r, actor)
    } else {
        0.2 * learned_value(r, actor) + 0.8 * human_value(r, actor)
    }
}

fn pool(r: &GameRunner, limit: usize) -> Result<(usize, Vec<RuleActionScore>), String> {
    let ranked = rank_rule_actions(
        r,
        &RuleDecisionConfig {
            lookahead_depth: 1,
            ..RuleDecisionConfig::default()
        },
    )?;
    let total = ranked.len();
    let mut buckets: Vec<VecDeque<RuleActionScore>> = Vec::new();
    let mut names = Vec::new();
    let mut variants = HashMap::<String, usize>::new();
    for c in ranked {
        let count = variants.entry(c.action.card_invariant_key()).or_default();
        if *count >= 2 {
            continue;
        }
        *count += 1;
        let family = if c.action.intent.action_type == ActionType::BuildBuilding {
            format!("build:{:?}", c.action.intent.selected_industry)
        } else {
            format!("{:?}", c.action.intent.action_type)
        };
        let index = names.iter().position(|n| *n == family).unwrap_or_else(|| {
            names.push(family);
            buckets.push(VecDeque::new());
            names.len() - 1
        });
        buckets[index].push_back(c);
    }
    let mut result = Vec::new();
    while result.len() < limit {
        let mut added = false;
        for b in &mut buckets {
            if let Some(c) = b.pop_front() {
                result.push(c);
                added = true;
                if result.len() == limit {
                    break;
                }
            }
        }
        if !added {
            break;
        }
    }
    Ok((total, result))
}

// Remove actual hidden card ordering before seeded sampling. Only card counts,
// our own hand and publicly acquired wild cards influence the imagined game.
fn imagined_root(r: &GameRunner) -> Result<GameRunner, String> {
    let mut copy = r.clone();
    let actor = r.framework.current_player;
    let s = &mut copy.framework.board.state;
    let mut hidden = s.deck.cards.clone();
    let mut counts = Vec::new();
    for (p, pl) in s.players.iter_mut().enumerate() {
        if p == actor {
            continue;
        }
        let normal = pl
            .hand
            .cards
            .iter()
            .filter(|c| !matches!(c.card_type, CardType::WildLocation | CardType::WildIndustry))
            .cloned()
            .collect::<Vec<_>>();
        counts.push((p, normal.len()));
        hidden.extend(normal);
        pl.hand
            .cards
            .retain(|c| matches!(c.card_type, CardType::WildLocation | CardType::WildIndustry));
    }
    // FixedBitSet's Debug representation includes its allocation address.
    // Sort semantic card contents so fresh cloud replays sample the same hands.
    hidden.sort_by_key(|c| match &c.card_type {
        CardType::Location(town) => (0, town.as_usize() as u64),
        CardType::Industry(types) => (1, types.ones().fold(0_u64, |mask, i| mask | (1 << i))),
        CardType::WildLocation => (2, 0),
        CardType::WildIndustry => (3, 0),
    });
    let mut offset = 0;
    for (p, count) in counts {
        s.players[p]
            .hand
            .cards
            .extend_from_slice(&hidden[offset..offset + count]);
        offset += count;
    }
    s.deck.cards = hidden[offset..].to_vec();
    let mut rng = StdRng::seed_from_u64(
        s.seed ^ u64::from(r.turn_count).rotate_left(19) ^ u64::from(r.actions_remaining_in_turn),
    );
    determinize_hidden_information(&mut copy, actor, &mut rng)?;
    Ok(copy)
}

pub fn trained_decision_report(r: &GameRunner, top_n: usize) -> Result<RootSearchReport, String> {
    if !(1..=10).contains(&top_n) {
        return Err("choose 1–10 recommendations".into());
    }
    let actor = r.framework.current_player;
    let imagined = imagined_root(r)?;
    let (total, candidates) = pool(&imagined, 64)?;
    let evaluated = candidates.len();
    let mut root = Vec::new();
    let mut calls = 0;
    let mut neural_calls = 0;
    for candidate in candidates {
        let mut after = imagined.clone();
        candidate.action.apply(&mut after)?;
        advance_successor_to_decision(&mut after)?;
        let score = evaluate(&after, actor);
        calls += 1;
        neural_calls += u64::from(!after.is_game_finished());
        root.push((candidate, after, score));
    }
    root.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.action.key().cmp(&b.0.action.key()))
    });
    root.truncate(8.max(top_n));
    for (_, after, score) in &mut root {
        if after.is_game_finished() {
            continue;
        }
        let next_actor = after.framework.current_player;
        let (_, followups) = pool(after, 12)?;
        let mut best = f64::NEG_INFINITY;
        let mut value = *score;
        for next in followups {
            let mut leaf = after.clone();
            next.action.apply(&mut leaf)?;
            advance_successor_to_decision(&mut leaf)?;
            let own = evaluate(&leaf, next_actor);
            calls += 1;
            neural_calls += u64::from(!leaf.is_game_finished());
            if own > best {
                best = own;
                value = if next_actor == actor {
                    own
                } else {
                    calls += 1;
                    neural_calls += u64::from(!leaf.is_game_finished());
                    evaluate(&leaf, actor)
                };
            }
        }
        if best.is_finite() {
            *score = value;
        }
    }
    root.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.action.key().cmp(&b.0.action.key()))
    });
    let recommendations = root
        .into_iter()
        .take(top_n)
        .enumerate()
        .map(|(i, (c, _, score))| RootActionEstimate {
            rank: i + 1,
            action_key: c.action.key(),
            action: c.action.clone(),
            visits: 1,
            visit_share: 0.0,
            value_source: VALUE_SOURCE.into(),
            value_sample_count: 1,
            estimated_shared_win_rate: 0.0,
            estimated_outright_win_rate: None,
            estimated_tied_first_rate: None,
            average_final_victory_points: None,
            average_victory_point_margin: c.after_actor_victory_point_margin,
            shared_win_rate_standard_error: None,
            policy_probability: None,
            calibrated_win_rate: None,
            rule_score: Some(score),
            rule_score_breakdown: None,
            immediate_effect: {
                let mut after = r.clone();
                c.action.apply(&mut after).expect("native root legality");
                immediate_effect_between(r, &after)
            },
            sample_random_continuation: SampleRandomContinuation {
                steps: Vec::new(),
                final_victory_points: Vec::new(),
                official_winners: Vec::new(),
            },
        })
        .collect();
    Ok(RootSearchReport {
        method: METHOD.into(),
        value_source: VALUE_SOURCE.into(),
        model_id: Some(MODEL_ID.into()),
        root_model_shared_win_rate: None,
        root_model_victory_point_margin: None,
        root_policy_probabilities: None,
        root_player: actor,
        requested_simulations: calls,
        completed_simulations: calls,
        root_action_count: total,
        evaluated_action_count: evaluated,
        visited_action_count: evaluated,
        all_root_actions_evaluated: evaluated == total,
        max_search_depth: Some(2),
        neural_leaf_evaluations: Some(neural_calls),
        inference_batches: None,
        root_action_model_shared_win_rates: None,
        root_action_model_victory_point_margins: None,
        root_action_model_actor_victory_points: None,
        root_action_model_shared_win_standard_errors: None,
        root_action_model_sample_counts: None,
        recommendations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::runner::GamePhase;
    #[test]
    fn native_semantic_features_match_independent_js_schema_projection() {
        use crate::core::{player::PlayerId, types::IndustryLevel};
        let mut r = GameRunner::new(2, Some(11));
        r.framework.current_player = 0;
        r.round_in_phase = 2;
        r.actions_remaining_in_turn = 1;
        r.personal_turns_taken[0] = 2;
        let s = &mut r.framework.board.state;
        s.deck.cards.truncate(12);
        s.remaining_market_coal = 7;
        s.remaining_market_iron = 5;
        let p = &mut s.players[0];
        p.money = 42;
        p.income_level = 35;
        p.victory_points = 13;
        p.spent_this_turn = 8;
        p.hand.cards.truncate(6);
        for _ in 0..2 {
            p.industry_mat.pop_tile(IndustryType::Beer);
        }
        for _ in 0..4 {
            p.industry_mat.pop_tile(IndustryType::Cotton);
        }
        s.players[1].industry_mat.pop_tile(IndustryType::Cotton);
        for (loc, owner, industry, level, flipped, resource_amt) in [
            (0, 0, IndustryType::Beer, IndustryLevel::II, false, 2),
            (6, 0, IndustryType::Cotton, IndustryLevel::II, true, 0),
            (12, 1, IndustryType::Cotton, IndustryLevel::I, true, 0),
        ] {
            s.bl_to_building.insert(
                loc,
                BuiltBuilding {
                    loc: loc as u8,
                    owner: PlayerId::from_usize(owner),
                    industry,
                    level,
                    flipped,
                    resource_amt,
                },
            );
        }
        s.player_road_mask[0].insert(10);
        s.player_road_mask[1].insert(0);
        let expected: Vec<f64> =
            serde_json::from_str(include_str!("models/teacher-native-feature-fixture.json"))
                .unwrap();
        for (i, (actual, expected)) in value_features(&r, 0).iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "feature {i}: native {actual}, JS {expected}"
            );
        }
    }
    #[test]
    fn frozen_teacher_matches_javascript_inference() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("models/teacher-value-fixture.json")).unwrap();
        for case in fixture.as_array().unwrap() {
            let input = case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>();
            assert!((forward(&input) - case["expected"].as_f64().unwrap()).abs() < 1e-8);
        }
    }
    #[test]
    fn teacher_runs_for_all_player_counts_and_recommendations_are_native_legal() {
        for n in 2..=4 {
            let mut r = GameRunner::new(n, Some(9821));
            r.start_turn();
            let before = value_features(&r, r.framework.current_player);
            assert_eq!(before.len(), 125);
            assert!(learned_value(&r, r.framework.current_player).is_finite());
            let report = trained_decision_report(&r, 3).unwrap();
            assert_eq!(report.model_id.as_deref(), Some(MODEL_ID));
            assert!(report.neural_leaf_evaluations.unwrap() > 0);
            assert_eq!(value_features(&r, r.framework.current_player), before);
            for c in report.recommendations {
                let mut copy = r.clone();
                c.action.apply(&mut copy).unwrap();
                assert!(c.rule_score.unwrap().is_finite());
            }
        }
    }
    #[test]
    fn hidden_deck_order_and_opponent_allocations_do_not_change_decisions() {
        let mut a = GameRunner::new(2, Some(31));
        a.start_turn();
        let mut b = a.clone();
        b.framework.board.state.deck.cards.reverse();
        let s = &mut b.framework.board.state;
        std::mem::swap(&mut s.deck.cards[0], &mut s.players[1].hand.cards[0]);
        let left = trained_decision_report(&a, 3).unwrap();
        let right = trained_decision_report(&b, 3).unwrap();
        assert_eq!(
            left.recommendations
                .iter()
                .map(|c| (&c.action_key, c.rule_score))
                .collect::<Vec<_>>(),
            right
                .recommendations
                .iter()
                .map(|c| (&c.action_key, c.rule_score))
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn terminal_value_is_official_score_not_prediction() {
        let mut r = GameRunner::new(3, Some(19));
        r.game_phase = GamePhase::GameEnd;
        r.framework.board.state.players[0].victory_points = 151;
        r.framework.board.state.players[1].victory_points = 120;
        assert_eq!(learned_value(&r, 0), 151.0);
        assert_eq!(human_value(&r, 0), 133.0);
    }
}
