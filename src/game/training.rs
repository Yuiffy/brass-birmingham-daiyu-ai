use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::board::resources::{BeerSellSource, BreweryBeerSource, ResourceSource};
use crate::consts::{
    MAX_MARKET_COAL, MAX_MARKET_IRON, NUM_TRADE_POSTS, N_BL, N_LOCATIONS, N_PLAYERS,
    N_ROAD_LOCATIONS,
};
use crate::core::static_data::{BUILD_LOCATION_MASK, LINK_LOCATIONS};
use crate::core::types::{ActionType, BitSetWrapper, Card, CardType, Era, IndustryType};
use crate::game::framework::{ActionChoice, NetworkMode, ShortfallResolutionSession};
use crate::game::hidden_information::determinize_hidden_information;
use crate::game::legal_actions::{enumerate_legal_actions, LegalAction};
use crate::game::runner::{GamePhase, GameRunner};
use crate::market::merchants::MerchantTileType;

pub const TRAINING_FEATURE_VERSION: u32 = 1;
pub const CARD_TYPE_DIM: usize = 29;

const ROOT_ACTION_DIM: usize = 9;
const EXACT_ACTION_DIM: usize = 9;
const PLAYER_PUBLIC_DIM: usize = 11;
const BUILDING_FEATURE_DIM: usize = 20;
const ROAD_FEATURE_DIM: usize = 10;
const INDUSTRY_MAT_FEATURE_DIM: usize = 18;
const MERCHANT_SLOT_DIM: usize = 6;

const STATE_GLOBAL_DIM: usize = 41;
const STATE_DISCARD_DIM: usize = CARD_TYPE_DIM;
const STATE_MERCHANT_DIM: usize = (NUM_TRADE_POSTS * 2) * (MERCHANT_SLOT_DIM + 1);
const STATE_BUILDINGS_DIM: usize = N_BL * BUILDING_FEATURE_DIM;
const STATE_ROADS_DIM: usize = N_ROAD_LOCATIONS * ROAD_FEATURE_DIM;
const STATE_PLAYERS_DIM: usize = N_PLAYERS * PLAYER_PUBLIC_DIM;
const STATE_INDUSTRY_MATS_DIM: usize = N_PLAYERS * INDUSTRY_MAT_FEATURE_DIM;
const STATE_PLAYER_BUILDINGS_DIM: usize = N_PLAYERS * N_BL;
const STATE_PLAYER_ROADS_DIM: usize = N_PLAYERS * N_ROAD_LOCATIONS;
const STATE_SELF_HAND_DIM: usize = CARD_TYPE_DIM;
const STATE_SHORTFALL_DIM: usize = 1 + N_PLAYERS + 1 + N_BL + N_BL;

pub const STATE_FEATURE_DIM: usize = STATE_GLOBAL_DIM
    + STATE_DISCARD_DIM
    + STATE_MERCHANT_DIM
    + STATE_BUILDINGS_DIM
    + STATE_ROADS_DIM
    + STATE_PLAYERS_DIM
    + STATE_INDUSTRY_MATS_DIM
    + STATE_PLAYER_BUILDINGS_DIM
    + STATE_PLAYER_ROADS_DIM
    + STATE_SELF_HAND_DIM
    + STATE_SHORTFALL_DIM;

const MAX_ACTION_CARDS: usize = 3;
const MAX_ACTION_INDUSTRIES: usize = 2;
const MAX_ACTION_ROADS: usize = 2;
const MAX_ACTION_COAL_SOURCES: usize = 4;
const MAX_ACTION_IRON_SOURCES: usize = 4;
const MAX_ACTION_BEER_SOURCES: usize = 8;
const MAX_ACTION_SELL_TARGETS: usize = 8;
const RESOURCE_SOURCE_DIM: usize = N_BL + 1;
const BEER_SELL_SOURCE_DIM: usize = N_BL + NUM_TRADE_POSTS * 2;
const ACTION_BEER_SOURCE_DIM: usize = N_BL * 2;

const ACTION_BIAS_OFFSET: usize = 0;
const ACTION_ROOT_OFFSET: usize = ACTION_BIAS_OFFSET + 1;
const ACTION_EXACT_OFFSET: usize = ACTION_ROOT_OFFSET + ROOT_ACTION_DIM;
const ACTION_CARD_OFFSET: usize = ACTION_EXACT_OFFSET + EXACT_ACTION_DIM;
const ACTION_INDUSTRY_OFFSET: usize = ACTION_CARD_OFFSET + MAX_ACTION_CARDS * CARD_TYPE_DIM;
const ACTION_BUILD_OFFSET: usize = ACTION_INDUSTRY_OFFSET + MAX_ACTION_INDUSTRIES * 6;
const ACTION_NETWORK_MODE_OFFSET: usize = ACTION_BUILD_OFFSET + N_BL;
const ACTION_ROAD_OFFSET: usize = ACTION_NETWORK_MODE_OFFSET + 2;
const ACTION_COAL_OFFSET: usize = ACTION_ROAD_OFFSET + MAX_ACTION_ROADS * N_ROAD_LOCATIONS;
const ACTION_IRON_OFFSET: usize =
    ACTION_COAL_OFFSET + MAX_ACTION_COAL_SOURCES * RESOURCE_SOURCE_DIM;
const ACTION_BEER_OFFSET: usize =
    ACTION_IRON_OFFSET + MAX_ACTION_IRON_SOURCES * RESOURCE_SOURCE_DIM;
const ACTION_ACTION_BEER_OFFSET: usize =
    ACTION_BEER_OFFSET + MAX_ACTION_BEER_SOURCES * BEER_SELL_SOURCE_DIM;
const ACTION_SELL_OFFSET: usize = ACTION_ACTION_BEER_OFFSET + ACTION_BEER_SOURCE_DIM;
const ACTION_FREE_DEVELOPMENT_OFFSET: usize = ACTION_SELL_OFFSET + MAX_ACTION_SELL_TARGETS * N_BL;

pub const ACTION_FEATURE_DIM: usize = ACTION_FREE_DEVELOPMENT_OFFSET + 6;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FeatureBlock {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrainingFeatureSchema {
    pub version: u32,
    pub state_dim: usize,
    pub action_dim: usize,
    pub card_type_dim: usize,
    pub max_players: usize,
    pub state_blocks: Vec<FeatureBlock>,
    pub action_blocks: Vec<FeatureBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootActionSuccessorBatch {
    pub num_players: usize,
    pub root_player: usize,
    pub action_keys: Vec<String>,
    pub determinizations_per_action: usize,
    pub states: Vec<SuccessorValueState>,
    pub samples: Vec<RootActionSuccessorSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SuccessorValueState {
    pub observer_idx: usize,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootActionSuccessorSample {
    pub action_index: usize,
    pub action_key: String,
    pub sample_index: usize,
    pub evaluation_player: usize,
    pub state_index: Option<usize>,
    pub terminal_root_shared_win_rate: Option<f64>,
    pub terminal_root_victory_point_margin: Option<f64>,
    pub terminal_root_actor_victory_points: Option<f64>,
}

pub fn training_feature_schema() -> TrainingFeatureSchema {
    let mut state_offset = 0usize;
    let state_blocks = [
        ("global", STATE_GLOBAL_DIM),
        ("discard_counts", STATE_DISCARD_DIM),
        ("merchants", STATE_MERCHANT_DIM),
        ("buildings", STATE_BUILDINGS_DIM),
        ("roads", STATE_ROADS_DIM),
        ("players", STATE_PLAYERS_DIM),
        ("industry_mats", STATE_INDUSTRY_MATS_DIM),
        ("player_building_masks", STATE_PLAYER_BUILDINGS_DIM),
        ("player_road_masks", STATE_PLAYER_ROADS_DIM),
        ("self_hand_counts", STATE_SELF_HAND_DIM),
        ("shortfall", STATE_SHORTFALL_DIM),
    ]
    .into_iter()
    .map(|(name, size)| {
        let block = FeatureBlock {
            name,
            offset: state_offset,
            size,
        };
        state_offset += size;
        block
    })
    .collect::<Vec<_>>();

    let action_blocks = vec![
        FeatureBlock {
            name: "bias",
            offset: ACTION_BIAS_OFFSET,
            size: 1,
        },
        FeatureBlock {
            name: "root_action",
            offset: ACTION_ROOT_OFFSET,
            size: ROOT_ACTION_DIM,
        },
        FeatureBlock {
            name: "exact_action",
            offset: ACTION_EXACT_OFFSET,
            size: EXACT_ACTION_DIM,
        },
        FeatureBlock {
            name: "discard_cards",
            offset: ACTION_CARD_OFFSET,
            size: MAX_ACTION_CARDS * CARD_TYPE_DIM,
        },
        FeatureBlock {
            name: "industries",
            offset: ACTION_INDUSTRY_OFFSET,
            size: MAX_ACTION_INDUSTRIES * 6,
        },
        FeatureBlock {
            name: "build_location",
            offset: ACTION_BUILD_OFFSET,
            size: N_BL,
        },
        FeatureBlock {
            name: "network_mode",
            offset: ACTION_NETWORK_MODE_OFFSET,
            size: 2,
        },
        FeatureBlock {
            name: "roads",
            offset: ACTION_ROAD_OFFSET,
            size: MAX_ACTION_ROADS * N_ROAD_LOCATIONS,
        },
        FeatureBlock {
            name: "coal_sources",
            offset: ACTION_COAL_OFFSET,
            size: MAX_ACTION_COAL_SOURCES * RESOURCE_SOURCE_DIM,
        },
        FeatureBlock {
            name: "iron_sources",
            offset: ACTION_IRON_OFFSET,
            size: MAX_ACTION_IRON_SOURCES * RESOURCE_SOURCE_DIM,
        },
        FeatureBlock {
            name: "beer_sources",
            offset: ACTION_BEER_OFFSET,
            size: MAX_ACTION_BEER_SOURCES * BEER_SELL_SOURCE_DIM,
        },
        FeatureBlock {
            name: "rail_beer_source",
            offset: ACTION_ACTION_BEER_OFFSET,
            size: ACTION_BEER_SOURCE_DIM,
        },
        FeatureBlock {
            name: "sell_targets",
            offset: ACTION_SELL_OFFSET,
            size: MAX_ACTION_SELL_TARGETS * N_BL,
        },
        FeatureBlock {
            name: "free_development",
            offset: ACTION_FREE_DEVELOPMENT_OFFSET,
            size: 6,
        },
    ];

    debug_assert_eq!(state_offset, STATE_FEATURE_DIM);
    TrainingFeatureSchema {
        version: TRAINING_FEATURE_VERSION,
        state_dim: STATE_FEATURE_DIM,
        action_dim: ACTION_FEATURE_DIM,
        card_type_dim: CARD_TYPE_DIM,
        max_players: N_PLAYERS,
        state_blocks,
        action_blocks,
    }
}

/// Builds hidden-information-consistent successor states for every stable root action.
/// Non-terminal states are encoded for the player who acts next; callers must convert
/// two-player values back to `root_player` before using them as root action values.
pub fn encode_root_action_successor_batch(
    runner: &GameRunner,
    determinizations_per_action: usize,
    seed: u64,
) -> Result<RootActionSuccessorBatch, String> {
    if runner.is_game_finished() {
        return Err("cannot encode successors for a finished game".to_string());
    }
    if !(1..=64).contains(&determinizations_per_action) {
        return Err("determinizations_per_action must be between 1 and 64".to_string());
    }
    let root_player = runner.framework.current_player;
    let actions = enumerate_legal_actions(runner)?;
    if actions.is_empty() {
        return Err("cannot encode successors without legal root actions".to_string());
    }
    let action_keys = actions.iter().map(LegalAction::key).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut determinizations = Vec::with_capacity(determinizations_per_action);
    for _ in 0..determinizations_per_action {
        let mut determinized = runner.clone();
        determinize_hidden_information(&mut determinized, root_player, &mut rng)?;
        determinizations.push(determinized);
    }

    let mut states = Vec::with_capacity(actions.len() * determinizations_per_action);
    let mut samples = Vec::with_capacity(actions.len() * determinizations_per_action);
    for (action_index, action) in actions.iter().enumerate() {
        for (sample_index, determinized) in determinizations.iter().enumerate() {
            let mut successor = determinized.clone();
            action.apply(&mut successor).map_err(|error| {
                format!(
                    "root action {} failed in successor determinization {sample_index}: {error}",
                    action_keys[action_index]
                )
            })?;
            advance_successor_to_decision(&mut successor)?;

            if successor.is_game_finished() {
                let (shared_win_rate, victory_point_margin, actor_victory_points) =
                    terminal_value_for_player(&successor, root_player)?;
                samples.push(RootActionSuccessorSample {
                    action_index,
                    action_key: action_keys[action_index].clone(),
                    sample_index,
                    evaluation_player: root_player,
                    state_index: None,
                    terminal_root_shared_win_rate: Some(shared_win_rate),
                    terminal_root_victory_point_margin: Some(victory_point_margin),
                    terminal_root_actor_victory_points: Some(actor_victory_points),
                });
            } else {
                let evaluation_player = successor.framework.current_player;
                let state_index = states.len();
                states.push(SuccessorValueState {
                    observer_idx: evaluation_player,
                    features: encode_training_state(&successor, evaluation_player)?,
                });
                samples.push(RootActionSuccessorSample {
                    action_index,
                    action_key: action_keys[action_index].clone(),
                    sample_index,
                    evaluation_player,
                    state_index: Some(state_index),
                    terminal_root_shared_win_rate: None,
                    terminal_root_victory_point_margin: None,
                    terminal_root_actor_victory_points: None,
                });
            }
        }
    }

    Ok(RootActionSuccessorBatch {
        num_players: runner.framework.board.state.players.len(),
        root_player,
        action_keys,
        determinizations_per_action,
        states,
        samples,
    })
}

pub(crate) fn advance_successor_to_decision(runner: &mut GameRunner) -> Result<(), String> {
    for _ in 0..256 {
        resolve_shortfalls_deterministically(runner);
        if runner.is_game_finished() {
            return Ok(());
        }
        if runner.actions_remaining_in_turn == 0 {
            let _ = runner.start_turn();
        }
        if !runner.framework.get_valid_root_actions().is_empty() {
            return Ok(());
        }
        runner.end_action_slot();
        if runner.actions_remaining_in_turn == 0 {
            runner.end_turn();
        }
    }
    Err("successor advance exceeded 256 forced action slots".to_string())
}

fn resolve_shortfalls_deterministically(runner: &mut GameRunner) {
    for session in runner.take_shortfall_sessions() {
        let mut tiles = session.removable_tiles.clone();
        tiles.sort_by_key(|tile| (tile.liquidation_value, tile.build_location_idx));
        runner.resolve_shortfall_with_tiles(
            session,
            tiles
                .into_iter()
                .map(|tile| tile.build_location_idx)
                .collect(),
        );
    }
}

fn terminal_value_for_player(
    runner: &GameRunner,
    player_idx: usize,
) -> Result<(f64, f64, f64), String> {
    if !runner.is_game_finished() {
        return Err("terminal value requires a finished game".to_string());
    }
    let players = &runner.framework.board.state.players;
    if player_idx >= players.len() {
        return Err(format!(
            "terminal value player {player_idx} out of bounds for {} players",
            players.len()
        ));
    }
    let best_key = players
        .iter()
        .map(|player| player.final_ranking_key())
        .max()
        .ok_or_else(|| "cannot value a game with no players".to_string())?;
    let winners = players
        .iter()
        .enumerate()
        .filter_map(|(index, player)| (player.final_ranking_key() == best_key).then_some(index))
        .collect::<Vec<_>>();
    let shared_win_rate = if winners.contains(&player_idx) {
        1.0 / winners.len() as f64
    } else {
        0.0
    };
    let best_opponent_vp = players
        .iter()
        .enumerate()
        .filter_map(|(index, player)| (index != player_idx).then_some(player.victory_points))
        .max()
        .unwrap_or(players[player_idx].victory_points);
    Ok((
        shared_win_rate,
        players[player_idx].victory_points as f64 - best_opponent_vp as f64,
        players[player_idx].victory_points as f64,
    ))
}

pub fn encode_training_state(runner: &GameRunner, observer_idx: usize) -> Result<Vec<f32>, String> {
    let state = &runner.framework.board.state;
    let num_players = state.players.len();
    validate_observer(observer_idx, num_players)?;
    let shortfall = runner.pending_shortfall_sessions.first();
    let decision_player = shortfall
        .map(|session| session.player_idx)
        .unwrap_or(runner.framework.current_player);

    let mut features = Vec::with_capacity(STATE_FEATURE_DIM);

    push_one_hot(
        &mut features,
        3,
        Some(match runner.game_phase {
            GamePhase::Canal => 0,
            GamePhase::Railroad => 1,
            GamePhase::GameEnd => 2,
        }),
    );
    push_one_hot(
        &mut features,
        2,
        Some(match state.era {
            Era::Canal => 0,
            Era::Railroad => 1,
        }),
    );
    push_one_hot(&mut features, 3, Some(num_players - 2));
    features.extend([
        runner.turn_count as f32 / 200.0,
        runner.round_in_phase as f32 / 16.0,
        runner.actions_remaining_in_turn as f32 / 2.0,
        state.wild_location_cards_available as f32 / 4.0,
        state.wild_industry_cards_available as f32 / 4.0,
        state.remaining_market_coal as f32 / MAX_MARKET_COAL as f32,
        state.remaining_market_iron as f32 / MAX_MARKET_IRON as f32,
        state.deck.len() as f32 / 80.0,
        f32::from(shortfall.is_some()),
    ]);
    push_one_hot(
        &mut features,
        N_PLAYERS,
        Some(relative_player(observer_idx, decision_player, num_players)?),
    );
    push_one_hot(
        &mut features,
        N_PLAYERS,
        Some(relative_player(
            observer_idx,
            runner.framework.current_player,
            num_players,
        )?),
    );
    for turn_position in 0..N_PLAYERS {
        let player = state.turn_order.get(turn_position).copied();
        let relative = player
            .map(|player_idx| relative_player(observer_idx, player_idx, num_players))
            .transpose()?;
        push_one_hot(&mut features, N_PLAYERS, relative);
    }
    debug_assert_eq!(features.len(), STATE_GLOBAL_DIM);

    let mut discard_counts = [0.0f32; CARD_TYPE_DIM];
    for card in &state.discard_pile {
        discard_counts[card_type_index(card)] += 1.0 / 8.0;
    }
    features.extend(discard_counts);

    for slot_idx in 0..(NUM_TRADE_POSTS * 2) {
        let merchant_idx = state
            .trade_post_slots
            .get(slot_idx)
            .and_then(|slot| slot.as_ref())
            .map(|merchant| merchant_type_index(&merchant.tile_type));
        push_one_hot(&mut features, MERCHANT_SLOT_DIM, merchant_idx);
        features.push(f32::from(state.trade_post_beer.contains(slot_idx)));
    }

    for build_location in 0..N_BL {
        if let Some(building) = state.bl_to_building.get(&build_location) {
            features.push(1.0);
            push_one_hot(
                &mut features,
                N_PLAYERS,
                Some(relative_player(
                    observer_idx,
                    building.owner.as_usize(),
                    num_players,
                )?),
            );
            push_one_hot(&mut features, 6, Some(building.industry as usize));
            features.extend([
                (building.level.as_usize() as f32 + 1.0) / 8.0,
                building.resource_amt as f32 / 5.0,
                f32::from(building.flipped),
            ]);
        } else {
            features.extend(std::iter::repeat_n(0.0, 1 + N_PLAYERS + 6 + 3));
        }
        for industry_idx in 0..6 {
            features.push(f32::from(
                BUILD_LOCATION_MASK[build_location].contains(industry_idx),
            ));
        }
    }

    for road_idx in 0..N_ROAD_LOCATIONS {
        let built = state.built_roads.contains(road_idx);
        features.push(f32::from(built));
        let owner = if built {
            (0..num_players)
                .find(|player_idx| state.player_road_mask[*player_idx].contains(road_idx))
        } else {
            None
        };
        let relative_owner = owner
            .map(|player_idx| relative_player(observer_idx, player_idx, num_players))
            .transpose()?;
        push_one_hot(&mut features, N_PLAYERS, relative_owner);
        features.extend([
            f32::from(LINK_LOCATIONS[road_idx].can_build_canal),
            f32::from(LINK_LOCATIONS[road_idx].can_build_rail),
        ]);
        let mut endpoints = [-1.0f32; 3];
        for (endpoint_idx, location_idx) in LINK_LOCATIONS[road_idx]
            .locations
            .ones()
            .take(3)
            .enumerate()
        {
            endpoints[endpoint_idx] = location_idx as f32 / (N_LOCATIONS - 1) as f32;
        }
        features.extend(endpoints);
    }

    for relative_idx in 0..N_PLAYERS {
        if let Some(player_idx) = absolute_player(observer_idx, relative_idx, num_players) {
            let player = &state.players[player_idx];
            features.extend([
                1.0,
                f32::from(relative_idx == 0),
                f32::from(player_idx == runner.framework.current_player),
                f32::from(player_idx == decision_player),
                player.money as f32 / 100.0,
                player.income_level as f32 / 100.0,
                player.get_income_amount(player.income_level) as f32 / 30.0,
                player.victory_points as f32 / 100.0,
                state.visible_vps[player_idx] as f32 / 100.0,
                player.hand.cards.len() as f32 / 8.0,
                player.spent_this_turn as f32 / 100.0,
            ]);
        } else {
            features.extend(std::iter::repeat_n(0.0, PLAYER_PUBLIC_DIM));
        }
    }

    for relative_idx in 0..N_PLAYERS {
        if let Some(player_idx) = absolute_player(observer_idx, relative_idx, num_players) {
            let mat = &state.players[player_idx].industry_mat;
            for industry_idx in 0..6 {
                let industry = IndustryType::from_usize(industry_idx);
                let level = mat.get_lowest_level(industry);
                features.extend([
                    (level.as_usize() as f32 + 1.0) / 8.0,
                    mat.get_remaining_tiles_at_level(industry) as f32 / 3.0,
                    f32::from(mat.has_tiles_left(industry)),
                ]);
            }
        } else {
            features.extend(std::iter::repeat_n(0.0, INDUSTRY_MAT_FEATURE_DIM));
        }
    }

    for relative_idx in 0..N_PLAYERS {
        let absolute = absolute_player(observer_idx, relative_idx, num_players);
        for build_location in 0..N_BL {
            features.push(f32::from(absolute.is_some_and(|player_idx| {
                state.player_building_mask[player_idx].contains(build_location)
            })));
        }
    }
    for relative_idx in 0..N_PLAYERS {
        let absolute = absolute_player(observer_idx, relative_idx, num_players);
        for road_idx in 0..N_ROAD_LOCATIONS {
            features.push(f32::from(absolute.is_some_and(|player_idx| {
                state.player_road_mask[player_idx].contains(road_idx)
            })));
        }
    }

    let mut self_hand_counts = [0.0f32; CARD_TYPE_DIM];
    for card in &state.players[observer_idx].hand.cards {
        self_hand_counts[card_type_index(card)] += 1.0 / 8.0;
    }
    features.extend(self_hand_counts);

    encode_shortfall(&mut features, shortfall, observer_idx, num_players)?;

    if features.len() != STATE_FEATURE_DIM {
        return Err(format!(
            "training state encoder produced {} features; schema requires {STATE_FEATURE_DIM}",
            features.len()
        ));
    }
    if features.iter().any(|value| !value.is_finite()) {
        return Err("training state encoder produced a non-finite value".to_string());
    }
    Ok(features)
}

pub fn encode_training_action(
    runner: &GameRunner,
    observer_idx: usize,
    action: &LegalAction,
) -> Result<Vec<u16>, String> {
    let state = &runner.framework.board.state;
    let num_players = state.players.len();
    validate_observer(observer_idx, num_players)?;
    if runner.has_pending_shortfall() {
        return Err("turn actions cannot be encoded during shortfall resolution".to_string());
    }
    if observer_idx != runner.framework.current_player {
        return Err(format!(
            "action features require the decision player as observer; expected {}, got {observer_idx}",
            runner.framework.current_player
        ));
    }

    let mut indices = vec![ACTION_BIAS_OFFSET as u16];
    indices.push((ACTION_ROOT_OFFSET + action_type_index(action.root)) as u16);
    indices.push((ACTION_EXACT_OFFSET + action_type_index(action.intent.action_type)) as u16);

    let mut card_position = 0usize;
    let mut industry_position = 0usize;
    let mut road_position = 0usize;
    let mut coal_position = 0usize;
    let mut iron_position = 0usize;
    let mut beer_position = 0usize;
    let mut sell_position = 0usize;

    for choice in &action.choices {
        match choice {
            ActionChoice::Card(card_idx) => {
                let card = state.players[observer_idx]
                    .hand
                    .cards
                    .get(*card_idx)
                    .ok_or_else(|| format!("action references missing hand card {card_idx}"))?;
                push_positioned_feature(
                    &mut indices,
                    ACTION_CARD_OFFSET,
                    card_position,
                    MAX_ACTION_CARDS,
                    CARD_TYPE_DIM,
                    card_type_index(card),
                    "discard card",
                )?;
                card_position += 1;
            }
            ActionChoice::Industry(industry) => {
                push_positioned_feature(
                    &mut indices,
                    ACTION_INDUSTRY_OFFSET,
                    industry_position,
                    MAX_ACTION_INDUSTRIES,
                    6,
                    industry.as_usize(),
                    "industry",
                )?;
                industry_position += 1;
            }
            ActionChoice::BuildLocation(location) => {
                push_single_feature(
                    &mut indices,
                    ACTION_BUILD_OFFSET,
                    N_BL,
                    *location,
                    "build location",
                )?;
            }
            ActionChoice::NetworkMode(mode) => {
                push_single_feature(
                    &mut indices,
                    ACTION_NETWORK_MODE_OFFSET,
                    2,
                    match mode {
                        NetworkMode::Single => 0,
                        NetworkMode::Double => 1,
                    },
                    "network mode",
                )?;
            }
            ActionChoice::Road(road_idx) => {
                push_positioned_feature(
                    &mut indices,
                    ACTION_ROAD_OFFSET,
                    road_position,
                    MAX_ACTION_ROADS,
                    N_ROAD_LOCATIONS,
                    *road_idx,
                    "road",
                )?;
                road_position += 1;
            }
            ActionChoice::CoalSource(source) => {
                push_positioned_feature(
                    &mut indices,
                    ACTION_COAL_OFFSET,
                    coal_position,
                    MAX_ACTION_COAL_SOURCES,
                    RESOURCE_SOURCE_DIM,
                    resource_source_index(*source)?,
                    "coal source",
                )?;
                coal_position += 1;
            }
            ActionChoice::IronSource(source) => {
                push_positioned_feature(
                    &mut indices,
                    ACTION_IRON_OFFSET,
                    iron_position,
                    MAX_ACTION_IRON_SOURCES,
                    RESOURCE_SOURCE_DIM,
                    resource_source_index(*source)?,
                    "iron source",
                )?;
                iron_position += 1;
            }
            ActionChoice::BeerSource(source) => {
                push_positioned_feature(
                    &mut indices,
                    ACTION_BEER_OFFSET,
                    beer_position,
                    MAX_ACTION_BEER_SOURCES,
                    BEER_SELL_SOURCE_DIM,
                    beer_sell_source_index(*source)?,
                    "beer source",
                )?;
                beer_position += 1;
            }
            ActionChoice::ActionBeerSource(source) => {
                push_single_feature(
                    &mut indices,
                    ACTION_ACTION_BEER_OFFSET,
                    ACTION_BEER_SOURCE_DIM,
                    action_beer_source_index(*source)?,
                    "rail beer source",
                )?;
            }
            ActionChoice::SellTarget(location) => {
                push_positioned_feature(
                    &mut indices,
                    ACTION_SELL_OFFSET,
                    sell_position,
                    MAX_ACTION_SELL_TARGETS,
                    N_BL,
                    *location,
                    "sell target",
                )?;
                sell_position += 1;
            }
            ActionChoice::FreeDevelopment(industry) => {
                push_single_feature(
                    &mut indices,
                    ACTION_FREE_DEVELOPMENT_OFFSET,
                    6,
                    industry.as_usize(),
                    "free development",
                )?;
            }
            ActionChoice::Confirm => {}
            ActionChoice::Cancel => {
                return Err("cancel is not a complete legal action feature".to_string())
            }
        }
    }

    indices.sort_unstable();
    indices.dedup();
    if indices
        .iter()
        .any(|feature_idx| *feature_idx as usize >= ACTION_FEATURE_DIM)
    {
        return Err("action encoder produced an out-of-range feature".to_string());
    }
    Ok(indices)
}

pub fn card_type_index(card: &Card) -> usize {
    match &card.card_type {
        CardType::Location(town) => town.as_usize(),
        CardType::Industry(industry_set) => {
            let industries = industry_set.to_industry_types();
            if industries.len() == 2
                && industries.contains(&IndustryType::Cotton)
                && industries.contains(&IndustryType::Goods)
            {
                26
            } else if industries.len() == 1 {
                20 + industries[0].as_usize()
            } else {
                26
            }
        }
        CardType::WildLocation => 27,
        CardType::WildIndustry => 28,
    }
}

fn encode_shortfall(
    features: &mut Vec<f32>,
    shortfall: Option<&ShortfallResolutionSession>,
    observer_idx: usize,
    num_players: usize,
) -> Result<(), String> {
    features.push(f32::from(shortfall.is_some()));
    let relative_player = shortfall
        .map(|session| relative_player(observer_idx, session.player_idx, num_players))
        .transpose()?;
    push_one_hot(features, N_PLAYERS, relative_player);
    features.push(
        shortfall
            .map(|session| session.shortfall as f32 / 20.0)
            .unwrap_or(0.0),
    );

    let mut tile_mask = [0.0f32; N_BL];
    let mut liquidation_values = [0.0f32; N_BL];
    if let Some(session) = shortfall {
        for tile in &session.removable_tiles {
            if tile.build_location_idx < N_BL {
                tile_mask[tile.build_location_idx] = 1.0;
                liquidation_values[tile.build_location_idx] = tile.liquidation_value as f32 / 30.0;
            }
        }
    }
    features.extend(tile_mask);
    features.extend(liquidation_values);
    Ok(())
}

fn validate_observer(observer_idx: usize, num_players: usize) -> Result<(), String> {
    if observer_idx >= num_players {
        return Err(format!(
            "observer index {observer_idx} is out of range for {num_players} players"
        ));
    }
    if !(2..=N_PLAYERS).contains(&num_players) {
        return Err(format!("unsupported player count {num_players}"));
    }
    Ok(())
}

fn relative_player(
    observer_idx: usize,
    player_idx: usize,
    num_players: usize,
) -> Result<usize, String> {
    if player_idx >= num_players {
        return Err(format!(
            "player index {player_idx} is out of range for {num_players} players"
        ));
    }
    Ok((player_idx + num_players - observer_idx) % num_players)
}

fn absolute_player(observer_idx: usize, relative_idx: usize, num_players: usize) -> Option<usize> {
    (relative_idx < num_players).then_some((observer_idx + relative_idx) % num_players)
}

fn push_one_hot(features: &mut Vec<f32>, width: usize, index: Option<usize>) {
    for feature_idx in 0..width {
        features.push(f32::from(index == Some(feature_idx)));
    }
}

#[allow(clippy::too_many_arguments)]
fn push_positioned_feature(
    features: &mut Vec<u16>,
    offset: usize,
    position: usize,
    max_positions: usize,
    width: usize,
    value: usize,
    label: &str,
) -> Result<(), String> {
    if position >= max_positions {
        return Err(format!(
            "{label} uses position {position}, but the schema supports {max_positions}"
        ));
    }
    if value >= width {
        return Err(format!(
            "{label} value {value} is outside feature width {width}"
        ));
    }
    features.push((offset + position * width + value) as u16);
    Ok(())
}

fn push_single_feature(
    features: &mut Vec<u16>,
    offset: usize,
    width: usize,
    value: usize,
    label: &str,
) -> Result<(), String> {
    if value >= width {
        return Err(format!(
            "{label} value {value} is outside feature width {width}"
        ));
    }
    features.push((offset + value) as u16);
    Ok(())
}

fn action_type_index(action_type: ActionType) -> usize {
    match action_type {
        ActionType::BuildBuilding => 0,
        ActionType::BuildRailroad => 1,
        ActionType::BuildDoubleRailroad => 2,
        ActionType::Develop => 3,
        ActionType::DevelopDouble => 4,
        ActionType::Sell => 5,
        ActionType::Loan => 6,
        ActionType::Scout => 7,
        ActionType::Pass => 8,
    }
}

fn merchant_type_index(tile: &MerchantTileType) -> usize {
    match tile {
        MerchantTileType::All => 1,
        MerchantTileType::Cotton => 2,
        MerchantTileType::Goods => 3,
        MerchantTileType::Pottery => 4,
        MerchantTileType::Blank => 5,
    }
}

fn resource_source_index(source: ResourceSource) -> Result<usize, String> {
    match source {
        ResourceSource::Building(location) if location < N_BL => Ok(location),
        ResourceSource::Building(location) => Err(format!(
            "resource source location {location} is out of range"
        )),
        ResourceSource::Market => Ok(N_BL),
    }
}

fn beer_sell_source_index(source: BeerSellSource) -> Result<usize, String> {
    match source {
        BeerSellSource::Building(location) if location < N_BL => Ok(location),
        BeerSellSource::Building(location) => {
            Err(format!("beer source location {location} is out of range"))
        }
        BeerSellSource::TradePost(slot) if slot < NUM_TRADE_POSTS * 2 => Ok(N_BL + slot),
        BeerSellSource::TradePost(slot) => {
            Err(format!("merchant beer slot {slot} is out of range"))
        }
    }
}

fn action_beer_source_index(source: BreweryBeerSource) -> Result<usize, String> {
    match source {
        BreweryBeerSource::OwnBrewery(location) if location < N_BL => Ok(location),
        BreweryBeerSource::OwnBrewery(location) => {
            Err(format!("own brewery location {location} is out of range"))
        }
        BreweryBeerSource::OpponentBrewery(location) if location < N_BL => Ok(N_BL + location),
        BreweryBeerSource::OpponentBrewery(location) => Err(format!(
            "opponent brewery location {location} is out of range"
        )),
    }
}
