use serde::Serialize;

use crate::board::state::BoardState;
use crate::core::building::BuiltBuilding;
use crate::core::static_data::{
    INDUSTRY_MAT, MAX_LEVELS_PER_INDUSTRY, NUM_INDUSTRIES, TOWNS_RANGES,
};
use crate::core::types::ActionType;
use crate::core::types::*;
use crate::game::framework::{ActionIntent, ChoiceSet};
use crate::game::runner::{GamePhase, GameRunner};

#[derive(Serialize)]
pub struct PendingDevelopmentJson {
    pub industry: &'static str,
    pub level: usize,
}

#[derive(Serialize)]
pub struct FullGameState {
    pub seed: u64,
    pub current_player: usize,
    pub phase: &'static str,
    pub era: &'static str,
    pub turn_count: u32,
    pub round_in_phase: u32,
    pub actions_remaining: u8,
    pub players: Vec<PlayerJson>,
    pub buildings: Vec<BuildingJson>,
    pub roads: Vec<RoadJson>,
    pub coal_market: u8,
    pub iron_market: u8,
    pub trade_posts: Vec<TradePostSlotJson>,
    pub available_actions: Option<Vec<&'static str>>,
    pub choice_set: Option<ChoiceSetJson>,
    pub pending_developments: Vec<PendingDevelopmentJson>,
    pub turn_action_history: Vec<TurnActionJson>,
    pub current_action_selections: Option<TurnActionJson>,
    pub discard_history: Vec<DiscardEntryJson>,
    pub has_pending_shortfall: bool,
    pub game_over: bool,
    pub turn_order: Vec<usize>,
}

#[derive(Serialize)]
pub struct TurnActionJson {
    pub action_type: &'static str,
    pub selections: Vec<String>,
}

#[derive(Serialize)]
pub struct DiscardEntryJson {
    pub order: usize,
    pub player_index: usize,
    pub player_name: &'static str,
    pub round_in_phase: u32,
    pub turn_count: u32,
    pub card_type: String,
    pub card_label: String,
}

#[derive(Serialize)]
pub struct PlayerJson {
    pub index: usize,
    pub name: &'static str,
    pub color: &'static str,
    pub money: u16,
    pub income_level: u8,
    pub income_amount: i8,
    pub victory_points: u16,
    pub hand: Vec<CardJson>,
    pub hand_size: usize,
    pub industry_mat: Vec<IndustryTileJson>,
}

#[derive(Serialize)]
pub struct IndustryTileJson {
    pub industry: &'static str,
    pub level: usize,
    pub tiles_remaining: u8,
    pub money_cost: u16,
    pub coal_cost: u8,
    pub iron_cost: u8,
    pub beer_needed: u8,
    pub vp_on_flip: u8,
    pub road_vp: u8,
    pub resource_amt: u8,
    pub income: i8,
    pub removed_after_phase1: bool,
    pub can_develop: bool,
    pub exhausted: bool,
}

#[derive(Serialize)]
pub struct CardJson {
    pub index: usize,
    pub card_type: String,
    pub label: String,
}

#[derive(Serialize)]
pub struct BuildingJson {
    pub location: usize,
    pub town: String,
    pub industry: &'static str,
    pub level: usize,
    pub owner: usize,
    pub flipped: bool,
    pub resource_amt: u8,
    pub road_vp: u8,
    pub vp_on_flip: u8,
}

#[derive(Serialize)]
pub struct RoadJson {
    pub index: usize,
    pub owner: usize,
}

#[derive(Serialize)]
pub struct TradePostSlotJson {
    pub slot_index: usize,
    pub trade_post: &'static str,
    pub tile_type: Option<String>,
    pub has_beer: bool,
}

#[derive(Serialize)]
pub struct ChoiceSetJson {
    pub kind: &'static str,
    pub options: Vec<ChoiceOptionJson>,
}

#[derive(Serialize)]
pub struct ChoiceOptionJson {
    pub value: serde_json::Value,
    pub label: String,
}

pub fn observer_player_index(runner: &GameRunner, observer_player: Option<usize>) -> usize {
    let current_player = runner.framework.current_player;
    observer_player
        .filter(|&player_idx| player_idx < runner.framework.board.state.players.len())
        .unwrap_or(current_player)
}

pub fn serialize_game_state(runner: &GameRunner, observer_player: Option<usize>) -> FullGameState {
    let state = &runner.framework.board.state;
    let num_players = state.players.len();
    let observer_player = observer_player_index(runner, observer_player);
    let observer_controls_current = observer_player == runner.framework.current_player;

    let players: Vec<PlayerJson> = state
        .players
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let hand: Vec<CardJson> = if i == observer_player {
                p.hand
                    .cards
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| CardJson {
                        index: ci,
                        card_type: format_card_type(&c.card_type),
                        label: format_card_label(&c.card_type),
                    })
                    .collect()
            } else {
                vec![]
            };
            let hand_size = p.hand.cards.len();
            let industry_mat = serialize_industry_mat(&p.industry_mat);
            PlayerJson {
                index: i,
                name: player_name(i),
                color: player_color(i),
                money: p.money,
                income_level: p.income_level,
                income_amount: p.get_income_amount(p.income_level),
                victory_points: p.victory_points,
                hand,
                hand_size,
                industry_mat,
            }
        })
        .collect();

    let buildings: Vec<BuildingJson> = state
        .bl_to_building
        .iter()
        .map(|(&loc, b)| serialize_building(loc, b))
        .collect();

    let mut roads = Vec::new();
    for road_idx in state.built_roads.ones() {
        for p_idx in 0..num_players {
            if state.player_road_mask[p_idx].contains(road_idx) {
                roads.push(RoadJson {
                    index: road_idx,
                    owner: p_idx,
                });
                break;
            }
        }
    }

    let trade_posts = serialize_trade_posts(state);

    let current_hand = &state.players[runner.framework.current_player].hand.cards;
    let choice_set = observer_controls_current
        .then(|| runner.framework.get_next_choice_set())
        .flatten()
        .map(|cs| serialize_choice_set(&cs, current_hand));

    let pending_developments = if observer_controls_current {
        serialize_pending_developments(runner)
    } else {
        Vec::new()
    };
    let turn_action_history = runner
        .turn_action_history()
        .iter()
        .map(|a| serialize_turn_action(a, state, runner.framework.current_player, None))
        .collect();
    let current_action_selections = observer_controls_current
        .then(|| runner.framework.current_session())
        .flatten()
        .map(|s| {
            serialize_turn_action(
                &s.intent,
                state,
                runner.framework.current_player,
                Some(current_hand),
            )
        });
    let discard_history = runner
        .discard_history()
        .iter()
        .map(|d| DiscardEntryJson {
            order: d.order,
            player_index: d.player_idx,
            player_name: player_name(d.player_idx),
            round_in_phase: d.round_in_phase,
            turn_count: d.turn_count,
            card_type: format_card_type(&d.card.card_type),
            card_label: format_card_label(&d.card.card_type),
        })
        .collect();

    FullGameState {
        seed: state.seed,
        current_player: runner.framework.current_player,
        phase: phase_str(runner.game_phase),
        era: era_str(state.era),
        turn_count: runner.turn_count,
        round_in_phase: runner.round_in_phase,
        actions_remaining: runner.actions_remaining_in_turn,
        players,
        buildings,
        roads,
        coal_market: state.remaining_market_coal,
        iron_market: state.remaining_market_iron,
        trade_posts,
        available_actions: None,
        choice_set,
        pending_developments,
        turn_action_history,
        current_action_selections,
        discard_history,
        has_pending_shortfall: runner.has_pending_shortfall(),
        game_over: runner.is_game_finished(),
        turn_order: state.turn_order.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_observer_sees_only_their_own_hand() {
        let runner = GameRunner::new(2, Some(20_260_816));
        let current_player = runner.framework.current_player;
        let observer_player = 1 - current_player;

        let state = serialize_game_state(&runner, Some(observer_player));

        assert_eq!(
            state.players[observer_player].hand.len(),
            state.players[observer_player].hand_size
        );
        assert!(!state.players[observer_player].hand.is_empty());
        assert!(state.players[current_player].hand.is_empty());
    }

    #[test]
    fn observer_cannot_see_an_opponents_active_action_session() {
        let mut runner = GameRunner::new(2, Some(20_260_817));
        let current_player = runner.framework.current_player;
        let observer_player = 1 - current_player;
        let _ = runner.start_turn();
        let _ = runner.start_action(ActionType::Pass);

        let acting_view = serialize_game_state(&runner, Some(current_player));
        let observer_view = serialize_game_state(&runner, Some(observer_player));

        assert!(acting_view.choice_set.is_some());
        assert!(acting_view.current_action_selections.is_some());
        assert!(observer_view.choice_set.is_none());
        assert!(observer_view.current_action_selections.is_none());
    }
}

fn serialize_turn_action(
    intent: &ActionIntent,
    state: &BoardState,
    _player_idx: usize,
    hand: Option<&[Card]>,
) -> TurnActionJson {
    let mut selections: Vec<String> = Vec::new();
    if let Some(ind) = intent.selected_industry {
        selections.push(format!("Industry {}", industry_str(ind)));
    }
    if let Some(ind) = intent.selected_second_industry {
        selections.push(format!("Second {}", industry_str(ind)));
    }
    if let Some(card_idx) = intent.selected_card_idx {
        let card_label = hand
            .and_then(|h| h.get(card_idx))
            .map(|c| format_card_label(&c.card_type))
            .unwrap_or_else(|| format!("Card slot #{}", card_idx));
        selections.push(format!("🃏 {}", card_label));
    }
    if let Some(loc) = intent.selected_build_location {
        selections.push(format!("Build {}", town_name_for_bl(loc)));
    }
    if let Some(road_idx) = intent.selected_road_idx {
        selections.push(format!("Road {}", road_idx));
    }
    if let Some(road_idx) = intent.selected_second_road_idx {
        selections.push(format!("Road2 {}", road_idx));
    }
    if let Some(mode) = intent.selected_network_mode {
        let mode_s = match mode {
            crate::game::framework::NetworkMode::Single => "Single",
            crate::game::framework::NetworkMode::Double => "Double",
        };
        selections.push(format!("Mode {}", mode_s));
    }
    if !intent.chosen_coal_sources.is_empty() {
        selections.push(format!("Coal x{}", intent.chosen_coal_sources.len()));
    }
    if !intent.chosen_iron_sources.is_empty() {
        for src in &intent.chosen_iron_sources {
            match src {
                crate::board::resources::ResourceSource::Market => {
                    selections.push("⛓️ Iron from Market".to_string());
                }
                crate::board::resources::ResourceSource::Building(loc) => {
                    let town = town_name_for_bl(*loc);
                    let source_name = state
                        .bl_to_building
                        .get(loc)
                        .map(|b| format!("{} Works", industry_str(b.industry)))
                        .unwrap_or_else(|| "Building".to_string());
                    selections.push(format!("⛓️ Iron from {} in {}", source_name, town));
                }
            }
        }
    }
    if !intent.chosen_beer_sources.is_empty() {
        selections.push(format!("Beer x{}", intent.chosen_beer_sources.len()));
    }
    if !intent.sell_choices.is_empty() {
        selections.push(format!("Sell targets {}", intent.sell_choices.len()));
    }
    TurnActionJson {
        action_type: action_type_str(intent.action_type),
        selections,
    }
}

fn serialize_industry_mat(
    mat: &crate::core::industry_mat::PlayerIndustryMat,
) -> Vec<IndustryTileJson> {
    use crate::core::types::IndustryType;
    const INDUSTRIES: [IndustryType; 6] = [
        IndustryType::Coal,
        IndustryType::Iron,
        IndustryType::Beer,
        IndustryType::Goods,
        IndustryType::Pottery,
        IndustryType::Cotton,
    ];
    INDUSTRIES
        .iter()
        .map(|&ind| {
            let exhausted = !mat.has_tiles_left(ind);
            let level = mat.get_lowest_level(ind);
            let remaining = mat.get_remaining_tiles_at_level(ind);
            let data = &INDUSTRY_MAT[ind as usize][level.as_usize()];
            IndustryTileJson {
                industry: industry_str(ind),
                level: level.as_usize() + 1,
                tiles_remaining: remaining,
                money_cost: data.money_cost,
                coal_cost: data.coal_cost,
                iron_cost: data.iron_cost,
                beer_needed: data.beer_needed,
                vp_on_flip: data.vp_on_flip,
                road_vp: data.road_vp,
                resource_amt: data.resource_amt,
                income: data.income,
                removed_after_phase1: data.removed_after_phase1,
                can_develop: data.can_develop,
                exhausted,
            }
        })
        .collect()
}

fn serialize_building(loc: usize, b: &BuiltBuilding) -> BuildingJson {
    let data = &INDUSTRY_MAT[b.industry as usize][b.level.as_usize()];
    BuildingJson {
        location: loc,
        town: town_name_for_bl(loc),
        industry: industry_str(b.industry),
        level: b.level.as_usize() + 1,
        owner: b.owner.as_usize(),
        flipped: b.flipped,
        resource_amt: b.resource_amt,
        road_vp: data.road_vp,
        vp_on_flip: data.vp_on_flip,
    }
}

fn serialize_pending_developments(runner: &GameRunner) -> Vec<PendingDevelopmentJson> {
    let mut out = Vec::new();
    let Some(session) = runner.framework.current_session() else {
        return out;
    };
    if session.action_type != ActionType::Develop
        && session.action_type != ActionType::DevelopDouble
    {
        return out;
    }
    let player_idx = session.player_idx;
    let mat = &runner.framework.board.state.players[player_idx].industry_mat;

    for industry in [
        session.intent.selected_industry,
        session.intent.selected_second_industry,
    ] {
        if let Some(ind) = industry {
            let current_level = mat.get_lowest_level(ind);
            let develop_to_level = current_level.as_usize() + 2; // 1-based display: we develop to next level
            out.push(PendingDevelopmentJson {
                industry: industry_str(ind),
                level: develop_to_level,
            });
        }
    }
    out
}

fn serialize_trade_posts(state: &BoardState) -> Vec<TradePostSlotJson> {
    let mut result = Vec::new();
    for (i, slot) in state.trade_post_slots.iter().enumerate() {
        let tp_name = trade_post_for_slot(i);
        result.push(TradePostSlotJson {
            slot_index: i,
            trade_post: tp_name,
            tile_type: slot.as_ref().map(|m| format!("{:?}", m.tile_type)),
            // Source of truth for UI: whether this slot currently has a merchant beer barrel.
            has_beer: state.trade_post_beer.contains(i),
        });
    }
    result
}

fn serialize_choice_set(cs: &ChoiceSet, hand: &[Card]) -> ChoiceSetJson {
    match cs {
        ChoiceSet::Industry(opts) => ChoiceSetJson {
            kind: "industry",
            options: opts
                .iter()
                .map(|i| ChoiceOptionJson {
                    value: serde_json::json!(industry_str(*i)),
                    label: industry_str(*i).to_string(),
                })
                .collect(),
        },
        ChoiceSet::Card(opts) => ChoiceSetJson {
            kind: "card",
            options: opts
                .iter()
                .map(|&i| {
                    let label = if i < hand.len() {
                        format!("#{}: {}", i, format_card_label(&hand[i].card_type))
                    } else {
                        format!("Card #{}", i)
                    };
                    ChoiceOptionJson {
                        value: serde_json::json!(i),
                        label,
                    }
                })
                .collect(),
        },
        ChoiceSet::BuildLocation(opts) => ChoiceSetJson {
            kind: "build_location",
            options: opts
                .iter()
                .map(|&loc| ChoiceOptionJson {
                    value: serde_json::json!(loc),
                    label: format!("{} (slot {})", town_name_for_bl(loc), loc),
                })
                .collect(),
        },
        ChoiceSet::Road(opts) => ChoiceSetJson {
            kind: "road",
            options: opts
                .iter()
                .map(|&r| ChoiceOptionJson {
                    value: serde_json::json!(r),
                    label: crate::core::static_data::road_label(r),
                })
                .collect(),
        },
        ChoiceSet::SecondRoad(opts) => ChoiceSetJson {
            kind: "second_road",
            options: opts
                .iter()
                .map(|&r| ChoiceOptionJson {
                    value: serde_json::json!(r),
                    label: crate::core::static_data::road_label(r),
                })
                .collect(),
        },
        ChoiceSet::CoalSource(opts) => ChoiceSetJson {
            kind: "coal_source",
            options: opts
                .iter()
                .map(|s| {
                    let (val, lbl) = match s {
                        crate::board::resources::ResourceSource::Building(loc) => (
                            serde_json::json!({"Building": loc}),
                            format!("Coal at loc {}", loc),
                        ),
                        crate::board::resources::ResourceSource::Market => {
                            (serde_json::json!("Market"), "Market".to_string())
                        }
                    };
                    ChoiceOptionJson {
                        value: val,
                        label: lbl,
                    }
                })
                .collect(),
        },
        ChoiceSet::IronSource(opts) => ChoiceSetJson {
            kind: "iron_source",
            options: opts
                .iter()
                .map(|s| {
                    let (val, lbl) = match s {
                        crate::board::resources::ResourceSource::Building(loc) => (
                            serde_json::json!({"Building": loc}),
                            format!("Iron at loc {}", loc),
                        ),
                        crate::board::resources::ResourceSource::Market => {
                            (serde_json::json!("Market"), "Market".to_string())
                        }
                    };
                    ChoiceOptionJson {
                        value: val,
                        label: lbl,
                    }
                })
                .collect(),
        },
        ChoiceSet::BeerSource(opts) => ChoiceSetJson {
            kind: "beer_source",
            options: opts
                .iter()
                .map(|s| {
                    let (val, lbl) = match s {
                        crate::board::resources::BeerSellSource::Building(loc) => (
                            serde_json::json!({"Building": loc}),
                            format!("Brewery at {}", loc),
                        ),
                        crate::board::resources::BeerSellSource::TradePost(slot) => (
                            serde_json::json!({"TradePost": slot}),
                            format!("Merchant slot {}", slot),
                        ),
                    };
                    ChoiceOptionJson {
                        value: val,
                        label: lbl,
                    }
                })
                .collect(),
        },
        ChoiceSet::ActionBeerSource(opts) => ChoiceSetJson {
            kind: "action_beer_source",
            options: opts
                .iter()
                .map(|s| {
                    let (val, lbl) = match s {
                        crate::board::resources::BreweryBeerSource::OwnBrewery(loc) => (
                            serde_json::json!({"OwnBrewery": loc}),
                            format!("Own brewery at {}", loc),
                        ),
                        crate::board::resources::BreweryBeerSource::OpponentBrewery(loc) => (
                            serde_json::json!({"OpponentBrewery": loc}),
                            format!("Opponent brewery at {}", loc),
                        ),
                    };
                    ChoiceOptionJson {
                        value: val,
                        label: lbl,
                    }
                })
                .collect(),
        },
        ChoiceSet::SellTarget(opts) => ChoiceSetJson {
            kind: "sell_target",
            options: opts
                .iter()
                .map(|&loc| ChoiceOptionJson {
                    value: serde_json::json!(loc),
                    label: format!("{} (slot {})", town_name_for_bl(loc), loc),
                })
                .collect(),
        },
        ChoiceSet::FreeDevelopment(opts) => ChoiceSetJson {
            kind: "free_development",
            options: opts
                .iter()
                .map(|i| ChoiceOptionJson {
                    value: serde_json::json!(industry_str(*i)),
                    label: industry_str(*i).to_string(),
                })
                .collect(),
        },
        ChoiceSet::SecondIndustry(opts) => ChoiceSetJson {
            kind: "second_industry",
            options: opts
                .iter()
                .map(|i| ChoiceOptionJson {
                    value: serde_json::json!(industry_str(*i)),
                    label: industry_str(*i).to_string(),
                })
                .collect(),
        },
        ChoiceSet::NetworkMode(opts) => ChoiceSetJson {
            kind: "network_mode",
            options: opts
                .iter()
                .map(|m| {
                    let s = match m {
                        crate::game::framework::NetworkMode::Single => "Single",
                        crate::game::framework::NetworkMode::Double => "Double",
                    };
                    ChoiceOptionJson {
                        value: serde_json::json!(s),
                        label: s.to_string(),
                    }
                })
                .collect(),
        },
        ChoiceSet::ConfirmOnly => ChoiceSetJson {
            kind: "confirm",
            options: vec![ChoiceOptionJson {
                value: serde_json::json!("confirm"),
                label: "Confirm".to_string(),
            }],
        },
    }
}

fn phase_str(p: GamePhase) -> &'static str {
    match p {
        GamePhase::Canal => "Canal",
        GamePhase::Railroad => "Railroad",
        GamePhase::GameEnd => "GameEnd",
    }
}

fn era_str(e: Era) -> &'static str {
    match e {
        Era::Canal => "Canal",
        Era::Railroad => "Railroad",
    }
}

pub fn industry_str(i: IndustryType) -> &'static str {
    match i {
        IndustryType::Coal => "Coal",
        IndustryType::Iron => "Iron",
        IndustryType::Beer => "Beer",
        IndustryType::Goods => "Goods",
        IndustryType::Pottery => "Pottery",
        IndustryType::Cotton => "Cotton",
    }
}

pub fn action_type_str(a: ActionType) -> &'static str {
    match a {
        ActionType::BuildBuilding => "BuildBuilding",
        ActionType::BuildRailroad => "BuildRailroad",
        ActionType::BuildDoubleRailroad => "BuildDoubleRailroad",
        ActionType::Develop => "Develop",
        ActionType::DevelopDouble => "DevelopDouble",
        ActionType::Sell => "Sell",
        ActionType::Loan => "Loan",
        ActionType::Scout => "Scout",
        ActionType::Pass => "Pass",
    }
}

pub fn action_type_from_str(s: &str) -> Option<ActionType> {
    match s {
        "BuildBuilding" => Some(ActionType::BuildBuilding),
        "BuildRailroad" => Some(ActionType::BuildRailroad),
        "BuildDoubleRailroad" => Some(ActionType::BuildDoubleRailroad),
        "Develop" => Some(ActionType::Develop),
        "DevelopDouble" => Some(ActionType::DevelopDouble),
        "Sell" => Some(ActionType::Sell),
        "Loan" => Some(ActionType::Loan),
        "Scout" => Some(ActionType::Scout),
        "Pass" => Some(ActionType::Pass),
        _ => None,
    }
}

fn player_name(idx: usize) -> &'static str {
    match idx {
        0 => "Coade",
        1 => "Brunel",
        2 => "Arkwright",
        3 => "Tinsley",
        _ => "Unknown",
    }
}

fn player_color(idx: usize) -> &'static str {
    match idx {
        0 => "#c7a750",
        1 => "#9c79c6",
        2 => "#a44529",
        3 => "#b6c3ca",
        _ => "#888888",
    }
}

pub(crate) fn town_name_for_bl(bl_idx: usize) -> String {
    static TOWN_NAMES: [&str; 20] = [
        "Stafford",
        "Burton-Upon-Trent",
        "Cannock",
        "Tamworth",
        "Walsall",
        "Leek",
        "Stoke-On-Trent",
        "Stone",
        "Uttoxeter",
        "Belper",
        "Derby",
        "Coalbrookdale",
        "Wolverhampton",
        "Dudley",
        "Kidderminster",
        "Worcester",
        "Birmingham",
        "Nuneaton",
        "Coventry",
        "Redditch",
    ];
    if bl_idx == 47 {
        return "beer1".to_string();
    }
    if bl_idx == 48 {
        return "beer2".to_string();
    }
    for (i, &(start, end)) in TOWNS_RANGES.iter().enumerate() {
        if bl_idx >= start && bl_idx < end {
            return TOWN_NAMES[i].to_string();
        }
    }
    format!("BL_{}", bl_idx)
}

fn trade_post_for_slot(slot_idx: usize) -> &'static str {
    match slot_idx {
        0 => "Shrewbury",
        1 | 2 => "Oxford",
        3 | 4 => "Gloucester",
        5 | 6 => "Warrington",
        7 | 8 => "Nottingham",
        _ => "Unknown",
    }
}

fn format_card_type(ct: &CardType) -> String {
    match ct {
        CardType::Location(town) => format!("Location({:?})", town),
        CardType::Industry(ind_set) => {
            let industries: Vec<&str> = ind_set
                .ones()
                .map(|i| industry_str(IndustryType::from_usize(i)))
                .collect();
            format!("Industry({})", industries.join(","))
        }
        CardType::WildLocation => "WildLocation".to_string(),
        CardType::WildIndustry => "WildIndustry".to_string(),
    }
}

pub(crate) fn format_card_label(ct: &CardType) -> String {
    match ct {
        CardType::Location(town) => format!("{:?}", town),
        CardType::Industry(ind_set) => {
            let industries: Vec<&str> = ind_set
                .ones()
                .map(|i| industry_str(IndustryType::from_usize(i)))
                .collect();
            industries.join("/")
        }
        CardType::WildLocation => "Wild Location".to_string(),
        CardType::WildIndustry => "Wild Industry".to_string(),
    }
}

#[derive(Serialize)]
pub struct IndustryLevelDataJson {
    pub industry: &'static str,
    pub level: usize,
    pub money_cost: u16,
    pub coal_cost: u8,
    pub iron_cost: u8,
    pub beer_needed: u8,
    pub vp_on_flip: u8,
    pub road_vp: u8,
    pub resource_amt: u8,
    pub income: i8,
    pub removed_after_phase1: bool,
    pub can_develop: bool,
    pub num_tiles: u8,
}

pub fn serialize_all_industry_data() -> serde_json::Value {
    use crate::core::types::IndustryType;
    const INDUSTRIES: [IndustryType; 6] = [
        IndustryType::Coal,
        IndustryType::Iron,
        IndustryType::Beer,
        IndustryType::Goods,
        IndustryType::Pottery,
        IndustryType::Cotton,
    ];
    let mut result = serde_json::Map::new();
    for &ind in &INDUSTRIES {
        let name = industry_str(ind);
        let max_level = MAX_LEVELS_PER_INDUSTRY[ind as usize];
        let mut levels = Vec::new();
        for lvl_idx in 0..=max_level.as_usize() {
            let data = &INDUSTRY_MAT[ind as usize][lvl_idx];
            levels.push(IndustryLevelDataJson {
                industry: name,
                level: lvl_idx + 1,
                money_cost: data.money_cost,
                coal_cost: data.coal_cost,
                iron_cost: data.iron_cost,
                beer_needed: data.beer_needed,
                vp_on_flip: data.vp_on_flip,
                road_vp: data.road_vp,
                resource_amt: data.resource_amt,
                income: data.income,
                removed_after_phase1: data.removed_after_phase1,
                can_develop: data.can_develop,
                num_tiles: data.num_tiles,
            });
        }
        result.insert(name.to_string(), serde_json::to_value(levels).unwrap());
    }
    serde_json::json!({"ok": true, "data": result})
}
