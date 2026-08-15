use fast_brass::actions::{SellChoice, SellOption, SingleRailroadOption};
use fast_brass::board::resources::BeerSellSource;
use fast_brass::board::Board;
use fast_brass::core::building::BuiltBuilding;
use fast_brass::core::player::PlayerId;
use fast_brass::core::types::{ActionType, Era, IndustryLevel, IndustryType, NextActionChoiceKind};
use fast_brass::game::framework::{ActionChoice, ChoiceSet, GameFramework};
use fast_brass::game::runner::GameRunner;
use fixedbitset::FixedBitSet;
use std::collections::VecDeque;

#[test]
fn test_network_mode_choice_is_exposed_in_railroad() {
    let board = Board::new(2, Some(7));
    let current_player = board.state.turn_order[0];
    let mut framework = GameFramework::new(board, current_player);
    framework.board.state.era = Era::Railroad;
    framework.start_action_session(ActionType::BuildRailroad);

    let mut coal_sources = FixedBitSet::with_capacity(1);
    coal_sources.insert(0);
    if let Some(ctx) = framework.action_context.as_mut() {
        ctx.choices_needed = VecDeque::from([NextActionChoiceKind::ChooseRoad]);
        ctx.initial_single_rail_options =
            Some(vec![SingleRailroadOption::new(1, coal_sources.clone())]);
        ctx.initial_double_rail_first_link_options =
            Some(vec![SingleRailroadOption::new(2, coal_sources)]);
        ctx.selected_network_mode = None;
        ctx.action_type = ActionType::BuildRailroad;
    }

    let next = framework.get_next_choice_set();
    assert!(matches!(next, Some(ChoiceSet::NetworkMode(_))));
}

#[test]
fn test_remove_iteration_clears_sell_iteration_state() {
    let board = Board::new(2, Some(8));
    let current_player = board.state.turn_order[0];
    let mut framework = GameFramework::new(board, current_player);
    framework.start_action_session(ActionType::Sell);

    if let Some(ctx) = framework.action_context.as_mut() {
        let mut beer_locs = FixedBitSet::with_capacity(0);
        beer_locs.clear();
        let mut tp = FixedBitSet::with_capacity(0);
        tp.clear();
        ctx.current_sell_choices = vec![SellChoice::new(27, vec![BeerSellSource::Building(29)])];
        ctx.current_filtered_sell_options = vec![SellOption {
            location: 27,
            trade_posts: tp,
            beer_locations: beer_locs,
            beer_counter: 1,
        }];
    }

    framework.remove_iteration(0).unwrap();
    let session = framework.current_session().unwrap();
    assert!(session.intent.sell_choices.is_empty());
}

#[test]
fn test_stale_session_revalidation_rejects_confirm() {
    let board = Board::new(2, Some(9));
    let current_player = board.state.turn_order[0];
    let mut framework = GameFramework::new(board, current_player);
    framework.start_action_session(ActionType::Loan);
    framework
        .apply_action_choice(ActionChoice::Card(0))
        .unwrap();

    // Loan becomes illegal at -10 income exactly.
    framework.board.state.players[current_player].income_level = 0;
    let result = framework.confirm_action_session();
    assert!(result.is_err());
}

#[test]
fn test_runner_exposes_shortfall_interrupt_and_applies_selected_tiles() {
    let mut runner = GameRunner::new(2, Some(11));
    let player_idx = runner.framework.current_player;
    let loc1 = 27usize;
    let loc2 = 28usize;

    let building1 = BuiltBuilding::build(
        IndustryType::Coal,
        IndustryLevel::I,
        loc1 as u8,
        PlayerId::from_usize(player_idx),
    );
    let building2 = BuiltBuilding::build(
        IndustryType::Iron,
        IndustryLevel::I,
        loc2 as u8,
        PlayerId::from_usize(player_idx),
    );
    runner
        .framework
        .board
        .state
        .bl_to_building
        .insert(loc1, building1);
    runner
        .framework
        .board
        .state
        .bl_to_building
        .insert(loc2, building2);
    runner
        .framework
        .board
        .state
        .build_locations_occupied
        .insert(loc1);
    runner
        .framework
        .board
        .state
        .build_locations_occupied
        .insert(loc2);
    runner.framework.board.state.player_building_mask[player_idx].insert(loc1);
    runner.framework.board.state.player_building_mask[player_idx].insert(loc2);

    runner.framework.board.state.players[player_idx].income_level = 0;
    runner.framework.board.state.players[player_idx].money = 0;

    runner.end_round();
    let sessions = runner.take_shortfall_sessions();
    assert!(!sessions.is_empty());

    let session = sessions.into_iter().next().unwrap();
    runner.resolve_shortfall_with_tiles(session, vec![loc2]);
    assert!(!runner
        .framework
        .board
        .state
        .bl_to_building
        .contains_key(&loc2));
}
