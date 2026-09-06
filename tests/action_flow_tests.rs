use fast_brass::board::resources::ResourceManager;
use fast_brass::board::Board;
use fast_brass::core::building::BuiltBuilding;
use fast_brass::core::player::PlayerId;
use fast_brass::core::types::{
    ActionType, Card, CardType, IndustryLevel, IndustrySet, IndustryType,
};
use fast_brass::game::framework::{ActionChoice, ChoiceSet, GameFramework};
use fast_brass::game::runner::GameRunner;
use fast_brass::locations::TownName;
use fast_brass::validation::{build_validation::BuildValidator, NetworkValidator};

fn do_pass_action(runner: &mut GameRunner) {
    let cs = runner.start_action(ActionType::Pass);
    assert!(
        matches!(cs, ChoiceSet::Card(_)),
        "Pass should require card choice, got {:?}",
        cs
    );
    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(cs, Some(ChoiceSet::ConfirmOnly)),
        "Pass should move to confirm after card, got {:?}",
        cs
    );
    runner
        .confirm_action()
        .expect("pass confirm should succeed");
}

#[test]
fn test_first_turn_has_single_action_budget() {
    let mut runner = GameRunner::new(2, Some(1234));
    let valid_actions = runner.start_turn();
    assert!(!valid_actions.is_empty());
    assert_eq!(runner.actions_remaining_in_turn, 1);
}

#[test]
fn duplicate_end_turn_cannot_advance_the_next_player() {
    let mut runner = GameRunner::new(3, Some(12_345));
    assert!(runner.try_end_turn().is_err());

    let first_player = runner.framework.current_player;
    runner.start_turn();
    do_pass_action(&mut runner);
    assert_eq!(runner.actions_remaining_in_turn, 0);

    runner
        .try_end_turn()
        .expect("a completed turn should end successfully");
    let next_player = runner.framework.current_player;
    let next_turn_count = runner.turn_count;
    assert_ne!(next_player, first_player);
    assert!(!runner.turn_started);

    let duplicate = runner.try_end_turn();
    assert!(duplicate.is_err(), "duplicate end_turn must be rejected");
    assert_eq!(runner.framework.current_player, next_player);
    assert_eq!(runner.turn_count, next_turn_count);
}

#[test]
fn strict_web_choice_rejects_invalid_value_without_mutating_session() {
    let mut runner = GameRunner::new(2, Some(12_346));
    runner.start_turn();
    let choice_set = runner
        .try_start_action(ActionType::Pass)
        .expect("pass should be legal");
    assert!(matches!(choice_set, ChoiceSet::Card(_)));
    let before = runner.framework.current_session().expect("session");

    assert!(runner
        .try_apply_choice(ActionChoice::Card(usize::MAX))
        .is_err());
    let after = runner.framework.current_session().expect("session remains");
    assert_eq!(before.intent, after.intent);
    assert_eq!(before.next_choices, after.next_choices);
}

#[test]
fn test_second_personal_turn_has_two_actions() {
    let mut runner = GameRunner::new(2, Some(1234));

    runner.start_turn();
    do_pass_action(&mut runner);
    runner.finish_turn_and_advance();

    runner.start_turn();
    do_pass_action(&mut runner);
    runner.finish_turn_and_advance();

    runner.start_turn();
    assert_eq!(runner.actions_remaining_in_turn, 2);
}

#[test]
fn test_action_session_can_cancel() {
    let board = Board::new(2, Some(42));
    let current_player = board.state.turn_order[0];
    let mut framework = GameFramework::new(board, current_player);

    let _session = framework.start_action_session(ActionType::Pass);
    assert!(matches!(
        framework.get_next_choice_set(),
        Some(ChoiceSet::Card(_))
    ));

    let result = framework.apply_action_choice(ActionChoice::Cancel).unwrap();
    assert!(result.is_none());
    assert!(framework.current_session().is_none());
}

#[test]
fn test_pass_requires_card_and_discards_it() {
    let mut runner = GameRunner::new(2, Some(777));
    let _ = runner.start_turn();

    let hand_before = runner.framework.board.state.players[runner.framework.current_player]
        .hand
        .cards
        .len();
    let discard_before = runner.framework.board.state.discard_pile.len();

    let cs = runner.start_action(ActionType::Pass);
    assert!(
        matches!(cs, ChoiceSet::Card(_)),
        "Pass must prompt for card selection"
    );

    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(cs, Some(ChoiceSet::ConfirmOnly)),
        "Pass should confirm only after card selection"
    );

    runner
        .confirm_action()
        .expect("pass confirm should succeed after selecting a card");

    let hand_after = runner.framework.board.state.players[runner.framework.current_player]
        .hand
        .cards
        .len();
    let discard_after = runner.framework.board.state.discard_pile.len();

    assert_eq!(
        hand_after, hand_before,
        "Pass should refill the hand after discarding"
    );
    assert_eq!(
        discard_after,
        discard_before + 1,
        "Pass should add one card to discard pile"
    );
}

#[test]
fn test_can_undo_last_confirmed_action_within_turn() {
    let mut runner = GameRunner::new(2, Some(42));
    let _ = runner.start_turn();
    assert_eq!(runner.actions_remaining_in_turn, 1);

    let cs = runner.start_action(ActionType::Pass);
    assert!(matches!(cs, ChoiceSet::Card(_)));
    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(matches!(cs, Some(ChoiceSet::ConfirmOnly)));
    runner
        .confirm_action()
        .expect("pass confirm should succeed");
    assert_eq!(runner.actions_remaining_in_turn, 0);
    assert_eq!(runner.turn_action_history().len(), 1);

    runner
        .undo_last_confirmed_action()
        .expect("undo last action should succeed");
    assert_eq!(runner.actions_remaining_in_turn, 1);
    assert_eq!(runner.turn_action_history().len(), 0);
}

#[test]
fn test_shortfall_resolution_removes_chosen_tile() {
    let mut board = Board::new(2, Some(99));
    let player_idx = 0usize;
    let loc = 27usize;

    let building = BuiltBuilding::build(
        IndustryType::Coal,
        IndustryLevel::I,
        loc as u8,
        PlayerId::from_usize(player_idx),
    );

    board.state.bl_to_building.insert(loc, building);
    board.state.build_locations_occupied.insert(loc);
    board.state.player_building_mask[player_idx].insert(loc);
    board.state.coal_locations.insert(loc);

    let mut framework = GameFramework::new(board, player_idx);
    framework.board.state.players[player_idx].money = 0;

    let session = framework.start_shortfall_resolution_session(player_idx, 3);
    assert!(!session.removable_tiles.is_empty());

    framework.resolve_shortfall_with_tile_choices(session, vec![loc]);
    assert!(!framework.board.state.bl_to_building.contains_key(&loc));
}

/// Full 2-player round test:
///   Player A does DevelopDouble (spends £4 on iron from market at £2 each)
///   Player B does Loan (spends £0, gains £30)
///   After the round, turn order updates: B goes first (spent less), A second
///   Current player switches to the first in the new turn order
#[test]
fn test_turn_order_updates_after_round_based_on_spending() {
    let mut runner = GameRunner::new(2, Some(42));

    let first_player = runner.framework.board.state.turn_order[0];
    let second_player = runner.framework.board.state.turn_order[1];
    assert_ne!(first_player, second_player);

    // --- First player's turn: DevelopDouble ---
    let actions = runner.start_turn();
    assert_eq!(
        runner.framework.current_player, first_player,
        "First turn should belong to first player in turn order"
    );
    assert!(
        actions.contains(&ActionType::DevelopDouble),
        "DevelopDouble should be available"
    );

    let money_before_p1 = runner.framework.board.state.players[first_player].money;

    // DevelopDouble flow: Card → Industry → SecondIndustry → IronSource(s) → Confirm
    let cs = runner.start_action(ActionType::DevelopDouble);
    assert!(
        matches!(cs, ChoiceSet::Card(_)),
        "First choice for DevelopDouble should be a card, got {:?}",
        cs
    );
    // Pick any card (index 0)
    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(cs, Some(ChoiceSet::Industry(_))),
        "After card, should choose first industry, got {:?}",
        cs
    );

    // Pick first industry — find one that's available
    let available_industries = match cs.as_ref().unwrap() {
        ChoiceSet::Industry(opts) => opts.clone(),
        _ => panic!("Expected Industry choice set"),
    };
    let first_industry = available_industries[0];
    let cs = runner.apply_choice(ActionChoice::Industry(first_industry));

    // Second industry (SecondIndustry choice set)
    let second_industry = match cs.as_ref().unwrap() {
        ChoiceSet::SecondIndustry(opts) => {
            // Same industry should be allowed for DevelopDouble.
            assert!(
                opts.contains(&first_industry),
                "SecondIndustry options should include first choice for same-industry double develop"
            );
            opts[0]
        }
        other => panic!(
            "Expected SecondIndustry for second industry, got {:?}",
            other
        ),
    };
    let cs = runner.apply_choice(ActionChoice::FreeDevelopment(second_industry));

    // Iron source(s) - need 2 iron
    // Could need 1 or 2 IronSource choices depending on how the market source works
    let mut cs = cs;
    while let Some(ChoiceSet::IronSource(ref opts)) = cs {
        assert!(!opts.is_empty(), "Iron source options should not be empty");
        cs = runner.apply_choice(ActionChoice::IronSource(opts[0].clone()));
    }

    // Should now be ConfirmOnly
    assert!(
        matches!(cs, Some(ChoiceSet::ConfirmOnly)),
        "Should be ready to confirm, got {:?}",
        cs
    );
    runner.confirm_action().unwrap();

    let money_after_p1 = runner.framework.board.state.players[first_player].money;
    let p1_spent = money_before_p1 - money_after_p1;
    assert_eq!(
        p1_spent, 4,
        "DevelopDouble with 2 market iron should cost £4 (iron at £2 each from initial market), got £{}",
        p1_spent
    );
    assert_eq!(
        runner.framework.board.state.players[first_player].spent_this_turn, 4,
        "spent_this_turn should be 4"
    );

    // First player's turn is done (only 1 action in first personal turn)
    assert_eq!(runner.actions_remaining_in_turn, 0);
    runner.finish_turn_and_advance();

    // --- Second player's turn: Loan ---
    assert_eq!(
        runner.framework.current_player, second_player,
        "Should now be second player's turn"
    );
    let actions = runner.start_turn();
    assert!(
        actions.contains(&ActionType::Loan),
        "Loan should be available"
    );

    let money_before_p2 = runner.framework.board.state.players[second_player].money;

    // Loan flow: Card → Confirm
    let cs = runner.start_action(ActionType::Loan);
    assert!(
        matches!(cs, ChoiceSet::Card(_)),
        "Loan first choice should be a card, got {:?}",
        cs
    );
    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(cs, Some(ChoiceSet::ConfirmOnly)),
        "After card, Loan should be ConfirmOnly, got {:?}",
        cs
    );
    runner.confirm_action().unwrap();

    let money_after_p2 = runner.framework.board.state.players[second_player].money;
    assert_eq!(money_after_p2, money_before_p2 + 30, "Loan should give £30");
    assert_eq!(
        runner.framework.board.state.players[second_player].spent_this_turn, 0,
        "Loan player should have spent £0"
    );

    assert_eq!(runner.actions_remaining_in_turn, 0);
    runner.finish_turn_and_advance();

    // --- After round: turn order should update ---
    // Second player spent £0, first player spent £4
    // Turn order: second_player first (less spending), first_player second
    let new_order = &runner.framework.board.state.turn_order;
    assert_eq!(
        new_order[0], second_player,
        "Player who spent less (Loan £0) should be first in new turn order. \
         Got order: {:?}, expected {} first",
        new_order, second_player
    );
    assert_eq!(
        new_order[1], first_player,
        "Player who spent more (£4 from DevelopDouble) should be second in new turn order"
    );

    // Current player should be the first in the new turn order
    assert_eq!(
        runner.framework.current_player, new_order[0],
        "Current player should be first in new turn order ({}) but is {}",
        new_order[0], runner.framework.current_player
    );
}

/// Test scenario: After building roads connecting Coalbrookdale to Shrewbury trade post,
/// and doing DevelopDouble on Cotton, Brunel should be able to build Iron Works I at
/// Coalbrookdale using a Coalbrookdale location card.
///
/// Iron Works I: money_cost=5, coal_cost=1, iron_cost=0
/// Coal from market at initial price = £1 (13 cubes remaining)
/// Total = £6, player has £6 → should be affordable
#[test]
fn test_iron_available_at_coalbrookdale_after_develop_and_road_build() {
    let mut runner = GameRunner::new(2, Some(42));

    let brunel = runner.framework.board.state.turn_order[0];
    let coade = runner.framework.board.state.turn_order[1];

    // Give both players enough cards for their actions.
    // Brunel needs: 1 card for Network, 1 card for DevelopDouble, 1 Coalbrookdale card for Build
    // Total minimum: 3 cards, but we give extras so hand isn't empty after discards
    runner.framework.board.state.players[brunel].hand.cards = vec![
        Card::new(CardType::Location(TownName::Coalbrookdale)),
        Card::new(CardType::Location(TownName::Coalbrookdale)),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Iron,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Coal,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Beer,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Location(TownName::Birmingham)),
    ];
    runner.framework.board.state.players[coade].hand.cards = vec![
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Coal,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Iron,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Beer,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Location(TownName::Birmingham)),
        Card::new(CardType::Location(TownName::Stafford)),
        Card::new(CardType::Location(TownName::Derby)),
    ];

    // === Round 1: Both players build canals ===
    // Brunel builds road 20 (Shrewbury-Coalbrookdale)
    let actions = runner.start_turn();
    assert_eq!(runner.framework.current_player, brunel);
    assert!(
        actions.contains(&ActionType::BuildRailroad),
        "Network should be available"
    );

    let cs = runner.start_action(ActionType::BuildRailroad);
    assert!(
        matches!(cs, ChoiceSet::Card(_)),
        "First choice should be card, got {:?}",
        cs
    );
    // Use the Iron industry card (index 2) for the discard
    let cs = runner.apply_choice(ActionChoice::Card(2));
    assert!(
        matches!(cs, Some(ChoiceSet::Road(_))),
        "Should choose road, got {:?}",
        cs
    );
    let cs = runner.apply_choice(ActionChoice::Road(20));
    assert!(
        matches!(cs, Some(ChoiceSet::ConfirmOnly)),
        "Should confirm, got {:?}",
        cs
    );
    runner
        .confirm_action()
        .expect("Canal build should succeed for road 20");
    runner.finish_turn_and_advance();

    // Verify road 20 is built and connectivity updated
    assert!(
        runner.framework.board.state.built_roads.contains(20),
        "Road 20 should be built"
    );
    assert!(
        runner
            .framework
            .board
            .state
            .connectivity
            .are_towns_connected(
                fast_brass::locations::LocationName::Coalbrookdale,
                fast_brass::locations::LocationName::Shrewbury,
            ),
        "Coalbrookdale should be connected to Shrewbury after road 20"
    );

    // Coade builds road 21 (Kidderminster-Coalbrookdale)
    assert_eq!(runner.framework.current_player, coade);
    let actions = runner.start_turn();
    assert!(actions.contains(&ActionType::BuildRailroad));

    let cs = runner.start_action(ActionType::BuildRailroad);
    assert!(matches!(cs, ChoiceSet::Card(_)));
    let cs = runner.apply_choice(ActionChoice::Card(0)); // discard any card
    assert!(matches!(cs, Some(ChoiceSet::Road(_))));
    let cs = runner.apply_choice(ActionChoice::Road(21));
    assert!(matches!(cs, Some(ChoiceSet::ConfirmOnly)));
    runner
        .confirm_action()
        .expect("Canal build should succeed for road 21");
    runner.finish_turn_and_advance();

    // === Round 2: Brunel does DevelopDouble on Cotton, then Build ===
    // After round end, turn order may change. Find Brunel.
    let round2_first = runner.framework.board.state.turn_order[0];
    let _round2_second = runner.framework.board.state.turn_order[1];

    // Both players spent £3 in round 1, so turn order is preserved or tied.
    // We need Brunel to go first in this round for the test.
    // If Brunel is second, swap so Brunel is the acting player.
    if round2_first == brunel {
        // Brunel goes first - DevelopDouble
        runner.start_turn();
        assert_eq!(runner.framework.current_player, brunel);
        do_develop_double_cotton(&mut runner, brunel);
        runner.finish_turn_and_advance();

        // Coade passes
        runner.start_turn();
        do_pass_action(&mut runner);
        runner.finish_turn_and_advance();
    } else {
        // Coade goes first - pass
        runner.start_turn();
        assert_eq!(runner.framework.current_player, coade);
        do_pass_action(&mut runner);
        runner.finish_turn_and_advance();

        // Brunel goes second - DevelopDouble
        runner.start_turn();
        assert_eq!(runner.framework.current_player, brunel);
        do_develop_double_cotton(&mut runner, brunel);
        runner.finish_turn_and_advance();
    }

    // Now in round 3, Brunel should be able to build.
    // But we want to check the build options BEFORE the round wraps.
    // Instead, let's just check the board state directly.
    let brunel_money = runner.framework.board.state.players[brunel].money;
    println!("Brunel money after DevelopDouble: £{}", brunel_money);

    // Verify market access: Coalbrookdale connected to Shrewbury (trade post)
    assert!(
        runner.framework.board.state.is_connected_to_trade_post(25),
        "Coalbrookdale BL 25 should be connected to trade post"
    );

    // Verify Brunel still has a Coalbrookdale card
    let has_coalbrookdale_card = runner.framework.board.state.players[brunel]
        .hand
        .cards
        .iter()
        .any(|c| matches!(&c.card_type, CardType::Location(t) if *t == TownName::Coalbrookdale));
    assert!(
        has_coalbrookdale_card,
        "Brunel should still have a Coalbrookdale card"
    );

    // Check build options for Brunel
    let build_options =
        BuildValidator::get_valid_build_options(&runner.framework.board.state, brunel);

    let iron_options: Vec<_> = build_options
        .iter()
        .filter(|opt| opt.industry_type == IndustryType::Iron)
        .collect();

    assert!(
        !iron_options.is_empty(),
        "Iron Works I should be available. Brunel has £{}, Iron I costs £5 + £1 coal from market. \
         Coalbrookdale is connected to Shrewbury trade post. \
         Available industries: {:?}",
        brunel_money,
        build_options
            .iter()
            .map(|o| format!(
                "{:?} at BL {} (£{})",
                o.industry_type, o.build_location_idx, o.total_money_cost
            ))
            .collect::<Vec<_>>()
    );

    // Verify at least one Iron option is at Coalbrookdale (BLs 25-26 support Iron)
    let iron_at_coalbrookdale: Vec<_> = iron_options
        .iter()
        .filter(|opt| opt.build_location_idx >= 25 && opt.build_location_idx <= 26)
        .collect();

    assert!(
        !iron_at_coalbrookdale.is_empty(),
        "Iron Works I should be buildable at Coalbrookdale (BLs 25-26). \
         All iron options: {:?}",
        iron_options
            .iter()
            .map(|o| format!("BL {} (£{})", o.build_location_idx, o.total_money_cost))
            .collect::<Vec<_>>()
    );
}

fn do_develop_double_cotton(runner: &mut GameRunner, player_idx: usize) {
    assert!(
        runner
            .framework
            .get_valid_root_actions()
            .contains(&ActionType::DevelopDouble),
        "DevelopDouble should be available for player {}",
        player_idx
    );

    let cs = runner.start_action(ActionType::DevelopDouble);
    assert!(
        matches!(cs, ChoiceSet::Card(_)),
        "DevelopDouble first choice should be card, got {:?}",
        cs
    );
    // Discard first available card (index 0 = first Coalbrookdale card)
    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(cs, Some(ChoiceSet::Industry(_))),
        "After card, should choose first industry, got {:?}",
        cs
    );

    // Choose Cotton as first industry
    let cs = runner.apply_choice(ActionChoice::Industry(IndustryType::Cotton));

    // Choose second industry (same industry should be allowed)
    let second_industry = match cs.as_ref().unwrap() {
        ChoiceSet::SecondIndustry(opts) => {
            assert!(
                opts.contains(&IndustryType::Cotton),
                "SecondIndustry options should include Cotton after selecting Cotton first"
            );
            IndustryType::Cotton
        }
        other => panic!(
            "Expected SecondIndustry for second industry, got {:?}",
            other
        ),
    };
    let cs = runner.apply_choice(ActionChoice::FreeDevelopment(second_industry));

    // Iron source(s) - need 2 iron
    let mut cs = cs;
    while let Some(ChoiceSet::IronSource(ref opts)) = cs {
        assert!(!opts.is_empty(), "Iron source options should not be empty");
        cs = runner.apply_choice(ActionChoice::IronSource(opts[0].clone()));
    }

    assert!(
        matches!(cs, Some(ChoiceSet::ConfirmOnly)),
        "Should be ready to confirm DevelopDouble, got {:?}",
        cs
    );
    runner
        .confirm_action()
        .expect("DevelopDouble should succeed");
}

/// Bug: start_turn() was resetting actions_remaining_in_turn every time it was called.
/// The frontend calls start_turn() between actions (to refresh available actions),
/// which reset the counter and gave players unlimited actions.
#[test]
fn test_start_turn_does_not_reset_actions_remaining_mid_turn() {
    let mut runner = GameRunner::new(2, Some(42));
    let first_player = runner.framework.board.state.turn_order[0];
    let second_player = runner.framework.board.state.turn_order[1];

    // --- Round 1: both pass (1 action each) ---
    runner.start_turn();
    assert_eq!(runner.framework.current_player, first_player);
    assert_eq!(
        runner.actions_remaining_in_turn, 1,
        "First round = 1 action"
    );
    do_pass_action(&mut runner);
    assert_eq!(runner.actions_remaining_in_turn, 0);
    runner.finish_turn_and_advance();

    runner.start_turn();
    assert_eq!(runner.framework.current_player, second_player);
    do_pass_action(&mut runner);
    runner.finish_turn_and_advance();

    // --- Round 2: first player gets 2 actions ---
    let round2_first = runner.framework.board.state.turn_order[0];
    let round2_second = runner.framework.board.state.turn_order[1];

    runner.start_turn();
    assert_eq!(runner.framework.current_player, round2_first);
    assert_eq!(
        runner.actions_remaining_in_turn, 2,
        "Second round = 2 actions"
    );

    // Do first action (Pass)
    do_pass_action(&mut runner);
    assert_eq!(
        runner.actions_remaining_in_turn, 1,
        "After 1 action, 1 remaining"
    );

    // Simulate what the frontend does: call start_turn() again to refresh actions
    let _actions = runner.start_turn();
    assert_eq!(
        runner.actions_remaining_in_turn, 1,
        "start_turn() between actions must NOT reset counter. \
         Should still be 1, not 2"
    );

    // Do second action (Pass)
    do_pass_action(&mut runner);
    assert_eq!(
        runner.actions_remaining_in_turn, 0,
        "After 2 actions, 0 remaining"
    );

    // Calling start_turn again with 0 remaining should NOT give more actions
    // (the turn is done; end_turn must be called first)
    // Actually with 0 remaining, start_turn WILL reset for the same player.
    // But finish_turn_and_advance hasn't been called yet, so current_player is the same.
    // This is fine - in the actual flow, endTurn() is called when actions_remaining = 0.
    runner.finish_turn_and_advance();

    // Now it should be the other player's turn
    assert_eq!(
        runner.framework.current_player, round2_second,
        "After first player finishes 2 actions, second player should have the turn"
    );
}

/// Bug: after completing round 1 (1 action each) and round 2 (2 actions for first player),
/// the turn should switch to the second player, not back to the first.
#[test]
fn test_turn_switches_to_other_player_after_two_actions() {
    let mut runner = GameRunner::new(2, Some(99));

    // Round 1: each player passes
    runner.start_turn();
    do_pass_action(&mut runner);
    runner.finish_turn_and_advance();

    runner.start_turn();
    do_pass_action(&mut runner);
    runner.finish_turn_and_advance();

    // Round 2: first player does 2 passes
    let r2_first = runner.framework.board.state.turn_order[0];
    let r2_second = runner.framework.board.state.turn_order[1];
    assert_eq!(runner.framework.current_player, r2_first);

    runner.start_turn();
    assert_eq!(runner.actions_remaining_in_turn, 2);

    // Action 1
    do_pass_action(&mut runner);
    assert_eq!(runner.actions_remaining_in_turn, 1);

    // Simulate frontend: call start_turn between actions
    runner.start_turn();
    assert_eq!(
        runner.actions_remaining_in_turn, 1,
        "Must not reset mid-turn"
    );

    // Action 2
    do_pass_action(&mut runner);
    assert_eq!(runner.actions_remaining_in_turn, 0);

    // End turn
    runner.finish_turn_and_advance();
    assert_eq!(
        runner.framework.current_player, r2_second,
        "Turn must switch to the other player after 2 actions, not back to the same player"
    );

    // Second player should get 2 actions
    runner.start_turn();
    assert_eq!(runner.actions_remaining_in_turn, 2);
}

/// Refill only after the final action of a player's turn (rulebook p. 6).
#[test]
fn test_players_draw_cards_after_all_turn_actions() {
    let mut runner = GameRunner::new(2, Some(42));

    let p0 = runner.framework.board.state.turn_order[0];
    let p1 = runner.framework.board.state.turn_order[1];

    // Initial hand size should be 8
    assert_eq!(
        runner.framework.board.state.players[p0].hand.cards.len(),
        8,
        "Initial hand size should be 8"
    );
    assert_eq!(
        runner.framework.board.state.players[p1].hand.cards.len(),
        8,
        "Initial hand size should be 8"
    );

    let deck_size_before = runner.framework.board.state.deck.cards_left();

    // Round 1: each player uses 1 action (discards 1 card)
    // Player 0: Loan (discards 1 card, gains £30)
    runner.start_turn();
    let cs = runner.start_action(ActionType::Loan);
    assert!(matches!(cs, ChoiceSet::Card(_)));
    runner.apply_choice(ActionChoice::Card(0));
    runner.confirm_action().unwrap();
    runner.finish_turn_and_advance();

    assert_eq!(
        runner.framework.board.state.players[p0].hand.cards.len(),
        8,
        "P0 should refill immediately after its action"
    );

    // Player 1: Loan
    runner.start_turn();
    let cs = runner.start_action(ActionType::Loan);
    assert!(matches!(cs, ChoiceSet::Card(_)));
    runner.apply_choice(ActionChoice::Card(0));
    runner.confirm_action().unwrap();

    // Both players refill before the round advances.
    assert_eq!(
        runner.framework.board.state.players[p0].hand.cards.len(),
        8,
        "P0 should still have 8 cards before round end"
    );
    assert_eq!(
        runner.framework.board.state.players[p1].hand.cards.len(),
        8,
        "P1 should refill immediately after its action"
    );

    // Advancing the round does not draw any additional cards.
    runner.finish_turn_and_advance();

    // After round end: both hands should be back to 8
    assert_eq!(
        runner.framework.board.state.players[p0].hand.cards.len(),
        8,
        "P0 should have 8 cards after round end"
    );
    assert_eq!(
        runner.framework.board.state.players[p1].hand.cards.len(),
        8,
        "P1 should have 8 cards after round end"
    );

    // Deck should have 2 fewer cards (1 drawn per player)
    let deck_size_after = runner.framework.board.state.deck.cards_left();
    assert_eq!(
        deck_size_before - deck_size_after,
        2,
        "Deck should have lost 2 cards (1 per player). Before: {}, After: {}",
        deck_size_before,
        deck_size_after
    );

    runner.start_turn();
    assert_eq!(runner.actions_remaining_in_turn, 2);
    let actor = runner.framework.current_player;
    do_pass_action(&mut runner);
    assert_eq!(
        runner.framework.board.state.players[actor].hand.cards.len(),
        7
    );
    assert_eq!(
        runner.framework.board.state.deck.cards_left(),
        deck_size_after
    );
    do_pass_action(&mut runner);
    assert_eq!(
        runner.framework.board.state.players[actor].hand.cards.len(),
        8
    );
    assert_eq!(
        runner.framework.board.state.deck.cards_left(),
        deck_size_after - 2
    );
}

/// Bug: Iron Works (and other buildings) had income: 0 in static data, so
/// flipping a building never increased the owner's income level.
///
/// Scenario (3 players):
///   1. Player 1 does DevelopX2 (consumes 2 iron from market → 6 remaining)
///   2. Player 2 builds Coal Mine at Redditch (bl 45)
///   3. Player 3 builds Iron Works I at Redditch (bl 46), consuming coal from #2's mine
///      → Iron Works has 4 iron cubes, market has 4 free slots → all sell → auto-flip
///
/// Expected after flip:
///   - Building is flipped
///   - Player 3's income level increases by 3 (Iron I income) → from 10 to 13
///   - Income amount at level 13 = +2
///   - After round end, player receives +2 income
#[test]
fn test_iron_works_flip_increases_income() {
    let mut runner = GameRunner::new(3, Some(42));
    let board = &mut runner.framework.board.state;

    // Pick player indices (doesn't matter which, we directly manipulate)
    let coal_owner = 1;
    let iron_owner = 2;

    // Player 1: DevelopX2 consumed 2 iron from market
    board.remaining_market_iron -= 2; // 8 → 6

    // Player 2: Build Coal Mine I at Redditch bl 45
    let coal_bl = 45;
    let coal_building = BuiltBuilding::build(
        IndustryType::Coal,
        IndustryLevel::I,
        coal_bl as u8,
        PlayerId::from_usize(coal_owner),
    );
    board.bl_to_building.insert(coal_bl, coal_building);
    board.build_locations_occupied.insert(coal_bl);
    board.player_building_mask[coal_owner].insert(coal_bl);
    board.coal_locations.insert(coal_bl);

    // Player 3: Build Iron Works I at Redditch bl 46
    let iron_bl = 46;
    let iron_building = BuiltBuilding::build(
        IndustryType::Iron,
        IndustryLevel::I,
        iron_bl as u8,
        PlayerId::from_usize(iron_owner),
    );
    board.bl_to_building.insert(iron_bl, iron_building.clone());
    board.build_locations_occupied.insert(iron_bl);
    board.player_building_mask[iron_owner].insert(iron_bl);
    board.iron_locations.insert(iron_bl);

    assert_eq!(
        board.players[iron_owner].income_level, 10,
        "Starting income level"
    );
    let money_before = board.players[iron_owner].money;

    // Sell iron to market (this is called automatically after build)
    let gained = ResourceManager::sell_building_resources_to_market(board, iron_owner, iron_bl);
    assert!(gained > 0, "Should gain money from selling iron to market");

    // The building should be flipped (all 4 iron went to market which had 4 free slots)
    let building = board.bl_to_building.get(&iron_bl).unwrap();
    assert!(
        building.flipped,
        "Iron Works should be flipped after all iron sold to market"
    );

    // Income level should have increased by 3 (Iron Works I income = 3)
    assert_eq!(
        board.players[iron_owner].income_level, 13,
        "Income level should increase from 10 to 13 (Iron I gives +3). Got {}",
        board.players[iron_owner].income_level
    );

    // Income amount at level 13 = (13-10+1)/2 = 2
    let income_amount =
        board.players[iron_owner].get_income_amount(board.players[iron_owner].income_level);
    assert_eq!(income_amount, 2, "Income at level 13 should be +2");

    // Now simulate round end → player should receive +2 income
    let money_after_sell = board.players[iron_owner].money;
    runner.resolve_income_shortfall_for_player(iron_owner);
    let money_after_income = runner.framework.board.state.players[iron_owner].money;
    assert_eq!(
        money_after_income,
        money_after_sell + 2,
        "Player should receive +2 income at round end. Money before: {}, after: {}",
        money_after_sell,
        money_after_income
    );
}

/// Bug: get_roads_adjacent_to_player_network used global connectivity,
/// so a player with a building in Coventry could "see" roads via another
/// player's Birmingham–Coventry link.
///
/// Scenario (2 players, canal era):
///   1. Player 1 builds coal mine at Coventry (bl 43)
///   2. Player 2 builds road 30 (Birmingham–Coventry)
///   3. Player 1 should have NO canal road options (road 30 occupied,
///      road 27 Nuneaton–Coventry is rail-only)
#[test]
fn test_player_cannot_build_roads_via_other_players_links() {
    let mut runner = GameRunner::new(2, Some(42));
    let board = &mut runner.framework.board.state;

    let p1 = 0;
    let p2 = 1;

    // Player 1 builds coal mine at Coventry (bl 43 = Goods/Coal slot)
    let coventry_bl = 43;
    let coal_building = BuiltBuilding::build(
        IndustryType::Coal,
        IndustryLevel::I,
        coventry_bl as u8,
        PlayerId::from_usize(p1),
    );
    board.bl_to_building.insert(coventry_bl, coal_building);
    board.build_locations_occupied.insert(coventry_bl);
    board.player_building_mask[p1].insert(coventry_bl);
    board.coal_locations.insert(coventry_bl);

    // Player 2 builds road 30 (Birmingham–Coventry)
    board.place_link(p2, 30);

    // Player 1 now has a building at Coventry but no roads of their own.
    // The only canal-eligible road touching Coventry is road 30, which is occupied.
    // Road 27 (Nuneaton–Coventry) is rail-only.
    // Player 1 should have ZERO canal options.
    let canal_options = NetworkValidator::get_valid_canal_options(board, p1);
    assert!(
        canal_options.is_empty(),
        "Player 1 should have no canal options — only adjacent road (30) is occupied and road 27 is rail-only. \
         Got {} options: {:?}",
        canal_options.len(), canal_options
    );
}

/// Bug: After choosing a building to sell, the framework tried to look up the
/// merchant beer slot using `beer_source_loc_idx` (which is `N_BL + slot_idx`)
/// directly as an index into `trade_post_slots` (which only has ~10 entries).
/// This caused `get()` to return None, the beer source list stayed empty,
/// and the framework looped back to ChooseSellTargets instead of advancing
/// to ChooseBeerSource.
///
/// Scenario (2 players):
///   1. Player 1 builds road 31 (Birmingham–Oxford)
///   2. Player 2 builds Cotton I at Birmingham (bl 36)
///   3. Oxford has a Goods-accepting merchant with beer at slot 1
///   4. Player 2 sells → chooses card → chooses cotton building →
///      NEXT CHOICE SHOULD BE BeerSource, NOT SellTarget again
#[test]
fn test_sell_advances_to_beer_source_after_choosing_building() {
    use fast_brass::board::resources::BeerSellSource;
    use fast_brass::market::merchants::{MerchantTile, MerchantTileType};

    let mut runner = GameRunner::new(2, Some(42));
    let board = &mut runner.framework.board.state;

    let p1 = 0;
    let p2 = 1;

    // Player 1 builds road 31 (Birmingham–Oxford)
    board.place_link(p1, 31);

    // Ensure Oxford slot 1 has a Cotton-accepting merchant with beer
    let cotton_merchant = MerchantTile::from_type(MerchantTileType::Cotton);
    board.trade_post_slots[1] = Some(cotton_merchant);
    board.trade_post_beer.insert(1);

    // Player 2 builds Cotton I at Birmingham bl 36
    let cotton_bl = 36;
    let cotton_building = BuiltBuilding::build(
        IndustryType::Cotton,
        IndustryLevel::I,
        cotton_bl as u8,
        PlayerId::from_usize(p2),
    );
    board.bl_to_building.insert(cotton_bl, cotton_building);
    board.build_locations_occupied.insert(cotton_bl);
    board.player_building_mask[p2].insert(cotton_bl);

    // Give player 2 a card to discard for selling
    board.players[p2].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Cotton]),
    ))];
    board.players[p2].money = 50;

    // Set current player to p2
    runner.framework.current_player = p2;
    runner.actions_remaining_in_turn = 2;

    // Start sell action
    let first_choice = runner.start_action(ActionType::Sell);
    // First should be choosing a card
    assert!(
        matches!(first_choice, ChoiceSet::Card(_)),
        "First choice should be Card, got {:?}",
        first_choice
    );

    // Choose card index 0
    let after_card = runner.apply_choice(ActionChoice::Card(0));
    // Next should be choosing a sell target (the cotton building)
    assert!(
        matches!(&after_card, Some(ChoiceSet::SellTarget(targets)) if targets.contains(&cotton_bl)),
        "After card choice, should get SellTarget containing bl {}, got {:?}",
        cotton_bl,
        after_card
    );

    // Choose the cotton building as sell target
    let after_sell_target = runner.apply_choice(ActionChoice::SellTarget(cotton_bl));
    // Should be BeerSource, NOT SellTarget again
    assert!(
        matches!(&after_sell_target, Some(ChoiceSet::BeerSource(_))),
        "After choosing sell target, next choice should be BeerSource (merchant), \
         but got {:?}",
        after_sell_target
    );

    // Extract the beer sources offered and choose the first one
    let beer_sources = match after_sell_target.unwrap() {
        ChoiceSet::BeerSource(sources) => sources,
        other => panic!("Expected BeerSource, got {:?}", other),
    };
    assert!(
        !beer_sources.is_empty(),
        "Should have at least one beer source"
    );

    let after_beer = runner.apply_choice(ActionChoice::BeerSource(beer_sources[0]));
    // After choosing beer, the building should be removed from sell targets.
    // With only one sellable building, the next step should be ConfirmOnly (no more targets).
    assert!(
        !matches!(&after_beer, Some(ChoiceSet::SellTarget(_))),
        "After choosing beer for the only sellable building, should NOT loop back to SellTarget. Got {:?}",
        after_beer
    );
}

#[test]
fn test_sell_confirm_discards_selected_card() {
    use fast_brass::board::resources::BeerSellSource;
    use fast_brass::market::merchants::{MerchantTile, MerchantTileType};

    let mut runner = GameRunner::new(2, Some(4242));
    let board = &mut runner.framework.board.state;

    let p1 = 0usize;
    let p2 = 1usize;
    let cotton_bl = 36usize;

    // Make a sellable cotton building for p2 at Birmingham.
    board.bl_to_building.insert(
        cotton_bl,
        BuiltBuilding::build(
            IndustryType::Cotton,
            IndustryLevel::I,
            cotton_bl as u8,
            PlayerId::from_usize(p2),
        ),
    );
    board.build_locations_occupied.insert(cotton_bl);
    board.player_building_mask[p2].insert(cotton_bl);

    // Connect Birmingham -> Oxford and ensure Oxford has cotton merchant beer.
    board.place_link(p1, 31);
    board.trade_post_slots[1] = Some(MerchantTile::from_type(MerchantTileType::Cotton));
    board.trade_post_beer.insert(1);

    // Two cards remain at the start of this two-action turn.
    board.players[p2].hand.cards =
        vec![Card::new(CardType::Location(TownName::BurtonUponTrent)); 2];
    board.players[p2].money = 20;

    runner.framework.current_player = p2;
    runner.actions_remaining_in_turn = 2;

    let discard_before = board.discard_pile.len();

    let first = runner.start_action(ActionType::Sell);
    assert!(
        matches!(first, ChoiceSet::Card(_)),
        "Sell should start with card choice"
    );
    let after_card = runner.apply_choice(ActionChoice::Card(0));
    assert!(matches!(&after_card, Some(ChoiceSet::SellTarget(t)) if t.contains(&cotton_bl)));
    let after_target = runner.apply_choice(ActionChoice::SellTarget(cotton_bl));
    let sources = match after_target {
        Some(ChoiceSet::BeerSource(s)) => s,
        other => panic!("Expected BeerSource after choosing target, got {:?}", other),
    };
    assert!(
        sources.contains(&BeerSellSource::TradePost(1)),
        "Expected Oxford beer source (slot 1), got {:?}",
        sources
    );
    let after_beer = runner.apply_choice(ActionChoice::BeerSource(BeerSellSource::TradePost(1)));
    assert!(matches!(after_beer, Some(ChoiceSet::ConfirmOnly)));

    runner
        .confirm_action()
        .expect("Sell confirm should succeed");

    let hand_after = runner.framework.board.state.players[p2].hand.cards.len();
    let discard_after = runner.framework.board.state.discard_pile.len();
    assert_eq!(
        hand_after, 1,
        "The first action must not draw the second action's card"
    );
    assert_eq!(
        discard_after,
        discard_before + 1,
        "Sell should add one card to discard pile"
    );
}

/// Bug: When selling via a Gloucester merchant (FreeDevelopment bonus), the
/// free development industry selection was empty because `initial_dev_options`
/// is only populated for Develop actions, not Sell.
///
/// Scenario (2 players):
///   1. Player 1 builds roads 33 (Birmingham–Worcester) and 37 (Worcester–Gloucester)
///   2. Player 2 builds Cotton I at Birmingham (bl 36)
///   3. Gloucester slot 3 has a Cotton merchant with beer (FreeDevelopment bonus)
///   4. Player 2 sells cotton → picks beer from Gloucester →
///      NEXT CHOICE SHOULD BE FreeDevelopment with non-empty industry list
#[test]
fn test_sell_with_gloucester_merchant_offers_free_development() {
    use fast_brass::board::resources::BeerSellSource;
    use fast_brass::market::merchants::{MerchantTile, MerchantTileType};

    let mut runner = GameRunner::new(2, Some(42));
    let board = &mut runner.framework.board.state;

    let p1 = 0;
    let p2 = 1;

    // Player 1 builds roads 33 (Birmingham–Worcester) and 37 (Worcester–Gloucester)
    board.place_link(p1, 33);
    board.place_link(p1, 37);

    // Ensure Gloucester slot 3 has a Cotton-accepting merchant with beer
    let cotton_merchant = MerchantTile::from_type(MerchantTileType::Cotton);
    board.trade_post_slots[3] = Some(cotton_merchant);
    board.trade_post_beer.insert(3);

    // Player 2 builds Cotton I at Birmingham bl 36
    let cotton_bl = 36;
    let cotton_building = BuiltBuilding::build(
        IndustryType::Cotton,
        IndustryLevel::I,
        cotton_bl as u8,
        PlayerId::from_usize(p2),
    );
    board.bl_to_building.insert(cotton_bl, cotton_building);
    board.build_locations_occupied.insert(cotton_bl);
    board.player_building_mask[p2].insert(cotton_bl);

    // Give player 2 a card to discard
    board.players[p2].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Cotton]),
    ))];
    board.players[p2].money = 50;

    runner.framework.current_player = p2;
    runner.actions_remaining_in_turn = 2;

    // Start sell action
    let first_choice = runner.start_action(ActionType::Sell);
    assert!(matches!(first_choice, ChoiceSet::Card(_)), "First: Card");

    let after_card = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(&after_card, Some(ChoiceSet::SellTarget(t)) if t.contains(&cotton_bl)),
        "After card: SellTarget"
    );

    let after_target = runner.apply_choice(ActionChoice::SellTarget(cotton_bl));
    let beer_sources = match after_target {
        Some(ChoiceSet::BeerSource(sources)) => sources,
        other => panic!("Expected BeerSource, got {:?}", other),
    };

    // The only beer source should be from Gloucester slot 3
    let gloucester_source = beer_sources
        .iter()
        .find(|s| matches!(s, BeerSellSource::TradePost(3)));
    assert!(
        gloucester_source.is_some(),
        "Should have Gloucester beer source (slot 3)"
    );

    // Choose the Gloucester merchant beer
    let after_beer = runner.apply_choice(ActionChoice::BeerSource(BeerSellSource::TradePost(3)));

    // Since Gloucester gives FreeDevelopment, should offer industry selection
    match &after_beer {
        Some(ChoiceSet::FreeDevelopment(industries)) => {
            assert!(
                !industries.is_empty(),
                "Free development industry list should NOT be empty. \
                 Player has tiles to develop on their mat."
            );
        }
        other => panic!(
            "After using Gloucester merchant beer, expected FreeDevelopment choice, got {:?}",
            other
        ),
    }
}

/// Regression: Gloucester free development must still offer industries even when the
/// player cannot afford normal Develop iron costs.
#[test]
fn test_sell_with_gloucester_free_development_ignores_iron_affordability() {
    use fast_brass::board::resources::BeerSellSource;
    use fast_brass::market::merchants::{MerchantTile, MerchantTileType};

    let mut runner = GameRunner::new(2, Some(1234));
    let board = &mut runner.framework.board.state;

    let p1 = 0;
    let p2 = 1;

    // Connect Birmingham to Gloucester.
    board.place_link(p1, 33); // Birmingham-Worcester
    board.place_link(p1, 37); // Worcester-Gloucester

    // Gloucester cotton merchant with beer (free development bonus).
    board.trade_post_slots[3] = Some(MerchantTile::from_type(MerchantTileType::Cotton));
    board.trade_post_beer.insert(3);

    // Sellable cotton for p2 at Birmingham.
    let cotton_bl = 36usize;
    board.bl_to_building.insert(
        cotton_bl,
        BuiltBuilding::build(
            IndustryType::Cotton,
            IndustryLevel::I,
            cotton_bl as u8,
            PlayerId::from_usize(p2),
        ),
    );
    board.build_locations_occupied.insert(cotton_bl);
    board.player_building_mask[p2].insert(cotton_bl);

    // Low money + no board iron + empty iron market => normal Develop options would be empty.
    board.remaining_market_iron = 0;
    board.players[p2].money = 3;
    board.players[p2].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Cotton]),
    ))];

    runner.framework.current_player = p2;
    runner.actions_remaining_in_turn = 2;

    let first_choice = runner.start_action(ActionType::Sell);
    assert!(matches!(first_choice, ChoiceSet::Card(_)));
    let after_card = runner.apply_choice(ActionChoice::Card(0));
    assert!(matches!(&after_card, Some(ChoiceSet::SellTarget(t)) if t.contains(&cotton_bl)));

    let after_target = runner.apply_choice(ActionChoice::SellTarget(cotton_bl));
    let beer_sources = match after_target {
        Some(ChoiceSet::BeerSource(sources)) => sources,
        other => panic!("Expected BeerSource, got {:?}", other),
    };
    assert!(beer_sources.contains(&BeerSellSource::TradePost(3)));

    let after_beer = runner.apply_choice(ActionChoice::BeerSource(BeerSellSource::TradePost(3)));
    match after_beer {
        Some(ChoiceSet::FreeDevelopment(industries)) => {
            assert!(
                !industries.is_empty(),
                "Free development industries should be selectable even with no iron affordability"
            );
        }
        other => panic!(
            "Expected FreeDevelopment after Gloucester beer, got {:?}",
            other
        ),
    }
}

/// Regression: when selling, own brewery beer must always be available, while
/// opponent brewery beer must require connectivity, and only matching-merchant
/// beer slots should be offered.
///
/// Scenario (4 players):
/// - Gloucester has Cotton merchant at slot 3 (with beer) and Goods merchant at slot 4 (with beer)
/// - Roads:
///   - Tinsley: 37 (Worcester-Gloucester)
///   - Brunel: 38 (Worcester-LoneBrewery2-Kidderminster)
/// - Breweries:
///   - Arkwright at Coalbrookdale
///   - Coade at Uttoxeter (own beer, should always be available)
///   - Brunel at LoneBrewery2 (connected to Worcester, should be available)
/// - Coade has Cotton at Worcester and attempts to sell it.
///
/// Expected beer options for Coade's sale:
/// - include Gloucester slot 3 (Cotton merchant beer)
/// - exclude Gloucester slot 4 (Goods merchant beer)
/// - include own brewery at Uttoxeter
/// - include Brunel brewery at LoneBrewery2
/// - exclude Arkwright brewery at Coalbrookdale (not connected)
#[test]
fn test_sell_beer_accessibility_own_and_connected_only() {
    use fast_brass::board::resources::BeerSellSource;
    use fast_brass::market::merchants::{MerchantTile, MerchantTileType};

    let mut runner = GameRunner::new(4, Some(42));
    let board = &mut runner.framework.board.state;

    // Player identities in this codebase:
    // 0=Coade, 1=Brunel, 2=Arkwright, 3=Tinsley
    let coade = 0usize;
    let brunel = 1usize;
    let arkwright = 2usize;
    let tinsley = 3usize;

    // Roads from the scenario.
    board.place_link(tinsley, 37); // Worcester-Gloucester
    board.place_link(brunel, 38); // Worcester-LoneBrewery2-Kidderminster

    // Gloucester slots:
    // - slot 3: Cotton merchant with beer (valid for cotton sell)
    // - slot 4: Goods merchant with beer (must NOT be offered for cotton sell)
    board.trade_post_slots[3] = Some(MerchantTile::from_type(MerchantTileType::Cotton));
    board.trade_post_slots[4] = Some(MerchantTile::from_type(MerchantTileType::Goods));
    board.trade_post_beer.insert(3);
    board.trade_post_beer.insert(4);

    // Build locations:
    let coade_uttoxeter_brewery_bl = 17usize; // Uttoxeter
    let arkwright_coalbrookdale_brewery_bl = 25usize; // Coalbrookdale
    let brunel_lone_brewery_bl = 48usize; // LoneBrewery2
    let coade_worcester_cotton_bl = 34usize; // Worcester

    // Coade brewery at Uttoxeter (own beer should always be available).
    let coade_brewery = BuiltBuilding::build(
        IndustryType::Beer,
        IndustryLevel::I,
        coade_uttoxeter_brewery_bl as u8,
        PlayerId::from_usize(coade),
    );
    board
        .bl_to_building
        .insert(coade_uttoxeter_brewery_bl, coade_brewery);
    board
        .build_locations_occupied
        .insert(coade_uttoxeter_brewery_bl);
    board.player_building_mask[coade].insert(coade_uttoxeter_brewery_bl);

    // Brunel brewery at LoneBrewery2 (connected to Worcester via road 38).
    let brunel_brewery = BuiltBuilding::build(
        IndustryType::Beer,
        IndustryLevel::I,
        brunel_lone_brewery_bl as u8,
        PlayerId::from_usize(brunel),
    );
    board
        .bl_to_building
        .insert(brunel_lone_brewery_bl, brunel_brewery);
    board
        .build_locations_occupied
        .insert(brunel_lone_brewery_bl);
    board.player_building_mask[brunel].insert(brunel_lone_brewery_bl);

    // Arkwright brewery at Coalbrookdale (not connected to Worcester in this setup).
    let arkwright_brewery = BuiltBuilding::build(
        IndustryType::Beer,
        IndustryLevel::I,
        arkwright_coalbrookdale_brewery_bl as u8,
        PlayerId::from_usize(arkwright),
    );
    board
        .bl_to_building
        .insert(arkwright_coalbrookdale_brewery_bl, arkwright_brewery);
    board
        .build_locations_occupied
        .insert(arkwright_coalbrookdale_brewery_bl);
    board.player_building_mask[arkwright].insert(arkwright_coalbrookdale_brewery_bl);

    // Coade cotton building at Worcester to sell.
    let coade_cotton = BuiltBuilding::build(
        IndustryType::Cotton,
        IndustryLevel::I,
        coade_worcester_cotton_bl as u8,
        PlayerId::from_usize(coade),
    );
    board
        .bl_to_building
        .insert(coade_worcester_cotton_bl, coade_cotton);
    board
        .build_locations_occupied
        .insert(coade_worcester_cotton_bl);
    board.player_building_mask[coade].insert(coade_worcester_cotton_bl);

    // Give Coade a card to discard for Sell action.
    board.players[coade].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Cotton]),
    ))];
    board.players[coade].money = 50;

    runner.framework.current_player = coade;
    runner.actions_remaining_in_turn = 2;

    let first = runner.start_action(ActionType::Sell);
    assert!(
        matches!(first, ChoiceSet::Card(_)),
        "Expected first sell step to be Card"
    );

    let after_card = runner.apply_choice(ActionChoice::Card(0));
    assert!(
        matches!(&after_card, Some(ChoiceSet::SellTarget(opts)) if opts.contains(&coade_worcester_cotton_bl)),
        "After card, expected SellTarget including Worcester cotton building, got {:?}",
        after_card
    );

    let after_target = runner.apply_choice(ActionChoice::SellTarget(coade_worcester_cotton_bl));
    let beer_sources = match after_target {
        Some(ChoiceSet::BeerSource(sources)) => sources,
        other => panic!(
            "Expected BeerSource after picking sell target, got {:?}",
            other
        ),
    };

    assert!(
        beer_sources.contains(&BeerSellSource::TradePost(3)),
        "Expected Gloucester Cotton merchant beer (slot 3) to be available. Got {:?}",
        beer_sources
    );
    assert!(
        !beer_sources.contains(&BeerSellSource::TradePost(4)),
        "Did not expect Gloucester Goods merchant beer (slot 4) for cotton sell. Got {:?}",
        beer_sources
    );
    assert!(
        beer_sources.contains(&BeerSellSource::Building(coade_uttoxeter_brewery_bl)),
        "Expected own Uttoxeter brewery beer to be available. Got {:?}",
        beer_sources
    );
    assert!(
        beer_sources.contains(&BeerSellSource::Building(brunel_lone_brewery_bl)),
        "Expected connected Brunel lone-brewery beer to be available. Got {:?}",
        beer_sources
    );
    assert!(
        !beer_sources.contains(&BeerSellSource::Building(
            arkwright_coalbrookdale_brewery_bl
        )),
        "Did not expect unconnected Arkwright Coalbrookdale brewery beer. Got {:?}",
        beer_sources
    );
}

/// Regression reproducing reported economics bug:
/// - one market iron left
/// - player starts action at £21
/// - build Iron Works II at Redditch using market coal
/// - new iron works should auto-sell 4 iron to market immediately
/// Expected net +£5 => £26 after confirm.
#[test]
fn test_build_iron_ii_market_refill_profit_after_undo() {
    use fast_brass::board::resources::ResourceSource;
    use fast_brass::validation::build_validation::BuildValidator;

    let mut runner = GameRunner::new(2, Some(1234));
    let coade = runner.framework.current_player;

    // Create the same turn context as the report: a confirmed action was undone first.
    let _ = runner.start_turn();
    let cs = runner.start_action(ActionType::Pass);
    assert!(matches!(cs, ChoiceSet::Card(_)));
    let cs = runner.apply_choice(ActionChoice::Card(0));
    assert!(matches!(cs, Some(ChoiceSet::ConfirmOnly)));
    runner
        .confirm_action()
        .expect("pass confirm should succeed");
    runner
        .undo_last_confirmed_action()
        .expect("undo should restore turn state");

    // Economic setup from repro.
    let board = &mut runner.framework.board.state;
    board.players[coade].money = 21;
    board.remaining_market_iron = 1; // one iron left in market
    board.remaining_market_coal = 8; // market coal price for 1 cube = £4 in current price table

    // Force next iron build to be Iron II.
    board.players[coade]
        .industry_mat
        .pop_tile(IndustryType::Iron);

    // Give exactly the "Iron card" flavor from report.
    board.players[coade].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Iron]),
    ))];

    // Connect Redditch to Gloucester (road 36) for coal market legality.
    let redditch_gloucester = 36usize;
    let iron_redditch_bl = 46usize; // "slot 47" in 1-based UI
    board.place_link(coade, redditch_gloucester);

    // Sanity-check intended build option economics before driving the action session.
    let opts = BuildValidator::get_valid_build_options(board, coade);
    let iron_redditch = opts
        .iter()
        .find(|o| {
            o.industry_type == IndustryType::Iron
                && o.build_location_idx == iron_redditch_bl
                && o.card_used_idx == 0
        })
        .cloned()
        .expect("Expected an Iron build option at Redditch slot 47 with the Iron card");
    assert_eq!(
        iron_redditch.level,
        IndustryLevel::II,
        "Expected Iron II build at Redditch"
    );
    assert_eq!(
        iron_redditch.building_data.money_cost, 7,
        "Iron II base cost mismatch"
    );
    assert_eq!(
        iron_redditch.total_coal_cost, 1,
        "Iron II should require one coal in current data"
    );
    assert_eq!(
        iron_redditch.total_money_cost, 11,
        "Expected £11 total (7 base + 4 coal market)"
    );
    // Drive the Build action session to confirmation.
    let mut cs = Some(runner.start_action(ActionType::BuildBuilding));
    loop {
        match cs {
            Some(ChoiceSet::Card(_)) => {
                cs = runner.apply_choice(ActionChoice::Card(0));
            }
            Some(ChoiceSet::Industry(_)) => {
                cs = runner.apply_choice(ActionChoice::Industry(IndustryType::Iron));
            }
            Some(ChoiceSet::BuildLocation(_)) => {
                cs = runner.apply_choice(ActionChoice::BuildLocation(iron_redditch_bl));
            }
            Some(ChoiceSet::CoalSource(ref opts)) => {
                let market = opts
                    .iter()
                    .find(|s| matches!(s, ResourceSource::Market))
                    .cloned()
                    .expect("Expected market coal source");
                cs = runner.apply_choice(ActionChoice::CoalSource(market));
            }
            Some(ChoiceSet::ConfirmOnly) => break,
            other => panic!("Unexpected choice sequence for build flow: {:?}", other),
        }
    }
    runner
        .confirm_action()
        .expect("Build should confirm successfully");

    let money_after = runner.framework.board.state.players[coade].money;
    assert_eq!(
        money_after, 26,
        "Expected £26 after Iron II build at Redditch with one market iron left (start £21, net +£5), got £{}",
        money_after
    );
}
