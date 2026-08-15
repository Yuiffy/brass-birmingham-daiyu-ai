// tests/connectivity_tests.rs

#[cfg(test)]
mod connectivity_tests {
    use fast_brass::board::Board;
    use fast_brass::core::building::BuiltBuilding;
    use fast_brass::core::player::PlayerId;
    use fast_brass::core::types::IndustryLevel;
    use fast_brass::core::types::*;
    use fast_brass::game::framework::{ActionChoice, ChoiceSet, GameFramework};
    use fast_brass::game::runner::GameRunner;

    // Helper function to create a board with a specific seed for reproducible tests
    fn setup_board(num_players: usize) -> Board {
        Board::new(num_players, Some(42))
    }

    #[test]
    fn test_board_connectivity_initial_state() {
        let board = setup_board(4);

        // Board should have proper data structures initialized
        assert_eq!(
            board.state.player_building_mask.len(),
            4,
            "Should have building mask for each player"
        );
        assert_eq!(
            board.state.player_road_mask.len(),
            4,
            "Should have road mask for each player"
        );
        assert_eq!(
            board.state.player_network_mask.len(),
            4,
            "Should have network mask for each player"
        );
    }

    #[test]
    fn test_canal_options_available() {
        let board = setup_board(4);
        let player_idx = 0;

        // In canal era with no network, player can build canal anywhere
        let canal_options = board.get_valid_canal_options(player_idx);

        // Should have some canal options available
        assert!(
            !canal_options.is_empty(),
            "Should have canal options available initially"
        );
    }

    #[test]
    fn test_double_railroad_basic_validation() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Set up rail era
        board.state.era = Era::Railroad;

        // Give player enough money
        board.state.players[player_idx].money = 30;

        // Add some coal availability
        use fast_brass::core::building::BuiltBuilding;
        use fast_brass::core::player::PlayerId;

        board.state.coal_locations.insert(27);
        board.state.bl_to_building.insert(
            27,
            BuiltBuilding::build(
                IndustryType::Coal,
                IndustryLevel::from_usize(0),
                27,
                PlayerId::from_usize(0),
            ),
        );

        // Add beer availability
        board.state.beer_locations.insert(26);
        let mut brewery = BuiltBuilding::build(
            IndustryType::Beer,
            IndustryLevel::from_usize(0),
            26,
            PlayerId::from_usize(player_idx),
        );
        brewery.resource_amt = 1;
        board.state.bl_to_building.insert(26, brewery);
        board.state.player_building_mask[player_idx].insert(26);

        // Test basic validation - should be able to build double railroad with resources
        let can_double = board.can_double_railroad(player_idx);
        // This depends on connectivity setup, but the function should not panic
        println!("Can build double railroad: {}", can_double);
    }

    #[test]
    fn test_double_railroad_insufficient_money() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Set up rail era
        board.state.era = Era::Railroad;

        // Give player insufficient money (less than £15 required)
        board.state.players[player_idx].money = 10;

        let can_double = board.can_double_railroad(player_idx);
        assert!(
            !can_double,
            "Player should not be able to build double railroad with insufficient money"
        );
    }

    #[test]
    fn test_loan_availability() {
        let board = setup_board(4);
        let player_idx = 0;

        // Initial state - player should be able to take loan
        let can_loan = board.can_take_loan(player_idx);
        assert!(can_loan, "Player should be able to take loan initially");
    }

    #[test]
    fn test_scout_availability() {
        let board = setup_board(4);
        let player_idx = 0;

        // For scout, player needs at least 3 cards and wild cards available
        // Initial state should have wild cards available
        assert!(board.state.wild_location_cards_available > 0);
        assert!(board.state.wild_industry_cards_available > 0);

        // Player should have 8 cards initially, so should be able to scout
        let can_scout = board.can_scout(player_idx);
        // Note: Scout requires player to NOT already have both wild cards
        println!("Can scout: {}", can_scout);
    }

    #[test]
    fn test_canal_vs_rail_era_state() {
        let mut board = setup_board(4);

        // Initial era should be Canal
        assert_eq!(board.state.era, Era::Canal);

        // Switch to Railroad era
        board.state.era = Era::Railroad;
        assert_eq!(board.state.era, Era::Railroad);
    }

    #[test]
    fn test_market_state() {
        let board = setup_board(4);

        // Market should be reasonably full at game start
        assert!(
            board.state.remaining_market_coal >= 10,
            "Should have most coal in market"
        );
        assert!(
            board.state.remaining_market_iron >= 5,
            "Should have most iron in market"
        );
    }

    #[test]
    fn test_trade_post_beer_initial_state() {
        let board = setup_board(4);

        // Trade posts should have merchant tiles
        assert!(!board.state.trade_post_slots.is_empty());
    }

    #[test]
    fn test_player_network_connectivity() {
        let board = setup_board(4);

        // Initial connectivity should be set up
        // The connectivity DSU structure should be initialized
        // Each player should have their own network connectivity tracker
        assert_eq!(
            board.state.player_network_mask.len(),
            4,
            "Should have connectivity for each player"
        );
    }

    #[test]
    fn test_build_locations_initial_state() {
        let board = setup_board(4);

        // Build locations data structure should be initialized
        // Note: Some game setups may place initial buildings
        let total_bl = board.state.build_locations_occupied.count_ones();
        println!("Build locations occupied at start: {}", total_bl);

        // Resource location tracking should be initialized
        let coal_count = board.state.coal_locations.count_ones();
        let iron_count = board.state.iron_locations.count_ones();
        let beer_count = board.state.beer_locations.count_ones();

        println!(
            "Resource locations - coal: {}, iron: {}, beer: {}",
            coal_count, iron_count, beer_count
        );

        // At minimum, the data structures should exist and be usable
        assert!(total_bl <= 49, "Should not exceed max build locations");
    }

    #[test]
    fn test_first_turn_has_single_action_budget() {
        let mut runner = GameRunner::new(2, Some(1234));
        let valid_actions = runner.start_turn();
        assert!(!valid_actions.is_empty());
        assert_eq!(runner.actions_remaining_in_turn, 1);
    }

    #[test]
    fn test_second_personal_turn_has_two_actions() {
        let mut runner = GameRunner::new(2, Some(1234));

        runner.start_turn();
        let cs = runner.start_action(ActionType::Pass);
        assert!(matches!(cs, ChoiceSet::Card(_)));
        runner.apply_choice(ActionChoice::Card(0));
        runner.confirm_action().unwrap();
        runner.finish_turn_and_advance();

        runner.start_turn();
        let cs = runner.start_action(ActionType::Pass);
        assert!(matches!(cs, ChoiceSet::Card(_)));
        runner.apply_choice(ActionChoice::Card(0));
        runner.confirm_action().unwrap();
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
}
