use fast_brass::board::BoardState;
use fast_brass::core::building::BuiltBuilding;
use fast_brass::core::player::PlayerId;
use fast_brass::core::types::{Era, IndustryLevel, IndustryType};
use fast_brass::validation::{BuildValidator, DevelopValidator, NetworkValidator, SellValidator};

/// Helper function to create a fresh board state for testing
fn create_test_board(num_players: usize, seed: u64) -> BoardState {
    BoardState::new(num_players, Some(seed))
}

// =========== BUILD VALIDATION TESTS ===========

#[test]
fn test_build_validator_initial_state() {
    let board = create_test_board(2, 12345);

    // At the start of the game, players should have some build options available
    let build_options = BuildValidator::get_valid_build_options(&board, 0);

    // Players have cards and money, so should have some valid build options
    // The exact number depends on starting hand, but should be > 0
    assert!(
        !build_options.is_empty(),
        "Players should have build options at game start"
    );
}

#[test]
fn test_build_validator_can_build() {
    let board = create_test_board(2, 12345);

    // At game start, players should be able to build something
    let can_build = BuildValidator::can_build(&board, 0);
    assert!(can_build, "Player should be able to build at game start");
}

#[test]
fn test_build_validator_respects_player_money() {
    let mut board = create_test_board(2, 12345);

    // Drain player's money
    let player = &mut board.players[0];
    let current_money = player.money;
    player.pay(current_money);

    // With no money, build options should be limited to free buildings or none
    let build_options = BuildValidator::get_valid_build_options(&board, 0);

    // All options should have zero cost if player has no money
    for opt in &build_options {
        assert_eq!(
            opt.total_money_cost, 0,
            "No-money player should only have free build options"
        );
    }
}

// =========== SELL VALIDATION TESTS ===========

#[test]
fn test_sell_validator_empty_at_start() {
    let board = create_test_board(2, 12345);

    // At game start, no buildings on board, so no sell options
    let sell_options = SellValidator::get_valid_sell_options(&board, 0);

    assert!(
        sell_options.is_empty(),
        "No sell options should exist at game start (no buildings)"
    );
}

// =========== DEVELOP VALIDATION TESTS ===========

#[test]
fn test_develop_validator_initial_state() {
    let board = create_test_board(2, 12345);

    // Get valid development options
    let dev_options = DevelopValidator::get_valid_development_options(&board, 0);

    // At game start, there's iron in the market (8 cubes), so development should be possible
    // if player has money and developable tiles
    // Note: Some industries might not be developable at level 1 (e.g., pottery has lightbulb)
    println!(
        "Development options available: {}",
        dev_options.count_ones(..)
    );
}

#[test]
fn test_develop_validator_can_develop() {
    let board = create_test_board(2, 12345);

    let can_develop = DevelopValidator::can_develop(&board, 0);
    // At game start with money and iron available, should be able to develop
    println!("Can develop at start: {}", can_develop);
}

#[test]
fn test_develop_validator_iron_sources() {
    let board = create_test_board(2, 12345);

    // Get iron sources for 1 iron
    let iron_sources = DevelopValidator::get_iron_sources_for_develop(&board, 0, 1);

    // At game start, iron market has cubes, so market should be a source
    // (no iron works on board yet)
    assert!(
        !iron_sources.is_empty(),
        "Should have iron sources available at game start"
    );
}

#[test]
fn test_develop_validator_iron_cost() {
    // Test iron cost calculation
    assert_eq!(DevelopValidator::get_iron_cost_for_develop(1), 1);
    assert_eq!(DevelopValidator::get_iron_cost_for_develop(2), 2);
    assert_eq!(DevelopValidator::get_iron_cost_for_develop(0), 0);
}

// =========== NETWORK VALIDATION TESTS ===========

#[test]
fn test_network_validator_canal_options_in_canal_era() {
    let board = create_test_board(2, 12345);

    // Board starts in Canal era
    assert_eq!(board.era, Era::Canal, "Board should start in Canal era");

    // Get valid canal options
    let canal_options = NetworkValidator::get_valid_canal_options(&board, 0);

    // At game start, player has no network, so any canal should be buildable
    // (if player can afford it)
    println!("Canal options available: {}", canal_options.len());
}

#[test]
fn test_network_validator_rail_options_in_canal_era() {
    let board = create_test_board(2, 12345);

    // In canal era, rail options should be empty
    let rail_options = NetworkValidator::get_valid_single_rail_options(&board, 0);

    assert!(
        rail_options.is_empty(),
        "No rail options should be available in canal era"
    );
}

#[test]
fn test_network_validator_double_rail_in_canal_era() {
    let board = create_test_board(2, 12345);

    // In canal era, double rail should not be possible
    let can_double_rail = NetworkValidator::can_double_railroad(&board, 0);

    assert!(
        !can_double_rail,
        "Cannot build double railroad in canal era"
    );
}

#[test]
fn test_network_validator_canal_affordability() {
    let mut board = create_test_board(2, 12345);

    // Drain player's money
    let player = &mut board.players[0];
    let current_money = player.money;
    player.pay(current_money);

    // With no money, canal options should be empty
    let canal_options = NetworkValidator::get_valid_canal_options(&board, 0);

    assert!(
        canal_options.is_empty(),
        "Player with no money should have no canal options"
    );
}

// =========== INTEGRATION TESTS ===========

#[test]
fn test_all_validators_work_on_fresh_board() {
    let board = create_test_board(4, 54321);

    // Test all validators can run without panicking on a fresh 4-player board
    for player_idx in 0..4 {
        let _build = BuildValidator::get_valid_build_options(&board, player_idx);
        let _sell = SellValidator::get_valid_sell_options(&board, player_idx);
        let _develop = DevelopValidator::get_valid_development_options(&board, player_idx);
        let _canal = NetworkValidator::get_valid_canal_options(&board, player_idx);
        let _rail = NetworkValidator::get_valid_single_rail_options(&board, player_idx);
        let _double = NetworkValidator::can_double_railroad(&board, player_idx);
    }
}

#[test]
fn test_validators_with_different_player_counts() {
    // Test with 2, 3, and 4 players
    for num_players in 2..=4 {
        let board = create_test_board(num_players, 99999);

        for player_idx in 0..num_players {
            // These should not panic
            let _ = BuildValidator::get_valid_build_options(&board, player_idx);
            let _ = SellValidator::get_valid_sell_options(&board, player_idx);
            let _ = DevelopValidator::get_valid_development_options(&board, player_idx);
            let _ = NetworkValidator::get_valid_canal_options(&board, player_idx);
        }
    }
}

/// Regression test: SellValidator must not panic when checking trade-post
/// connectivity for a sellable building. Previously, `TradePost` enum values
/// (starting at NUM_BL=49) were passed directly to `LocationName::from_usize`
/// which only accepts 0-26, causing a panic on index 50 (Oxford).
#[test]
fn test_sell_validator_does_not_panic_with_building_on_board() {
    let mut board = create_test_board(2, 42);
    let player_idx = 0;
    let bl_idx = 36; // Birmingham Cotton/Goods slot

    let building = BuiltBuilding::build(
        IndustryType::Cotton,
        IndustryLevel::I,
        bl_idx as u8,
        PlayerId::from_usize(player_idx),
    );
    board.bl_to_building.insert(bl_idx, building);
    board.build_locations_occupied.insert(bl_idx);
    board.player_building_mask[player_idx].insert(bl_idx);

    // This used to panic with "Invalid location index: 50" because
    // TradePost::Oxford (value 50) was cast to usize and passed to
    // LocationName::from_usize instead of using to_location_name().
    let sell_options = SellValidator::get_valid_sell_options(&board, player_idx);

    // Cotton I is unflipped, so the validator should process it without panic.
    // Whether sell options exist depends on merchant placement / connectivity,
    // but the key assertion is no panic.
    let _ = sell_options;
}
