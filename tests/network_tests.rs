use fast_brass::actions::network::{NetworkActions, NetworkError};
use fast_brass::board::resources::{BreweryBeerSource, ResourceSource};
use fast_brass::board::BoardState;
use fast_brass::core::building::BuiltBuilding;
use fast_brass::core::player::PlayerId;
use fast_brass::core::static_data::{road_label, CANAL_ONLY, LINK_LOCATIONS, RAIL_ONLY};
use fast_brass::core::types::*;
use fast_brass::game::framework::{ActionChoice, ChoiceSet};
use fast_brass::game::runner::GameRunner;
use fast_brass::validation::NetworkValidator;

fn create_board(num_players: usize, seed: u64) -> BoardState {
    BoardState::new(num_players, Some(seed))
}

// Road 0:  Warrington–StokeOnTrent      canal=true  rail=true
// Road 2:  Leek–Belper                  canal=false rail=true   (rail-only)
// Road 15: Walsall–BurtonUponTrent      canal=true  rail=false  (canal-only)
// Road 10: Stone–Stafford               canal=true  rail=true

// =========================================================================
// Static data sanity checks
// =========================================================================

#[test]
fn test_canal_only_contains_all_canal_capable_roads() {
    for (idx, link) in LINK_LOCATIONS.iter().enumerate() {
        assert_eq!(
            CANAL_ONLY.contains(idx),
            link.can_build_canal,
            "CANAL_ONLY mismatch at road {idx}"
        );
    }
}

#[test]
fn test_rail_only_contains_all_rail_capable_roads() {
    for (idx, link) in LINK_LOCATIONS.iter().enumerate() {
        assert_eq!(
            RAIL_ONLY.contains(idx),
            link.can_build_rail,
            "RAIL_ONLY mismatch at road {idx}"
        );
    }
}

#[test]
fn test_road_15_is_canal_only() {
    assert!(CANAL_ONLY.contains(15));
    assert!(!RAIL_ONLY.contains(15));
}

#[test]
fn test_road_2_is_rail_only() {
    assert!(!CANAL_ONLY.contains(2));
    assert!(RAIL_ONLY.contains(2));
}

// =========================================================================
// Validation: get_valid_road_options era filtering
// =========================================================================

#[test]
fn test_canal_era_excludes_rail_only_roads() {
    let board = create_board(2, 42);
    let valid = NetworkValidator::get_valid_road_options(&board, 0, Era::Canal);
    let rail_only_indices = [2, 5, 9, 13, 25, 27, 29, 32];
    for &idx in &rail_only_indices {
        assert!(
            !valid.contains(idx),
            "Rail-only road {idx} should not appear in canal era options"
        );
    }
}

#[test]
fn test_canal_era_includes_canal_capable_roads() {
    let board = create_board(2, 42);
    let valid = NetworkValidator::get_valid_road_options(&board, 0, Era::Canal);
    // Road 0 (Warrington–StokeOnTrent) is canal+rail, must appear
    assert!(
        valid.contains(0),
        "Canal-capable road 0 should appear in canal era"
    );
    // Road 15 (canal-only) must appear
    assert!(
        valid.contains(15),
        "Canal-only road 15 should appear in canal era"
    );
}

#[test]
fn test_railroad_era_excludes_canal_only_road() {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    let valid = NetworkValidator::get_valid_road_options(&board, 0, Era::Railroad);
    assert!(
        !valid.contains(15),
        "Canal-only road 15 should not appear in railroad era"
    );
}

#[test]
fn test_railroad_era_includes_rail_capable_roads() {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    let valid = NetworkValidator::get_valid_road_options(&board, 0, Era::Railroad);
    assert!(
        valid.contains(2),
        "Rail-only road 2 should appear in railroad era"
    );
    assert!(
        valid.contains(0),
        "Canal+rail road 0 should appear in railroad era"
    );
}

#[test]
fn test_already_built_roads_excluded() {
    let mut board = create_board(2, 42);
    board.place_link(0, 0); // build road 0
    let valid = NetworkValidator::get_valid_road_options(&board, 0, Era::Canal);
    assert!(
        !valid.contains(0),
        "Already-built road 0 should be excluded"
    );
}

// =========================================================================
// Validation: get_valid_canal_options
// =========================================================================

#[test]
fn test_canal_options_empty_in_railroad_era() {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    let opts = NetworkValidator::get_valid_canal_options(&board, 0);
    assert!(opts.is_empty());
}

#[test]
fn test_canal_options_nonempty_at_game_start() {
    let board = create_board(2, 42);
    let opts = NetworkValidator::get_valid_canal_options(&board, 0);
    assert!(
        opts.len() > 20,
        "Fresh game should have many canal options, got {}",
        opts.len()
    );
}

#[test]
fn test_canal_options_empty_when_broke() {
    let mut board = create_board(2, 42);
    let money = board.players[0].money;
    board.players[0].pay(money);
    let opts = NetworkValidator::get_valid_canal_options(&board, 0);
    assert!(opts.is_empty(), "Broke player should have no canal options");
}

#[test]
fn test_canal_options_never_contain_rail_only_road() {
    let board = create_board(4, 99);
    let opts = NetworkValidator::get_valid_canal_options(&board, 0);
    for &road_idx in &opts {
        assert!(
            CANAL_ONLY.contains(road_idx),
            "Canal option road {road_idx} is not canal-capable"
        );
    }
}

// =========================================================================
// Validation: no-network player can build anywhere
// =========================================================================

#[test]
fn test_no_network_player_gets_all_canal_roads() {
    let board = create_board(2, 42);
    let opts = NetworkValidator::get_valid_canal_options(&board, 0);
    let expected_count = CANAL_ONLY.count_ones();
    assert_eq!(
        opts.len(),
        expected_count,
        "No-network player should see all {} canal-capable roads, got {}",
        expected_count,
        opts.len()
    );
}

// =========================================================================
// Validation: get_valid_single_rail_options
// =========================================================================

#[test]
fn test_rail_options_empty_in_canal_era() {
    let board = create_board(2, 42);
    let opts = NetworkValidator::get_valid_single_rail_options(&board, 0);
    assert!(opts.is_empty(), "No rail options should exist in canal era");
}

// =========================================================================
// Validation: can_double_railroad
// =========================================================================

#[test]
fn test_double_railroad_impossible_in_canal_era() {
    let board = create_board(2, 42);
    assert!(!NetworkValidator::can_double_railroad(&board, 0));
}

#[test]
fn test_double_railroad_impossible_without_beer() {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    // no breweries on the board → no beer → can't double rail
    assert!(!NetworkValidator::can_double_railroad(&board, 0));
}

// =========================================================================
// Action execution: build_canal
// =========================================================================

#[test]
fn test_execute_canal_on_canal_capable_road_succeeds() {
    let mut board = create_board(2, 42);
    let money_before = board.players[0].money;
    let result = NetworkActions::execute_build_canal_action(&mut board, 0, 0, 0);
    assert!(result.is_ok(), "Building canal on road 0 should succeed");
    assert!(board.built_roads.contains(0));
    assert!(board.player_road_mask[0].contains(0));
    assert_eq!(board.players[0].money, money_before - CANAL_PRICE);
}

#[test]
fn test_execute_canal_on_rail_only_road_fails() {
    let mut board = create_board(2, 42);
    let result = NetworkActions::execute_build_canal_action(&mut board, 0, 2, 0);
    assert_eq!(result, Err(NetworkError::InvalidCanalBuild));
    assert!(!board.built_roads.contains(2));
}

#[test]
fn test_execute_canal_on_already_built_road_fails() {
    let mut board = create_board(2, 42);
    board.place_link(0, 10); // pre-build road 10
    let result = NetworkActions::execute_build_canal_action(&mut board, 0, 10, 0);
    assert_eq!(result, Err(NetworkError::InvalidCanalBuild));
}

#[test]
fn test_execute_canal_in_railroad_era_fails() {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    let result = NetworkActions::execute_build_canal_action(&mut board, 0, 0, 0);
    assert_eq!(result, Err(NetworkError::InvalidCanalBuild));
}

#[test]
fn test_execute_canal_discards_card() {
    let mut board = create_board(2, 42);
    let hand_before = board.players[0].hand.cards.len();
    let _ = NetworkActions::execute_build_canal_action(&mut board, 0, 0, 0);
    assert_eq!(board.players[0].hand.cards.len(), hand_before - 1);
}

#[test]
fn test_execute_canal_on_canal_only_road_succeeds() {
    let mut board = create_board(2, 42);
    let result = NetworkActions::execute_build_canal_action(&mut board, 0, 15, 0);
    assert!(
        result.is_ok(),
        "Building canal on canal-only road 15 should succeed"
    );
    assert!(board.built_roads.contains(15));
}

// =========================================================================
// Action execution: build_single_rail
// =========================================================================

fn setup_rail_board() -> BoardState {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    board
}

#[test]
fn test_execute_rail_on_rail_capable_road_succeeds() {
    let mut board = setup_rail_board();
    let money_before = board.players[0].money;
    let result = NetworkActions::execute_build_single_rail_action(
        &mut board,
        0,
        0,
        ResourceSource::Market,
        0,
    );
    assert!(result.is_ok(), "Rail build on road 0 should succeed");
    assert!(board.built_roads.contains(0));
    assert!(board.player_road_mask[0].contains(0));
    assert!(board.players[0].money < money_before);
}

#[test]
fn test_execute_rail_on_canal_only_road_fails() {
    let mut board = setup_rail_board();
    let result = NetworkActions::execute_build_single_rail_action(
        &mut board,
        0,
        15,
        ResourceSource::Market,
        0,
    );
    assert_eq!(result, Err(NetworkError::InvalidRailPlacement));
    assert!(!board.built_roads.contains(15));
}

#[test]
fn test_execute_rail_on_already_built_road_fails() {
    let mut board = setup_rail_board();
    board.place_link(0, 3);
    let result = NetworkActions::execute_build_single_rail_action(
        &mut board,
        0,
        3,
        ResourceSource::Market,
        0,
    );
    assert_eq!(result, Err(NetworkError::InvalidRailPlacement));
}

#[test]
fn test_execute_rail_in_canal_era_fails() {
    let mut board = create_board(2, 42);
    let result = NetworkActions::execute_build_single_rail_action(
        &mut board,
        0,
        0,
        ResourceSource::Market,
        0,
    );
    assert_eq!(result, Err(NetworkError::InvalidRailPlacement));
}

#[test]
fn test_execute_rail_on_rail_only_road_succeeds() {
    let mut board = setup_rail_board();
    let result = NetworkActions::execute_build_single_rail_action(
        &mut board,
        0,
        2,
        ResourceSource::Market,
        0,
    );
    assert!(
        result.is_ok(),
        "Rail build on rail-only road 2 should succeed"
    );
    assert!(board.built_roads.contains(2));
}

// =========================================================================
// Action execution: build_double_rail
// =========================================================================

fn setup_double_rail_board() -> BoardState {
    let mut board = create_board(2, 42);
    board.era = Era::Railroad;
    board.players[0].gain_money(50); // ensure enough for £15 + 2× coal from market
                                     // Place a brewery with beer for player 0 at BL 0 (Stafford)
    let brewery = BuiltBuilding::build(
        IndustryType::Beer,
        IndustryLevel::I,
        0,
        PlayerId::from_usize(0),
    );
    board.bl_to_building.insert(0, brewery);
    board.beer_locations.insert(0);
    board.player_building_mask[0].insert(0);
    board.build_locations_occupied.insert(0);
    board
}

#[test]
fn test_execute_double_rail_succeeds() {
    let mut board = setup_double_rail_board();
    let money_before = board.players[0].money;
    let result = NetworkActions::execute_build_double_rail_action(
        &mut board,
        0,
        0, // road 0
        1, // road 1
        ResourceSource::Market,
        ResourceSource::Market,
        BreweryBeerSource::OwnBrewery(0),
        0,
    );
    assert!(
        result.is_ok(),
        "Double rail build should succeed: {:?}",
        result
    );
    assert!(board.built_roads.contains(0));
    assert!(board.built_roads.contains(1));
    assert!(board.players[0].money < money_before);
}

#[test]
fn test_execute_double_rail_no_beer_fails() {
    let mut board = setup_rail_board();
    // No brewery on the board
    let result = NetworkActions::execute_build_double_rail_action(
        &mut board,
        0,
        0,
        1,
        ResourceSource::Market,
        ResourceSource::Market,
        BreweryBeerSource::OwnBrewery(99), // non-existent
        0,
    );
    assert_eq!(result, Err(NetworkError::MissingActionBeer));
    // Money should be refunded
    assert_eq!(board.players[0].money, 17);
}

// =========================================================================
// Connectivity: placing link updates connectivity
// =========================================================================

#[test]
fn test_place_link_updates_connectivity() {
    let mut board = create_board(2, 42);
    // Road 0 connects Warrington (25) and StokeOnTrent (6)
    board.place_link(0, 0);
    assert!(board.connectivity.are_towns_connected(
        fast_brass::core::locations::LocationName::Warrington,
        fast_brass::core::locations::LocationName::StokeOnTrent,
    ));
}

#[test]
fn test_place_link_updates_player_network() {
    let mut board = create_board(2, 42);
    board.place_link(0, 0);
    assert!(board.player_road_mask[0].contains(0));
    assert!(board.player_network_mask[0].are_towns_connected(
        fast_brass::core::locations::LocationName::Warrington,
        fast_brass::core::locations::LocationName::StokeOnTrent,
    ));
}

// =========================================================================
// Validation: network-restricted roads after building
// =========================================================================

#[test]
fn test_player_with_building_gets_adjacent_roads() {
    let mut board = create_board(2, 42);
    // Place a building for player 0 at BL 0 (Stafford)
    let building = BuiltBuilding::build(
        IndustryType::Coal,
        IndustryLevel::I,
        0,
        PlayerId::from_usize(0),
    );
    board.bl_to_building.insert(0, building);
    board.player_building_mask[0].insert(0);
    board.build_locations_occupied.insert(0);
    board.coal_locations.insert(0);

    let opts = NetworkValidator::get_valid_canal_options(&board, 0);
    // Stafford is connected to roads 10 (Stone–Stafford) and 12 (Stafford–Cannock)
    assert!(
        opts.contains(&10),
        "Should include road 10 (Stone–Stafford)"
    );
    assert!(
        opts.contains(&12),
        "Should include road 12 (Stafford–Cannock)"
    );
    // Should NOT include a distant road like 37 (Worcester–Gloucester)
    assert!(
        !opts.contains(&37),
        "Should not include distant road 37 without connectivity"
    );
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_all_canal_options_do_not_contain_rail_only_roads() {
    let board = create_board(4, 42);
    let opts = NetworkValidator::get_valid_canal_options(&board, 0);

    // All rail-only road indices from LINK_LOCATIONS (can_build_canal=false, can_build_rail=true)
    let rail_only_roads: [usize; 8] = [
        2,  // Leek–Belper
        5,  // Derby–Uttoxeter
        9,  // Stone–Uttoxeter
        13, // Cannock–BurtonUponTrent
        25, // Tamworth–Walsall
        27, // Nuneaton–Coventry
        29, // Birmingham–Nuneaton
        32, // Birmingham–Redditch
    ];

    for &road_idx in &opts {
        assert!(
            !rail_only_roads.contains(&road_idx),
            "Canal option {} is rail-only",
            road_idx
        );
    }
}

#[test]
fn test_all_rail_options_do_not_contain_canal_only_roads() {
    let mut board = create_board(4, 42);
    board.era = Era::Railroad;
    let opts = NetworkValidator::get_valid_single_rail_options(&board, 0);

    // Road 15 (Walsall–BurtonUponTrent) is the only canal-only road
    let canal_only_road: usize = 15;
    assert!(
        LINK_LOCATIONS[canal_only_road].can_build_canal,
        "Road 15 should be canal-capable"
    );
    assert!(
        !LINK_LOCATIONS[canal_only_road].can_build_rail,
        "Road 15 should NOT be rail-capable"
    );

    for opt in &opts {
        assert!(
            opt.road_idx != canal_only_road,
            "Rail option {} is canal-only (Walsall-BurtonUponTrent)",
            opt.road_idx
        );
    }
}

#[test]
fn test_all_canal_options_are_canal_capable_across_seeds() {
    for seed in 0..10 {
        let board = create_board(4, seed);
        let opts = NetworkValidator::get_valid_canal_options(&board, 0);
        for &road_idx in &opts {
            assert!(
                LINK_LOCATIONS[road_idx].can_build_canal,
                "Seed {seed}: canal option {road_idx} is not canal-capable"
            );
        }
    }
}

#[test]
fn test_all_rail_options_are_rail_capable_across_seeds() {
    for seed in 0..10 {
        let mut board = create_board(4, seed);
        board.era = Era::Railroad;
        let opts = NetworkValidator::get_valid_single_rail_options(&board, 0);
        for opt in &opts {
            assert!(
                LINK_LOCATIONS[opt.road_idx].can_build_rail,
                "Seed {seed}: rail option {} is not rail-capable",
                opt.road_idx
            );
        }
    }
}

// =========================================================================
// End-to-end: canal build through full GameRunner pipeline
// =========================================================================

/// Tests the full pipeline: GameRunner → Framework → Validation → Execution
/// for canal building in Canal era. Every road offered as a choice must:
///   1) be canal-capable (can_build_canal == true)
///   2) not be rail-only
///   3) be confirmable without error
#[test]
fn test_canal_build_e2e_all_offered_roads_are_valid_and_confirmable() {
    for seed in 0..5 {
        let mut runner = GameRunner::new(2, Some(seed));
        runner.start_turn();

        let cs = runner.start_action(ActionType::BuildRailroad);

        // First choice should be a card
        let road_opts = match cs {
            ChoiceSet::Card(ref cards) => {
                assert!(!cards.is_empty(), "Seed {seed}: should have cards");
                let cs2 = runner.apply_choice(ActionChoice::Card(cards[0]));
                match cs2 {
                    Some(ChoiceSet::Road(opts)) => opts,
                    other => panic!(
                        "Seed {seed}: expected Road choices after card, got {:?}",
                        other
                    ),
                }
            }
            ChoiceSet::Road(opts) => opts,
            other => panic!("Seed {seed}: unexpected first choice set {:?}", other),
        };

        // Verify every offered road is canal-capable
        for &road_idx in &road_opts {
            assert!(
                LINK_LOCATIONS[road_idx].can_build_canal,
                "Seed {seed}: offered {} but it is NOT canal-capable",
                road_label(road_idx),
            );
            assert!(
                !RAIL_ONLY.contains(road_idx) || CANAL_ONLY.contains(road_idx),
                "Seed {seed}: offered {} which is rail-only",
                road_label(road_idx),
            );
        }

        // Pick the first offered road and confirm the whole action succeeds
        let road_to_build = road_opts[0];
        let cs3 = runner.apply_choice(ActionChoice::Road(road_to_build));
        assert!(
            matches!(cs3, Some(ChoiceSet::ConfirmOnly)),
            "Seed {seed}: after choosing road {}, expected ConfirmOnly, got {:?}",
            road_label(road_to_build),
            cs3,
        );

        let result = runner.confirm_action();
        assert!(
            result.is_ok(),
            "Seed {seed}: confirm canal build on {} failed: {:?}",
            road_label(road_to_build),
            result.err(),
        );

        // Verify road was actually built
        assert!(
            runner
                .framework
                .board
                .state
                .built_roads
                .contains(road_to_build),
            "Seed {seed}: road {} should be marked as built after confirm",
            road_label(road_to_build),
        );
    }
}

/// Verify that when a road is offered in canal era and built, the execution
/// doesn't reject it. This test tries EVERY offered road for a no-network player.
#[test]
fn test_canal_build_every_offered_road_executes_successfully() {
    let board = create_board(2, 42);
    let canal_opts = NetworkValidator::get_valid_canal_options(&board, 0);

    for &road_idx in &canal_opts {
        let mut fresh_board = create_board(2, 42);
        let result = NetworkActions::execute_build_canal_action(&mut fresh_board, 0, road_idx, 0);
        assert!(
            result.is_ok(),
            "Execution rejected canal build on {} which was offered as valid: {:?}",
            road_label(road_idx),
            result.err(),
        );
    }
}
