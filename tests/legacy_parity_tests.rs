use fast_brass::actions::network::NetworkActions;
use fast_brass::board::resources::ResourceManager;
use fast_brass::board::Board;
use fast_brass::consts::{
    CANAL_PRICE, STARTING_CARDS_2P_LEN, STARTING_CARDS_3P_LEN, STARTING_CARDS_4P_LEN,
    STARTING_HAND_SIZE,
};
use fast_brass::core::building::BuiltBuilding;
use fast_brass::core::player::PlayerId;
use fast_brass::core::types::{
    ActionType, Card, CardType, Era, IndustryLevel, IndustrySet, IndustryType,
};
use fast_brass::game::framework::ActionChoice;
use fast_brass::game::runner::{GamePhase, GameRunner};
use fast_brass::locations::LocationName;
use fast_brass::validation::{BuildValidator, NetworkValidator};

#[test]
fn test_legacy_initial_setup_deck_market_and_players() {
    // Functional parity with legacy setup checks:
    // verify total deck composition per player-count and post-deal deck size.
    let cases = [
        (2usize, *STARTING_CARDS_2P_LEN, 40usize),
        (3usize, *STARTING_CARDS_3P_LEN, 54usize),
        (4usize, *STARTING_CARDS_4P_LEN, 64usize),
    ];

    for (num_players, total_cards, expected_total) in cases {
        assert_eq!(total_cards, expected_total);
        let board = Board::new(num_players, Some(42));
        let expected_deck_left = total_cards - (num_players * STARTING_HAND_SIZE as usize) - 1;

        assert_eq!(
            board.state.deck.cards_left(),
            expected_deck_left,
            "Deck size should account for hands and the face-down discard for {num_players} players"
        );
        assert_eq!(
            board.state.discard_pile.len(),
            1,
            "Setup should place one card face down in the discard pile"
        );
        assert_eq!(
            board.state.remaining_market_coal, 13,
            "Initial coal market should match legacy expectation"
        );
        assert_eq!(
            board.state.remaining_market_iron, 8,
            "Initial iron market should match legacy expectation"
        );
        assert_eq!(
            board.state.players.len(),
            num_players,
            "Player count should match game setup"
        );
    }
}

#[test]
fn test_legacy_resource_market_price_curve() {
    // These exact values mirror the legacy Python test vectors.
    assert_eq!(ResourceManager::get_coal_price(0, 4), 32);
    assert_eq!(ResourceManager::get_iron_price(0, 5), 30);

    assert_eq!(ResourceManager::get_coal_price(8, 1), 4);
    assert_eq!(ResourceManager::get_iron_price(8, 1), 2);

    assert_eq!(ResourceManager::get_coal_price(8, 3), 13);
    assert_eq!(ResourceManager::get_iron_price(8, 3), 7);

    assert_eq!(ResourceManager::get_coal_price(13, 10), 35);
    assert_eq!(ResourceManager::get_iron_price(8, 10), 40);
}

#[test]
fn test_legacy_canal_build_connectivity_and_cost() {
    let mut board = Board::new(2, Some(7));
    let player_idx = 0usize;
    let road_idx = 10usize; // Stone <-> Stafford

    assert!(
        !board
            .state
            .connectivity
            .are_towns_connected(LocationName::Stone, LocationName::Stafford),
        "Towns should start disconnected before canal build"
    );

    let money_before = board.state.players[player_idx].money;
    let res = NetworkActions::execute_build_canal_action(&mut board.state, player_idx, road_idx, 0);
    assert!(
        res.is_ok(),
        "Canal build should succeed on a canal-capable road"
    );

    assert!(
        board.state.built_roads.contains(road_idx),
        "Road should be marked built after canal action"
    );
    assert!(
        board
            .state
            .connectivity
            .are_towns_connected(LocationName::Stone, LocationName::Stafford),
        "Towns should become connected after canal build"
    );
    assert_eq!(
        board.state.players[player_idx].money,
        money_before - CANAL_PRICE,
        "Canal build should deduct legacy-expected canal price"
    );
}

#[test]
fn test_legacy_canal_to_railroad_transition_clears_roads_and_retires_level1() {
    let mut runner = GameRunner::new(2, Some(11));
    let state = &mut runner.framework.board.state;

    // Seed roads/buildings so transition effects are observable.
    state.place_link(0, 0);
    state.place_link(0, 10);

    // Coal I should be removed after canal era; Coal II should stay.
    let coal_i_bl = 27usize;
    let coal_ii_bl = 45usize;
    state.bl_to_building.insert(
        coal_i_bl,
        BuiltBuilding::build(
            IndustryType::Coal,
            IndustryLevel::I,
            coal_i_bl as u8,
            PlayerId::from_usize(0),
        ),
    );
    state.bl_to_building.insert(
        coal_ii_bl,
        BuiltBuilding::build(
            IndustryType::Coal,
            IndustryLevel::II,
            coal_ii_bl as u8,
            PlayerId::from_usize(0),
        ),
    );
    state.build_locations_occupied.insert(coal_i_bl);
    state.build_locations_occupied.insert(coal_ii_bl);
    state.player_building_mask[0].insert(coal_i_bl);
    state.player_building_mask[0].insert(coal_ii_bl);

    // Run the same transition branch used when canal phase ends.
    runner.game_phase = GamePhase::Railroad;
    runner.framework.board.state.era = Era::Railroad;
    runner.end_era();

    assert!(
        runner.framework.board.state.built_roads.is_clear(),
        "All roads should be cleared during canal->railroad transition"
    );
    assert!(
        runner.framework.board.state.player_road_mask[0].is_clear(),
        "Player road ownership should be reset during transition"
    );

    assert!(
        !runner
            .framework
            .board
            .state
            .bl_to_building
            .contains_key(&coal_i_bl),
        "Level I coal tile should retire at canal->railroad transition"
    );
    assert!(
        runner
            .framework
            .board
            .state
            .bl_to_building
            .contains_key(&coal_ii_bl),
        "Level II coal tile should remain after transition"
    );
}

#[test]
fn test_legacy_income_level_mapping_and_decrease() {
    let mut board = Board::new(2, Some(42));
    let p = &mut board.state.players[0];

    assert_eq!(p.get_income_amount(10), 0);
    assert_eq!(p.get_income_amount(19), 5);
    assert_eq!(p.get_income_amount(35), 12);
    assert_eq!(p.get_income_amount(68), 22);
    assert_eq!(p.get_income_amount(99), 30);

    p.income_level = 17;
    p.decrease_income_level(2);
    assert_eq!(
        p.income_level, 9,
        "Income-level decrement should follow current Rust ladder semantics"
    );
}

#[test]
fn test_legacy_victory_points_gain_from_flipped_building_scoring() {
    let mut runner = GameRunner::new(2, Some(42));
    let bl = 36usize; // Birmingham cotton slot
    let mut cotton = BuiltBuilding::build(
        IndustryType::Cotton,
        IndustryLevel::I,
        bl as u8,
        PlayerId::from_usize(0),
    );
    cotton.flipped = true;

    let state = &mut runner.framework.board.state;
    state.bl_to_building.insert(bl, cotton);
    state.build_locations_occupied.insert(bl);
    state.player_building_mask[0].insert(bl);

    let vp_before = state.players[0].victory_points;
    runner.end_era();
    let vp_after = runner.framework.board.state.players[0].victory_points;

    assert!(
        vp_after > vp_before,
        "Flipped building should contribute VPs during era scoring"
    );
}

#[test]
fn test_legacy_end_canal_era_refills_hands_and_switches_phase() {
    let mut runner = GameRunner::new(2, Some(42));
    runner.game_phase = GamePhase::Canal;
    runner.framework.board.state.era = Era::Canal;
    let state = &mut runner.framework.board.state;
    let mut discarded = std::mem::take(&mut state.deck.cards);
    discarded.append(&mut state.discard_pile);
    for player in &mut state.players {
        discarded.append(&mut player.hand.cards);
    }
    state.discard_pile = discarded;

    runner.end_round();

    assert_eq!(runner.game_phase, GamePhase::Railroad);
    assert_eq!(runner.framework.board.state.era, Era::Railroad);
    assert_eq!(
        runner.framework.board.state.players[0].hand.cards.len(),
        STARTING_HAND_SIZE as usize
    );
    assert_eq!(
        runner.framework.board.state.players[1].hand.cards.len(),
        STARTING_HAND_SIZE as usize
    );
    assert!(runner.framework.board.state.discard_pile.is_empty());
    assert_eq!(runner.round_in_phase, 0);
}

#[test]
fn test_final_railroad_round_skips_income() {
    let mut runner = GameRunner::new(2, Some(91));
    runner.game_phase = GamePhase::Railroad;
    runner.framework.board.state.era = Era::Railroad;

    let state = &mut runner.framework.board.state;
    state.deck.cards.clear();
    for player in &mut state.players {
        player.hand.cards.clear();
    }
    state.players[0].income_level = 20;
    let money_before = state.players[0].money;

    runner.end_round();

    assert_eq!(runner.game_phase, GamePhase::GameEnd);
    assert_eq!(runner.framework.board.state.players[0].money, money_before);
}

#[test]
fn test_round_order_uses_only_current_round_spending() {
    let mut runner = GameRunner::new(2, Some(92));
    let original_order = runner.framework.board.state.turn_order.clone();
    runner.framework.board.state.players[original_order[0]].spent_this_turn = 9;
    runner.framework.board.state.players[original_order[1]].spent_this_turn = 2;

    runner.end_round();

    assert_eq!(
        runner.framework.board.state.turn_order,
        vec![original_order[1], original_order[0]]
    );
    assert!(runner
        .framework
        .board
        .state
        .players
        .iter()
        .all(|player| player.spent_this_turn == 0));
}

#[test]
fn test_legacy_railroad_rules_double_requires_beer() {
    let mut board = Board::new(2, Some(42));
    board.state.era = Era::Railroad;
    board.state.players[0].gain_money(50);

    assert!(
        !NetworkValidator::can_double_railroad(&board.state, 0),
        "Without beer source, double railroad should be unavailable"
    );

    let brewery_bl = 47usize; // lone brewery location
    board.state.bl_to_building.insert(
        brewery_bl,
        BuiltBuilding::build(
            IndustryType::Beer,
            IndustryLevel::I,
            brewery_bl as u8,
            PlayerId::from_usize(0),
        ),
    );
    board.state.build_locations_occupied.insert(brewery_bl);
    board.state.player_building_mask[0].insert(brewery_bl);
    board.state.beer_locations.insert(brewery_bl);

    assert!(
        NetworkValidator::can_double_railroad(&board.state, 0),
        "With own beer and enough money, double railroad should be available"
    );
}

#[test]
fn test_legacy_cards_industry_card_can_be_network_gated() {
    let mut board = Board::new(2, Some(42));
    let p = 0usize;
    board.state.players[p].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Iron]),
    ))];

    let opts = BuildValidator::get_valid_build_options(&board.state, p);
    assert!(
        opts.is_empty(),
        "Legacy parity: pure industry card with no network should have no initial build options"
    );
}

#[test]
fn test_legacy_location_card_builds_exclude_occupied_slots() {
    let mut board = Board::new(2, Some(42));
    let p1 = 0usize;
    let p2 = 1usize;
    let occupied_bl = 27usize; // Coalbrookdale coal slot

    board.state.players[p1].hand.cards = vec![Card::new(CardType::Location(
        fast_brass::locations::TownName::Coalbrookdale,
    ))];
    board.state.players[p2].hand.cards = vec![Card::new(CardType::Location(
        fast_brass::locations::TownName::Coalbrookdale,
    ))];

    board.state.bl_to_building.insert(
        occupied_bl,
        BuiltBuilding::build(
            IndustryType::Coal,
            IndustryLevel::I,
            occupied_bl as u8,
            PlayerId::from_usize(p2),
        ),
    );
    board.state.build_locations_occupied.insert(occupied_bl);
    board.state.player_building_mask[p2].insert(occupied_bl);
    board.state.coal_locations.insert(occupied_bl);

    let p1_opts = BuildValidator::get_valid_build_options(&board.state, p1);
    assert!(
        !p1_opts.iter().any(|o| o.build_location_idx == occupied_bl),
        "Occupied build slot should not be offered to another player"
    );
}

#[test]
fn test_legacy_scout_action_discards_three_and_adds_two_wilds() {
    let mut runner = GameRunner::new(2, Some(42));
    let p = runner.framework.current_player;

    runner.framework.board.state.players[p].hand.cards = vec![
        Card::new(CardType::Location(
            fast_brass::locations::TownName::Birmingham,
        )),
        Card::new(CardType::Location(
            fast_brass::locations::TownName::Coventry,
        )),
        Card::new(CardType::Location(fast_brass::locations::TownName::Dudley)),
        Card::new(CardType::Location(
            fast_brass::locations::TownName::Worcester,
        )),
    ];
    let actions = runner.start_turn();
    assert!(actions.contains(&ActionType::Scout));
    let _ = runner.start_action(ActionType::Scout);
    let _ = runner.apply_choice(ActionChoice::Card(0));
    let _ = runner.apply_choice(ActionChoice::Card(1));
    let _ = runner.apply_choice(ActionChoice::Card(2));
    runner
        .confirm_action()
        .expect("Scout action should confirm");

    let hand = &runner.framework.board.state.players[p].hand.cards;
    assert_eq!(
        hand.len(),
        STARTING_HAND_SIZE as usize,
        "Scout should refill to eight after replacing three cards with two wilds"
    );
    assert!(
        hand.iter()
            .any(|c| matches!(c.card_type, CardType::WildLocation)),
        "Scout should add wild-location card"
    );
    assert!(
        hand.iter()
            .any(|c| matches!(c.card_type, CardType::WildIndustry)),
        "Scout should add wild-industry card"
    );
}

#[test]
fn test_legacy_overbuilding_restriction_on_opponent_cotton() {
    let mut board = Board::new(2, Some(42));
    let bl = 36usize; // Birmingham cotton slot

    board.state.bl_to_building.insert(
        bl,
        BuiltBuilding::build(
            IndustryType::Cotton,
            IndustryLevel::I,
            bl as u8,
            PlayerId::from_usize(1),
        ),
    );
    board.state.build_locations_occupied.insert(bl);
    board.state.player_building_mask[1].insert(bl);

    let old = board.state.bl_to_building.get(&bl).unwrap();
    assert!(
        !BuildValidator::can_overbuild(&board.state, old, 0, IndustryType::Cotton),
        "Player should not be able to overbuild opponent cotton tile"
    );
}

#[test]
fn test_legacy_all_networks_era_filtering_equivalent() {
    let mut board = Board::new(2, Some(42));
    let canal_opts = NetworkValidator::get_valid_canal_options(&board.state, 0);
    assert!(
        !canal_opts.is_empty(),
        "Canal era should expose at least some canal build options"
    );

    board.state.era = Era::Railroad;
    let rail_opts = NetworkValidator::get_valid_single_rail_options(&board.state, 0);
    assert!(
        !rail_opts.is_empty(),
        "Railroad era should expose single-rail options"
    );
}

#[test]
fn test_legacy_second_phase_equivalent_non_retired_and_rail_available() {
    let mut runner = GameRunner::new(2, Some(77));
    let p = 0usize;
    let coal_ii_bl = 45usize; // a non-retired coal level

    runner.framework.board.state.bl_to_building.insert(
        coal_ii_bl,
        BuiltBuilding::build(
            IndustryType::Coal,
            IndustryLevel::II,
            coal_ii_bl as u8,
            PlayerId::from_usize(p),
        ),
    );
    runner
        .framework
        .board
        .state
        .build_locations_occupied
        .insert(coal_ii_bl);
    runner.framework.board.state.player_building_mask[p].insert(coal_ii_bl);
    runner
        .framework
        .board
        .state
        .coal_locations
        .insert(coal_ii_bl);

    runner.game_phase = GamePhase::Railroad;
    runner.framework.board.state.era = Era::Railroad;
    runner.end_era();

    assert!(
        runner
            .framework
            .board
            .state
            .bl_to_building
            .contains_key(&coal_ii_bl),
        "Non-retired higher-level buildings should remain after canal->rail transition"
    );
    let rail_opts =
        NetworkValidator::get_valid_single_rail_options(&runner.framework.board.state, p);
    assert!(
        !rail_opts.is_empty(),
        "After railroad transition, rail options should be available"
    );
}
