// tests/board_tests.rs

// Allow access to internal components for testing
#[cfg(test)]
mod tests {
    use fast_brass::board::Board;
    use fast_brass::consts::*;
    use fast_brass::locations::TownName;
    use fast_brass::BuildOption;
    use fast_brass::Era;
    use fast_brass::{Card, CardType, IndustrySet};

    fn industry_card(industry: IndustryType) -> Card {
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            industry,
        ])))
    }

    // Helper function to create a board with a specific seed for reproducible tests
    fn setup_board(num_players: usize) -> Board {
        // Using a fixed seed ensures the deck shuffle is the same every time
        Board::new(num_players, Some(12345))
    }

    // Helper to find a specific build option in the results
    fn find_option(options: &[BuildOption], loc: usize, ind: IndustryType) -> Option<&BuildOption> {
        options
            .iter()
            .find(|opt| opt.build_location_idx == loc && opt.industry_type == ind)
    }

    #[test]
    fn test_initial_coal_build_location_card() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Give player 0 a Coalbrookdale card
        board.state.players[player_idx].hand.cards =
            vec![Card::new(CardType::Location(TownName::Coalbrookdale))];
        let card_idx = board.state.players[player_idx].hand.cards.len() - 1;

        let options = board.get_valid_build_options(player_idx);

        // Coalbrookdale locations: 25, 26, 27
        // Location 25: Iron
        // Location 26: Iron, Beer
        // Location 27: Coal
        // Expect option to build Level I Coal at loc 27
        let coal_option = find_option(&options, 27, IndustryType::Coal);

        assert!(
            coal_option.is_some(),
            "Should find option for Coal Lvl I at loc 27"
        );
        let opt = coal_option.unwrap();
        assert_eq!(opt.card_used_idx, card_idx);
        assert_eq!(opt.level.as_usize(), 0); // Level I is index 0
        assert_eq!(opt.building_data.money_cost, 5);
        assert_eq!(opt.building_data.coal_cost, 0);
        assert_eq!(opt.building_data.iron_cost, 0);
    }

    #[test]
    fn test_initial_coal_build_industry_card() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Give player 0 a Beer industry card
        board.state.players[player_idx].hand.cards = vec![industry_card(IndustryType::Beer)];

        let options = board.get_valid_build_options(player_idx);

        // With industry card and no network, should have some beer build options
        // Check lonely brewery locations which are always available for beer
        let lonely_beer_1 = find_option(&options, 47, IndustryType::Beer);
        let lonely_beer_2 = find_option(&options, 48, IndustryType::Beer);

        // At least one lonely brewery should be available
        assert!(
            lonely_beer_1.is_some() || lonely_beer_2.is_some(),
            "Should find option for Beer at lonely brewery locations"
        );
    }

    #[test]
    fn test_build_location_card_allows_build_anywhere() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Give player 0 a Tamworth location card (no network needed with location card)
        board.state.players[player_idx].hand.cards =
            vec![Card::new(CardType::Location(TownName::Tamworth))];
        let card_idx = board.state.players[player_idx].hand.cards.len() - 1;

        let options = board.get_valid_build_options(player_idx);

        // Tamworth locations 6, 7
        // Loc 6: Coal
        // Loc 7: Cotton, Coal
        // Should be able to build Coal at loc 6 using Tamworth card, even with no network
        let coal_option = find_option(&options, 6, IndustryType::Coal);
        assert!(
            coal_option.is_some(),
            "Should find option for Coal Lvl I at loc 6 with location card"
        );
        assert_eq!(coal_option.unwrap().card_used_idx, card_idx);
    }

    #[test]
    fn test_insufficient_funds_blocks_expensive_builds() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Set player money low (only 5)
        board.state.players[player_idx].money = 5;

        // Give player 0 a Coalbrookdale card
        board.state.players[player_idx].hand.cards =
            vec![Card::new(CardType::Location(TownName::Coalbrookdale))];

        let options = board.get_valid_build_options(player_idx);

        // Beer at loc 26 costs money + iron from market, should be too expensive
        let beer_option = find_option(&options, 26, IndustryType::Beer);
        assert!(
            beer_option.is_none(),
            "Should not have option to build beer with insufficient funds"
        );
    }

    #[test]
    fn test_board_creation() {
        let board = setup_board(4);

        // Check that board was created with correct number of players
        assert_eq!(board.state.players.len(), 4);

        // Check initial state
        assert_eq!(board.state.era, Era::Canal);

        // Market should be mostly full (some may be consumed by initial iron works)
        assert!(
            board.state.remaining_market_coal >= 10,
            "Market should have coal"
        );
        assert!(
            board.state.remaining_market_iron >= 5,
            "Market should have iron"
        );

        // Each player should have cards
        for player in &board.state.players {
            assert!(
                !player.hand.cards.is_empty(),
                "Players should have starting cards"
            );
        }
    }

    #[test]
    fn test_get_coal_price() {
        let board = setup_board(4);

        // Price depends on how much coal is in the market
        // Higher market = cheaper prices (COAL_PRICE_TABLE starts from expensive at index 0)
        let coal_price = board.get_coal_price(1);
        assert!(
            coal_price > 0 && coal_price <= 7,
            "Coal price should be between 1-7"
        );

        // Price for multiple cubes
        let price_for_2 = board.get_coal_price(2);
        assert!(
            price_for_2 >= coal_price,
            "Price for 2 cubes should be at least price for 1"
        );
    }

    #[test]
    fn test_get_iron_price() {
        let board = setup_board(4);

        // Price depends on how much iron is in the market
        // Higher market = cheaper prices (IRON_PRICE_TABLE starts from expensive at index 0)
        let iron_price = board.get_iron_price(1);
        assert!(
            iron_price > 0 && iron_price <= 5,
            "Iron price should be between 1-5"
        );

        // Price for multiple cubes
        let price_for_2 = board.get_iron_price(2);
        assert!(
            price_for_2 >= iron_price,
            "Price for 2 cubes should be at least price for 1"
        );
    }
}
