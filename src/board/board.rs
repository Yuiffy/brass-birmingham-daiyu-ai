use fixedbitset::FixedBitSet;

// Refactored Board that uses the new modular structure
use crate::actions::*;
use crate::board::state::BoardState;
use crate::core::building::BuiltBuilding;
use crate::core::types::*;
use crate::validation::*;
use crate::Player;

/// Main Board struct that orchestrates all game operations
/// This is a wrapper around BoardState that provides high-level game operations
pub struct Board {
    pub state: BoardState,
}

impl Board {
    pub fn new(num_players: usize, seed: Option<u64>) -> Self {
        Self {
            state: BoardState::new(num_players, seed),
        }
    }

    // === Action Execution Methods ===

    /// Execute a building action using the sources from the BuildOption
    pub fn execute_build(
        &mut self,
        player_idx: usize,
        build_opt: BuildOption,
    ) -> Result<(), crate::actions::build::BuildError> {
        BuildActions::execute(&mut self.state, player_idx, build_opt)
    }

    /// Execute a building action with explicitly chosen resource sources
    pub fn build_building(
        &mut self,
        player_idx: usize,
        build_opt: BuildOption,
        chosen_coal_sources: Vec<crate::board::resources::ResourceSource>,
        chosen_iron_sources: Vec<crate::board::resources::ResourceSource>,
    ) -> Result<(), crate::actions::build::BuildError> {
        BuildActions::execute_with_sources(
            &mut self.state,
            player_idx,
            build_opt,
            chosen_coal_sources,
            chosen_iron_sources,
        )
    }

    /// Execute selling buildings
    pub fn sell_all_buildings(
        &mut self,
        player_idx: usize,
        sell_choices: Vec<SellChoice>,
        free_dev_choice: Option<IndustryType>,
    ) -> Result<(), crate::actions::sell::SellError> {
        SellActions::execute_sell_all_buildings(
            &mut self.state,
            player_idx,
            sell_choices,
            free_dev_choice,
        )
    }

    /// Execute development action
    pub fn develop_action(
        &mut self,
        player_idx: usize,
        industries: Vec<IndustryType>,
        iron_sources: Vec<crate::board::resources::ResourceSource>,
    ) -> Result<(), crate::actions::develop::DevelopError> {
        DevelopActions::execute_develop_action(
            &mut self.state,
            player_idx,
            industries,
            iron_sources,
        )
    }

    /// Execute canal building
    pub fn build_canal_action(
        &mut self,
        player_idx: usize,
        road_idx: usize,
        card_to_discard_idx: usize,
    ) -> Result<(), crate::actions::network::NetworkError> {
        NetworkActions::execute_build_canal_action(
            &mut self.state,
            player_idx,
            road_idx,
            card_to_discard_idx,
        )
    }

    /// Execute single railroad building
    pub fn build_single_rail_action(
        &mut self,
        player_idx: usize,
        road_idx: usize,
        coal_source: crate::board::resources::ResourceSource,
        card_to_discard_idx: usize,
    ) -> Result<(), crate::actions::network::NetworkError> {
        NetworkActions::execute_build_single_rail_action(
            &mut self.state,
            player_idx,
            road_idx,
            coal_source,
            card_to_discard_idx,
        )
    }

    /// Execute double railroad building
    pub fn build_double_rail_action(
        &mut self,
        player_idx: usize,
        road1_idx: usize,
        road2_idx: usize,
        coal_source1: crate::board::resources::ResourceSource,
        coal_source2: crate::board::resources::ResourceSource,
        action_beer_source: crate::board::resources::BreweryBeerSource,
        card_to_discard_idx: usize,
    ) -> Result<(), crate::actions::network::NetworkError> {
        NetworkActions::execute_build_double_rail_action(
            &mut self.state,
            player_idx,
            road1_idx,
            road2_idx,
            coal_source1,
            coal_source2,
            action_beer_source,
            card_to_discard_idx,
        )
    }

    /// Execute loan action
    pub fn loan_action(&mut self, player_idx: usize, card_to_discard_idx: usize) {
        SpecialActions::execute_loan_action(&mut self.state, player_idx, card_to_discard_idx);
    }

    /// Execute scout action
    pub fn scout_action(
        &mut self,
        player_idx: usize,
        card_action_idx: usize,
        card_add1_idx: usize,
        card_add2_idx: usize,
    ) {
        SpecialActions::execute_scout_action(
            &mut self.state,
            player_idx,
            card_action_idx,
            card_add1_idx,
            card_add2_idx,
        );
    }

    // === Validation Methods ===

    /// Get valid build options for a player
    pub fn get_valid_build_options(&self, player_idx: usize) -> Vec<BuildOption> {
        BuildValidator::get_valid_build_options(&self.state, player_idx)
    }

    /// Public view of the shared discard pile.
    pub fn public_discard_pile(&self) -> &[Card] {
        &self.state.discard_pile
    }

    /// Public view of a player's industry mat.
    pub fn public_player_industry_mat(
        &self,
        player_idx: usize,
    ) -> Option<&crate::core::industry_mat::PlayerIndustryMat> {
        self.state.players.get(player_idx).map(|p| &p.industry_mat)
    }

    /// Get valid sell options for a player
    pub fn get_valid_sell_options(&self, player_idx: usize) -> Vec<SellOption> {
        SellValidator::get_valid_sell_options(&self.state, player_idx)
    }

    /// Get valid development options for a player
    pub fn get_valid_development_options(&self, player_idx: usize) -> FixedBitSet {
        DevelopValidator::get_valid_development_options(&self.state, player_idx)
    }

    /// Get valid industries for free development bonuses (no iron payment required).
    pub fn get_valid_free_development_options(&self, player_idx: usize) -> FixedBitSet {
        DevelopValidator::get_valid_free_development_options(&self.state, player_idx)
    }

    /// Get valid canal options for a player
    pub fn get_valid_canal_options(&self, player_idx: usize) -> Vec<usize> {
        NetworkValidator::get_valid_canal_options(&self.state, player_idx)
    }

    /// Get valid single rail options for a player
    pub fn get_valid_single_rail_options(&self, player_idx: usize) -> Vec<SingleRailroadOption> {
        NetworkValidator::get_valid_single_rail_options(&self.state, player_idx)
    }

    /// Get valid double rail first link options for a player
    pub fn get_valid_double_rail_first_link_options(
        &self,
        player_idx: usize,
    ) -> Vec<SingleRailroadOption> {
        NetworkValidator::get_valid_double_rail_first_link_options(&self.state, player_idx)
    }

    /// Check if player can take a loan
    pub fn can_take_loan(&self, player_idx: usize) -> bool {
        SpecialActions::can_take_loan(&self.state, player_idx)
    }

    /// Check if player can scout
    pub fn can_scout(&self, player_idx: usize) -> bool {
        SpecialActions::can_scout(&self.state, player_idx)
    }

    /// Check if player can build double railroad
    pub fn can_double_railroad(&self, player_idx: usize) -> bool {
        NetworkValidator::can_double_railroad(&self.state, player_idx)
    }

    // === Utility Methods ===

    /// Get iron sources for development
    pub fn get_iron_sources_for_develop(
        &self,
        player_idx: usize,
        num_needed: u8,
    ) -> Vec<crate::board::resources::ResourceSource> {
        DevelopValidator::get_iron_sources_for_develop(&self.state, player_idx, num_needed)
    }

    pub fn can_source_iron(&self, player_idx: usize, num_needed: u8) -> bool {
        DevelopValidator::can_source_iron(&self.state, player_idx, num_needed)
    }

    /// Get iron cost for development
    pub fn get_iron_cost_for_develop(&self, num_develops: u8) -> u8 {
        DevelopValidator::get_iron_cost_for_develop(num_develops)
    }

    /// Get options for second rail link in double railroad
    pub fn get_options_for_second_rail_link(
        &self,
        player_idx: usize,
        first_road_idx: usize,
        first_road_coal_source: &crate::board::resources::ResourceSource,
        hypothetical_money_remaining: u16,
    ) -> Vec<DoubleRailroadSecondLinkOption> {
        NetworkValidator::get_options_for_second_rail_link(
            &self.state,
            player_idx,
            first_road_idx,
            first_road_coal_source,
            hypothetical_money_remaining,
        )
    }

    /// Liquidate player assets
    pub fn liquidate_for_player(&mut self, player_idx: usize) {
        SpecialActions::liquidate_for_player(&mut self.state, player_idx);
    }

    /// Update turn order
    pub fn turn_order_next(&mut self) {
        SpecialActions::update_turn_order(&mut self.state);
    }

    /// Discard a card
    pub fn discard_card(&mut self, player_idx: usize, card_idx_in_hand: usize) {
        if card_idx_in_hand >= self.state.players[player_idx].hand.cards.len() {
            eprintln!(
                "Error: Invalid card index {} to discard for player {}. Hand size: {}.",
                card_idx_in_hand,
                player_idx,
                self.state.players[player_idx].hand.cards.len()
            );
            return;
        }

        let card = self.state.players[player_idx]
            .hand
            .cards
            .remove(card_idx_in_hand);

        match card.card_type {
            CardType::WildLocation => {
                self.state.wild_location_cards_available =
                    self.state.wild_location_cards_available.saturating_add(1)
            }
            CardType::WildIndustry => {
                self.state.wild_industry_cards_available =
                    self.state.wild_industry_cards_available.saturating_add(1)
            }
            _ => self.state.discard_pile.push(card),
        }
    }

    // === State Access Methods ===

    /// Get coal price
    pub fn get_coal_price(&self, cubes: u8) -> u16 {
        crate::board::resources::ResourceManager::get_coal_price(
            self.state.remaining_market_coal,
            cubes,
        )
    }

    /// Get iron price
    pub fn get_iron_price(&self, cubes: u8) -> u16 {
        crate::board::resources::ResourceManager::get_iron_price(
            self.state.remaining_market_iron,
            cubes,
        )
    }

    /// Get tile at location
    pub fn get_tile_at_loc(&self, loc: usize) -> Option<BuildingTypeData> {
        self.state.get_tile_at_loc(loc)
    }

    // Direct access to state for compatibility
    pub fn era(&self) -> Era {
        self.state.era
    }
    pub fn players(&self) -> &Vec<Player> {
        &self.state.players
    }
    pub fn players_mut(&mut self) -> &mut Vec<Player> {
        &mut self.state.players
    }
    pub fn current_player_idx(&self) -> usize {
        self.state.current_player_idx
    }
    pub fn set_current_player_idx(&mut self, idx: usize) {
        self.state.current_player_idx = idx;
    }
    pub fn player_network_mask(&self) -> &Vec<crate::board::connectivity::Connectivity> {
        &self.state.player_network_mask
    }
    pub fn bl_to_building(&self) -> &std::collections::HashMap<usize, BuiltBuilding> {
        &self.state.bl_to_building
    }
    pub fn iron_locations(&self) -> &BuildLocationSet {
        &self.state.iron_locations
    }
    pub fn coal_locations(&self) -> &BuildLocationSet {
        &self.state.coal_locations
    }
    pub fn beer_locations(&self) -> &BuildLocationSet {
        &self.state.beer_locations
    }
    pub fn trade_post_slots(&self) -> &Vec<Option<crate::market::merchants::MerchantTile>> {
        &self.state.trade_post_slots
    }
    pub fn trade_post_beer(&self) -> &FixedBitSet {
        &self.state.trade_post_beer
    }
    pub fn remaining_market_coal(&self) -> u8 {
        self.state.remaining_market_coal
    }
    pub fn remaining_market_iron(&self) -> u8 {
        self.state.remaining_market_iron
    }
    pub fn connectivity(&self) -> &crate::board::connectivity::Connectivity {
        &self.state.connectivity
    }
    pub fn built_roads(&self) -> &RoadSet {
        &self.state.built_roads
    }
    pub fn build_locations_occupied(&self) -> &BuildLocationSet {
        &self.state.build_locations_occupied
    }
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board")
            .field("era", &self.state.era)
            .field("current_player_idx", &self.state.current_player_idx)
            .finish()
    }
}
