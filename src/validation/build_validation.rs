//! Build Validation Module
//!
//! Determines valid build options for a player based on:
//! - Player's industry mat (lowest level buildings available)
//! - Player's hand (cards that enable building)
//! - Player's network (for industry cards) or lack thereof
//! - Resource availability (coal requires connection, iron does not)
//! - Era rules (Canal: 1 building per town, Railroad: unlimited)

use crate::actions::build::BuildOption;
use crate::board::resources::ResourceSource;
use crate::board::BoardState;
use crate::consts::{BEER_BREWERY_1, BEER_BREWERY_2};
use crate::core::building::BuiltBuilding;
use crate::core::static_data::BUILD_LOCATION_MASK;
use crate::core::types::*;
use crate::locations::LocationName;
use crate::PlayerId;

// =============================================================================
// Resource Context - Pre-computed resource availability for a build location
// =============================================================================

/// Tracks available resources for building at a specific location
#[derive(Debug, Clone)]
pub struct ResourceContext {
    /// Connected coal sources: (location, amount_available)
    pub connected_coal: Vec<(usize, u8)>,
    /// Total coal available from connected mines
    pub total_connected_coal: u8,
    /// Whether this location can access market (connected to trade post)
    pub has_market_access: bool,
    /// All iron sources on board: (location, amount_available)  
    /// Iron doesn't require connection
    pub iron_on_board: Vec<(usize, u8)>,
    /// Total iron available on board
    pub total_iron_on_board: u8,
}

impl ResourceContext {
    /// Compute resource context for a specific build location
    pub fn for_location(board_state: &BoardState, build_loc_idx: usize) -> Self {
        let connected_coal = board_state.get_connected_coal_sources(build_loc_idx);
        let total_connected_coal: u8 = connected_coal.iter().map(|(_, amt)| *amt).sum();
        let has_market_access = board_state.is_connected_to_trade_post(build_loc_idx);
        let iron_on_board = board_state.get_all_iron_sources();
        let total_iron_on_board: u8 = iron_on_board.iter().map(|(_, amt)| *amt).sum();

        ResourceContext {
            connected_coal,
            total_connected_coal,
            has_market_access,
            iron_on_board,
            total_iron_on_board,
        }
    }

    /// Check if we can satisfy coal requirement at this location
    pub fn can_get_coal(&self, amount: u8, player_money: u16, market_coal: u8) -> bool {
        if amount == 0 {
            return true;
        }

        // First use free coal from connected mines
        let free_coal = self.total_connected_coal.min(amount);
        let coal_from_market = amount - free_coal;

        if coal_from_market > 0 {
            // Need market access and enough money
            if !self.has_market_access {
                return false;
            }

            let market_cost = compute_coal_cost(market_coal, coal_from_market);
            if market_cost > player_money {
                return false;
            }
        }

        true
    }

    /// Check if we can satisfy iron requirement (no connection needed)
    pub fn can_get_iron(&self, amount: u8, player_money: u16, market_iron: u8) -> bool {
        if amount == 0 {
            return true;
        }

        // First use free iron from board
        let free_iron = self.total_iron_on_board.min(amount);
        let iron_from_market = amount - free_iron;

        if iron_from_market > 0 {
            let market_cost = compute_iron_cost(market_iron, iron_from_market);
            if market_cost > player_money {
                return false;
            }
        }

        true
    }

    /// Compute total resource cost and return sources
    pub fn compute_resource_plan(
        &self,
        coal_needed: u8,
        iron_needed: u8,
        market_coal: u8,
        market_iron: u8,
    ) -> (u16, Vec<ResourceSource>, Vec<ResourceSource>) {
        let mut total_cost = 0u16;
        let mut coal_sources = Vec::new();
        let mut iron_sources = Vec::new();

        // === Coal sourcing (requires connection) ===
        let mut coal_remaining = coal_needed;

        // First, take from connected coal mines (free)
        for &(loc, amt) in &self.connected_coal {
            if coal_remaining == 0 {
                break;
            }
            let take = coal_remaining.min(amt);
            coal_sources.push(ResourceSource::Building(loc));
            coal_remaining -= take;
        }

        // Then, buy from market if needed and connected
        if coal_remaining > 0 && self.has_market_access {
            total_cost += compute_coal_cost(market_coal, coal_remaining);
            coal_sources.push(ResourceSource::Market);
        }

        // === Iron sourcing (no connection required) ===
        let mut iron_remaining = iron_needed;

        // First, take from any iron works on board (free)
        for &(loc, amt) in &self.iron_on_board {
            if iron_remaining == 0 {
                break;
            }
            let take = iron_remaining.min(amt);
            iron_sources.push(ResourceSource::Building(loc));
            iron_remaining -= take;
        }

        // Then, buy from market if needed
        if iron_remaining > 0 {
            total_cost += compute_iron_cost(market_iron, iron_remaining);
            iron_sources.push(ResourceSource::Market);
        }

        (total_cost, coal_sources, iron_sources)
    }
}

// =============================================================================
// Helper Functions - Pure, focused computations
// =============================================================================

/// Compute coal cost from market
pub fn compute_coal_cost(remaining_market: u8, cubes_needed: u8) -> u16 {
    use crate::board::resources::ResourceManager;
    ResourceManager::get_coal_price(remaining_market, cubes_needed)
}

/// Compute iron cost from market
pub fn compute_iron_cost(remaining_market: u8, cubes_needed: u8) -> u16 {
    use crate::board::resources::ResourceManager;
    ResourceManager::get_iron_price(remaining_market, cubes_needed)
}

// =============================================================================
// Overbuild Rules - When can we build over an existing building?
// =============================================================================

/// Check if we can build at this location (empty or valid overbuild)
pub fn can_build_at_location(
    board_state: &BoardState,
    player_idx: usize,
    loc_idx: usize,
    new_industry: IndustryType,
    new_level: IndustryLevel,
) -> bool {
    // Empty slot - always valid (if industry type matches the slot)
    if !board_state.build_locations_occupied.contains(loc_idx) {
        return true;
    }

    let existing = match board_state.bl_to_building.get(&loc_idx) {
        Some(b) => b,
        None => return false, // Inconsistent state
    };

    // === Own building: Can overbuild any of own tiles with higher level ===
    if existing.owner.as_usize() == player_idx {
        // Must be same industry and higher level
        if new_industry != existing.industry {
            return false;
        }
        if new_level <= existing.level {
            return false;
        }
        return true;
    }

    // === Opponent's building: Can only overbuild depleted coal/iron ===
    // Must be coal or iron
    if existing.industry != IndustryType::Coal && existing.industry != IndustryType::Iron {
        return false;
    }

    // Must be same industry type
    if new_industry != existing.industry {
        return false;
    }

    // Check if ALL resources of this type are depleted (market + board)
    board_state.is_resource_depleted(existing.industry)
}

// =============================================================================
// Card-Based Location Filtering
// =============================================================================

/// Check if card is valid for lonely brewery locations
pub fn card_valid_for_lonely_brewery(card: &Card, industry: IndustryType) -> bool {
    if industry != IndustryType::Beer {
        return true; // Only relevant for beer
    }

    match &card.card_type {
        CardType::Industry(ind_set) => ind_set.contains(IndustryType::Beer as usize),
        CardType::WildIndustry => true,
        _ => false,
    }
}

/// Determine which build locations a card allows for a specific player
pub fn get_build_locations_for_card(
    board_state: &BoardState,
    player_idx: usize,
    card: &Card,
) -> BuildLocationSet {
    let mut build_locations = BuildLocationSet::new();
    let has_network = board_state.player_has_network(player_idx);

    match &card.card_type {
        CardType::Location(town_name) => {
            // Location card: Build anywhere in that town (ignores network)
            build_locations.union_with(&town_name.to_bl_set());
        }

        CardType::Industry(industry_set) => {
            if has_network {
                // Build only in network locations that match the industry
                let locations_in_network =
                    board_state.players[player_idx].get_locations_in_network(board_state);
                for location in locations_in_network.ones() {
                    let bl_set = LocationName::from_usize(location).to_bl_set();
                    for bl_idx in bl_set.ones() {
                        let industries: &IndustrySet = &BUILD_LOCATION_MASK[bl_idx];
                        if !industries.is_disjoint(industry_set) {
                            build_locations.insert(bl_idx);
                        }
                    }
                }
            } else {
                // No network - can build at any location that matches the industry
                for bl_idx in 0..NUM_BL {
                    let industries: &IndustrySet = &BUILD_LOCATION_MASK[bl_idx];
                    if !industries.is_disjoint(industry_set) {
                        build_locations.insert(bl_idx);
                    }
                }
            }
        }

        CardType::WildLocation => {
            // All Locations are valid (ignores network)
            build_locations.insert_range(0..NUM_BL);
        }

        CardType::WildIndustry => {
            if has_network {
                // Build only in network
                let locations_in_network =
                    board_state.players[player_idx].get_locations_in_network(board_state);
                for location in locations_in_network.ones() {
                    let bl_set = LocationName::from_usize(location).to_bl_set();
                    build_locations.union_with(&bl_set);
                }
            } else {
                // No network - can build anywhere
                build_locations.insert_range(0..NUM_BL);
            }
        }
    }

    build_locations
}
// =============================================================================
// Main Validation Logic
// =============================================================================

/// Build action validation
pub struct BuildValidator;

impl BuildValidator {
    /// Check if a player can overbuild an existing building at a location
    pub fn can_overbuild(
        board_state: &BoardState,
        old_building: &BuiltBuilding,
        player_idx: usize,
        industry: IndustryType,
    ) -> bool {
        let level = board_state.players[player_idx]
            .industry_mat
            .get_lowest_level(industry);
        let building_data = board_state.players[player_idx]
            .industry_mat
            .get_tile_for_industry(industry);

        // No tiles available for this industry
        if building_data.is_none() {
            return false;
        }

        // Must be same industry to overbuild
        if old_building.industry != industry {
            return false;
        }

        // Must be higher level
        if old_building.level >= level {
            return false;
        }

        // If it's our own building, can always overbuild with higher level
        if old_building.owner == PlayerId::from_usize(player_idx) {
            return true;
        }

        // Opponent's building - can only overbuild coal/iron when ALL of that resource is depleted
        if !industry.is_market_resource() {
            return false;
        }

        // Check if market resource is depleted
        match industry {
            IndustryType::Coal => {
                board_state.remaining_market_coal == 0 && board_state.get_total_coal_on_board() == 0
            }
            IndustryType::Iron => {
                board_state.remaining_market_iron == 0 && board_state.get_total_iron_on_board() == 0
            }
            _ => false,
        }
    }

    /// Get all industries that the player has tiles available for and can build in the current era
    pub fn get_buildable_industries(board_state: &BoardState, player_idx: usize) -> IndustrySet {
        let player = &board_state.players[player_idx];
        let mut valid_industries = IndustrySet::new();

        for industry_idx in 0..N_INDUSTRIES {
            let industry = IndustryType::from_usize(industry_idx);

            // Must have tiles left
            if !player.industry_mat.has_tiles_left(industry) {
                continue;
            }

            // Must be buildable in current era
            let building_data = player
                .industry_mat
                .get_current_level_building_data(industry);
            if !building_data.can_build_in_era(board_state.era) {
                continue;
            }

            valid_industries.insert(industry_idx);
        }

        valid_industries
    }

    /// Get all valid build options for a player
    pub fn get_valid_build_options(
        board_state: &BoardState,
        player_idx: usize,
    ) -> Vec<BuildOption> {
        let mut valid_options = Vec::new();
        let player = &board_state.players[player_idx];
        let has_network = board_state.player_has_network(player_idx);

        // For each industry type (6 industries)
        for industry_idx in 0..N_INDUSTRIES {
            let industry = IndustryType::from_usize(industry_idx);

            // Skip if no tiles left for this industry
            if !player.industry_mat.has_tiles_left(industry) {
                continue;
            }

            let level = player.industry_mat.get_lowest_level(industry);
            let building_data = player
                .industry_mat
                .get_current_level_building_data(industry);

            // Skip if building can't be built in current era
            if !building_data.can_build_in_era(board_state.era) {
                continue;
            }

            // Skip if player can't afford base money cost
            if !player.can_afford(building_data.money_cost) {
                continue;
            }

            // Process each card in player's hand
            for (card_idx, card) in player.hand.cards.iter().enumerate() {
                Self::process_card_for_industry(
                    board_state,
                    player_idx,
                    &mut valid_options,
                    industry,
                    level,
                    building_data,
                    card_idx,
                    card,
                    has_network,
                );
            }
        }

        valid_options
    }

    /// Process a single card for a specific industry
    fn process_card_for_industry(
        board_state: &BoardState,
        player_idx: usize,
        valid_options: &mut Vec<BuildOption>,
        industry: IndustryType,
        level: IndustryLevel,
        building_data: &'static BuildingTypeData,
        card_idx: usize,
        card: &Card,
        _has_network: bool,
    ) {
        // First check if the card allows building this industry
        match &card.card_type {
            CardType::Industry(industry_set) => {
                if !industry_set.contains(industry as usize) {
                    return; // Card doesn't allow this industry
                }
            }
            CardType::WildIndustry => {
                // Wild card allows any industry
            }
            CardType::Location(_) | CardType::WildLocation => {
                // Location cards allow any industry at that location
            }
        }

        let player = &board_state.players[player_idx];

        // Get locations this card allows for this industry
        let potential_locations = get_build_locations_for_card(board_state, player_idx, card);

        // No valid locations for this card/industry combo
        if potential_locations.is_clear() {
            return;
        }

        // Check each potential location
        for loc_idx in potential_locations.ones() {
            // === Filter 1: Industry type must match build location ===
            if !BUILD_LOCATION_MASK[loc_idx].contains(industry as usize) {
                continue;
            }

            // === Filter 2: Lonely brewery special rule ===
            if (loc_idx == BEER_BREWERY_1 as usize || loc_idx == BEER_BREWERY_2 as usize)
                && !card_valid_for_lonely_brewery(card, industry)
            {
                continue;
            }

            // === Filter 3: Overbuild rules ===
            if !can_build_at_location(board_state, player_idx, loc_idx, industry, level) {
                continue;
            }

            // === Filter 4: Canal era - one building per town per player ===
            if board_state.era == Era::Canal {
                let is_overbuilding_own = board_state
                    .bl_to_building
                    .get(&loc_idx)
                    .map_or(false, |b| b.owner.as_usize() == player_idx);

                if !is_overbuilding_own {
                    if let Some(town_idx) = board_state.get_town_for_bl(loc_idx) {
                        if board_state.player_has_building_in_town(player_idx, town_idx) {
                            continue;
                        }
                    }
                }
            }

            // === Filter 5: Resource availability and cost ===
            let resource_ctx = ResourceContext::for_location(board_state, loc_idx);

            // Check if resources are available
            let coal_needed = building_data.coal_cost;
            let iron_needed = building_data.iron_cost;

            // Calculate maximum money available for resources
            let money_for_resources = player.money.saturating_sub(building_data.money_cost);

            // Quick check: Can we get the resources?
            if !resource_ctx.can_get_coal(
                coal_needed,
                money_for_resources,
                board_state.remaining_market_coal,
            ) {
                continue;
            }
            if !resource_ctx.can_get_iron(
                iron_needed,
                money_for_resources,
                board_state.remaining_market_iron,
            ) {
                continue;
            }

            // Compute actual resource plan and total cost
            let (resource_cost, coal_sources, iron_sources) = resource_ctx.compute_resource_plan(
                coal_needed,
                iron_needed,
                board_state.remaining_market_coal,
                board_state.remaining_market_iron,
            );

            let total_money_cost = building_data.money_cost + resource_cost;

            // Final affordability check
            if !player.can_afford(total_money_cost) {
                continue;
            }

            // === Valid build option found! ===
            valid_options.push(BuildOption {
                industry_type: industry,
                build_location_idx: loc_idx,
                card_used_idx: card_idx,
                card_used: card.clone(),
                level,
                building_data,
                total_money_cost,
                total_coal_cost: coal_needed,
                total_iron_cost: iron_needed,
                coal_sources,
                iron_sources,
            });
        }
    }

    /// Check if player can perform any build action
    /// Returns true if the player has at least one valid build option
    pub fn can_build(board_state: &BoardState, player_idx: usize) -> bool {
        let player = &board_state.players[player_idx];

        // Quick check: player needs at least one card
        if player.hand.cards.is_empty() {
            return false;
        }

        // Check if any industry is potentially buildable
        let buildable_industries = Self::get_buildable_industries(board_state, player_idx);
        if buildable_industries.is_clear() {
            return false;
        }

        // Check if player can afford the cheapest available building
        let mut cheapest_affordable = false;
        for industry_idx in buildable_industries.ones() {
            let industry = IndustryType::from_usize(industry_idx);
            let building_data = player
                .industry_mat
                .get_current_level_building_data(industry);

            // For quick check, just verify base money cost
            // Resource costs are checked more thoroughly in get_valid_build_options
            if player.can_afford(building_data.money_cost) {
                cheapest_affordable = true;
                break;
            }
        }

        if !cheapest_affordable {
            return false;
        }

        // Do the full check - this is more expensive but accurate
        !Self::get_valid_build_options(board_state, player_idx).is_empty()
    }

    /// Quick check if player has any potentially buildable industries
    /// This is a fast check that doesn't account for all constraints (cards, locations, resources)
    pub fn has_buildable_industries(board_state: &BoardState, player_idx: usize) -> bool {
        !Self::get_buildable_industries(board_state, player_idx).is_clear()
    }
}

// =============================================================================
// Top-Level Convenience Functions
// =============================================================================

/// Check if a player can build anything given their current state
///
/// This considers:
/// - Player's hand (must have at least one card)
/// - Player's industry mat (lowest level buildings available in current era)
/// - Player's money (must afford base cost + resource costs)
/// - Available build locations (empty or valid overbuild targets)
/// - Resource availability (coal requires connection, iron does not)
/// - Era rules (Canal: 1 building per town, Railroad: unlimited)
///
/// Returns true if there is at least one valid build option.
pub fn player_can_build_anything(board_state: &BoardState, player_idx: usize) -> bool {
    BuildValidator::can_build(board_state, player_idx)
}

/// Get all valid build options for a player
///
/// This is the comprehensive validation function that returns all possible
/// build actions a player can take given the current board state.
///
/// Each BuildOption contains:
/// - Which industry to build
/// - Where to build it
/// - Which card to use
/// - Total cost (money + resource purchases)
/// - Where to source resources from
pub fn get_all_build_options(board_state: &BoardState, player_idx: usize) -> Vec<BuildOption> {
    BuildValidator::get_valid_build_options(board_state, player_idx)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::core::building::BuiltBuilding;
    use crate::core::locations::LocationName;
    use crate::core::player::PlayerId;
    use crate::locations::TownName;

    fn setup_board(num_players: usize) -> Board {
        Board::new(num_players, Some(12345))
    }

    #[test]
    fn test_no_network_industry_card_builds_anywhere() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Clear any initial buildings for this player to ensure no network
        board.state.player_building_mask[player_idx].clear();
        board.state.player_road_mask[player_idx].clear();

        // Player has no buildings or roads (no network)
        // Give them a Coal industry card - Coal Level 1 requires no resources!
        board.state.players[player_idx].hand.cards = vec![Card::new(CardType::Industry(
            IndustrySet::new_from_industry_types(&[IndustryType::Coal]),
        ))];
        board.state.players[player_idx].money = 100; // Plenty of money

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Should be able to build Coal at ANY Coal location on the map
        // because player has no network yet (and Coal Level 1 needs no resources)
        assert!(
            !options.is_empty(),
            "Should have build options with no network"
        );

        // All options should be for Coal industry
        for opt in &options {
            assert_eq!(opt.industry_type, IndustryType::Coal);
        }

        println!("Build options with no network: {}", options.len());
    }

    #[test]
    fn test_with_network_industry_card_restricted() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Build a building in Birmingham to establish network
        // Birmingham build locations: 36-39
        let birmingham_loc = 36;
        let building = BuiltBuilding::build(
            IndustryType::Cotton,
            IndustryLevel::I,
            birmingham_loc as u8,
            PlayerId::from_usize(player_idx),
        );
        board.state.bl_to_building.insert(birmingham_loc, building);
        board.state.build_locations_occupied.insert(birmingham_loc);
        board.state.player_building_mask[player_idx].insert(birmingham_loc);

        // Build a link from Birmingham to Coventry
        // Link index 11 connects Birmingham <-> Coventry
        let link_idx = 11;
        board.state.built_roads.insert(link_idx);
        board.state.player_road_mask[player_idx].insert(link_idx);
        board.state.connectivity.add_road(link_idx);

        // Give player a Goods industry card
        board.state.players[player_idx].hand.cards = vec![Card::new(CardType::Industry(
            IndustrySet::new_from_industry_types(&[IndustryType::Goods]),
        ))];
        board.state.players[player_idx].money = 100;

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // All options should be in player's network (Birmingham + connected towns)
        for opt in &options {
            let is_in_network = board
                .state
                .is_in_player_network(player_idx, opt.build_location_idx);
            assert!(
                is_in_network,
                "Build location {} should be in player's network",
                opt.build_location_idx
            );
        }

        println!("Build options with network: {}", options.len());
    }

    #[test]
    fn test_location_card_ignores_network() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Build a building in Birmingham to establish network
        let birmingham_loc = 36;
        let building = BuiltBuilding::build(
            IndustryType::Cotton,
            IndustryLevel::I,
            birmingham_loc as u8,
            PlayerId::from_usize(player_idx),
        );
        board.state.bl_to_building.insert(birmingham_loc, building);
        board.state.build_locations_occupied.insert(birmingham_loc);
        board.state.player_building_mask[player_idx].insert(birmingham_loc);

        // Give player a Coalbrookdale location card
        // Coalbrookdale is NOT connected to Birmingham
        board.state.players[player_idx].hand.cards =
            vec![Card::new(CardType::Location(TownName::Coalbrookdale))];
        board.state.players[player_idx].money = 100;

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Should have options in Coalbrookdale despite having network elsewhere
        // Coalbrookdale build locations: 25-27
        let has_coalbrookdale_option = options
            .iter()
            .any(|opt| opt.build_location_idx >= 25 && opt.build_location_idx <= 27);

        assert!(
            has_coalbrookdale_option,
            "Location card should allow building in Coalbrookdale even with network elsewhere"
        );
    }

    #[test]
    fn test_coal_requirement_needs_connection() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Set up: Player wants to build Iron Works (needs coal)
        // Iron Works level 1 costs 5 money + 1 coal

        // Place a coal mine in Coalbrookdale (loc 27) with coal
        let coal_loc = 27;
        let mut coal_building = BuiltBuilding::build(
            IndustryType::Coal,
            IndustryLevel::I,
            coal_loc as u8,
            PlayerId::from_usize(1), // Opponent owns it
        );
        coal_building.resource_amt = 2;
        board.state.bl_to_building.insert(coal_loc, coal_building);
        board.state.build_locations_occupied.insert(coal_loc);
        board.state.coal_locations.insert(coal_loc);

        // Give player an Iron industry card and enough money
        board.state.players[player_idx].hand.cards = vec![Card::new(CardType::Industry(
            IndustrySet::new_from_industry_types(&[IndustryType::Iron]),
        ))];
        board.state.players[player_idx].money = 20;

        // Empty the market so coal can only come from mines
        board.state.remaining_market_coal = 0;

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Should only have options at locations connected to Coalbrookdale's coal mine
        // Since player has no network, they can build anywhere, but coal requirement
        // means only locations connected to coal mine (Coalbrookdale area)
        for opt in &options {
            if opt.industry_type == IndustryType::Iron && opt.building_data.coal_cost > 0 {
                // This location must be connected to coal
                let build_loc = LocationName::from_bl_idx(opt.build_location_idx);
                let coal_mine_loc = LocationName::from_bl_idx(coal_loc);
                let is_connected = board
                    .state
                    .connectivity
                    .are_towns_connected(build_loc, coal_mine_loc);
                assert!(
                    is_connected
                        || opt
                            .coal_sources
                            .iter()
                            .any(|s| matches!(s, ResourceSource::Market)),
                    "Iron build at {} must be connected to coal source",
                    opt.build_location_idx
                );
            }
        }

        println!(
            "Iron build options (need coal): {}",
            options
                .iter()
                .filter(|o| o.industry_type == IndustryType::Iron)
                .count()
        );
    }

    #[test]
    fn test_iron_no_connection_required() {
        let mut board = setup_board(4);
        let player_idx = 0;

        // Place an iron works far away (not connected)
        let iron_loc = 0; // Stafford
        let mut iron_building = BuiltBuilding::build(
            IndustryType::Iron,
            IndustryLevel::I,
            iron_loc as u8,
            PlayerId::from_usize(1), // Opponent owns it
        );
        iron_building.resource_amt = 2;
        board.state.bl_to_building.insert(iron_loc, iron_building);
        board.state.build_locations_occupied.insert(iron_loc);
        board.state.iron_locations.insert(iron_loc);

        // Give player a Beer industry card (Beer needs iron)
        board.state.players[player_idx].hand.cards = vec![Card::new(CardType::Industry(
            IndustrySet::new_from_industry_types(&[IndustryType::Beer]),
        ))];
        board.state.players[player_idx].money = 20;

        // Empty market so iron can only come from the iron works
        board.state.remaining_market_iron = 0;

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Should have beer options anywhere (iron doesn't need connection)
        let beer_options: Vec<_> = options
            .iter()
            .filter(|o| o.industry_type == IndustryType::Beer)
            .collect();

        assert!(
            !beer_options.is_empty(),
            "Should have beer options even when iron is not connected"
        );

        // Iron sources should include the remote iron works
        for opt in &beer_options {
            if opt.building_data.iron_cost > 0 {
                let has_iron_source = opt
                    .iron_sources
                    .iter()
                    .any(|s| matches!(s, ResourceSource::Building(loc) if *loc == iron_loc));
                assert!(
                    has_iron_source,
                    "Should be able to take iron from unconnected iron works"
                );
            }
        }
    }

    #[test]
    fn test_canal_era_one_building_per_town() {
        let mut board = setup_board(4);
        let player_idx = 0;
        board.state.era = Era::Canal;

        // Build a building in Birmingham
        let birmingham_loc = 36;
        let building = BuiltBuilding::build(
            IndustryType::Cotton,
            IndustryLevel::I,
            birmingham_loc as u8,
            PlayerId::from_usize(player_idx),
        );
        board.state.bl_to_building.insert(birmingham_loc, building);
        board.state.build_locations_occupied.insert(birmingham_loc);
        board.state.player_building_mask[player_idx].insert(birmingham_loc);

        // Give player a Birmingham location card
        board.state.players[player_idx].hand.cards =
            vec![Card::new(CardType::Location(TownName::Birmingham))];
        board.state.players[player_idx].money = 100;

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Should NOT have options for new buildings in Birmingham (already have one)
        // Exception: Can overbuild own building
        for opt in &options {
            if opt.build_location_idx >= 36 && opt.build_location_idx <= 39 {
                // Must be overbuilding own building at loc 36
                assert_eq!(
                    opt.build_location_idx, birmingham_loc,
                    "In canal era, can only overbuild own building, not build new one in same town"
                );
            }
        }
    }

    #[test]
    fn test_railroad_era_multiple_buildings_per_town() {
        let mut board = setup_board(4);
        let player_idx = 0;
        board.state.era = Era::Railroad;

        // In Railroad era, Level 1 buildings of most industries are removed.
        // We need to advance the industry mat to Level 2 to have buildable tiles.
        // Pop tiles from Iron to get to Level 2 (Level 1 has 1 tile)
        board.state.players[player_idx]
            .industry_mat
            .pop_tile(IndustryType::Iron);

        // Build a building in Birmingham at location 38 (Iron)
        let birmingham_loc = 38;
        let building = BuiltBuilding::build(
            IndustryType::Iron,
            IndustryLevel::II, // Level 2
            birmingham_loc as u8,
            PlayerId::from_usize(player_idx),
        );
        board.state.bl_to_building.insert(birmingham_loc, building);
        board.state.build_locations_occupied.insert(birmingham_loc);
        board.state.player_building_mask[player_idx].insert(birmingham_loc);
        board.state.iron_locations.insert(birmingham_loc);

        // Connect Birmingham to Oxford (trade post) so player can buy coal from market
        // Link 12 connects Birmingham <-> Oxford
        let link_idx = 12;
        board.state.built_roads.insert(link_idx);
        board.state.connectivity.add_road(link_idx);

        // Give player a Birmingham location card
        // Birmingham allows: 36=Cotton/Goods, 37=Goods, 38=Iron, 39=Goods
        board.state.players[player_idx].hand.cards =
            vec![Card::new(CardType::Location(TownName::Birmingham))];
        board.state.players[player_idx].money = 100;

        // Also pop Goods to Level 2 (Level 1 removed in Railroad era)
        board.state.players[player_idx]
            .industry_mat
            .pop_tile(IndustryType::Goods);

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Debug output
        println!("Railroad era - Birmingham options:");
        for opt in &options {
            println!(
                "  Location {}: {:?} level {:?}",
                opt.build_location_idx, opt.industry_type, opt.level
            );
        }

        // Should have options for other locations in Birmingham (36, 37, 39)
        // Location 38 is occupied
        let other_birmingham: Vec<_> = options
            .iter()
            .filter(|o| {
                let is_birmingham = o.build_location_idx >= 36 && o.build_location_idx <= 39;
                let is_not_occupied = o.build_location_idx != birmingham_loc;
                is_birmingham && is_not_occupied
            })
            .collect();

        assert!(!other_birmingham.is_empty(),
            "In railroad era, should be able to build multiple buildings in same town. Options: {:?}",
            options.iter().map(|o| (o.build_location_idx, format!("{:?}", o.industry_type))).collect::<Vec<_>>());
    }

    #[test]
    fn test_complex_coal_scenario() {
        // Test case from user's description:
        // Player wants to build Iron Works needing 2 coal
        // Coalbrookdale has: coal mine with 1 coal, connected to Shrewsbury trade post
        // Player has 7 money, Iron Works costs 6
        // Should be able to build: 1 free coal from mine + 1 from market = 6 + 1 = 7 total

        let mut board = setup_board(4);
        let player_idx = 0;

        // Set up coal mine in Coalbrookdale (loc 27) with 1 coal
        let coal_loc = 27;
        let mut coal_building = BuiltBuilding::build(
            IndustryType::Coal,
            IndustryLevel::I,
            coal_loc as u8,
            PlayerId::from_usize(1), // Player 2 owns it
        );
        coal_building.resource_amt = 1; // Only 1 coal
        board.state.bl_to_building.insert(coal_loc, coal_building);
        board.state.build_locations_occupied.insert(coal_loc);
        board.state.coal_locations.insert(coal_loc);

        // Connect Coalbrookdale to Shrewsbury (trade post)
        // Need to find the correct road index - let's use road that connects these
        // Road 20 connects Coalbrookdale to Shrewsbury
        board.state.built_roads.insert(20);
        board.state.connectivity.add_road(20);

        // Set market coal to price level 1 (last cube at that price)
        board.state.remaining_market_coal = 2; // Low but available

        // Give player an Iron industry card
        board.state.players[player_idx].hand.cards = vec![Card::new(CardType::Industry(
            IndustrySet::new_from_industry_types(&[IndustryType::Iron]),
        ))];

        // Player has exactly enough: building_cost + market_coal_price = 6 + 1 = 7
        // But if building needs 2 coal and mine only has 1, need to check affordability
        board.state.players[player_idx].money = 7;

        let options = BuildValidator::get_valid_build_options(&board.state, player_idx);

        // Check if we can find an Iron option in Coalbrookdale area (25-27)
        // that correctly accounts for the resource costs
        let iron_in_coalbrookdale: Vec<_> = options
            .iter()
            .filter(|o| {
                o.industry_type == IndustryType::Iron
                    && o.build_location_idx >= 25
                    && o.build_location_idx <= 27
            })
            .collect();

        println!(
            "Iron options in Coalbrookdale: {:?}",
            iron_in_coalbrookdale
                .iter()
                .map(|o| (o.build_location_idx, o.total_money_cost))
                .collect::<Vec<_>>()
        );

        // The validation should correctly account for:
        // - 1 coal from mine (free)
        // - Additional coal from market if needed (costs money)
    }
}
