use crate::board::BoardState;
use crate::{core::types::*, locations::LocationName};
use std::collections::{HashSet, VecDeque};

/// Resource source enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourceSource {
    Building(usize), // Build location index
    Market,
}

impl From<usize> for ResourceSource {
    fn from(index: usize) -> Self {
        if index < NUM_BL {
            ResourceSource::Building(index)
        } else {
            ResourceSource::Market
        }
    }
}

/// Beer source for selling actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeerSellSource {
    Building(usize),  // Build location index of a brewery
    TradePost(usize), // Merchant *slot* index
}

/// Brewery beer source for double railroad actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BreweryBeerSource {
    OwnBrewery(usize),      // loc_idx of player's own brewery
    OpponentBrewery(usize), // loc_idx of opponent's brewery
}

/// Resource management and pricing functions
pub struct ResourceManager;

impl ResourceManager {
    /// Get coal price for a given number of cubes
    pub fn get_coal_price(remaining_market_coal: u8, cubes: u8) -> u16 {
        if cubes == 0 {
            return 0;
        }

        let mut total_cost = 0;
        let mut market_idx = remaining_market_coal;

        for _ in 0..cubes {
            if market_idx > 0 {
                total_cost += COAL_PRICE_TABLE[(MAX_MARKET_COAL - market_idx) as usize] as u16;
                market_idx -= 1;
            } else {
                total_cost += 8; // Cost when market is empty
            }
        }
        total_cost
    }

    /// Check if beer is present on the board
    pub fn beer_present(board_state: &BoardState) -> bool {
        return !board_state.beer_locations.is_clear();
    }

    /// Get iron price for a given number of cubes
    pub fn get_iron_price(remaining_market_iron: u8, cubes: u8) -> u16 {
        if cubes == 0 {
            return 0;
        }

        let mut total_cost = 0;
        let mut market_idx = remaining_market_iron;

        for _ in 0..cubes {
            if market_idx > 0 {
                total_cost += IRON_PRICE_TABLE[(MAX_MARKET_IRON - market_idx) as usize] as u16;
                market_idx -= 1;
            } else {
                total_cost += 6; // Cost when market is empty
            }
        }
        total_cost
    }

    /// Consume coal from market and return cost
    pub fn consume_market_coal(remaining_market_coal: &mut u8, cubes: u8) -> u16 {
        let mut cost = 0;
        for _ in 0..cubes {
            if *remaining_market_coal > 0 {
                cost +=
                    COAL_PRICE_TABLE[(MAX_MARKET_COAL - *remaining_market_coal) as usize] as u16;
                *remaining_market_coal -= 1;
            } else {
                cost += 8; // Empty market price
            }
        }
        cost
    }

    /// Consume iron from market and return cost
    pub fn consume_market_iron(remaining_market_iron: &mut u8, cubes: u8) -> u16 {
        let mut cost = 0;
        for _ in 0..cubes {
            if *remaining_market_iron > 0 {
                cost +=
                    IRON_PRICE_TABLE[(MAX_MARKET_IRON - *remaining_market_iron) as usize] as u16;
                *remaining_market_iron -= 1;
            } else {
                cost += 6; // Empty market price
            }
        }
        cost
    }

    /// Check if market has space for a resource type
    pub fn has_market_space(
        industry: IndustryType,
        remaining_coal: u8,
        remaining_iron: u8,
    ) -> bool {
        match industry {
            IndustryType::Coal => remaining_coal < MAX_MARKET_COAL,
            IndustryType::Iron => remaining_iron < MAX_MARKET_IRON,
            _ => false,
        }
    }

    /// Get free market space for a resource type
    pub fn get_free_market_space(
        industry: IndustryType,
        remaining_coal: u8,
        remaining_iron: u8,
    ) -> u8 {
        match industry {
            IndustryType::Coal => MAX_MARKET_COAL - remaining_coal,
            IndustryType::Iron => MAX_MARKET_IRON - remaining_iron,
            _ => 0,
        }
    }

    /// Sell resource to market and return money gained
    pub fn sell_resource_to_market(
        industry: IndustryType,
        amount: u8,
        remaining_market_coal: &mut u8,
        remaining_market_iron: &mut u8,
    ) -> u16 {
        let (price_table, remaining_market, max_market) = match industry {
            IndustryType::Coal => (
                COAL_PRICE_TABLE.to_vec(),
                remaining_market_coal,
                MAX_MARKET_COAL,
            ),
            IndustryType::Iron => (
                IRON_PRICE_TABLE.to_vec(),
                remaining_market_iron,
                MAX_MARKET_IRON,
            ),
            _ => return 0,
        };

        let mut total_money_gained = 0u16;

        for _ in 0..amount {
            if *remaining_market < max_market {
                // Fill highest-value empty slots first when market is scarce.
                // Example (iron, max=10): remaining=1 -> indices 8,7,6,5 => 5,4,4,3.
                let price_idx = (max_market - *remaining_market - 1) as usize;
                total_money_gained += price_table[price_idx] as u16;
                *remaining_market += 1;
            }
        }
        total_money_gained
    }
    /// Find resources and their distances using BFS, explore towns
    /// and add corresponding buildlocations to the list in order of distance
    /// Returns Vec of (loc_idx, resource_amount, distance)
    pub fn find_connected_coal_sources(
        board_state: &BoardState,
        build_loc_idx: usize,
    ) -> Vec<(usize, u8, usize)> {
        let mut resources_found: Vec<(usize, u8, usize)> = Vec::new();

        if !board_state.is_connected_to_coal(build_loc_idx) {
            return resources_found;
        }

        let first_town = LocationName::from_bl_idx(build_loc_idx);
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(LocationName, usize)> = VecDeque::new();
        queue.push_back((first_town, 0));

        while let Some((current_loc, dist)) = queue.pop_front() {
            if !current_loc.is_town() {
                continue;
            }
            // Add all coal build locations in this town to the list
            let town_bl_set = current_loc.to_bl_set();

            for coal_bl_idx in board_state.coal_locations.intersection(&town_bl_set) {
                resources_found.push((
                    coal_bl_idx,
                    board_state.get_resource_amount_at_bl(coal_bl_idx),
                    dist,
                ));
            }
            visited.insert(current_loc);
            // Add Connected towns to exploration queue
            let neighbors = board_state.get_location_neighbors(current_loc);
            for neighbor in neighbors.ones() {
                let neighbor_loc = LocationName::from_usize(neighbor);
                if !visited.contains(&neighbor_loc) {
                    queue.push_back((neighbor_loc, dist + 1));
                }
            }
        }

        // Sort by distance (closest first), then by location index
        resources_found
    }

    // =========================================================================
    // Resource Consumption Functions (used by build, develop, network actions)
    // =========================================================================

    /// Consume coal from specified sources, paying for market purchases
    /// Returns the total cost paid to market
    pub fn consume_coal(
        board_state: &mut BoardState,
        player_idx: usize,
        sources: Vec<ResourceSource>,
        num_needed: u8,
    ) -> u16 {
        let mut remaining_needed = num_needed;
        let mut total_paid_market = 0u16;

        for source in sources {
            if remaining_needed == 0 {
                break;
            }
            match source {
                ResourceSource::Building(loc) => {
                    if let Some(b) = board_state.bl_to_building.get_mut(&loc) {
                        if b.industry == IndustryType::Coal && !b.flipped && b.resource_amt > 0 {
                            let consume_now = std::cmp::min(remaining_needed, b.resource_amt);
                            if b.consume_resource_and_check_flip(consume_now) {
                                board_state.handle_building_flip(loc);
                            }
                            remaining_needed -= consume_now;
                        }
                    }
                }
                ResourceSource::Market => {
                    total_paid_market += Self::consume_market_coal(
                        &mut board_state.remaining_market_coal,
                        remaining_needed,
                    );
                    remaining_needed = 0;
                    break;
                }
            }
        }

        if total_paid_market > 0 {
            board_state.players[player_idx].pay(total_paid_market);
        }
        if remaining_needed > 0 {
            eprintln!("Warning: Could not satisfy coal demand of {}", num_needed);
        }

        total_paid_market
    }

    /// Consume iron from specified sources, paying for market purchases
    /// Returns the total cost paid to market
    pub fn consume_iron(
        board_state: &mut BoardState,
        player_idx: usize,
        sources: Vec<ResourceSource>,
        num_needed: u8,
    ) -> u16 {
        let mut remaining_needed = num_needed;
        let mut total_paid_market = 0u16;

        for source in sources {
            if remaining_needed == 0 {
                break;
            }
            match source {
                ResourceSource::Building(loc) => {
                    if let Some(b) = board_state.bl_to_building.get_mut(&loc) {
                        if b.industry == IndustryType::Iron && !b.flipped && b.resource_amt > 0 {
                            let consume_now = std::cmp::min(remaining_needed, b.resource_amt);
                            if b.consume_resource_and_check_flip(consume_now) {
                                board_state.handle_building_flip(loc);
                            }
                            remaining_needed -= consume_now;
                        }
                    }
                }
                ResourceSource::Market => {
                    total_paid_market += Self::consume_market_iron(
                        &mut board_state.remaining_market_iron,
                        remaining_needed,
                    );
                    remaining_needed = 0;
                    break;
                }
            }
        }

        if total_paid_market > 0 {
            board_state.players[player_idx].pay(total_paid_market);
        }
        if remaining_needed > 0 {
            eprintln!("Warning: Could not satisfy iron demand of {}", num_needed);
        }

        total_paid_market
    }

    /// Consume beer from a brewery building
    /// Returns true if successfully consumed
    pub fn consume_beer_from_brewery(board_state: &mut BoardState, loc: usize) -> bool {
        if let Some(brewery) = board_state.bl_to_building.get_mut(&loc) {
            if brewery.industry == IndustryType::Beer
                && brewery.resource_amt > 0
                && !brewery.flipped
            {
                if brewery.consume_resource_and_check_flip(1) {
                    board_state.handle_building_flip(loc);
                }
                return true;
            }
        }
        false
    }

    // =========================================================================
    // Sell to Market Functions
    // =========================================================================

    /// Try to sell resources from a newly built coal/iron building to the market
    /// Returns money gained
    pub fn sell_building_resources_to_market(
        board_state: &mut BoardState,
        player_idx: usize,
        building_loc: usize,
    ) -> u16 {
        let (industry, resource_amt) = {
            let building = match board_state.bl_to_building.get(&building_loc) {
                Some(b) => b,
                None => return 0,
            };

            // Only coal and iron can be sold to market
            if building.industry != IndustryType::Coal && building.industry != IndustryType::Iron {
                return 0;
            }

            // Coal requires connection to trade post
            if building.industry == IndustryType::Coal
                && !board_state.is_connected_to_trade_post(building_loc)
            {
                return 0;
            }

            (building.industry, building.resource_amt)
        };

        let free_market_space = Self::get_free_market_space(
            industry,
            board_state.remaining_market_coal,
            board_state.remaining_market_iron,
        );

        if free_market_space == 0 || resource_amt == 0 {
            return 0;
        }

        let to_market = free_market_space.min(resource_amt);
        let money_gained = Self::sell_resource_to_market(
            industry,
            to_market,
            &mut board_state.remaining_market_coal,
            &mut board_state.remaining_market_iron,
        );

        board_state.players[player_idx].gain_money(money_gained);

        // Consume from building and handle flip if depleted
        if let Some(building) = board_state.bl_to_building.get_mut(&building_loc) {
            if building.consume_resource_and_check_flip(to_market) {
                board_state.handle_building_flip(building_loc);
            }
        }

        money_gained
    }
}

/// Discard a card from player's hand, returning wild cards to pool.
pub fn discard_card(board_state: &mut BoardState, player_idx: usize, card_idx: usize) {
    if card_idx >= board_state.players[player_idx].hand.cards.len() {
        return;
    }

    let card = board_state.players[player_idx].hand.cards.remove(card_idx);
    match card.card_type {
        CardType::WildLocation => {
            board_state.wild_location_cards_available =
                board_state.wild_location_cards_available.saturating_add(1)
        }
        CardType::WildIndustry => {
            board_state.wild_industry_cards_available =
                board_state.wild_industry_cards_available.saturating_add(1)
        }
        _ => board_state.discard_pile.push(card),
    }
}

/// Remove a building from board state (used by overbuild and liquidation flows).
pub fn remove_building_from_board(board_state: &mut BoardState, loc_idx: usize) {
    if let Some(removed_building) = board_state.bl_to_building.remove(&loc_idx) {
        let owner_idx = removed_building.owner.as_usize();
        board_state.build_locations_occupied.remove(loc_idx);
        board_state.building_types[loc_idx] = None;
        board_state.player_building_mask[owner_idx].remove(loc_idx);

        match removed_building.industry {
            IndustryType::Coal => board_state.coal_locations.remove(loc_idx),
            IndustryType::Iron => board_state.iron_locations.remove(loc_idx),
            IndustryType::Beer => board_state.beer_locations.remove(loc_idx),
            _ => {}
        }
    }
}
