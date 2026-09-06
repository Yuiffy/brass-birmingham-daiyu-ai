use super::connectivity::Connectivity;
use crate::{
    cards::Deck,
    core::{
        building::BuiltBuilding,
        player::{Player, PlayerId},
        static_data::INDUSTRY_MAT,
        types::*,
    },
    locations::LocationName,
    market::merchants::MerchantTile,
};
use fixedbitset::FixedBitSet;
use rand::prelude::*;
use std::collections::HashMap;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingType {
    Industry(IndustryType),
    Market,
}
use crate::core::static_data::LINK_LOCATIONS;
/// Core board state - contains only the essential game state without complex logic
#[derive(Debug, Clone)]
pub struct BoardState {
    pub seed: u64,
    pub era: Era,
    pub wild_location_cards_available: u8,
    pub wild_industry_cards_available: u8,
    pub built_roads: RoadSet,
    pub build_locations_occupied: BuildLocationSet,
    pub building_types: Vec<Option<BuildingType>>,
    pub player_road_mask: Vec<RoadSet>,
    pub player_network_mask: Vec<Connectivity>,
    pub player_building_mask: Vec<BuildLocationSet>,
    pub connectivity: Connectivity,
    pub trade_post_slots: Vec<Option<MerchantTile>>,
    pub num_trade_posts: usize,
    pub remaining_market_coal: u8,
    pub remaining_market_iron: u8,
    pub trade_post_beer: FixedBitSet,
    pub deck: Deck,
    pub discard_pile: Vec<Card>,
    pub players: Vec<Player>,
    pub current_player_idx: usize,
    pub bl_to_building: HashMap<usize, BuiltBuilding>,
    pub rng: StdRng,
    pub coal_locations: BuildLocationSet,
    pub iron_locations: BuildLocationSet,
    pub beer_locations: BuildLocationSet,
    pub visible_vps: [u16; N_PLAYERS],
    pub turn_order: Vec<usize>,
}

impl BoardState {
    pub fn new(num_players: usize, seed: Option<u64>) -> Self {
        use crate::market::merchants::*;

        let actual_seed = seed.unwrap_or_else(rand::random::<u64>);
        let mut rng = StdRng::seed_from_u64(actual_seed);
        let _connectivity = Connectivity::new();

        let wild_location_cards_available = 4;
        let wild_industry_cards_available = 4;

        let num_trade_posts_active = PLAYER_COUNT_TO_NUM_TRADE_POSTS[num_players - 2];
        let num_merchant_tiles_in_pool = NUM_PLAYERS_TO_MERCHANT_TILES_POOL[num_players - 2];

        let mut shuffled_tiles: Vec<MerchantTile> =
            MERCHANT_TILES[0..num_merchant_tiles_in_pool].to_vec();
        shuffled_tiles.shuffle(&mut rng);

        let mut trade_post_slots: Vec<Option<MerchantTile>> =
            Vec::with_capacity(NUM_TRADE_POSTS * 2);
        let mut current_slot_idx_counter = 0;

        // Populate slots only for active trade posts
        for tp_global_idx in 0..num_trade_posts_active {
            let trade_post_enum = TRADE_POST_ORDERED[tp_global_idx];
            let num_slots_for_this_post = if trade_post_enum == TradePost::Shrewbury {
                1
            } else {
                2
            };
            for _ in 0..num_slots_for_this_post {
                if current_slot_idx_counter < trade_post_slots.capacity() {
                    if !shuffled_tiles.is_empty() {
                        trade_post_slots.push(shuffled_tiles.pop());
                    } else {
                        trade_post_slots.push(None);
                    }
                    current_slot_idx_counter += 1;
                } else {
                    break;
                }
            }
        }

        while trade_post_slots.len() < trade_post_slots.capacity() {
            trade_post_slots.push(None);
        }

        let mut trade_post_beer = FixedBitSet::with_capacity(NUM_TRADE_POSTS * 2);
        for (idx, slot_content) in trade_post_slots.iter().enumerate() {
            if let Some(merchant) = slot_content {
                if merchant.tile_type != MerchantTileType::Blank {
                    trade_post_beer.insert(idx);
                }
            }
        }

        let mut deck = Deck::new(num_players, rng.clone());
        let players: Vec<Player> = (0..num_players)
            .map(|i| {
                Player::new(
                    PlayerId::from_usize(i),
                    deck.draw_n(STARTING_HAND_SIZE as usize),
                )
            })
            .collect();
        let discard_pile = deck.draw_n(num_players);

        let mut initial_turn_order: Vec<usize> = (0..num_players).collect();
        initial_turn_order.shuffle(&mut rng);

        Self {
            seed: actual_seed,
            era: Era::Canal,
            wild_location_cards_available,
            wild_industry_cards_available,
            built_roads: RoadSet::new(),
            build_locations_occupied: BuildLocationSet::new(),
            building_types: vec![None; NUM_BL],
            player_road_mask: vec![RoadSet::new(); num_players],
            player_network_mask: vec![Connectivity::new(); num_players],
            player_building_mask: vec![BuildLocationSet::new(); num_players],
            connectivity: Connectivity::new(),
            trade_post_slots,
            num_trade_posts: num_trade_posts_active,
            remaining_market_coal: MAX_MARKET_COAL - 1,
            remaining_market_iron: MAX_MARKET_IRON - 2,
            trade_post_beer,
            deck,
            discard_pile,
            players,
            current_player_idx: initial_turn_order[0],
            rng,
            coal_locations: BuildLocationSet::new(),
            iron_locations: BuildLocationSet::new(),
            beer_locations: BuildLocationSet::new(),
            bl_to_building: HashMap::new(),
            visible_vps: [0; N_PLAYERS],
            turn_order: initial_turn_order,
        }
    }

    /// Printed merchant icons score even when that location has no active merchant.
    pub fn link_victory_points(&self, road_idx: usize) -> u16 {
        LINK_LOCATIONS[road_idx]
            .locations
            .ones()
            .map(|town| {
                let location = LocationName::from_usize(town);
                let merchant_points = if town >= LocationName::Shrewbury.as_usize() {
                    2
                } else {
                    0
                };
                merchant_points
                    + location
                        .to_bl_set()
                        .ones()
                        .filter_map(|bl| {
                            self.bl_to_building.get(&bl).filter(|b| b.flipped).map(|b| {
                                u16::from(
                                    INDUSTRY_MAT[b.industry.as_usize()][b.level.as_usize()].road_vp,
                                )
                            })
                        })
                        .sum::<u16>()
            })
            .sum()
    }

    pub fn get_player_building_mask(&self, player_id: PlayerId) -> &BuildLocationSet {
        if player_id.as_usize() >= self.player_building_mask.len() {
            panic!("Player index out of bounds");
        }
        &self.player_building_mask[player_id.as_usize()]
    }

    pub fn get_player_road_mask(&self, player_id: PlayerId) -> &RoadSet {
        if player_id.as_usize() >= self.player_road_mask.len() {
            panic!("Player index out of bounds");
        }
        &self.player_road_mask[player_id.as_usize()]
    }

    pub fn get_building_data_at_loc(&self, loc: usize) -> &BuildingTypeData {
        if let Some(building) = self.bl_to_building.get(&loc) {
            return &INDUSTRY_MAT[building.industry as usize][building.level.as_usize()];
        } else {
            panic!("Building at location {} not found", loc);
        }
    }

    pub fn get_num_coal_on_board(&self) -> u8 {
        let mut num_coal = 0;
        for loc in self.coal_locations.ones() {
            if let Some(building) = self.bl_to_building.get(&loc) {
                num_coal += building.get_resource_amt() as u8;
            }
        }
        num_coal
    }

    pub fn get_tile_at_loc(&self, loc: usize) -> Option<BuildingTypeData> {
        self.bl_to_building
            .get(&loc)
            .map(|b| INDUSTRY_MAT[b.industry as usize][b.level.as_usize()])
    }

    pub fn is_location_connected_to_trade_post(&self, location: LocationName) -> bool {
        for trade_post in TRADE_POST_ORDERED {
            if self
                .connectivity
                .are_towns_connected(location, trade_post.to_location_name())
            {
                return true;
            }
        }
        false
    }

    pub fn is_bl_connected_to_trade_post(&self, build_loc_idx: usize) -> bool {
        self.is_location_connected_to_trade_post(LocationName::from_bl_idx(build_loc_idx))
    }

    // =========================================================================
    // Network Helpers
    // =========================================================================

    /// Get all locations connected to a given location by built roads
    pub fn get_location_neighbors(&self, location: LocationName) -> LocationSet {
        let mut neighbors = LocationSet::new();
        // Get all roads connected to this location
        for road in location.get_roads().intersection(&self.built_roads) {
            neighbors.union_with(&LINK_LOCATIONS[road as usize].locations);
        }
        neighbors
    }

    /// Check if player has any network presence (buildings or roads)
    pub fn player_has_network(&self, player_idx: usize) -> bool {
        !self.player_building_mask[player_idx].is_clear()
            || !self.player_road_mask[player_idx].is_clear()
    }

    /// Check if a build location is in the player's network
    /// A location is in player's network if:
    /// 1. Player has a building there, or
    /// 2. It's connected via player's connectivity (set of links)
    pub fn is_in_player_network(&self, player_idx: usize, bl_idx: usize) -> bool {
        self.players[player_idx]
            .get_locations_in_network(self)
            .contains(LocationName::from_bl_idx(bl_idx).as_usize())
    }

    /// Check if player already has a building in this town (for canal era rule)
    pub fn player_has_building_in_town(&self, player_idx: usize, town_idx: usize) -> bool {
        self.player_building_mask[player_idx]
            .contains_any_in_range(TOWNS_RANGES[town_idx].0..TOWNS_RANGES[town_idx].1)
    }

    // =========================================================================
    // Resource Helpers
    // =========================================================================

    pub fn is_connected_to_coal(&self, build_loc_idx: usize) -> bool {
        let connected_locations = self
            .connectivity
            .get_connected_locations(LocationName::from_bl_idx(build_loc_idx));
        !connected_locations.is_clear()
            || connected_locations
                .to_bl_set()
                .intersection_count(&self.coal_locations)
                > 0
    }

    pub fn get_resource_amount_at_bl(&self, bl_idx: usize) -> u8 {
        if let Some(building) = self.bl_to_building.get(&bl_idx) {
            building.resource_amt
        } else {
            0
        }
    }

    /// Get total coal available on board (from all coal mines)
    pub fn get_total_coal_on_board(&self) -> u8 {
        self.get_num_coal_on_board()
    }

    /// Get total iron available on board (from all iron works)
    pub fn get_total_iron_on_board(&self) -> u8 {
        let mut total = 0;
        for loc in self.iron_locations.ones() {
            if let Some(building) = self.bl_to_building.get(&loc) {
                if !building.flipped {
                    total += building.resource_amt;
                }
            }
        }
        total
    }

    /// Get all connected coal sources from a build location
    /// Returns Vec of (location_idx, resource_amount)
    pub fn get_connected_coal_sources(&self, build_loc_idx: usize) -> Vec<(usize, u8)> {
        let build_location = LocationName::from_bl_idx(build_loc_idx);
        let mut sources = Vec::new();

        for coal_loc in self.coal_locations.ones() {
            if let Some(building) = self.bl_to_building.get(&coal_loc) {
                if !building.flipped && building.resource_amt > 0 {
                    let coal_location = LocationName::from_bl_idx(coal_loc);
                    if self
                        .connectivity
                        .are_towns_connected(build_location, coal_location)
                    {
                        sources.push((coal_loc, building.resource_amt));
                    }
                }
            }
        }
        sources
    }

    /// Get all iron sources on board (iron doesn't need connection)
    /// Returns Vec of (location_idx, resource_amount)
    pub fn get_all_iron_sources(&self) -> Vec<(usize, u8)> {
        let mut sources = Vec::new();

        for iron_loc in self.iron_locations.ones() {
            if let Some(building) = self.bl_to_building.get(&iron_loc) {
                if !building.flipped && building.resource_amt > 0 {
                    sources.push((iron_loc, building.resource_amt));
                }
            }
        }
        sources
    }

    /// Check if all resources of a type are depleted (for overbuild rules)
    pub fn is_resource_depleted(&self, industry: IndustryType) -> bool {
        match industry {
            IndustryType::Coal => {
                self.remaining_market_coal == 0 && self.get_total_coal_on_board() == 0
            }
            IndustryType::Iron => {
                self.remaining_market_iron == 0 && self.get_total_iron_on_board() == 0
            }
            _ => false,
        }
    }

    /// Get the town index for a build location
    pub fn get_town_for_bl(&self, bl_idx: usize) -> Option<usize> {
        TOWNS_RANGES
            .iter()
            .position(|range| bl_idx >= range.0 && bl_idx < range.1)
    }

    /// Alias for is_bl_connected_to_trade_post for consistency
    pub fn is_connected_to_trade_post(&self, build_loc_idx: usize) -> bool {
        self.is_bl_connected_to_trade_post(build_loc_idx)
    }

    // =========================================================================
    // Actions
    // =========================================================================
    /// Discard a card from a player's hand, returning wild cards to pool
    pub fn discard_card(&mut self, player_idx: usize, card_idx: usize) {
        if card_idx >= self.players[player_idx].hand.cards.len() {
            eprintln!(
                "Error: Invalid card index {} to discard for player {}. Hand size: {}.",
                card_idx,
                player_idx,
                self.players[player_idx].hand.cards.len()
            );
            return;
        }

        let card = self.players[player_idx].hand.cards.remove(card_idx);

        match card.card_type {
            CardType::WildLocation => {
                self.wild_location_cards_available =
                    self.wild_location_cards_available.saturating_add(1)
            }
            CardType::WildIndustry => {
                self.wild_industry_cards_available =
                    self.wild_industry_cards_available.saturating_add(1)
            }
            _ => self.discard_pile.push(card),
        }
    }

    // =========================================================================
    // Network Actions
    // =========================================================================

    pub fn place_link(&mut self, player_idx: usize, road_idx: usize) {
        self.built_roads.insert(road_idx);

        // Update global connectivity and player network
        self.connectivity.add_road(road_idx);
        self.player_network_mask[player_idx].add_road(road_idx);

        self.player_road_mask[player_idx].insert(road_idx);
    }

    /// Handle a building flip - update income and resource tracking
    pub fn handle_building_flip(&mut self, loc: usize) {
        if let Some(building) = self.bl_to_building.get_mut(&loc) {
            if building.is_flipped() {
                let building_data =
                    &INDUSTRY_MAT[building.industry as usize][building.level.as_usize()];
                self.players[building.owner.as_usize()]
                    .increase_income_level(building_data.income as u8);
            }
        } else {
            panic!(
                "handle_building_flip called on a non-existent building at loc {}",
                loc
            );
        }
    }

    // =============================================================================
    // Building Removal Functions
    // =============================================================================

    /// Remove a building from the board (for overbuilding, era cleanup, etc.)
    pub fn remove_building_from_board(&mut self, loc_idx: usize) {
        if let Some(removed_building) = self.bl_to_building.remove(&loc_idx) {
            let owner_idx = removed_building.owner.as_usize();
            self.build_locations_occupied.remove(loc_idx);
            self.building_types[loc_idx] = None;
            self.player_building_mask[owner_idx].remove(loc_idx);

            match removed_building.industry {
                IndustryType::Coal => self.coal_locations.remove(loc_idx),
                IndustryType::Iron => self.iron_locations.remove(loc_idx),
                IndustryType::Beer => self.beer_locations.remove(loc_idx),
                _ => {}
            }
        }
    }
}
