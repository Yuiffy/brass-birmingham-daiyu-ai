use crate::actions::network::{DoubleRailroadSecondLinkOption, SingleRailroadOption};
use crate::board::connectivity::{
    calculate_connectivity_if_roads_built, calculate_global_connectivity_after_road_build,
    Connectivity,
};
use crate::board::resources::{BreweryBeerSource, ResourceManager, ResourceSource};
use crate::board::BoardState;
use crate::core::building::BuiltBuilding;
use crate::core::static_data::{CANAL_ONLY, LINK_LOCATIONS, N_LINK_LOCATIONS, RAIL_ONLY};
use crate::core::types::*;
use fixedbitset::FixedBitSet;
use std::collections::HashMap;

/// Network action validation logic
pub struct NetworkValidator;

impl NetworkValidator {
    /// Get valid road options for a player based on their network
    pub fn get_valid_road_options(
        board_state: &BoardState,
        player_idx: usize,
        era: Era,
    ) -> RoadSet {
        let mut valid_roads = RoadSet::new();
        let _player = &board_state.players[player_idx];

        // Get roads adjacent to player's network
        let roads_in_network = Self::get_roads_adjacent_to_player_network(board_state, player_idx);
        valid_roads.union_with(&roads_in_network);
        valid_roads.difference_with(&board_state.built_roads);

        match era {
            Era::Canal => valid_roads.intersect_with(&CANAL_ONLY),
            Era::Railroad => valid_roads.intersect_with(&RAIL_ONLY),
        }
        valid_roads
    }

    /// Get roads that are adjacent to player's own network (buildings + own roads).
    /// A road is adjacent to a player's network if it directly touches a town
    /// where the player has a building, or shares an endpoint with one of the
    /// player's own roads.  Global connectivity (other players' roads) is
    /// intentionally NOT used — only the player's own presence matters.
    fn get_roads_adjacent_to_player_network(
        board_state: &BoardState,
        player_idx: usize,
    ) -> RoadSet {
        use crate::core::locations::LocationName;

        let mut roads = RoadSet::new();
        let player_buildings = &board_state.player_building_mask[player_idx];

        // If player has no buildings and no roads, they can build anywhere
        if player_buildings.count_ones() == 0
            && board_state.player_road_mask[player_idx].count_ones() == 0
        {
            roads.insert_range(0..N_ROAD_LOCATIONS);
            return roads;
        }

        // Roads that directly touch a town where the player has a building
        for bl_idx in player_buildings.ones() {
            let town = LocationName::from_bl_idx(bl_idx);
            for road_idx in 0..N_LINK_LOCATIONS {
                if LINK_LOCATIONS[road_idx].locations.contains(town.as_usize()) {
                    roads.insert(road_idx);
                }
            }
        }

        // Roads that share an endpoint with one of the player's own roads
        for road_idx in board_state.player_road_mask[player_idx].ones() {
            let link = &LINK_LOCATIONS[road_idx];
            for loc in link.locations.ones() {
                for other_road_idx in 0..N_LINK_LOCATIONS {
                    if other_road_idx == road_idx {
                        continue;
                    }
                    if LINK_LOCATIONS[other_road_idx].locations.contains(loc) {
                        roads.insert(other_road_idx);
                    }
                }
            }
        }

        roads
    }

    /// Get valid canal options for a player
    pub fn get_valid_canal_options(board_state: &BoardState, player_idx: usize) -> Vec<usize> {
        if board_state.era != Era::Canal {
            return Vec::new();
        }

        let player = &board_state.players[player_idx];
        if !player.can_afford(CANAL_PRICE) {
            return Vec::new();
        }

        let valid_roads = Self::get_valid_road_options(board_state, player_idx, Era::Canal);
        valid_roads.ones().collect()
    }

    /// Get valid single railroad options for a player
    pub fn get_valid_single_rail_options(
        board_state: &BoardState,
        player_idx: usize,
    ) -> Vec<SingleRailroadOption> {
        let mut options = Vec::new();
        if board_state.era != Era::Railroad {
            return options;
        }

        let player = &board_state.players[player_idx];
        let valid_roads = Self::get_valid_road_options(board_state, player_idx, Era::Railroad);

        for road_idx in valid_roads.ones() {
            if !player.can_afford(ONE_RAILROAD_PRICE) {
                continue;
            }

            // Get coal sources that would be available after building this road
            let potential_coal_sources =
                Self::get_coal_sources_after_road_build(board_state, player_idx, road_idx);

            if potential_coal_sources.is_empty() {
                continue;
            }

            let can_execute_with_some_coal = potential_coal_sources.iter().any(|src| match src {
                ResourceSource::Building(_) => player.can_afford(ONE_RAILROAD_PRICE),
                ResourceSource::Market => player.can_afford(
                    ONE_RAILROAD_PRICE
                        + ResourceManager::get_coal_price(board_state.remaining_market_coal, 1),
                ),
            });

            if can_execute_with_some_coal {
                let mut sources_bs = FixedBitSet::with_capacity(N_LOCATIONS + 1);
                for source in &potential_coal_sources {
                    match source {
                        ResourceSource::Building(loc) => sources_bs.insert(*loc),
                        ResourceSource::Market => sources_bs.insert(N_LOCATIONS),
                    }
                }
                options.push(SingleRailroadOption::new(road_idx, sources_bs));
            }
        }
        options
    }

    /// Get valid double railroad first link options
    pub fn get_valid_double_rail_first_link_options(
        board_state: &BoardState,
        player_idx: usize,
    ) -> Vec<SingleRailroadOption> {
        if !Self::can_double_railroad(board_state, player_idx) {
            return Vec::new();
        }
        Self::get_valid_single_rail_options(board_state, player_idx)
    }

    /// Check if player can build double railroad
    pub fn can_double_railroad(board_state: &BoardState, player_idx: usize) -> bool {
        let player = &board_state.players[player_idx];

        if board_state.era != Era::Railroad {
            return false;
        }

        // 1. Affordability of base action cost
        if !player.can_afford(TWO_RAILROAD_PRICE) {
            return false;
        }

        // 2. Beer availability (any player's brewery with beer)
        let has_any_player_brewery_beer = board_state
            .bl_to_building
            .values()
            .any(|b| b.industry == IndustryType::Beer && !b.flipped && b.resource_amt > 0);

        if !has_any_player_brewery_beer {
            return false;
        }

        // 3. At least two available rail link slots on the board
        let mut available_rail_links = 0;
        for i in 0..N_ROAD_LOCATIONS {
            if !CANAL_ONLY.contains(i) && !board_state.built_roads.contains(i) {
                available_rail_links += 1;
                if available_rail_links >= 2 {
                    break;
                }
            }
        }
        if available_rail_links < 2 {
            return false;
        }

        // 4. Ability to source 2 coal (from board or market, total)
        let mut total_coal_available = 0u8;
        for coal_loc in board_state.coal_locations.ones() {
            if let Some(building) = board_state.bl_to_building.get(&coal_loc) {
                if !building.flipped {
                    total_coal_available += building.resource_amt;
                }
            }
        }

        let coal_needed_from_market = 2u8.saturating_sub(total_coal_available);
        if !player.can_afford(
            TWO_RAILROAD_PRICE
                + ResourceManager::get_coal_price(
                    board_state.remaining_market_coal,
                    coal_needed_from_market,
                ),
        ) {
            return false;
        }

        true
    }

    /// Get options for second rail link in double railroad action
    pub fn get_options_for_second_rail_link(
        board_state: &BoardState,
        player_idx: usize,
        first_road_idx: usize,
        first_road_coal_source: &ResourceSource,
        hypothetical_money_remaining: u16,
    ) -> Vec<DoubleRailroadSecondLinkOption> {
        let mut second_link_options = Vec::new();

        // Simulate state after first link and its coal consumption
        let mut hypothetical_bl_to_building = board_state.bl_to_building.clone();
        let mut hypothetical_market_coal = board_state.remaining_market_coal;

        match first_road_coal_source {
            ResourceSource::Building(loc) => {
                if let Some(building) = hypothetical_bl_to_building.get_mut(loc) {
                    if building.resource_amt > 0 {
                        building.resource_amt -= 1;
                    }
                }
            }
            ResourceSource::Market => {
                if hypothetical_market_coal > 0 {
                    hypothetical_market_coal -= 1;
                }
            }
        }

        // Connectivity after the first link
        let hypothetical_conn = calculate_global_connectivity_after_road_build(
            &board_state.connectivity,
            first_road_idx,
        );

        // Find valid second links
        let mut potential_second_roads = RoadSet::new();
        potential_second_roads.insert_range(0..N_ROAD_LOCATIONS);
        potential_second_roads.difference_with(&CANAL_ONLY);
        potential_second_roads.remove(first_road_idx);
        potential_second_roads.difference_with(&board_state.built_roads);

        // Filter to roads in player's network after first link
        let mut valid_second_roads = RoadSet::new();
        for road_idx in potential_second_roads.ones() {
            // Check if road is adjacent to player's network after first road
            if Self::is_road_adjacent_to_network_after_road(
                board_state,
                player_idx,
                first_road_idx,
                road_idx,
            ) {
                valid_second_roads.insert(road_idx);
            }
        }

        for second_road_idx in valid_second_roads.ones() {
            // Get coal sources for second link
            let coal_sources = Self::get_coal_sources_given_state(
                board_state,
                second_road_idx,
                &hypothetical_bl_to_building,
                hypothetical_market_coal,
                hypothetical_money_remaining,
                &hypothetical_conn,
            );

            if coal_sources.is_empty() {
                continue;
            }

            // Get beer sources for the double link action
            let (own_beer, opp_beer) = Self::get_beer_sources_for_double_rail(
                board_state,
                player_idx,
                first_road_idx,
                second_road_idx,
            );

            if own_beer.is_empty() && opp_beer.is_empty() {
                continue;
            }

            second_link_options.push(DoubleRailroadSecondLinkOption {
                second_road_idx,
                potential_coal_sources_for_second_link: coal_sources,
                potential_beer_sources_for_action: opp_beer,
                own_brewery_sources: own_beer,
            });
        }

        second_link_options
    }

    /// Get coal sources available after hypothetically building a road
    fn get_coal_sources_after_road_build(
        board_state: &BoardState,
        player_idx: usize,
        road_idx: usize,
    ) -> Vec<ResourceSource> {
        let hypothetical_conn =
            calculate_global_connectivity_after_road_build(&board_state.connectivity, road_idx);

        Self::get_coal_sources_given_state(
            board_state,
            road_idx,
            &board_state.bl_to_building,
            board_state.remaining_market_coal,
            board_state.players[player_idx].money,
            &hypothetical_conn,
        )
    }

    fn get_coal_sources_given_state(
        board_state: &BoardState,
        road_idx: usize,
        buildings: &HashMap<usize, BuiltBuilding>,
        market_coal: u8,
        player_money: u16,
        connectivity: &Connectivity,
    ) -> Vec<ResourceSource> {
        use crate::core::locations::LocationName;

        let mut sources = Vec::new();
        let link = &LINK_LOCATIONS[road_idx];

        // Get endpoints of this road
        let road_endpoints: Vec<usize> = link.locations.ones().collect();

        // Find coal mines connected to road endpoints
        for coal_loc in board_state.coal_locations.ones() {
            if let Some(building) = buildings.get(&coal_loc) {
                if !building.flipped && building.resource_amt > 0 {
                    let coal_town = LocationName::from_bl_idx(coal_loc);
                    // Check if any road endpoint is connected to this coal mine
                    for &endpoint in &road_endpoints {
                        if connectivity
                            .are_towns_connected(LocationName::from_usize(endpoint), coal_town)
                        {
                            sources.push(ResourceSource::Building(coal_loc));
                            break;
                        }
                    }
                }
            }
        }

        // Check market coal availability
        // Trade posts are at LocationName indices 22-26
        const TRADE_POST_START: usize = 22;
        let connected_to_market = road_endpoints.iter().any(|&endpoint| {
            for tp_idx in 0..NUM_TRADE_POSTS {
                let tp_loc = LocationName::from_usize(TRADE_POST_START + tp_idx);
                if connectivity.are_towns_connected(LocationName::from_usize(endpoint), tp_loc) {
                    return true;
                }
            }
            false
        });

        if connected_to_market {
            let coal_price = if market_coal > 0 {
                ResourceManager::get_coal_price(market_coal, 1)
            } else {
                8 // Empty market price
            };
            if player_money >= coal_price {
                sources.push(ResourceSource::Market);
            }
        }

        sources.sort_unstable();
        sources.dedup();
        sources
    }

    fn is_road_adjacent_to_network_after_road(
        board_state: &BoardState,
        player_idx: usize,
        first_road_idx: usize,
        second_road_idx: usize,
    ) -> bool {
        use crate::core::locations::LocationName;

        let first_link = &LINK_LOCATIONS[first_road_idx];
        let second_link = &LINK_LOCATIONS[second_road_idx];

        // Check if second road shares a location with first road
        for loc1 in first_link.locations.ones() {
            for loc2 in second_link.locations.ones() {
                if loc1 == loc2 {
                    return true;
                }
            }
        }

        // Check if second road is adjacent to player's existing network
        for bl_idx in board_state.player_building_mask[player_idx].ones() {
            let town = LocationName::from_bl_idx(bl_idx);
            for loc in second_link.locations.ones() {
                if board_state
                    .connectivity
                    .are_towns_connected(town, LocationName::from_usize(loc))
                {
                    return true;
                }
            }
        }

        false
    }

    fn get_beer_sources_for_double_rail(
        board_state: &BoardState,
        player_idx: usize,
        first_road_idx: usize,
        second_road_idx: usize,
    ) -> (Vec<BreweryBeerSource>, Vec<BreweryBeerSource>) {
        use crate::core::locations::LocationName;

        let mut own_beer = Vec::new();
        let mut opp_beer = Vec::new();

        let hypothetical_conn = calculate_connectivity_if_roads_built(
            &board_state.connectivity,
            &[first_road_idx, second_road_idx],
        );
        let second_link = &LINK_LOCATIONS[second_road_idx];
        let road_endpoints: Vec<usize> = second_link.locations.ones().collect();

        // Player's own breweries (don't need connection)
        for brewery_loc in board_state.player_building_mask[player_idx].ones() {
            if let Some(b) = board_state.bl_to_building.get(&brewery_loc) {
                if b.industry == IndustryType::Beer && !b.flipped && b.resource_amt > 0 {
                    own_beer.push(BreweryBeerSource::OwnBrewery(brewery_loc));
                }
            }
        }

        // Opponent's breweries (must be connected to second road)
        for p_other_idx in 0..board_state.players.len() {
            if p_other_idx == player_idx {
                continue;
            }
            for brewery_loc in board_state.player_building_mask[p_other_idx].ones() {
                if let Some(b) = board_state.bl_to_building.get(&brewery_loc) {
                    if b.industry == IndustryType::Beer && !b.flipped && b.resource_amt > 0 {
                        let brewery_town = LocationName::from_bl_idx(brewery_loc);
                        // Check if connected to second road
                        for &endpoint in &road_endpoints {
                            if hypothetical_conn.are_towns_connected(
                                brewery_town,
                                LocationName::from_usize(endpoint),
                            ) {
                                opp_beer.push(BreweryBeerSource::OpponentBrewery(brewery_loc));
                                break;
                            }
                        }
                    }
                }
            }
        }

        own_beer.sort_unstable();
        own_beer.dedup();
        opp_beer.sort_unstable();
        opp_beer.dedup();

        (own_beer, opp_beer)
    }
}
