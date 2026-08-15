use crate::board::resources::{BreweryBeerSource, ResourceManager, ResourceSource};
use crate::board::BoardState;
use crate::core::static_data::{CANAL_ONLY, RAIL_ONLY};
use crate::core::types::*;
use fixedbitset::FixedBitSet;
/// Network action options and data structures
#[derive(Debug, Clone)]
pub struct SingleRailroadOption {
    pub road_idx: usize,
    pub potential_coal_sources: FixedBitSet,
}

impl SingleRailroadOption {
    pub fn new(road_idx: usize, potential_coal_sources: FixedBitSet) -> Self {
        Self {
            road_idx,
            potential_coal_sources,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DoubleRailroadSecondLinkOption {
    pub second_road_idx: usize,
    pub potential_coal_sources_for_second_link: Vec<ResourceSource>,
    pub potential_beer_sources_for_action: Vec<BreweryBeerSource>,
    pub own_brewery_sources: Vec<BreweryBeerSource>,
}

impl DoubleRailroadSecondLinkOption {
    pub fn from_bitsets(
        second_road_idx: usize,
        coal_sources_bs: FixedBitSet,
        opponent_beer_bs: FixedBitSet,
        own_beer_bs: FixedBitSet,
    ) -> Self {
        Self {
            second_road_idx,
            potential_coal_sources_for_second_link: coal_sources_bs
                .ones()
                .map(ResourceSource::from)
                .collect(),
            potential_beer_sources_for_action: opponent_beer_bs
                .ones()
                .map(BreweryBeerSource::OpponentBrewery)
                .collect(),
            own_brewery_sources: own_beer_bs
                .ones()
                .map(BreweryBeerSource::OwnBrewery)
                .collect(),
        }
    }
}

/// Network action logic (canals and railroads)
pub struct NetworkActions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkError {
    InvalidCanalBuild,
    InvalidRailPlacement,
    MissingActionBeer,
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::InvalidCanalBuild => write!(f, "invalid canal build"),
            NetworkError::InvalidRailPlacement => write!(f, "invalid rail link placement"),
            NetworkError::MissingActionBeer => {
                write!(f, "failed to consume beer for double railroad")
            }
        }
    }
}

impl NetworkActions {
    /// Execute canal building action
    pub fn execute_build_canal_action(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        road_idx: usize,
        card_to_discard_idx: usize,
    ) -> Result<(), NetworkError> {
        // Discard card first
        board_state.discard_card(player_idx, card_to_discard_idx);

        // Execute canal building
        Self::build_canal(board_state, player_idx, road_idx)
    }

    /// Execute single railroad building action
    pub fn execute_build_single_rail_action(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        road_idx: usize,
        coal_source: ResourceSource,
        card_to_discard_idx: usize,
    ) -> Result<(), NetworkError> {
        board_state.discard_card(player_idx, card_to_discard_idx);
        Self::build_railroad(board_state, player_idx, road_idx, coal_source)
    }

    /// Execute double railroad building action
    pub fn execute_build_double_rail_action(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        road1_idx: usize,
        road2_idx: usize,
        coal_source1: ResourceSource,
        coal_source2: ResourceSource,
        action_beer_source: BreweryBeerSource,
        card_to_discard_idx: usize,
    ) -> Result<(), NetworkError> {
        board_state.discard_card(player_idx, card_to_discard_idx);
        Self::build_double_railroad(
            board_state,
            player_idx,
            vec![road1_idx, road2_idx],
            vec![coal_source1, coal_source2],
            action_beer_source,
        )
    }

    fn build_canal(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        road_idx: usize,
    ) -> Result<(), NetworkError> {
        if board_state.era != Era::Canal
            || !CANAL_ONLY.contains(road_idx)
            || board_state.built_roads.contains(road_idx)
        {
            return Err(NetworkError::InvalidCanalBuild);
        }

        board_state.players[player_idx].pay(CANAL_PRICE);
        board_state.place_link(player_idx, road_idx);
        Ok(())
    }

    fn build_railroad(
        board_state: &mut BoardState,
        player_idx: usize,
        road_idx: usize,
        coal_source: ResourceSource,
    ) -> Result<(), NetworkError> {
        board_state.players[player_idx].pay(ONE_RAILROAD_PRICE);
        Self::place_single_rail_link_and_consume_coal(
            board_state,
            player_idx,
            road_idx,
            coal_source,
        )
    }

    fn build_double_railroad(
        board_state: &mut BoardState,
        player_idx: usize,
        roads: Vec<usize>,
        coal_sources: Vec<ResourceSource>,
        beer_source_for_action: BreweryBeerSource,
    ) -> Result<(), NetworkError> {
        board_state.players[player_idx].pay(TWO_RAILROAD_PRICE);

        // Consume beer for action using centralized function
        let beer_loc = match beer_source_for_action {
            BreweryBeerSource::OwnBrewery(loc) | BreweryBeerSource::OpponentBrewery(loc) => loc,
        };

        if !ResourceManager::consume_beer_from_brewery(board_state, beer_loc) {
            board_state.players[player_idx].gain_money(TWO_RAILROAD_PRICE);
            return Err(NetworkError::MissingActionBeer);
        }

        // Place both links
        Self::place_single_rail_link_and_consume_coal(
            board_state,
            player_idx,
            roads[0],
            coal_sources[0],
        )?;
        Self::place_single_rail_link_and_consume_coal(
            board_state,
            player_idx,
            roads[1],
            coal_sources[1],
        )?;
        Ok(())
    }

    fn place_single_rail_link_and_consume_coal(
        board_state: &mut BoardState,
        player_idx: usize,
        road_idx: usize,
        coal_source: ResourceSource,
    ) -> Result<(), NetworkError> {
        if board_state.era != Era::Railroad
            || board_state.built_roads.contains(road_idx)
            || !RAIL_ONLY.contains(road_idx)
        {
            return Err(NetworkError::InvalidRailPlacement);
        }

        board_state.place_link(player_idx, road_idx);

        // Consume coal using centralized function
        ResourceManager::consume_coal(board_state, player_idx, vec![coal_source], 1);
        Ok(())
    }
}
