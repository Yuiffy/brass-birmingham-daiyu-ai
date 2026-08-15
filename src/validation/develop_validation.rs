use crate::board::resources::ResourceSource;
use crate::core::static_data::INDUSTRY_MAT;
use crate::core::types::{IndustryType, N_INDUSTRIES, N_LEVELS};
use fixedbitset::FixedBitSet;

pub struct DevelopValidator;

impl DevelopValidator {
    pub fn can_source_iron(
        board_state: &crate::board::BoardState,
        player_idx: usize,
        num_needed: u8,
    ) -> bool {
        use crate::board::resources::ResourceManager;

        let board_iron = board_state
            .iron_locations
            .ones()
            .filter_map(|location| board_state.bl_to_building.get(&location))
            .filter(|building| !building.flipped)
            .map(|building| building.resource_amt)
            .sum::<u8>();
        let market_needed = num_needed.saturating_sub(board_iron);
        market_needed == 0
            || board_state.players[player_idx].can_afford(ResourceManager::get_iron_price(
                board_state.remaining_market_iron,
                market_needed,
            ))
    }

    /// Get valid development options for a player
    /// Returns a bitset where each bit represents an industry that can be developed
    pub fn get_valid_development_options(
        board_state: &crate::board::BoardState,
        player_idx: usize,
    ) -> FixedBitSet {
        use crate::board::resources::ResourceManager;

        let mut valid_options = FixedBitSet::with_capacity(N_INDUSTRIES);
        let player = &board_state.players[player_idx];

        // Check if player can afford at least one iron from market
        let can_afford_one_market_iron = player.can_afford(ResourceManager::get_iron_price(
            board_state.remaining_market_iron,
            1,
        ));

        // Check if there's any iron on the board
        let has_board_iron = board_state.iron_locations.ones().any(|loc_idx| {
            board_state
                .bl_to_building
                .get(&loc_idx)
                .map_or(false, |b| !b.flipped && b.resource_amt > 0)
        });

        // Need either board iron or ability to buy from market
        if !has_board_iron && !can_afford_one_market_iron {
            return valid_options;
        }

        for industry_idx in 0..N_INDUSTRIES {
            let industry = IndustryType::from_usize(industry_idx);
            if player.industry_mat.has_tiles_left(industry) {
                let current_level = player.industry_mat.get_lowest_level(industry);
                let current_level_usize = current_level.as_usize();
                if current_level_usize < N_LEVELS
                    && current_level_usize < INDUSTRY_MAT[industry_idx].len()
                {
                    let building_data = &INDUSTRY_MAT[industry_idx][current_level_usize];
                    if building_data.can_develop {
                        valid_options.insert(industry_idx);
                    }
                }
            }
        }

        valid_options
    }

    /// Get industries eligible for a *free* development bonus (e.g. Gloucester).
    /// Unlike normal Develop, this intentionally does not require iron availability
    /// or iron affordability because no iron is spent for this bonus.
    pub fn get_valid_free_development_options(
        board_state: &crate::board::BoardState,
        player_idx: usize,
    ) -> FixedBitSet {
        let mut valid_options = FixedBitSet::with_capacity(N_INDUSTRIES);
        let player = &board_state.players[player_idx];

        for industry_idx in 0..N_INDUSTRIES {
            let industry = IndustryType::from_usize(industry_idx);
            if player.industry_mat.has_tiles_left(industry) {
                let current_level = player.industry_mat.get_lowest_level(industry);
                let current_level_usize = current_level.as_usize();
                if current_level_usize < N_LEVELS
                    && current_level_usize < INDUSTRY_MAT[industry_idx].len()
                {
                    let building_data = &INDUSTRY_MAT[industry_idx][current_level_usize];
                    if building_data.can_develop {
                        valid_options.insert(industry_idx);
                    }
                }
            }
        }

        valid_options
    }

    /// Get iron sources for development action
    pub fn get_iron_sources_for_develop(
        board_state: &crate::board::BoardState,
        player_idx: usize,
        num_needed: u8,
    ) -> Vec<ResourceSource> {
        use crate::board::resources::ResourceManager;

        let mut sources = Vec::new();
        let player = &board_state.players[player_idx];
        let mut iron_found_on_board = 0u8;

        // Board Iron (any player, free, no network needed)
        for iron_loc in board_state.iron_locations.ones() {
            if let Some(building) = board_state.bl_to_building.get(&iron_loc) {
                if !building.flipped && building.resource_amt > 0 {
                    sources.push(ResourceSource::Building(iron_loc));
                    iron_found_on_board += building.resource_amt;
                }
            }
        }

        // Market Iron (if needed and affordable)
        if iron_found_on_board < num_needed {
            let needed_from_market = num_needed - iron_found_on_board;
            if player.can_afford(ResourceManager::get_iron_price(
                board_state.remaining_market_iron,
                needed_from_market,
            )) {
                sources.push(ResourceSource::Market);
            }
        } else if num_needed == 0 {
            sources.push(ResourceSource::Market);
        }

        sources.sort_unstable();
        sources.dedup();
        sources
    }

    /// Get iron cost for development (1 iron per tile developed)
    pub fn get_iron_cost_for_develop(num_develops: u8) -> u8 {
        num_develops
    }

    /// Check if player can perform develop action
    pub fn can_develop(board_state: &crate::board::BoardState, player_idx: usize) -> bool {
        let valid_options = Self::get_valid_development_options(board_state, player_idx);
        valid_options.count_ones(..) > 0
    }
}
