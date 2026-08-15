use crate::board::resources::{BeerSellSource, ResourceManager};
use crate::core::types::*;
use crate::market::merchants::{slot_to_trade_post, TradePostBonus, TRADE_POST_TO_BONUS};
use fixedbitset::FixedBitSet;

/// Options for selling actions
#[derive(Debug, Clone)]
pub struct SellOption {
    pub location: usize,
    pub trade_posts: FixedBitSet, // Trade Post *indices* (0-4 for Shrewbury etc.)
    pub beer_locations: FixedBitSet, // Locations of available beer (breweries or merchant slots)
    pub beer_counter: u8,         // Total available beer units for this specific sell option
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SellChoice {
    pub location: usize,
    pub beer_sources: Vec<BeerSellSource>,
}

impl SellChoice {
    pub fn new(location: usize, beer_sources: Vec<BeerSellSource>) -> Self {
        Self {
            location,
            beer_sources,
        }
    }
}

/// Selling action logic
pub struct SellActions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SellError {
    BuildingNotSellable(usize),
    BuildingDataMissing(usize),
    InvalidTradePostBeerSource(usize),
    MissingFreeDevelopmentChoice,
    InsufficientBeer {
        location: usize,
        needed: u8,
        consumed: u8,
        sources: Vec<BeerSellSource>,
    },
}

impl std::fmt::Display for SellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SellError::BuildingNotSellable(loc) => write!(f, "cannot sell building at {}", loc),
            SellError::BuildingDataMissing(loc) => write!(f, "building data missing at {}", loc),
            SellError::InvalidTradePostBeerSource(slot) => {
                write!(f, "invalid trade post beer source {}", slot)
            }
            SellError::MissingFreeDevelopmentChoice => {
                write!(f, "free development bonus requires industry choice")
            }
            SellError::InsufficientBeer {
                location,
                needed,
                consumed,
                sources,
            } => write!(
                f,
                "sale at {location} consumed {consumed} of {needed} beer from {sources:?}"
            ),
        }
    }
}

impl SellActions {
    /// Execute selling of buildings
    pub fn execute_sell_all_buildings(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        sell_choices: Vec<SellChoice>,
        free_dev_choice: Option<IndustryType>,
    ) -> Result<(), SellError> {
        use crate::core::static_data::INDUSTRY_MAT;

        for choice in sell_choices {
            let loc = choice.location;

            // Get building data
            let building_data_opt = board_state
                .bl_to_building
                .get(&loc)
                .map(|b| &INDUSTRY_MAT[b.industry as usize][b.level.as_usize()]);

            if let Some(building_data) = building_data_opt {
                if board_state
                    .bl_to_building
                    .get(&loc)
                    .map_or(true, |b| b.owner as usize != player_idx || b.flipped)
                {
                    return Err(SellError::BuildingNotSellable(loc));
                }

                if building_data.beer_needed > 0 {
                    Self::consume_beer_from_sources(
                        board_state,
                        player_idx,
                        loc,
                        choice.beer_sources,
                        building_data.beer_needed,
                        free_dev_choice,
                    )?;
                }

                if let Some(building_to_flip) = board_state.bl_to_building.get_mut(&loc) {
                    building_to_flip.flip(); // Mark as flipped
                }

                board_state.handle_building_flip(loc); // Process VP/income etc.
            } else {
                return Err(SellError::BuildingDataMissing(loc));
            }
        }
        Ok(())
    }

    fn consume_beer_from_sources(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        location: usize,
        beer_sources: Vec<BeerSellSource>,
        beer_needed: u8,
        free_dev_choice: Option<IndustryType>,
    ) -> Result<(), SellError> {
        let mut beer_to_consume = beer_needed;
        let attempted_sources = beer_sources.clone();

        for source in beer_sources {
            if beer_to_consume == 0 {
                break;
            }

            match source {
                BeerSellSource::Building(loc) => {
                    // Use centralized beer consumption function
                    if ResourceManager::consume_beer_from_brewery(board_state, loc) {
                        beer_to_consume -= 1;
                    }
                }
                BeerSellSource::TradePost(slot_idx) => {
                    if !board_state.trade_post_beer.contains(slot_idx) {
                        return Err(SellError::InvalidTradePostBeerSource(slot_idx));
                    }

                    board_state.trade_post_beer.set(slot_idx, false); // Consume it
                    let tp_enum = slot_to_trade_post(slot_idx);
                    let bonus = TRADE_POST_TO_BONUS[tp_enum.to_index()];

                    match bonus {
                        TradePostBonus::FreeDevelopment => {
                            if let Some(dev_ind) = free_dev_choice {
                                board_state.players[player_idx]
                                    .industry_mat
                                    .pop_tile(dev_ind);
                            } else {
                                return Err(SellError::MissingFreeDevelopmentChoice);
                            }
                        }
                        TradePostBonus::Income2 => {
                            board_state.players[player_idx].increase_income_level(2)
                        }
                        TradePostBonus::Vp3 => board_state.players[player_idx].victory_points += 3,
                        TradePostBonus::Vp4 => board_state.players[player_idx].victory_points += 4,
                        TradePostBonus::Money5 => board_state.players[player_idx].gain_money(5),
                    }
                    beer_to_consume -= 1;
                }
            }
        }

        if beer_to_consume > 0 {
            return Err(SellError::InsufficientBeer {
                location,
                needed: beer_needed,
                consumed: beer_needed - beer_to_consume,
                sources: attempted_sources,
            });
        }
        Ok(())
    }
}
