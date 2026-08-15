use crate::actions::sell::SellOption;
use crate::core::locations::LocationName;
use crate::core::static_data::INDUSTRY_MAT;
use crate::core::types::{IndustryType, NUM_BL, NUM_TRADE_POSTS};
use crate::market::merchants::slot_to_trade_post;
use fixedbitset::FixedBitSet;

/// Sell action validation logic
pub struct SellValidator;

impl SellValidator {
    /// Get all valid sell options for a player
    pub fn get_valid_sell_options(
        board_state: &crate::board::BoardState,
        player_idx: usize,
    ) -> Vec<SellOption> {
        let mut all_sell_options = Vec::new();

        for building_loc_idx in board_state.player_building_mask[player_idx].ones() {
            let building = match board_state.bl_to_building.get(&building_loc_idx) {
                Some(b) if !b.flipped => b,
                _ => continue, // Not player's, or no building, or already flipped
            };

            let is_sellable_industry = matches!(
                building.industry,
                IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
            );
            if !is_sellable_industry {
                continue;
            }

            let building_data =
                &INDUSTRY_MAT[building.industry as usize][building.level.as_usize()];
            let beer_needed_for_sale = building_data.beer_needed;

            // Find all distinct merchant *slots* it can sell to
            let mut potential_target_merchant_slots =
                FixedBitSet::with_capacity(NUM_TRADE_POSTS * 2);
            let building_location = LocationName::from_bl_idx(building_loc_idx);

            for (slot_idx, merchant_opt) in board_state.trade_post_slots.iter().enumerate() {
                if let Some(merchant) = merchant_opt {
                    if merchant.industries.contains(building.industry as usize) {
                        let trade_post_enum = slot_to_trade_post(slot_idx);
                        let trade_post_location = trade_post_enum.to_location_name();
                        if board_state
                            .connectivity
                            .are_towns_connected(building_location, trade_post_location)
                        {
                            potential_target_merchant_slots.insert(slot_idx);
                        }
                    }
                }
            }

            if potential_target_merchant_slots.count_ones(..) == 0 {
                continue; // No connected merchants of the right type for this building
            }

            // Aggregate ALL potential beer sources relevant for selling THIS building
            let mut available_beer_units_total = 0u8;
            let mut beer_source_locations_for_this_building =
                FixedBitSet::with_capacity(NUM_BL + NUM_TRADE_POSTS * 2);

            // 1. Player's own breweries (anywhere, unflipped, with beer)
            for brewery_loc in board_state.player_building_mask[player_idx].ones() {
                if let Some(b) = board_state.bl_to_building.get(&brewery_loc) {
                    if b.industry == IndustryType::Beer && !b.flipped && b.resource_amt > 0 {
                        available_beer_units_total += b.resource_amt;
                        beer_source_locations_for_this_building.insert(brewery_loc);
                    }
                }
            }

            // 2. Opponent's breweries (must be connected to building_loc_idx, unflipped, with beer)
            for p_other_idx in 0..board_state.players.len() {
                if p_other_idx == player_idx {
                    continue;
                }
                for brewery_loc in board_state.player_building_mask[p_other_idx].ones() {
                    let brewery_location = LocationName::from_bl_idx(brewery_loc);
                    if board_state
                        .connectivity
                        .are_towns_connected(building_location, brewery_location)
                    {
                        if let Some(b) = board_state.bl_to_building.get(&brewery_loc) {
                            if b.industry == IndustryType::Beer && !b.flipped && b.resource_amt > 0
                            {
                                available_beer_units_total += b.resource_amt;
                                beer_source_locations_for_this_building.insert(brewery_loc);
                            }
                        }
                    }
                }
            }

            // 3. Merchant beer from any of the potential_target_merchant_slots that currently have beer
            for slot_idx in potential_target_merchant_slots.ones() {
                if board_state.trade_post_beer.contains(slot_idx) {
                    // Check if beer barrel is present at this slot
                    available_beer_units_total += 1;
                    beer_source_locations_for_this_building.insert(NUM_BL + slot_idx);
                }
            }

            if beer_needed_for_sale > 0 && available_beer_units_total < beer_needed_for_sale {
                continue; // Not enough beer from any combined source for this building to make a sale
            }

            // Convert merchant slots to trade post enums
            let mut valid_trade_post_enums = FixedBitSet::with_capacity(NUM_TRADE_POSTS);
            for slot_idx in potential_target_merchant_slots.ones() {
                valid_trade_post_enums.insert(slot_to_trade_post(slot_idx).to_index());
            }

            all_sell_options.push(SellOption {
                location: building_loc_idx,
                trade_posts: valid_trade_post_enums,
                beer_locations: beer_source_locations_for_this_building,
                beer_counter: available_beer_units_total,
            });
        }

        all_sell_options
    }
}
