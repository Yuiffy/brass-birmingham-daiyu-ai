//! Network utilities for build location and connectivity queries
//!
//! This module provides utility functions for working with player networks,
//! build locations, and connectivity in Brass Birmingham.

use crate::board::BoardState;
use crate::core::locations::LocationName;
use crate::core::types::*;
use fixedbitset::FixedBitSet;

/// Get all build locations for a specific industry type
pub fn get_industry_locations(industry: IndustryType) -> FixedBitSet {
    use crate::core::static_data::BUILD_LOCATION_MASK;

    let mut locations = FixedBitSet::with_capacity(NUM_BL);
    for (loc_idx, mask) in BUILD_LOCATION_MASK.iter().enumerate() {
        if mask.contains(industry as usize) {
            locations.insert(loc_idx);
        }
    }
    locations
}

/// Get all unoccupied build locations for a specific industry type
pub fn get_available_industry_locations(
    board_state: &BoardState,
    industry: IndustryType,
) -> FixedBitSet {
    let mut locations = get_industry_locations(industry);

    // Remove occupied locations (unless overbuildable)
    for loc in board_state.build_locations_occupied.ones() {
        locations.remove(loc);
    }

    locations
}

/// Check if a build location can be used for a specific industry
pub fn can_build_industry_at_location(industry: IndustryType, loc_idx: usize) -> bool {
    use crate::core::static_data::BUILD_LOCATION_MASK;
    if loc_idx >= NUM_BL {
        return false;
    }
    BUILD_LOCATION_MASK[loc_idx].contains(industry as usize)
}

/// Get the town name for a build location
pub fn get_location_town(bl_idx: usize) -> Option<LocationName> {
    if bl_idx >= NUM_BL {
        return None;
    }
    Some(LocationName::from_bl_idx(bl_idx))
}
