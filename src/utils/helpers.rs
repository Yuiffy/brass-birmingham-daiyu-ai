use crate::core::locations::{LocationName, TownName};
use crate::core::static_data::LOCATION_TO_ROADS;
use crate::core::types::*;
use fixedbitset::FixedBitSet;
use std::ops::Range;

pub fn is_town_bl(loc_idx: usize) -> bool {
    loc_idx < NUM_TOWN_BL
}

/// Convert a number to TownName if valid
pub fn town_from_number(n: usize) -> Option<TownName> {
    if n < 20 {
        Some(TownName::from_usize(n))
    } else {
        None
    }
}

pub fn get_towns_from_bl_mask(bl_mask: &FixedBitSet) -> FixedBitSet {
    let mut towns = FixedBitSet::with_capacity(N_LOCATIONS);
    for bl_idx in bl_mask.ones() {
        towns.union_with(&LocationName::from_bl_idx(bl_idx).to_location_set());
    }
    towns
}

/// Binary search for the town index for a given location index
pub fn find_town_idx_for_loc(loc_idx: usize) -> usize {
    if loc_idx >= NUM_TOWN_BL {
        panic!("Invalid location index: {}", loc_idx);
    }

    let mut low = 0;
    let mut high = N_TOWNS - 1;
    while low <= high {
        let mid = (low + high) / 2;
        let (start, end) = TOWNS_RANGES[mid];
        if loc_idx >= start && loc_idx < end {
            return mid;
        } else if loc_idx < start {
            if mid == 0 {
                break;
            } // Prevent underflow
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    panic!("Invalid location index: {}", loc_idx);
}

/// Helper: Finds the town range containing a build location index
pub fn find_town_range(loc_idx: usize) -> Option<Range<usize>> {
    if loc_idx >= NUM_TOWN_BL {
        return None;
    }

    for (_town_idx, &(start, end)) in TOWNS_RANGES.iter().enumerate() {
        if loc_idx >= start && loc_idx < end {
            return Some(start..end);
        }
    }
    None
}

/// Check if an industry type includes merchant industry (Cotton or Goods)
pub fn includes_merchant_industry(industry: IndustryType) -> bool {
    matches!(industry, IndustryType::Cotton | IndustryType::Goods)
}

/// Check if a building is a market building (Coal or Iron)
pub fn is_market_building(industry: IndustryType) -> bool {
    matches!(industry, IndustryType::Coal | IndustryType::Iron)
}

/// Convert build location index to road location index
/// This is used for connectivity calculations
pub fn bl_to_road_loc(bl_idx: usize) -> Option<usize> {
    if bl_idx < NUM_BL {
        Some(bl_idx)
    } else {
        None
    }
}
/// Check if a build location index is valid
pub fn is_valid_build_location(loc_idx: usize) -> bool {
    loc_idx < NUM_BL
}

/// Check if a road location index is valid
pub fn is_valid_road_location(road_idx: usize) -> bool {
    road_idx < N_ROAD_LOCATIONS
}

pub fn bl_to_road_idxs(bl_idx: usize) -> RoadSet {
    if bl_idx >= N_LOCATIONS {
        panic!("Invalid build location index: {}", bl_idx);
    }

    LOCATION_TO_ROADS[LocationName::from_bl_idx(bl_idx).as_usize()].clone()
}

pub fn location_to_road_idxs(location_idx: usize) -> RoadSet {
    if location_idx >= LOCATION_TO_ROADS.len() {
        panic!("Invalid build location index: {}", location_idx);
    }
    LOCATION_TO_ROADS[location_idx].clone()
}

/// Calculate distance between two build locations
pub fn calculate_location_distance(loc1: usize, loc2: usize) -> usize {
    // This is a simplified distance calculation
    // In a real implementation, you might want to use actual board coordinates
    if loc1 == loc2 {
        0
    } else {
        // Find which towns these locations belong to
        let town1_opt = find_town_range(loc1);
        let town2_opt = find_town_range(loc2);

        match (town1_opt, town2_opt) {
            (Some(range1), Some(range2)) => {
                if range1.start == range2.start {
                    // Same town
                    1
                } else {
                    // Different towns - use a simple heuristic
                    ((loc1 as i32 - loc2 as i32).abs() / 10 + 1) as usize
                }
            }
            _ => usize::MAX, // Invalid locations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_town_from_number() {
        assert_eq!(town_from_number(0), Some(TownName::Stafford));
        assert_eq!(town_from_number(16), Some(TownName::Birmingham));
        assert_eq!(town_from_number(19), Some(TownName::Redditch));
        assert_eq!(town_from_number(20), None);
    }

    #[test]
    fn test_includes_merchant_industry() {
        assert!(includes_merchant_industry(IndustryType::Cotton));
        assert!(includes_merchant_industry(IndustryType::Goods));
        assert!(!includes_merchant_industry(IndustryType::Coal));
        assert!(!includes_merchant_industry(IndustryType::Iron));
        assert!(!includes_merchant_industry(IndustryType::Beer));
    }

    #[test]
    fn test_is_market_building() {
        assert!(is_market_building(IndustryType::Coal));
        assert!(is_market_building(IndustryType::Iron));
        assert!(!is_market_building(IndustryType::Cotton));
        assert!(!is_market_building(IndustryType::Goods));
        assert!(!is_market_building(IndustryType::Beer));
    }

    #[test]
    fn test_is_valid_locations() {
        assert!(is_valid_build_location(0));
        assert!(is_valid_build_location(NUM_BL - 1));
        assert!(!is_valid_build_location(NUM_BL));

        assert!(is_valid_road_location(0));
        assert!(is_valid_road_location(N_ROAD_LOCATIONS - 1));
        assert!(!is_valid_road_location(N_ROAD_LOCATIONS));
    }
}
