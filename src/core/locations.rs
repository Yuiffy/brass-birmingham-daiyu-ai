// Core types and constants extracted from consts.rs

use fixedbitset::FixedBitSet;
// Price tables
pub static COAL_PRICE_TABLE: [u8; 14] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
pub static IRON_PRICE_TABLE: [u8; 10] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
use crate::core::static_data::{LOCATION_TO_ROADS, TOWN_TO_BL_SET};

// This will need to be moved to a separate constants file or lazy static
// For now, we'll reference the original location
// Re-export consts that are NOT also defined in types.rs to avoid ambiguity
pub use crate::consts::{
    BEER_BREWERY_1, BEER_BREWERY_2, INDUSTRY_TO_BYTES, STARTING_CARDS_2P, STARTING_CARDS_2P_LEN,
    STARTING_CARDS_3P, STARTING_CARDS_3P_LEN, STARTING_CARDS_4P, STARTING_CARDS_4P_LEN,
    TOTAL_TOWNS,
};
// INDUSTRY_MAT is in static_data, not consts
pub use crate::core::static_data::INDUSTRY_MAT;
// Note: N_BL, N_ROAD_LOCATIONS, RAIL_ONLY, CANAL_ONLY are defined in types.rs
// Note: LOCATION_TO_ROADS, BUILD_LOCATION_TO_TOWN, BUILD_LOCATION_MASK imported at use sites to avoid circular dependency

use crate::core::types::*;
pub const NUM_TOTAL_BUILD_TOWNS: usize = 20;
pub const NUM_BL_WITH_TRADE_POSTS: usize = N_BL + 5;
// Define which build locations belong to each town as ranges
// pub static TOWNS_RANGES: [(usize, usize); N_TOWNS] = [
//     (0, 2),    // Stafford (build locations 0-1)
//     (2, 4),    // BurtonUponTrent (build locations 2-3)
//     (4, 6),    // Cannock (build locations 4-5)
//     (6, 8),    // Tamworth (build locations 6-7)
//     (8, 10),   // Walsall (build locations 8-9)
//     (10, 12),  // Leek (build locations 10-11)
//     (12, 15),  // StokeOnTrent (build locations 12-14)
//     (15, 17),  // Stone (build locations 15-16)
//     (17, 19),  // Uttoxeter (build locations 17-18)
//     (19, 22),  // Belper (build locations 19-21)
//     (22, 25),  // Derby (build locations 22-24)
//     (25, 28),  // Coalbrookdale (build locations 25-27)
//     (28, 30),  // Wolverhampton (build locations 28-29)
//     (30, 32),  // Dudley (build locations 30-31)
//     (32, 34),  // Kidderminster (build locations 32-33)
//     (34, 36),  // Worcester (build locations 34-35)
//     (36, 40),  // Birmingham (build locations 36-39)
//     (40, 42),  // Nuneaton (build locations 40-41)
//     (42, 45),  // Coventry (build locations 42-44)
//     (45, 47),  // Redditch (build locations 45-46)
// ];

const NUM_NAMED_LOCATIONS: usize = 27;
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocationName {
    Stafford = 0,
    BurtonUponTrent = 1,
    Cannock = 2,
    Tamworth = 3,
    Walsall = 4,
    Leek = 5,
    StokeOnTrent = 6,
    Stone = 7,
    Uttoxeter = 8,
    Belper = 9,
    Derby = 10,
    Coalbrookdale = 11,
    Wolverhampton = 12,
    Dudley = 13,
    Kidderminster = 14,
    Worcester = 15,
    Birmingham = 16,
    Nuneaton = 17,
    Coventry = 18,
    Redditch = 19,
    LoneBrewery1 = 20,
    LoneBrewery2 = 21,
    Shrewbury = 22,
    Oxford = 23,
    Gloucester = 24,
    Warrington = 25,
    Nottingham = 26,
}

impl std::fmt::Display for LocationName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocationName::Stafford => write!(f, "Stafford"),
            LocationName::BurtonUponTrent => write!(f, "BurtonUponTrent"),
            LocationName::Cannock => write!(f, "Cannock"),
            LocationName::Tamworth => write!(f, "Tamworth"),
            LocationName::Walsall => write!(f, "Walsall"),
            LocationName::Leek => write!(f, "Leek"),
            LocationName::StokeOnTrent => write!(f, "StokeOnTrent"),
            LocationName::Stone => write!(f, "Stone"),
            LocationName::Uttoxeter => write!(f, "Uttoxeter"),
            LocationName::Belper => write!(f, "Belper"),
            LocationName::Derby => write!(f, "Derby"),
            LocationName::Coalbrookdale => write!(f, "Coalbrookdale"),
            LocationName::Wolverhampton => write!(f, "Wolverhampton"),
            LocationName::Dudley => write!(f, "Dudley"),
            LocationName::Kidderminster => write!(f, "Kidderminster"),
            LocationName::Worcester => write!(f, "Worcester"),
            LocationName::Birmingham => write!(f, "Birmingham"),
            LocationName::Nuneaton => write!(f, "Nuneaton"),
            LocationName::Coventry => write!(f, "Coventry"),
            LocationName::Redditch => write!(f, "Redditch"),
            LocationName::LoneBrewery1 => write!(f, "LoneBrewery1"),
            LocationName::LoneBrewery2 => write!(f, "LoneBrewery2"),
            LocationName::Shrewbury => write!(f, "Shrewbury"),
            LocationName::Oxford => write!(f, "Oxford"),
            LocationName::Gloucester => write!(f, "Gloucester"),
            LocationName::Warrington => write!(f, "Warrington"),
            LocationName::Nottingham => write!(f, "Nottingham"),
        }
    }
}

impl LocationName {
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => LocationName::Stafford,
            1 => LocationName::BurtonUponTrent,
            2 => LocationName::Cannock,
            3 => LocationName::Tamworth,
            4 => LocationName::Walsall,
            5 => LocationName::Leek,
            6 => LocationName::StokeOnTrent,
            7 => LocationName::Stone,
            8 => LocationName::Uttoxeter,
            9 => LocationName::Belper,
            10 => LocationName::Derby,
            11 => LocationName::Coalbrookdale,
            12 => LocationName::Wolverhampton,
            13 => LocationName::Dudley,
            14 => LocationName::Kidderminster,
            15 => LocationName::Worcester,
            16 => LocationName::Birmingham,
            17 => LocationName::Nuneaton,
            18 => LocationName::Coventry,
            19 => LocationName::Redditch,
            20 => LocationName::LoneBrewery1,
            21 => LocationName::LoneBrewery2,
            22 => LocationName::Shrewbury,
            23 => LocationName::Oxford,
            24 => LocationName::Gloucester,
            25 => LocationName::Warrington,
            26 => LocationName::Nottingham,
            _ => panic!("Invalid location index: {}", idx),
        }
    }

    pub fn to_location_set(&self) -> LocationSet {
        let mut location_set = LocationSet::new();
        location_set.insert(self.as_usize());
        location_set
    }

    pub fn as_usize(&self) -> usize {
        match self {
            LocationName::Stafford => 0,
            LocationName::BurtonUponTrent => 1,
            LocationName::Cannock => 2,
            LocationName::Tamworth => 3,
            LocationName::Walsall => 4,
            LocationName::Leek => 5,
            LocationName::StokeOnTrent => 6,
            LocationName::Stone => 7,
            LocationName::Uttoxeter => 8,
            LocationName::Belper => 9,
            LocationName::Derby => 10,
            LocationName::Coalbrookdale => 11,
            LocationName::Wolverhampton => 12,
            LocationName::Dudley => 13,
            LocationName::Kidderminster => 14,
            LocationName::Worcester => 15,
            LocationName::Birmingham => 16,
            LocationName::Nuneaton => 17,
            LocationName::Coventry => 18,
            LocationName::Redditch => 19,
            LocationName::LoneBrewery1 => 20,
            LocationName::LoneBrewery2 => 21,
            LocationName::Shrewbury => 22,
            LocationName::Oxford => 23,
            LocationName::Gloucester => 24,
            LocationName::Warrington => 25,
            LocationName::Nottingham => 26,
        }
    }

    pub fn is_town(self) -> bool {
        // Only the 20 named towns are indexed by TOWN_TO_BL_SET.
        // Lone breweries are separate location kinds and must not be treated as towns.
        self.as_usize() < NUM_TOTAL_BUILD_TOWNS as usize
    }

    pub fn is_trade_post(self) -> bool {
        self.as_usize() >= NUM_TOTAL_BUILD_TOWNS as usize + 2
    }

    pub fn get_roads(&self) -> &RoadSet {
        &LOCATION_TO_ROADS[self.as_usize()]
    }

    pub fn from_bl_idx(location_idx: usize) -> Self {
        if location_idx >= NUM_BL_WITH_TRADE_POSTS {
            panic!(
                "Invalid location index: {}. Max NUM_BL_WITH_TRADE_POSTS: {}",
                location_idx, NUM_BL_WITH_TRADE_POSTS
            );
        }

        if location_idx >= N_BL {
            return LocationName::from_usize(NUM_TOTAL_BUILD_TOWNS + 2 + (location_idx - N_BL));
            // 2 is for the two lone breweries
        }

        use super::static_data::BUILD_LOCATION_TO_TOWN;
        let town_mask = &BUILD_LOCATION_TO_TOWN[location_idx];
        let town_idx = town_mask.ones().next().unwrap() as usize;
        return LocationName::from_usize(town_idx);
    }

    pub fn to_bl_set(&self) -> BuildLocationSet {
        let mut bl_set = BuildLocationSet::new();

        if self.is_town() {
            return bl_set.union(&TOWN_TO_BL_SET[self.as_usize()]);
        }
        if *self == LocationName::LoneBrewery1 {
            bl_set.insert(BEER_BREWERY_1 as usize);
        }
        if *self == LocationName::LoneBrewery2 {
            bl_set.insert(BEER_BREWERY_2 as usize);
        }
        return bl_set;
    }
}
// Town Colors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TownColors {
    Red = 0,
    Blue = 1,
    Green = 2,
    Yellow = 3,
    Purple = 4,
}

// Town Names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TownName {
    Stafford = 0,
    BurtonUponTrent = 1,
    Cannock = 2,
    Tamworth = 3,
    Walsall = 4,
    Leek = 5,
    StokeOnTrent = 6,
    Stone = 7,
    Uttoxeter = 8,
    Belper = 9,
    Derby = 10,
    Coalbrookdale = 11,
    Wolverhampton = 12,
    Dudley = 13,
    Kidderminster = 14,
    Worcester = 15,
    Birmingham = 16,
    Nuneaton = 17,
    Coventry = 18,
    Redditch = 19,
}

impl TownName {
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => TownName::Stafford,
            1 => TownName::BurtonUponTrent,
            2 => TownName::Cannock,
            3 => TownName::Tamworth,
            4 => TownName::Walsall,
            5 => TownName::Leek,
            6 => TownName::StokeOnTrent,
            7 => TownName::Stone,
            8 => TownName::Uttoxeter,
            9 => TownName::Belper,
            10 => TownName::Derby,
            11 => TownName::Coalbrookdale,
            12 => TownName::Wolverhampton,
            13 => TownName::Dudley,
            14 => TownName::Kidderminster,
            15 => TownName::Worcester,
            16 => TownName::Birmingham,
            17 => TownName::Nuneaton,
            18 => TownName::Coventry,
            19 => TownName::Redditch,
            _ => panic!("Invalid town index: {}", idx),
        }
    }

    pub fn as_usize(&self) -> usize {
        *self as usize
    }

    pub fn is_town_name(idx: usize) -> bool {
        idx < TOTAL_TOWNS as usize
    }

    pub fn to_bl_set(&self) -> &BuildLocationSet {
        &TOWN_TO_BL_SET[self.as_usize()]
    }

    pub fn to_location_set(&self) -> LocationSet {
        let mut location_set = LocationSet::new();
        location_set.insert(self.as_usize());
        location_set
    }
}

#[derive(Debug, Clone)]
pub struct BuildTown {
    pub town_name: TownName,
    pub bl_range: (u8, u8),
    pub town_color: TownColors,
}

impl BuildTown {
    pub fn new(town_name: TownName, bl_range: (u8, u8), town_color: TownColors) -> Self {
        Self {
            town_name,
            bl_range,
            town_color,
        }
    }

    pub fn get_build_locations(&self) -> FixedBitSet {
        let mut bl_range = FixedBitSet::with_capacity(N_BL);
        bl_range.set_range(self.bl_range.0 as usize..self.bl_range.1 as usize, true);
        bl_range
    }

    pub fn get_location(&self) -> LocationSet {
        self.town_name.to_location_set()
    }
}

pub fn get_bl_by_industry(industry: IndustryType) -> FixedBitSet {
    use super::static_data::BUILD_LOCATION_MASK;
    let mut bl_range = FixedBitSet::with_capacity(N_BL);
    for (bl_idx, bl) in BUILD_LOCATION_MASK.iter().enumerate() {
        if bl.contains(industry as usize) {
            bl_range.insert(bl_idx);
        }
    }
    bl_range
}

mod tests {
    use crate::core::locations::LocationName;
    use crate::core::static_data::BEER_BREWERY_1;
    use crate::core::static_data::BEER_BREWERY_2;
    #[test]
    fn test_get_locationname_from_bl_idx() {
        let location_name = LocationName::from_bl_idx(36);
        assert_eq!(location_name, LocationName::Birmingham);

        let location_name = LocationName::from_bl_idx(BEER_BREWERY_1 as usize);
        assert_eq!(location_name, LocationName::LoneBrewery1);
        let location_name = LocationName::from_bl_idx(BEER_BREWERY_2 as usize);
        assert_eq!(location_name, LocationName::LoneBrewery2);
    }

    #[test]
    fn test_lone_brewery_to_bl_set_does_not_index_town_table() {
        let lb1 = LocationName::LoneBrewery1;
        let lb2 = LocationName::LoneBrewery2;

        assert!(
            !lb1.is_town(),
            "Lone brewery 1 should not be classified as a town"
        );
        assert!(
            !lb2.is_town(),
            "Lone brewery 2 should not be classified as a town"
        );

        let lb1_set = lb1.to_bl_set();
        let lb2_set = lb2.to_bl_set();

        assert!(lb1_set.contains(BEER_BREWERY_1 as usize));
        assert!(!lb1_set.contains(BEER_BREWERY_2 as usize));
        assert!(lb2_set.contains(BEER_BREWERY_2 as usize));
        assert!(!lb2_set.contains(BEER_BREWERY_1 as usize));
    }
}
