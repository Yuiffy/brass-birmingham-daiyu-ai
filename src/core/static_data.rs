//  This Represents all Static Data that is used in the game

pub const N_LINK_LOCATIONS: usize = 39;
use crate::core::links::Link;
use crate::core::locations::LocationName;
use crate::core::locations::TOTAL_TOWNS;
use crate::core::types::BuildingTypeData;
use crate::core::types::IndustryLevel;
use crate::core::types::*;
use once_cell::sync::Lazy;
//  Beer Brewery Locations
pub const BEER_BREWERY_1: u8 = (NUM_BL - 2) as u8;
pub const BEER_BREWERY_2: u8 = (NUM_BL - 1) as u8;
// Local helper to avoid circular dependency with utils
fn find_town_idx_for_loc(loc_idx: usize) -> usize {
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
            }
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    panic!("Town not found for location index: {}", loc_idx)
}

pub const NUM_INDUSTRIES: usize = 6;
pub const NUM_LOCATIONS: usize = 19 + 2 + 5; // Towns + Trade Posts + Special Locations
pub const BEER_BREWERY_BL_1: usize = 47;
pub const BEER_BREWERY_BL_2: usize = 48;

pub static BUILD_LOCATION_MASK: Lazy<[IndustrySet; NUM_BL]> = Lazy::new(|| {
    let mut masks = [(); NUM_BL].map(|_| IndustrySet::new());
    masks[0].insert(IndustryType::Goods as usize);
    masks[0].insert(IndustryType::Beer as usize); // 1
    masks[1].insert(IndustryType::Pottery as usize); // 2
    masks[2].insert(IndustryType::Goods as usize);
    masks[2].insert(IndustryType::Coal as usize); // 3
    masks[3].insert(IndustryType::Beer as usize); // 4
    masks[4].insert(IndustryType::Goods as usize);
    masks[4].insert(IndustryType::Coal as usize); // 5
    masks[5].insert(IndustryType::Coal as usize); // 6
    masks[6].insert(IndustryType::Cotton as usize);
    masks[6].insert(IndustryType::Coal as usize); // 7
    masks[7].insert(IndustryType::Cotton as usize);
    masks[7].insert(IndustryType::Coal as usize); // 8
    masks[8].insert(IndustryType::Iron as usize);
    masks[8].insert(IndustryType::Goods as usize); // 9
    masks[9].insert(IndustryType::Goods as usize);
    masks[9].insert(IndustryType::Beer as usize); // 10
    masks[10].insert(IndustryType::Cotton as usize);
    masks[10].insert(IndustryType::Goods as usize); // 11
    masks[11].insert(IndustryType::Cotton as usize);
    masks[11].insert(IndustryType::Coal as usize); // 12
    masks[12].insert(IndustryType::Cotton as usize);
    masks[12].insert(IndustryType::Goods as usize); // 13
    masks[13].insert(IndustryType::Pottery as usize);
    masks[13].insert(IndustryType::Iron as usize); // 14
    masks[14].insert(IndustryType::Goods as usize); // 15
    masks[15].insert(IndustryType::Cotton as usize);
    masks[15].insert(IndustryType::Beer as usize); // 16
    masks[16].insert(IndustryType::Goods as usize);
    masks[16].insert(IndustryType::Coal as usize); // 17
    masks[17].insert(IndustryType::Goods as usize);
    masks[17].insert(IndustryType::Beer as usize); // 18
    masks[18].insert(IndustryType::Cotton as usize);
    masks[18].insert(IndustryType::Beer as usize); // 19
    masks[19].insert(IndustryType::Cotton as usize);
    masks[19].insert(IndustryType::Goods as usize); // 20
    masks[20].insert(IndustryType::Coal as usize); // 21
    masks[21].insert(IndustryType::Pottery as usize); // 22
    masks[22].insert(IndustryType::Cotton as usize);
    masks[22].insert(IndustryType::Beer as usize); // 23
    masks[23].insert(IndustryType::Cotton as usize);
    masks[23].insert(IndustryType::Goods as usize); // 24
    masks[24].insert(IndustryType::Iron as usize); // 25
    masks[25].insert(IndustryType::Iron as usize);
    masks[25].insert(IndustryType::Beer as usize); // 26
    masks[26].insert(IndustryType::Iron as usize); // 27
    masks[27].insert(IndustryType::Coal as usize); // 28
    masks[28].insert(IndustryType::Goods as usize); // 29
    masks[29].insert(IndustryType::Goods as usize);
    masks[29].insert(IndustryType::Coal as usize); // 30
    masks[30].insert(IndustryType::Coal as usize); // 31
    masks[31].insert(IndustryType::Iron as usize); // 32
    masks[32].insert(IndustryType::Cotton as usize);
    masks[32].insert(IndustryType::Coal as usize); // 33
    masks[33].insert(IndustryType::Cotton as usize); // 34
    masks[34].insert(IndustryType::Cotton as usize); // 35
    masks[35].insert(IndustryType::Cotton as usize); // 36
    masks[36].insert(IndustryType::Cotton as usize);
    masks[36].insert(IndustryType::Goods as usize); // 37
    masks[37].insert(IndustryType::Goods as usize); // 38
    masks[38].insert(IndustryType::Iron as usize); // 39
    masks[39].insert(IndustryType::Goods as usize); // 40
    masks[40].insert(IndustryType::Goods as usize);
    masks[40].insert(IndustryType::Beer as usize); // 41
    masks[41].insert(IndustryType::Cotton as usize);
    masks[41].insert(IndustryType::Coal as usize); // 42
    masks[42].insert(IndustryType::Pottery as usize); // 43
    masks[43].insert(IndustryType::Goods as usize);
    masks[43].insert(IndustryType::Coal as usize); // 44
    masks[44].insert(IndustryType::Iron as usize);
    masks[44].insert(IndustryType::Goods as usize); // 45
    masks[45].insert(IndustryType::Goods as usize);
    masks[45].insert(IndustryType::Coal as usize); // 46
    masks[46].insert(IndustryType::Iron as usize); // 47
    masks[47].insert(IndustryType::Beer as usize); // 48   // Brewery 1 from Cannock
    masks[48].insert(IndustryType::Beer as usize); // 49   // Brewery 2 Between Kidderminster and Worcester
    masks
});
// // Define which build locations belong to each town as ranges
pub static TOWNS_RANGES: [(usize, usize); N_TOWNS] = [
    (0, 2),   // Stafford (build locations 0-1)
    (2, 4),   // BurtonUponTrent (build locations 2-3)
    (4, 6),   // Cannock (build locations 4-5)
    (6, 8),   // Tamworth (build locations 6-7)
    (8, 10),  // Walsall (build locations 8-9)
    (10, 12), // Leek (build locations 10-11)
    (12, 15), // StokeOnTrent (build locations 12-14)
    (15, 17), // Stone (build locations 15-16)
    (17, 19), // Uttoxeter (build locations 17-18)
    (19, 22), // Belper (build locations 19-21)
    (22, 25), // Derby (build locations 22-24)
    (25, 28), // Coalbrookdale (build locations 25-27)
    (28, 30), // Wolverhampton (build locations 28-29)
    (30, 32), // Dudley (build locations 30-31)
    (32, 34), // Kidderminster (build locations 32-33)
    (34, 36), // Worcester (build locations 34-35)
    (36, 40), // Birmingham (build locations 36-39)
    (40, 42), // Nuneaton (build locations 40-41)
    (42, 45), // Coventry (build locations 42-44)
    (45, 47), // Redditch (build locations 45-46)
];

pub static TOWN_TO_BL_SET: Lazy<[BuildLocationSet; N_TOWNS]> = Lazy::new(|| {
    std::array::from_fn(|town_idx| {
        return BuildLocationSet::new_from_range(TOWNS_RANGES[town_idx]);
    })
});

pub static LINK_LOCATIONS: Lazy<[Link; N_LINK_LOCATIONS]> = Lazy::new(|| {
    [
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Warrington,
                LocationName::StokeOnTrent,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 0
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::StokeOnTrent,
                LocationName::Leek,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 1
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Leek,
                LocationName::Belper,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 2
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Belper,
                LocationName::Derby,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 3
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Derby,
                LocationName::Nottingham,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 4
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Derby,
                LocationName::Uttoxeter,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 5
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Derby,
                LocationName::BurtonUponTrent,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 6
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Tamworth,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 7
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::StokeOnTrent,
                LocationName::Stone,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 8
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Stone,
                LocationName::Uttoxeter,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 9
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Stone,
                LocationName::Stafford,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 10
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Stone,
                LocationName::BurtonUponTrent,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 11
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Stafford,
                LocationName::Cannock,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 12
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Cannock,
                LocationName::BurtonUponTrent,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 13
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Tamworth,
                LocationName::BurtonUponTrent,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 14
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Walsall,
                LocationName::BurtonUponTrent,
            ]),
            can_build_canal: true,
            can_build_rail: false,
        }, // 15
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::LoneBrewery1,
                LocationName::Cannock,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 16
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Wolverhampton,
                LocationName::Cannock,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 17
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Walsall,
                LocationName::Cannock,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 18
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Wolverhampton,
                LocationName::Coalbrookdale,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 19
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Shrewbury,
                LocationName::Coalbrookdale,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 20
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Kidderminster,
                LocationName::Coalbrookdale,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 21
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Kidderminster,
                LocationName::Dudley,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 22
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Wolverhampton,
                LocationName::Walsall,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 23
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Wolverhampton,
                LocationName::Dudley,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 24
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Tamworth,
                LocationName::Walsall,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 25
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Tamworth,
                LocationName::Nuneaton,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 26
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Nuneaton,
                LocationName::Coventry,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 27
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Walsall,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 28
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Nuneaton,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 29
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Coventry,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 30
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Oxford,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 31
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Redditch,
            ]),
            can_build_canal: false,
            can_build_rail: true,
        }, // 32
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Worcester,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 33
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Birmingham,
                LocationName::Dudley,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 34
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Redditch,
                LocationName::Oxford,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 35
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Redditch,
                LocationName::Gloucester,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 36
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Worcester,
                LocationName::Gloucester,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 37
        Link {
            locations: LocationSet::new_from_locations(vec![
                LocationName::Worcester,
                LocationName::LoneBrewery2,
                LocationName::Kidderminster,
            ]),
            can_build_canal: true,
            can_build_rail: true,
        }, // 38
    ]
});

pub static BUILD_LOCATION_TO_TOWN: Lazy<[LocationSet; N_BL]> = Lazy::new(|| {
    std::array::from_fn(|bl_idx| {
        return match bl_idx {
            x if x == BEER_BREWERY_BL_1 as usize => LocationName::LoneBrewery1.to_location_set(),
            x if x == BEER_BREWERY_BL_2 as usize => LocationName::LoneBrewery2.to_location_set(),
            x if x < N_BL - 2 => {
                return LocationName::from_usize(find_town_idx_for_loc(bl_idx)).to_location_set();
            }
            _ => {
                panic!("Invalid location: {}", bl_idx);
            }
        };
    })
});

pub static LOCATION_TO_ROADS: Lazy<[RoadSet; TOTAL_TOWNS]> = Lazy::new(|| {
    let mut location_to_roads: [RoadSet; TOTAL_TOWNS] = std::array::from_fn(|_| RoadSet::new());
    for (i, link) in LINK_LOCATIONS.iter().enumerate() {
        for loc in link.locations.ones() {
            location_to_roads[loc].insert(i);
        }
    }
    location_to_roads
});

// Build a flat array:
pub static INDUSTRY_MAT: Lazy<[Vec<BuildingTypeData>; NUM_INDUSTRIES]> = Lazy::new(|| {
    [
        // For each industry, for levels 1..8; unused = None
        // e.g. [Some(BuildingTypeData{...}), ....],  // Coal
        // Coal
        vec![
            BuildingTypeData {
                money_cost: 5,
                iron_cost: 0,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 1,
                road_vp: 2,
                resource_amt: 2,
                income: 4,
                removed_after_phase1: true,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 7,
                iron_cost: 0,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 2,
                road_vp: 1,
                resource_amt: 3,
                income: 7,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 8,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 3,
                road_vp: 1,
                resource_amt: 4,
                income: 6,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 10,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 4,
                road_vp: 1,
                resource_amt: 5,
                income: 5,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
        ],
        // Iron
        vec![
            BuildingTypeData {
                money_cost: 5,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 0,
                vp_on_flip: 3,
                road_vp: 1,
                resource_amt: 4,
                income: 3,
                removed_after_phase1: true,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 7,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 0,
                vp_on_flip: 5,
                road_vp: 1,
                resource_amt: 4,
                income: 3,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 9,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 0,
                vp_on_flip: 7,
                road_vp: 1,
                resource_amt: 5,
                income: 2,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 12,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 0,
                vp_on_flip: 9,
                road_vp: 1,
                resource_amt: 6,
                income: 1,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
        ],
        // Beer
        vec![
            BuildingTypeData {
                money_cost: 5,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 4,
                road_vp: 2,
                resource_amt: 1,
                income: 4,
                removed_after_phase1: true,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 7,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 5,
                road_vp: 2,
                resource_amt: 1,
                income: 5,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 9,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 7,
                road_vp: 2,
                resource_amt: 1,
                income: 5,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 9,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 0,
                vp_on_flip: 10,
                road_vp: 2,
                resource_amt: 1,
                income: 5,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
        ],
        // Goods
        vec![
            BuildingTypeData {
                money_cost: 8,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 1,
                vp_on_flip: 3,
                road_vp: 2,
                resource_amt: 0,
                income: 5,
                removed_after_phase1: true,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 10,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 1,
                vp_on_flip: 5,
                road_vp: 1,
                resource_amt: 0,
                income: 0,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 12,
                iron_cost: 0,
                coal_cost: 2,
                beer_needed: 0,
                vp_on_flip: 4,
                road_vp: 0,
                resource_amt: 0,
                income: 4,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 8,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 1,
                vp_on_flip: 3,
                road_vp: 1,
                resource_amt: 0,
                income: 6,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 16,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 2,
                vp_on_flip: 8,
                road_vp: 2,
                resource_amt: 0,
                income: 2,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 20,
                iron_cost: 0,
                coal_cost: 0,
                beer_needed: 1,
                vp_on_flip: 7,
                road_vp: 1,
                resource_amt: 0,
                income: 6,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 16,
                iron_cost: 1,
                coal_cost: 1,
                beer_needed: 0,
                vp_on_flip: 9,
                road_vp: 0,
                resource_amt: 0,
                income: 4,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 20,
                iron_cost: 2,
                coal_cost: 0,
                beer_needed: 1,
                vp_on_flip: 11,
                road_vp: 1,
                resource_amt: 0,
                income: 1,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
        ],
        // Pottery
        vec![
            BuildingTypeData {
                money_cost: 17,
                iron_cost: 1,
                coal_cost: 0,
                beer_needed: 1,
                vp_on_flip: 10,
                road_vp: 1,
                resource_amt: 0,
                income: 5,
                removed_after_phase1: false,
                can_develop: false,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 0,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 1,
                vp_on_flip: 1,
                road_vp: 1,
                resource_amt: 0,
                income: 1,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 22,
                iron_cost: 0,
                coal_cost: 2,
                beer_needed: 2,
                vp_on_flip: 11,
                road_vp: 1,
                resource_amt: 0,
                income: 5,
                removed_after_phase1: false,
                can_develop: false,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 0,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 1,
                vp_on_flip: 1,
                road_vp: 1,
                resource_amt: 0,
                income: 1,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
            BuildingTypeData {
                money_cost: 24,
                iron_cost: 0,
                coal_cost: 2,
                beer_needed: 2,
                vp_on_flip: 20,
                road_vp: 1,
                resource_amt: 0,
                income: 5,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 1,
            },
        ],
        // Cotton
        vec![
            BuildingTypeData {
                money_cost: 12,
                iron_cost: 0,
                coal_cost: 0,
                beer_needed: 1,
                vp_on_flip: 5,
                road_vp: 1,
                resource_amt: 0,
                income: 5,
                removed_after_phase1: true,
                can_develop: true,
                num_tiles: 3,
            },
            BuildingTypeData {
                money_cost: 14,
                iron_cost: 0,
                coal_cost: 1,
                beer_needed: 1,
                vp_on_flip: 5,
                road_vp: 2,
                resource_amt: 0,
                income: 4,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 2,
            },
            BuildingTypeData {
                money_cost: 16,
                iron_cost: 1,
                coal_cost: 1,
                beer_needed: 1,
                vp_on_flip: 9,
                road_vp: 1,
                resource_amt: 0,
                income: 3,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 3,
            },
            BuildingTypeData {
                money_cost: 18,
                iron_cost: 1,
                coal_cost: 1,
                beer_needed: 1,
                vp_on_flip: 12,
                road_vp: 1,
                resource_amt: 0,
                income: 2,
                removed_after_phase1: false,
                can_develop: true,
                num_tiles: 3,
            },
        ],
    ]
});

pub static COAL_LOCATIONS: Lazy<BuildLocationSet> = Lazy::new(|| {
    let mut coal_locations = BuildLocationSet::new();
    for (bl_idx, mask) in BUILD_LOCATION_MASK.iter().enumerate() {
        if mask.contains(IndustryType::Coal as usize) {
            coal_locations.insert(bl_idx);
        }
    }
    coal_locations
});

pub static MAX_LEVELS_PER_INDUSTRY: Lazy<[IndustryLevel; NUM_INDUSTRIES]> =
    Lazy::new(|| std::array::from_fn(|i| IndustryLevel::from_usize(INDUSTRY_MAT[i].len() - 1)));

pub static RAIL_ONLY: Lazy<RoadSet> = Lazy::new(|| {
    let mut valid_roads = RoadSet::new();
    for (road_idx, link) in LINK_LOCATIONS.iter().enumerate() {
        if link.can_build_rail {
            valid_roads.insert(road_idx);
        }
    }
    valid_roads
});

pub static CANAL_ONLY: Lazy<RoadSet> = Lazy::new(|| {
    let mut valid_roads = RoadSet::new();
    for (road_idx, link) in LINK_LOCATIONS.iter().enumerate() {
        if link.can_build_canal {
            valid_roads.insert(road_idx);
        }
    }
    valid_roads
});

pub static LOCATION_SET_TO_ROADS: Lazy<std::collections::HashMap<LocationSet, usize>> =
    Lazy::new(|| {
        let mut map = std::collections::HashMap::new();
        for (i, link) in LINK_LOCATIONS.iter().enumerate() {
            map.entry(link.locations.clone()).or_insert(i);
        }
        map
    });

pub fn get_road_idx_by_connecting_locations(locations: Vec<LocationName>) -> Option<usize> {
    let location_set = LocationSet::new_from_locations(locations);
    LOCATION_SET_TO_ROADS.get(&location_set).cloned()
}

pub fn road_label(road_idx: usize) -> String {
    if road_idx < N_LINK_LOCATIONS {
        format!("Road {} <{}>", road_idx, LINK_LOCATIONS[road_idx])
    } else {
        format!("Road {} <invalid>", road_idx)
    }
}
