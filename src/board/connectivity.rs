use crate::consts::TOTAL_TOWNS;
use crate::core::locations::LocationName;
use crate::core::static_data::{LINK_LOCATIONS, N_LINK_LOCATIONS};
use crate::core::types::{BitSetWrapper, BuildLocationSet, LocationSet, RoadSet};
use crate::utils::dsu::DisjointSetUnion;

#[derive(Debug, Clone)]
pub struct Connectivity {
    pub sets: DisjointSetUnion,
}

impl Connectivity {
    pub fn new() -> Self {
        Self {
            sets: DisjointSetUnion::new(TOTAL_TOWNS),
        }
    }

    pub fn add_road(&mut self, road_idx: usize) {
        if road_idx >= N_LINK_LOCATIONS {
            panic!("Invalid road index: {}", road_idx);
        }
        let link = &LINK_LOCATIONS[road_idx];
        for loc in link.locations.ones() {
            for loc2 in link.locations.ones() {
                if loc2 != loc {
                    self.connect_two_towns(
                        LocationName::from_usize(loc),
                        LocationName::from_usize(loc2),
                    );
                }
            }
        }
    }

    pub fn get_connected_locations(&self, location: LocationName) -> LocationSet {
        let mut connected_locations = LocationSet::new();
        for loc_idx in self.sets.get_set_elements_immutable(location.as_usize()) {
            connected_locations.insert(loc_idx);
        }
        connected_locations
    }

    /// Find all locations in the same connected component as `location` (immutable)
    pub fn get_all_connected_to(&self, location: LocationName) -> LocationSet {
        let mut connected_locations = LocationSet::new();
        for loc_idx in self.sets.get_set_elements_immutable(location.as_usize()) {
            connected_locations.insert(loc_idx);
        }
        connected_locations
    }

    pub fn are_build_locations_connected(&self, location1: usize, location2: usize) -> bool {
        self.are_towns_connected(
            LocationName::from_bl_idx(location1),
            LocationName::from_bl_idx(location2),
        )
    }

    pub fn is_bl_connected_to_location(&self, bl_idx: usize, location: LocationName) -> bool {
        self.sets.same_set_immutable(
            LocationName::from_bl_idx(bl_idx).as_usize(),
            location.as_usize(),
        )
    }

    pub fn are_towns_connected(&self, town1: LocationName, town2: LocationName) -> bool {
        self.sets
            .same_set_immutable(town1.as_usize(), town2.as_usize())
    }

    fn connect_two_towns(&mut self, town1: LocationName, town2: LocationName) {
        self.sets.union_sets(town1.as_usize(), town2.as_usize());
    }

    /// Check if build location is connected to any trade post
    /// Trade posts are: Shrewbury (22), Oxford (23), Gloucester (24), Warrington (25), Nottingham (26)
    pub fn is_bl_connected_to_any_trade_post(&self, bl_idx: usize) -> bool {
        use crate::core::types::NUM_TRADE_POSTS;
        let bl_town = LocationName::from_bl_idx(bl_idx);
        // Trade posts start at LocationName::Shrewbury (22)
        const TRADE_POST_START: usize = 22;
        for tp_idx in 0..NUM_TRADE_POSTS {
            let tp_loc = LocationName::from_usize(TRADE_POST_START + tp_idx);
            if self
                .sets
                .same_set_immutable(bl_town.as_usize(), tp_loc.as_usize())
            {
                return true;
            }
        }
        false
    }

    /// Get all build locations connected to a given build location
    pub fn get_connected_build_locations(&self, bl_idx: usize) -> BuildLocationSet {
        let town = LocationName::from_bl_idx(bl_idx);
        self.get_all_connected_to(town).to_bl_set()
    }
}

// /// Calculate connectivity if a set of new roads were added to an existing connectivity map
// pub fn calculate_connectivity_if_roads_built(
//     initial_conn_map: &[FixedBitSet; N_LOCATIONS],
//     new_road_indices: &[usize]
// ) -> [FixedBitSet; N_LOCATIONS] {
//     let mut new_connectivity = initial_conn_map.clone();

//     for &road_idx in new_road_indices {
//         if road_idx >= ROAD_LOCATION_MASK.len() { continue; }

//         let road_conn_mask = &ROAD_LOCATION_MASK[road_idx];
//         let mut component_to_merge = FixedBitSet::with_capacity(N_LOCATIONS);

//         for loc_idx_in_road in road_conn_mask.ones() {
//             if loc_idx_in_road < N_LOCATIONS {
//                 component_to_merge.union_with(&new_connectivity[loc_idx_in_road]);
//             }
//         }
//         component_to_merge.union_with(road_conn_mask);

//         for loc_to_update in component_to_merge.ones() {
//             if loc_to_update < N_LOCATIONS {
//                 new_connectivity[loc_to_update].union_with(&component_to_merge);
//             }
//         }
//     }

//     new_connectivity
// }

// /// Update global connectivity after building a road (mutates state in place)
// pub fn update_global_connectivity_after_road_build_inplace(
//     connectivity: &mut [FixedBitSet; N_LOCATIONS],
//     road_idx: usize
// ) {
//     let road_conn_mask = &ROAD_LOCATION_MASK[road_idx];
//     let mut component_to_merge = FixedBitSet::with_capacity(N_LOCATIONS);

//     for loc_idx_in_road in road_conn_mask.ones() {
//         if loc_idx_in_road < N_LOCATIONS {
//             component_to_merge.union_with(&connectivity[loc_idx_in_road]);
//         }
//     }
//     component_to_merge.union_with(road_conn_mask);

//     for loc_to_update in component_to_merge.ones() {
//         if loc_to_update < N_LOCATIONS {
//             connectivity[loc_to_update].union_with(&component_to_merge);
//         }
//     }
// }

// /// Update player network after building a road
// pub fn update_player_network_after_road_build(
//     player_road_mask: &mut FixedBitSet,
//     player_network_mask: &mut FixedBitSet,
//     connectivity: &[FixedBitSet; N_LOCATIONS],
//     road_idx: usize
// ) {
//     player_road_mask.insert(road_idx);
//     let road_locations = &ROAD_LOCATION_MASK[road_idx];

//     for road_loc_node in road_locations.ones() {
//         if road_loc_node < N_LOCATIONS {
//             player_network_mask.union_with(&connectivity[road_loc_node]);
//         }
//     }
// }

/// Update player network after building a building
/// This adds the building location and all connected locations to the player's network
pub fn update_player_network_after_building(
    player_building_mask: &mut BuildLocationSet,
    player_network_mask: &mut Connectivity,
    connectivity: &Connectivity,
    building_loc: usize,
) {
    player_building_mask.insert(building_loc);
    // The player network is updated by union with the connectivity
    // Since we use DSU, connecting happens automatically
    let town = LocationName::from_bl_idx(building_loc);
    for loc in 0..TOTAL_TOWNS {
        if connectivity.sets.same_set_immutable(town.as_usize(), loc) {
            player_network_mask.sets.union_sets(town.as_usize(), loc);
        }
    }
}

/// Update global connectivity after building a road (mutates state in place)
pub fn update_global_connectivity_after_road_build_inplace(
    connectivity: &mut Connectivity,
    road_idx: usize,
) {
    connectivity.add_road(road_idx);
}

/// Update player network after building a road
pub fn update_player_network_after_road_build(
    player_road_mask: &mut RoadSet,
    player_network_mask: &mut Connectivity,
    connectivity: &Connectivity,
    road_idx: usize,
) {
    player_road_mask.insert(road_idx);
    // Connect the player's network to all locations connected by this road
    if road_idx < N_LINK_LOCATIONS {
        let link = &LINK_LOCATIONS[road_idx];
        for loc in link.locations.ones() {
            for loc2 in 0..TOTAL_TOWNS {
                if connectivity.sets.same_set_immutable(loc, loc2) {
                    player_network_mask.sets.union_sets(loc, loc2);
                }
            }
        }
    }
}

/// Calculate global connectivity after building a road (returns new connectivity state)
pub fn calculate_global_connectivity_after_road_build(
    connectivity: &Connectivity,
    road_idx: usize,
) -> Connectivity {
    let mut new_conn = connectivity.clone();
    new_conn.add_road(road_idx);
    new_conn
}

/// Calculate connectivity if a set of roads were built
pub fn calculate_connectivity_if_roads_built(
    connectivity: &Connectivity,
    road_indices: &[usize],
) -> Connectivity {
    let mut new_conn = connectivity.clone();
    for &road_idx in road_indices {
        new_conn.add_road(road_idx);
    }
    new_conn
}
