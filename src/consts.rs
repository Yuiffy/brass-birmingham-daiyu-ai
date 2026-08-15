use crate::core::locations::TownName;
use crate::core::types::{Card, CardType};
use fixedbitset::FixedBitSet;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ops::Range;
pub type Level = usize;

pub const N_PLAYERS: usize = 4;
pub const CANAL_PRICE: u16 = 3;
pub const ONE_RAILROAD_PRICE: u16 = 5;
pub const TWO_RAILROAD_PRICE: u16 = 15;
pub const MAX_MARKET_COAL: u8 = 14;
pub const MAX_MARKET_IRON: u8 = 10;
pub const STARTING_HAND_SIZE: u16 = 8;

pub const N_BL: usize = 49; // Build Locations
pub const N_ROAD_LOCATIONS: usize = 39; // Road Locations
pub const N_RAIL_ONLY_ROAD_LOCATIONS: usize = 10; // Rail-only road locations
pub const N_TOWNS: usize = 20; // Towns
pub const N_INDUSTRIES: usize = 6; // Industries
pub const N_COAL_SOURCES: usize = 16; // Coal Sources
pub const N_IRON_SOURCES: usize = 10; // Iron Sources
pub const MAX_RESORCES: usize = 6;
pub const MAX_TIER_BUILDINGS: usize = 3;
pub const N_LEVELS: usize = 8; // Levels
pub const MAX_TILES_PER_LEVEL: usize = 3;
pub const MAX_TOTAL_TILES: usize = 44; // Actual count of tiles

pub const NUM_BL: usize = 49;

pub const NUM_TRADE_POSTS: usize = 5;
pub const NUM_TOWNS: usize = 20;

//  Total towns = Towns + 2 (Lone breweries) + Trade Posts
pub const TOTAL_TOWNS: usize = NUM_TOWNS + 2 + NUM_TRADE_POSTS;
//  ALl build locations except 2 lonely breweries
pub const NUM_TOWN_BL: usize = 47;

//  2 -> 3, 3 -> 4, 4 -> 5
pub const PLAYER_COUNT_TO_NUM_TRADE_POSTS: [usize; 3] = [3, 4, 5];

//  Locations = Towns + Trade Posts + Special Locations
pub const N_LOCATIONS: usize = NUM_BL + NUM_TRADE_POSTS;

//  Industry Type to Byte - re-export from core::types
pub use crate::core::types::IndustryType;

pub static INDUSTRY_TO_BYTES: Lazy<[FixedBitSet; N_INDUSTRIES]> = Lazy::new(|| {
    let mut bitsets = [(); N_INDUSTRIES].map(|_| FixedBitSet::with_capacity(N_INDUSTRIES));
    bitsets[IndustryType::Coal as usize].insert(IndustryType::Coal as usize);
    bitsets[IndustryType::Iron as usize].insert(IndustryType::Iron as usize);
    bitsets[IndustryType::Beer as usize].insert(IndustryType::Beer as usize);
    bitsets[IndustryType::Goods as usize].insert(IndustryType::Goods as usize);
    bitsets[IndustryType::Pottery as usize].insert(IndustryType::Pottery as usize);
    bitsets[IndustryType::Cotton as usize].insert(IndustryType::Cotton as usize);
    bitsets
});

//  Town Colors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TownColors {
    Red = 0,
    Blue = 1,
    Green = 2,
    Yellow = 3,
    Purple = 4,
}

pub static COAL_PRICE_TABLE: [u8; 14] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
pub static IRON_PRICE_TABLE: [u8; 10] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]

// pub enum TownName {
//     Stafford = 0,
//     BurtonUponTrent = 1,
//     Cannock = 2,
//     Tamworth = 3,
//     Walsall = 4,
//     Leek = 5,
//     StokeOnTrent = 6,
//     Stone = 7,
//     Uttoxeter = 8,
//     Belper = 9,
//     Derby = 10,
//     Coalbrookdale = 11,
//     Wolverhampton = 12,
//     Dudley = 13,
//     Kidderminster = 14,
//     Worcester = 15,
//     Birmingham = 16,
//     Nuneaton = 17,
//     Coventry = 18,
//     Redditch = 19,
// }

// --- Static Deck Definitions ---
// Using lazy initialization for IndustrySet-based cards
use crate::core::types::IndustrySet;

fn location(town: TownName) -> Card {
    Card::new(CardType::Location(town))
}

fn industry(industry_type: IndustryType) -> Card {
    Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
        industry_type,
    ])))
}

pub static STARTING_CARDS_2P: Lazy<Vec<Card>> = Lazy::new(|| {
    vec![
        location(TownName::Stafford),
        location(TownName::Stafford),
        location(TownName::BurtonUponTrent),
        location(TownName::BurtonUponTrent),
        location(TownName::Cannock),
        location(TownName::Cannock),
        location(TownName::Tamworth),
        location(TownName::Walsall),
        location(TownName::Coalbrookdale),
        location(TownName::Coalbrookdale),
        location(TownName::Coalbrookdale),
        location(TownName::Dudley),
        location(TownName::Dudley),
        location(TownName::Kidderminster),
        location(TownName::Kidderminster),
        location(TownName::Wolverhampton),
        location(TownName::Wolverhampton),
        location(TownName::Worcester),
        location(TownName::Worcester),
        location(TownName::Birmingham),
        location(TownName::Birmingham),
        location(TownName::Birmingham),
        location(TownName::Coventry),
        location(TownName::Coventry),
        location(TownName::Coventry),
        location(TownName::Nuneaton),
        location(TownName::Redditch),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Coal),
        industry(IndustryType::Coal),
        industry(IndustryType::Pottery),
        industry(IndustryType::Pottery),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
    ]
});
pub static STARTING_CARDS_2P_LEN: Lazy<usize> = Lazy::new(|| STARTING_CARDS_2P.len());

pub static STARTING_CARDS_3P: Lazy<Vec<Card>> = Lazy::new(|| {
    vec![
        // --- 2P Cards ---
        location(TownName::Stafford),
        location(TownName::Stafford),
        location(TownName::BurtonUponTrent),
        location(TownName::BurtonUponTrent),
        location(TownName::Cannock),
        location(TownName::Cannock),
        location(TownName::Tamworth),
        location(TownName::Walsall),
        location(TownName::Coalbrookdale),
        location(TownName::Coalbrookdale),
        location(TownName::Coalbrookdale),
        location(TownName::Dudley),
        location(TownName::Dudley),
        location(TownName::Kidderminster),
        location(TownName::Kidderminster),
        location(TownName::Wolverhampton),
        location(TownName::Wolverhampton),
        location(TownName::Worcester),
        location(TownName::Worcester),
        location(TownName::Birmingham),
        location(TownName::Birmingham),
        location(TownName::Birmingham),
        location(TownName::Coventry),
        location(TownName::Coventry),
        location(TownName::Coventry),
        location(TownName::Nuneaton),
        location(TownName::Redditch),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Coal),
        industry(IndustryType::Coal),
        industry(IndustryType::Pottery),
        industry(IndustryType::Pottery),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        // --- 3P Additions ---
        location(TownName::Leek),
        location(TownName::Leek),
        location(TownName::StokeOnTrent),
        location(TownName::StokeOnTrent),
        location(TownName::StokeOnTrent),
        location(TownName::Stone),
        location(TownName::Stone),
        location(TownName::Uttoxeter),
        // 6 Cotton / Manufactured Goods dual-industry cards added for 3P
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
    ]
});
pub static STARTING_CARDS_3P_LEN: Lazy<usize> = Lazy::new(|| STARTING_CARDS_3P.len());

pub static STARTING_CARDS_4P: Lazy<Vec<Card>> = Lazy::new(|| {
    vec![
        // --- 3P Cards ---
        location(TownName::Stafford),
        location(TownName::Stafford),
        location(TownName::BurtonUponTrent),
        location(TownName::BurtonUponTrent),
        location(TownName::Cannock),
        location(TownName::Cannock),
        location(TownName::Tamworth),
        location(TownName::Walsall),
        location(TownName::Coalbrookdale),
        location(TownName::Coalbrookdale),
        location(TownName::Coalbrookdale),
        location(TownName::Dudley),
        location(TownName::Dudley),
        location(TownName::Kidderminster),
        location(TownName::Kidderminster),
        location(TownName::Wolverhampton),
        location(TownName::Wolverhampton),
        location(TownName::Worcester),
        location(TownName::Worcester),
        location(TownName::Birmingham),
        location(TownName::Birmingham),
        location(TownName::Birmingham),
        location(TownName::Coventry),
        location(TownName::Coventry),
        location(TownName::Coventry),
        location(TownName::Nuneaton),
        location(TownName::Redditch),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Iron),
        industry(IndustryType::Coal),
        industry(IndustryType::Coal),
        industry(IndustryType::Pottery),
        industry(IndustryType::Pottery),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        industry(IndustryType::Beer),
        location(TownName::Leek),
        location(TownName::Leek),
        location(TownName::StokeOnTrent),
        location(TownName::StokeOnTrent),
        location(TownName::StokeOnTrent),
        location(TownName::Stone),
        location(TownName::Stone),
        location(TownName::Uttoxeter),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        // --- 4P Additions ---
        location(TownName::Belper),
        location(TownName::Belper),
        location(TownName::Derby),
        location(TownName::Derby),
        location(TownName::Derby),
        location(TownName::Uttoxeter),   // Second Uttoxeter card
        industry(IndustryType::Coal),    // Third Coal card
        industry(IndustryType::Pottery), // Third Pottery card
        // 2 additional dual-industry cards bring the 4P total to 8.
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
        Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
            IndustryType::Cotton,
            IndustryType::Goods,
        ]))),
    ]
});
pub static STARTING_CARDS_4P_LEN: Lazy<usize> = Lazy::new(|| STARTING_CARDS_4P.len());

//  Beer Brewery Locations
pub const BEER_BREWERY_1: u8 = (NUM_BL - 2) as u8;
pub const BEER_BREWERY_2: u8 = (NUM_BL - 1) as u8;

pub const BEER_TOWN_1: u8 = (NUM_TOWNS - 2) as u8;
pub static NUM_PLAYERS_TO_N_ACTIVE_TRADE_POSTS: Lazy<HashMap<usize, usize>> =
    Lazy::new(|| HashMap::from([(2, 3), (3, 4), (4, 5)]));

// set bits for all rail only roads NO CANAL POSSIBLE
pub static RAIL_ONLY: Lazy<FixedBitSet> =
    Lazy::new(|| FixedBitSet::from_iter(vec![2, 5, 8, 12, 24, 26, 29, 32]));
// Helper function to convert (usize, usize) tuple to Range<usize>
const fn to_range(bounds: (usize, usize)) -> Range<usize> {
    bounds.0..bounds.1
}
