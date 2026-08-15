// Extracted from merchants.rs
use crate::core::types::*;
use fixedbitset::FixedBitSet;
use once_cell::sync::Lazy;

// Enum representing the *kind* of merchant tile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MerchantTileType {
    All, // Cotton, Goods, Pottery
    Cotton,
    Goods,
    Pottery,
    Blank, // No industries, no beer
}

// Struct representing a merchant tile *instance* placed on the board
#[derive(Debug, Clone)]
pub struct MerchantTile {
    pub tile_type: MerchantTileType,
    pub industries: FixedBitSet, // Precomputed based on tile_type
    pub has_beer: bool,          // Beer status for this specific slot
}

impl MerchantTile {
    // Creates a new MerchantTile instance from a type
    pub fn from_type(tile_type: MerchantTileType) -> Self {
        let mut industries = FixedBitSet::with_capacity(N_INDUSTRIES);
        let has_beer;

        match tile_type {
            MerchantTileType::All => {
                industries.insert(IndustryType::Cotton as usize);
                industries.insert(IndustryType::Goods as usize);
                industries.insert(IndustryType::Pottery as usize);
                has_beer = true;
            }
            MerchantTileType::Cotton => {
                industries.insert(IndustryType::Cotton as usize);
                has_beer = true;
            }
            MerchantTileType::Goods => {
                industries.insert(IndustryType::Goods as usize);
                has_beer = true;
            }
            MerchantTileType::Pottery => {
                industries.insert(IndustryType::Pottery as usize);
                has_beer = true;
            }
            MerchantTileType::Blank => {
                // No industries
                has_beer = false;
            }
        }

        Self {
            tile_type,
            industries,
            has_beer,
        }
    }
}

// Static definitions for all merchant tiles
pub static MERCHANT_TILES: Lazy<[MerchantTile; 9]> = Lazy::new(|| {
    [
        // 2P
        MerchantTile::from_type(MerchantTileType::All),
        MerchantTile::from_type(MerchantTileType::Cotton),
        MerchantTile::from_type(MerchantTileType::Goods),
        MerchantTile::from_type(MerchantTileType::Blank),
        MerchantTile::from_type(MerchantTileType::Blank),
        // 3P
        MerchantTile::from_type(MerchantTileType::Blank),
        MerchantTile::from_type(MerchantTileType::Pottery),
        // 4P
        MerchantTile::from_type(MerchantTileType::Cotton),
        MerchantTile::from_type(MerchantTileType::Goods),
    ]
});

pub const NUM_PLAYERS_TO_MERCHANT_TILES_POOL: [usize; 3] = [5, 7, 9];

pub fn slot_to_trade_post(slot_idx: usize) -> TradePost {
    match slot_idx {
        0 => TradePost::Shrewbury,
        1..3 => TradePost::Oxford,
        3..5 => TradePost::Gloucester,
        5..7 => TradePost::Warrington,
        7..9 => TradePost::Nottingham,
        _ => panic!("Invalid slot index: {}", slot_idx),
    }
}

// Trade Post Bonuses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradePostBonus {
    FreeDevelopment,
    Income2,
    Vp3,
    Vp4,
    Money5,
}

pub static TRADE_POST_TO_BONUS: [TradePostBonus; NUM_TRADE_POSTS] = [
    TradePostBonus::Vp4,             //Shrewbury (index 0 -> NUM_BL)
    TradePostBonus::Income2,         //Oxford (index 1 -> NUM_BL+1)
    TradePostBonus::FreeDevelopment, //Gloucester (index 2 -> NUM_BL+2)
    TradePostBonus::Money5,          //Warrington (index 3 -> NUM_BL+3)
    TradePostBonus::Vp3,             //Nottingham (index 4 -> NUM_BL+4)
];
