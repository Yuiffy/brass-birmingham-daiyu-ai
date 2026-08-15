// Market pricing logic
use crate::core::types::*;

pub struct MarketPricing;

impl MarketPricing {
    /// Calculate coal price for given number of cubes
    pub fn coal_price(remaining_market_coal: u8, cubes: u8) -> u16 {
        if cubes == 0 {
            return 0;
        }

        let mut total_cost = 0;
        let mut market_idx = remaining_market_coal;

        for _ in 0..cubes {
            if market_idx > 0 {
                total_cost += COAL_PRICE_TABLE[(MAX_MARKET_COAL - market_idx) as usize] as u16;
                market_idx -= 1;
            } else {
                total_cost += 8; // Cost when market is empty
            }
        }
        total_cost
    }

    /// Calculate iron price for given number of cubes
    pub fn iron_price(remaining_market_iron: u8, cubes: u8) -> u16 {
        if cubes == 0 {
            return 0;
        }

        let mut total_cost = 0;
        let mut market_idx = remaining_market_iron;

        for _ in 0..cubes {
            if market_idx > 0 {
                total_cost += IRON_PRICE_TABLE[(MAX_MARKET_IRON - market_idx) as usize] as u16;
                market_idx -= 1;
            } else {
                total_cost += 6; // Cost when market is empty
            }
        }
        total_cost
    }
}
