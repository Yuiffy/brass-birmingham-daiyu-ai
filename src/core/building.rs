use crate::core::locations::LocationName;
use crate::core::player::PlayerId;
use crate::core::types::{IndustryLevel, IndustryType};
// Built Building
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuiltBuilding {
    pub industry: IndustryType,
    pub level: IndustryLevel,
    pub loc: u8,
    pub owner: PlayerId,
    pub resource_amt: u8,
    pub flipped: bool,
}

impl BuiltBuilding {
    pub fn build(industry: IndustryType, level: IndustryLevel, loc: u8, owner: PlayerId) -> Self {
        // Get Resource Amount from INDUSTRY_MAT
        let resource_amt =
            crate::static_data::INDUSTRY_MAT[industry as usize][level.as_usize()].resource_amt;
        Self {
            industry,
            level,
            loc,
            owner,
            resource_amt,
            flipped: false,
        }
    }

    pub fn flip(&mut self) {
        self.flipped = true;
    }

    pub fn is_flipped(&self) -> bool {
        self.flipped
    }

    pub fn get_resource_amt(&self) -> u8 {
        self.resource_amt
    }

    /// Consume resource and return amount that was actually consumed
    /// If not enough, consume what's available and return that amount
    /// Flip if resource is depleted to 0
    pub fn consume_resource(&mut self, amount: u8) -> u8 {
        let to_consume = amount.min(self.resource_amt);
        self.resource_amt -= to_consume;
        if self.resource_amt == 0 {
            self.flipped = true;
        }
        to_consume
    }

    pub fn consume_resource_and_check_flip(&mut self, amount: u8) -> bool {
        self.consume_resource(amount);
        self.flipped
    }

    pub fn get_town_name(&self) -> LocationName {
        LocationName::from_bl_idx(self.loc as usize)
    }
}
