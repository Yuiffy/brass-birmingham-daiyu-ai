use super::types::IndustryLevel;
use crate::core::types::BuildingTypeData;
use crate::core::types::IndustryType;
use crate::static_data::{INDUSTRY_MAT, MAX_LEVELS_PER_INDUSTRY, NUM_INDUSTRIES};

type IndustryProgress = (IndustryLevel, u8);

#[derive(Debug, Clone)]
pub struct PlayerIndustryMat {
    // For each industry (0..5),
    //what's the lowest available *level*, and how many tiles left at that level
    progress: [IndustryProgress; NUM_INDUSTRIES],
}

// progress is a tuple of (level, remaining tiles)
// level is the current level of the industry
// remaining tiles is the number of tiles left at the current level
// if remaining tiles is 0, the industry is at the next level
// if remaining tiles is greater than 0, the industry is at the current level
// if remaining tiles is equal to the number of tiles at the current level, the industry is at the next level
// if remaining tiles is less than the number of tiles at the current level, the industry is at the current level
// if remaining tiles is greater than the number of tiles at the current level, the industry is at the next level
// if remaining tiles is less than the number of tiles at the current level, the industry is at the current level
// if remaining tiles is equal to the number of tiles at the current level, the industry is at the next level
impl PlayerIndustryMat {
    pub fn new() -> Self {
        let mut initial_progress = [(IndustryLevel::I, 0); NUM_INDUSTRIES];
        for industry_idx in 0..NUM_INDUSTRIES {
            let num_tiles = INDUSTRY_MAT[industry_idx][0].num_tiles;
            initial_progress[industry_idx] = (IndustryLevel::I, num_tiles);
        }

        PlayerIndustryMat {
            progress: initial_progress,
        }
    }

    // Returns all lowest level tiles for all industries
    pub fn get_lowest_level_tiles(&self) -> Vec<&'static BuildingTypeData> {
        let mut tiles = Vec::new();
        for industry in 0..NUM_INDUSTRIES {
            let tile = self.get_tile_for_industry(IndustryType::from_usize(industry));
            if let Some(tile) = tile {
                tiles.push(tile);
            }
        }
        tiles
    }

    pub fn get_tile_for_industry(
        &self,
        industry: IndustryType,
    ) -> Option<&'static BuildingTypeData> {
        if self.is_max_level(industry) {
            None
        } else {
            Some(&INDUSTRY_MAT[industry.as_usize()][self.get_lowest_level(industry).as_usize()])
        }
    }

    pub fn get_lowest_level(&self, industry: IndustryType) -> IndustryLevel {
        let (level, _) = self.progress[industry.as_usize()];
        level
    }

    pub fn get_remaining_tiles_at_level(&self, industry: IndustryType) -> u8 {
        let (_, remaining) = self.progress[industry.as_usize()];
        remaining
    }

    pub fn get_progress(&self, industry: IndustryType) -> IndustryProgress {
        self.progress[industry.as_usize()]
    }

    pub fn pop_tile(&mut self, industry: IndustryType) -> IndustryLevel {
        let (current_level, remaining) = self.progress[industry.as_usize()];

        let max_level = MAX_LEVELS_PER_INDUSTRY[industry.as_usize()];

        if remaining == 0 && current_level == max_level {
            return current_level;
        } // No tiles left at any level

        let new_remaining = remaining - 1;

        if new_remaining == 0 && current_level < max_level {
            let next_level = IndustryLevel::from_u8(current_level.as_u8() + 1);

            let num_tiles = INDUSTRY_MAT[industry.as_usize()][next_level.as_usize()].num_tiles;
            self.progress[industry.as_usize()] = (next_level, num_tiles);
            return next_level;
        }

        self.progress[industry.as_usize()] = (current_level, new_remaining);
        return current_level;
    }

    pub fn has_tiles_left(&self, industry: IndustryType) -> bool {
        let (_, remaining) = self.progress[industry.as_usize()];
        remaining > 0
    }

    pub fn is_max_level(&self, industry: IndustryType) -> bool {
        let (level, _) = self.progress[industry.as_usize()];
        level == MAX_LEVELS_PER_INDUSTRY[industry.as_usize()]
    }

    pub fn get_current_level_building_data(
        &self,
        industry: IndustryType,
    ) -> &'static BuildingTypeData {
        let (level, _) = self.progress[industry.as_usize()];
        &INDUSTRY_MAT[industry as usize][level.as_usize()]
    }
}
