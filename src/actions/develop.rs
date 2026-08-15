use crate::board::resources::{ResourceManager, ResourceSource};
use crate::core::types::*;

/// Development action logic
pub struct DevelopActions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevelopError {
    InvalidIndustryCount,
}

impl std::fmt::Display for DevelopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DevelopError::InvalidIndustryCount => {
                write!(f, "develop action requires 1 or 2 industries")
            }
        }
    }
}

impl DevelopActions {
    /// Execute development action for one or more industries
    pub fn execute_develop_action(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        industries: Vec<IndustryType>,
        iron_sources: Vec<ResourceSource>,
    ) -> Result<(), DevelopError> {
        if industries.is_empty() || industries.len() > 2 {
            return Err(DevelopError::InvalidIndustryCount);
        }

        let iron_needed = industries.len() as u8;

        // Consume iron resources using centralized function
        ResourceManager::consume_iron(board_state, player_idx, iron_sources, iron_needed);

        // Pop tiles from player's industry mat
        for industry_to_develop in industries {
            board_state.players[player_idx]
                .industry_mat
                .pop_tile(industry_to_develop);
        }
        Ok(())
    }
}
