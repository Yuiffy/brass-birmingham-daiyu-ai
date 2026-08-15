// Main library module structure

// Core modules - basic types and game concepts
pub mod core;

// Board modules - game state and board operations
pub mod board;

// Action modules - all game actions
pub mod actions;

// Validation modules - action validation logic
pub mod validation;

// Market modules - trading and merchants
pub mod market;

// Card modules - deck and hand management
pub mod cards;

// Game modules - high-level game management
pub mod game;

// Utility modules
pub mod utils;

// Web UI
pub mod web;

// Python / PyO3 bridge
#[cfg(feature = "python-bindings")]
pub mod python;

pub mod consts; // Will be gradually moved to core::types

// Re-exports for backward compatibility
pub use actions::*;
pub use board::*;
pub use cards::*;
pub use core::*;
pub use game::*;
pub use market::*;
pub use validation::*;

// Note: locations is already re-exported through core::*
