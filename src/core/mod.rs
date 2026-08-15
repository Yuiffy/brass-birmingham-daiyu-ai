pub mod building;
pub mod industry_mat;
pub mod links;
pub mod locations;
pub mod player;
pub mod static_data; // Must come before locations to break circular dependency
pub mod types;
pub use industry_mat::*;
pub use player::*;
pub use types::*;
// Note: Not re-exporting locations::* to avoid name conflicts
pub use building::*;
pub use links::*;
