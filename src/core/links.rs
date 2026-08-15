pub use crate::consts::TOTAL_TOWNS;
pub use crate::core::static_data::{LINK_LOCATIONS, N_LINK_LOCATIONS};
use crate::core::types::LocationSet;

pub struct Link {
    pub locations: LocationSet,
    pub can_build_canal: bool,
    pub can_build_rail: bool,
}

impl Link {
    pub fn new(locations: LocationSet, can_build_canal: bool, can_build_rail: bool) -> Self {
        Self {
            locations,
            can_build_canal,
            can_build_rail,
        }
    }
}

impl std::fmt::Display for Link {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<String> = self
            .locations
            .ones()
            .map(|idx| format!("{}", crate::core::locations::LocationName::from_usize(idx)))
            .collect();
        let mode = match (self.can_build_canal, self.can_build_rail) {
            (true, true) => "canal+rail",
            (true, false) => "canal-only",
            (false, true) => "rail-only",
            (false, false) => "none",
        };
        write!(f, "{}  [{}]", names.join("–"), mode)
    }
}
