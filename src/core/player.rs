use super::industry_mat::PlayerIndustryMat;
use super::static_data::LINK_LOCATIONS;
use super::types::*;
use crate::board::connectivity::Connectivity;
use crate::board::BoardState;
use crate::cards::Hand;
use crate::core::locations::LocationName;

// Define PlayerId Enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerId {
    Coade,
    Brunel,
    Arkwright,
    Tinsley,
}

impl PlayerId {
    pub fn as_usize(&self) -> usize {
        match self {
            PlayerId::Coade => 0,
            PlayerId::Brunel => 1,
            PlayerId::Arkwright => 2,
            PlayerId::Tinsley => 3,
        }
    }

    // returns tuple of hex and rgb values
    pub fn to_color(&self) -> (&str, u8, u8, u8) {
        match self {
            PlayerId::Coade => ("#c7a750", 199, 167, 80),
            PlayerId::Brunel => ("#9c79c6", 156, 121, 198),
            PlayerId::Arkwright => ("#a44529", 164, 69, 41),
            PlayerId::Tinsley => ("#b6c3ca", 182, 195, 202),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            PlayerId::Coade => "Coade",
            PlayerId::Brunel => "Brunel",
            PlayerId::Arkwright => "Arkwright",
            PlayerId::Tinsley => "Tinsley",
        }
    }

    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => PlayerId::Coade,
            1 => PlayerId::Brunel,
            2 => PlayerId::Arkwright,
            3 => PlayerId::Tinsley,
            _ => panic!("Invalid player index: {}", idx),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Player {
    pub id: PlayerId,
    pub money: u16,
    pub income_level: u8,
    pub victory_points: u16,
    pub hand: Hand,
    pub industry_mat: PlayerIndustryMat,
    pub spent_this_turn: u16,
    pub network: Connectivity,
}

impl Player {
    pub fn new(id: PlayerId, starting_hand: Vec<Card>) -> Self {
        Self {
            id,
            money: 17,        // Starting money
            income_level: 10, // Start at 10 points, which is 0 income by the formula
            victory_points: 0,
            hand: Hand::new(starting_hand),
            industry_mat: PlayerIndustryMat::new(),
            spent_this_turn: 0,
            network: Connectivity::new(),
        }
    }

    pub fn can_afford(&self, cost: u16) -> bool {
        self.money >= cost
    }

    pub fn final_ranking_key(&self) -> (u16, i8, u16) {
        (
            self.victory_points,
            self.get_income_amount(self.income_level),
            self.money,
        )
    }

    pub fn pay(&mut self, amount: u16) {
        // Safety against desync/edge-case callers: never underflow player money.
        let paid = amount.min(self.money);
        self.money -= paid;
        self.spent_this_turn = self.spent_this_turn.saturating_add(paid);
    }

    pub fn gain_money(&mut self, amount: u16) {
        self.money = self.money.saturating_add(amount);
    }

    // Gain income from income level
    pub fn gain_income(&mut self) {
        self.money += self.get_income_amount(self.income_level) as u16;
    }

    pub fn increase_income_level(&mut self, amount: u8) {
        self.income_level = self.income_level.saturating_add(amount);
        if self.income_level > 96 {
            self.income_level = 100;
        }
    }

    pub fn get_income_amount(&self, income_points: u8) -> i8 {
        let level = income_points as i32;

        let result = if level <= 10 {
            level - 10
        } else if level <= 30 {
            (level - 10 + 1) / 2
        } else if level <= 60 {
            (level + 2) / 3
        } else if level <= 96 {
            20 + (level - 60 + 3) / 4
        } else {
            30 // Max income
        };
        result as i8
    }

    pub fn pay_debt(&mut self) -> bool {
        let income_val = self.get_income_amount(self.income_level);
        if income_val >= 0 {
            return true; // No debt to pay
        }
        let debt_amount = income_val.abs() as u16;
        if self.money >= debt_amount {
            self.money -= debt_amount; // Direct money decrease for debt
            return true;
        } else {
            self.money = 0;
            return false; // Indicates liquidation is needed
        }
    }

    pub fn decrease_income_level(&mut self, levels: u8) {
        if levels == 0 {
            return;
        }
        let target = (i16::from(self.get_income_amount(self.income_level)) - i16::from(levels))
            .max(-10) as i8;
        // Loans land on the highest space in the target income band (rulebook p. 10).
        self.income_level = (0..=100)
            .rev()
            .find(|space| self.get_income_amount(*space) == target)
            .unwrap_or(0);
    }

    // A location on the board is considered to be
    // a part of your network if at least one of the
    // following is true:
    // • The location contains one or more of
    // your Industry tiles B ;
    // • The location is adjacent to one or more
    // of your Link tiles
    pub fn get_locations_in_network(&self, board_state: &BoardState) -> LocationSet {
        let mut locations = LocationSet::new();

        // Add all Towns where the player has a building
        let bl_mask = &board_state.player_building_mask[self.id.as_usize()];

        for bl_idx in bl_mask.ones() {
            locations.union_with(&LocationName::from_bl_idx(bl_idx).to_location_set());
        }
        // Add all Locations that are adjacent to the player's buildings
        let road_mask = &board_state.player_road_mask[self.id.as_usize()];

        for road_idx in road_mask.ones() {
            locations.union_with(&LINK_LOCATIONS[road_idx].locations);
        }

        locations
    }

    pub fn get_roads_in_network(&self, board_state: &BoardState) -> RoadSet {
        let mut roads = RoadSet::new();
        let locations_in_network = self.get_locations_in_network(board_state);
        for location in locations_in_network.ones() {
            roads.union_with(LocationName::from_usize(location).get_roads());
        }
        roads
    }
    // pub fn get_locations_in_network(&self, board_state: &BoardState) -> FixedBitSet {
    //     let mut locations = FixedBitSet::with_capacity(TOTAL_TOWNS);
    //     for link in crate::core::links::get_links_from_bitset(board_state.links) {
    //         locations.union_with(&link.locations);
    //     }
    //     locations
    // }
}
