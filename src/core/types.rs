// Core types and constants extracted from consts.rs
// Price tables
pub static COAL_PRICE_TABLE: [u8; 14] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
pub static IRON_PRICE_TABLE: [u8; 10] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];

pub use crate::core::static_data::*;
use fixedbitset::FixedBitSet;
// This will need to be moved to a separate constants file or lazy static
// For now, we'll reference the original location
// Note: INDUSTRY_MAT, RAIL_ONLY, CANAL_ONLY are NOT re-exported here to avoid conflict with static_data
pub use crate::consts::{
    BEER_BREWERY_1, BEER_BREWERY_2, INDUSTRY_TO_BYTES, STARTING_CARDS_2P, STARTING_CARDS_2P_LEN,
    STARTING_CARDS_3P, STARTING_CARDS_3P_LEN, STARTING_CARDS_4P, STARTING_CARDS_4P_LEN,
    TOTAL_TOWNS,
};

// Re-export commonly used types
pub use crate::cards::{Deck, Hand};
use crate::core::locations::LocationName;
use crate::locations::TownName;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndustryLevel {
    I,
    II,
    III,
    IV,
    V,
    VI,
    VII,
    VIII,
}
impl IndustryLevel {
    pub fn as_usize(&self) -> usize {
        match self {
            IndustryLevel::I => 0,
            IndustryLevel::II => 1,
            IndustryLevel::III => 2,
            IndustryLevel::IV => 3,
            IndustryLevel::V => 4,
            IndustryLevel::VI => 5,
            IndustryLevel::VII => 6,
            IndustryLevel::VIII => 7,
        }
    }
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => IndustryLevel::I,
            1 => IndustryLevel::II,
            2 => IndustryLevel::III,
            3 => IndustryLevel::IV,
            4 => IndustryLevel::V,
            5 => IndustryLevel::VI,
            6 => IndustryLevel::VII,
            7 => IndustryLevel::VIII,
            _ => panic!("Invalid level index: {}", idx),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            IndustryLevel::I => "I",
            IndustryLevel::II => "II",
            IndustryLevel::III => "III",
            IndustryLevel::IV => "IV",
            IndustryLevel::V => "V",
            IndustryLevel::VI => "VI",
            IndustryLevel::VII => "VII",
            IndustryLevel::VIII => "VIII",
        }
    }

    pub fn from_u8(u: u8) -> Self {
        match u {
            0 => IndustryLevel::I,
            1 => IndustryLevel::II,
            2 => IndustryLevel::III,
            3 => IndustryLevel::IV,
            4 => IndustryLevel::V,
            5 => IndustryLevel::VI,
            6 => IndustryLevel::VII,
            7 => IndustryLevel::VIII,
            _ => panic!("Invalid level index: {}", u),
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            IndustryLevel::I => 0,
            IndustryLevel::II => 1,
            IndustryLevel::III => 2,
            IndustryLevel::IV => 3,
            IndustryLevel::V => 4,
            IndustryLevel::VI => 5,
            IndustryLevel::VII => 6,
            IndustryLevel::VIII => 7,
        }
    }
}

// Game constants
pub const N_PLAYERS: usize = 4;
pub const CANAL_PRICE: u16 = 3;
pub const ONE_RAILROAD_PRICE: u16 = 5;
pub const TWO_RAILROAD_PRICE: u16 = 15;
pub const MAX_MARKET_COAL: u8 = 14;
pub const MAX_MARKET_IRON: u8 = 10;
pub const STARTING_HAND_SIZE: u16 = 8;

// Board constants
pub const N_BL: usize = 49; // Build Locations
pub const N_ROAD_LOCATIONS: usize = 39; // Road Locations
pub const N_RAIL_ONLY_ROAD_LOCATIONS: usize = 10; // Rail-only road locations
pub const N_TOWNS: usize = 20; // Towns
pub const N_INDUSTRIES: usize = 6; // Industries
pub const N_COAL_SOURCES: usize = 16; // Coal Sources
pub const N_IRON_SOURCES: usize = 10; // Iron Sources
pub const MAX_RESORCES: usize = 6;
pub const MAX_TIER_BUILDINGS: usize = 3;
pub const N_LEVELS: usize = 8; // IndustryLevels
pub const MAX_TILES_PER_LEVEL: usize = 3;
pub const MAX_TOTAL_TILES: usize = 44; // Actual count of tiles
pub const NUM_BL: usize = 49;
pub const NUM_TRADE_POSTS: usize = 5;
pub const NUM_TOWN_BL: usize = 47; // All build locations except 2 lonely breweries
pub const PLAYER_COUNT_TO_NUM_TRADE_POSTS: [usize; 3] = [3, 4, 5]; // 2 -> 3, 3 -> 4, 4 -> 5
pub const N_LOCATIONS: usize = NUM_BL + NUM_TRADE_POSTS; // Locations = Towns + Trade Posts + Special Locations

use std::ops::{Deref, DerefMut, Range};

/// Common trait for all bitset wrappers
pub trait BitSetWrapper: Sized {
    /// The capacity of this bitset type
    const CAPACITY: usize;

    /// Create a new empty bitset
    fn new() -> Self;

    /// Get a reference to the underlying FixedBitSet
    fn as_bitset(&self) -> &FixedBitSet;

    /// Get a mutable reference to the underlying FixedBitSet
    fn as_bitset_mut(&mut self) -> &mut FixedBitSet;

    /// Create from a FixedBitSet (useful for conversions)
    fn from_bitset(set: FixedBitSet) -> Self;

    // Provided methods with default implementations

    /// Insert a single element
    #[inline]
    fn insert(&mut self, bit: usize) {
        if bit >= Self::CAPACITY {
            panic!("Invalid bit index: {}", bit);
        }
        self.as_bitset_mut().insert(bit);
    }

    /// Remove a single element
    #[inline]
    fn remove(&mut self, bit: usize) {
        if bit >= Self::CAPACITY {
            panic!("Invalid bit index: {}", bit);
        }
        self.as_bitset_mut().remove(bit);
    }

    /// Check if contains an element
    #[inline]
    fn contains(&self, bit: usize) -> bool {
        self.as_bitset().contains(bit)
    }

    /// Clear all bits
    #[inline]
    fn clear(&mut self) {
        self.as_bitset_mut().clear();
    }

    /// Count the number of set bits
    #[inline]
    fn count_ones(&self) -> usize {
        self.as_bitset().count_ones(..)
    }

    /// Count the number of unset bits
    #[inline]
    fn count_zeroes(&self) -> usize {
        self.as_bitset().count_zeroes(..)
    }

    /// Check if empty (no bits set)
    #[inline]
    fn is_clear(&self) -> bool {
        self.as_bitset().is_clear()
    }

    /// Check if all bits are set
    #[inline]
    fn is_full(&self) -> bool {
        self.as_bitset().is_full()
    }

    /// Check if at least one bit is set in the given range
    #[inline]
    fn contains_any_in_range(&self, range: Range<usize>) -> bool {
        self.as_bitset().contains_any_in_range(range)
    }

    /// Count the number of set bits in a specific range
    #[inline]
    fn count_ones_in_range(&self, range: Range<usize>) -> usize {
        self.as_bitset().count_ones(range)
    }

    /// Iterate over all set bit indices
    #[inline]
    fn ones(&self) -> fixedbitset::Ones<'_> {
        self.as_bitset().ones()
    }

    /// Iterate over all unset bit indices
    #[inline]
    fn zeroes(&self) -> fixedbitset::Zeroes<'_> {
        self.as_bitset().zeroes()
    }

    /// Lazy iterator over the intersection of two sets
    #[inline]
    fn intersection<'a>(&'a self, other: &'a Self) -> fixedbitset::Intersection<'a> {
        self.as_bitset().intersection(other.as_bitset())
    }

    /// Union with another set (in-place)
    #[inline]
    fn union_with(&mut self, other: &Self) {
        self.as_bitset_mut().union_with(other.as_bitset());
    }

    /// Intersect with another set (in-place)
    #[inline]
    fn intersect_with(&mut self, other: &Self) {
        self.as_bitset_mut().intersect_with(other.as_bitset());
    }

    /// Difference with another set (in-place)
    #[inline]
    fn difference_with(&mut self, other: &Self) {
        self.as_bitset_mut().difference_with(other.as_bitset());
    }

    /// Symmetric difference with another set (in-place)
    #[inline]
    fn symmetric_difference_with(&mut self, other: &Self) {
        self.as_bitset_mut()
            .symmetric_difference_with(other.as_bitset());
    }

    /// Check if disjoint from another set
    #[inline]
    fn is_disjoint(&self, other: &Self) -> bool {
        self.as_bitset().is_disjoint(other.as_bitset())
    }

    /// Check if subset of another set
    #[inline]
    fn is_subset(&self, other: &Self) -> bool {
        self.as_bitset().is_subset(other.as_bitset())
    }

    /// Check if superset of another set
    #[inline]
    fn is_superset(&self, other: &Self) -> bool {
        self.as_bitset().is_superset(other.as_bitset())
    }

    /// Create a union with another set (returns new set)
    #[inline]
    fn union(&self, other: &Self) -> Self {
        let mut result = Self::new();
        result.as_bitset_mut().union_with(self.as_bitset());
        result.as_bitset_mut().union_with(other.as_bitset());
        result
    }

    /// Create a difference with another set (returns new set)
    #[inline]
    fn difference(&self, other: &Self) -> Self {
        let mut result = Self::new();
        result.as_bitset_mut().union_with(self.as_bitset());
        result.as_bitset_mut().difference_with(other.as_bitset());
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndustrySet {
    set: FixedBitSet,
}

impl IndustrySet {
    pub fn new() -> Self {
        Self {
            set: FixedBitSet::with_capacity(N_INDUSTRIES),
        }
    }

    pub fn new_from_industry_types(industry_types: &[IndustryType]) -> Self {
        let mut industry_set = Self::new();
        for industry_type in industry_types {
            industry_set.insert(industry_type.as_usize());
        }
        industry_set
    }

    pub fn has_industry(&self, industry_type: IndustryType) -> bool {
        self.contains(industry_type.as_usize())
    }

    pub fn to_industry_types(&self) -> Vec<IndustryType> {
        let mut industry_types = Vec::new();
        for industry_type in 0..N_INDUSTRIES {
            if self.contains(industry_type) {
                industry_types.push(IndustryType::from_usize(industry_type));
            }
        }
        industry_types
    }
}

impl BitSetWrapper for IndustrySet {
    const CAPACITY: usize = N_INDUSTRIES;

    fn new() -> Self {
        Self {
            set: FixedBitSet::with_capacity(Self::CAPACITY),
        }
    }

    fn as_bitset(&self) -> &FixedBitSet {
        &self.set
    }

    fn as_bitset_mut(&mut self) -> &mut FixedBitSet {
        &mut self.set
    }

    fn from_bitset(set: FixedBitSet) -> Self {
        Self { set }
    }
}

impl Deref for IndustrySet {
    type Target = FixedBitSet;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl DerefMut for IndustrySet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
    }
}

impl Default for IndustrySet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocationSet {
    set: FixedBitSet,
}

impl std::fmt::Display for LocationSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<String> = self
            .ones()
            .map(|idx| format!("{}", LocationName::from_usize(idx)))
            .collect();
        write!(f, "{{{}}}", names.join(", "))
    }
}

impl LocationSet {
    pub fn new_from_locations(vec: Vec<LocationName>) -> Self {
        let mut location_set = Self::new();
        for location in vec {
            location_set.insert(location.as_usize());
        }
        location_set
    }

    pub fn to_bl_set(&self) -> BuildLocationSet {
        let mut bl_set = BuildLocationSet::new();
        for location in self.ones() {
            bl_set.union_with(&LocationName::from_usize(location).to_bl_set());
        }
        bl_set
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildLocationSet {
    set: FixedBitSet,
}

impl BuildLocationSet {
    pub fn new_from_range(range: (usize, usize)) -> Self {
        let mut build_location_set = Self::new();
        build_location_set.insert_range(range.0..range.1);
        build_location_set
    }

    pub fn new_from_locations(vec: Vec<LocationName>) -> Self {
        let mut build_location_set = Self::new();
        for location in vec {
            build_location_set.insert(location.as_usize());
        }
        build_location_set
    }

    pub fn to_location_set(&self) -> LocationSet {
        let mut location_set = LocationSet::new();
        for (idx, town_range) in TOWNS_RANGES.iter().enumerate() {
            if self.contains_any_in_range(town_range.0..town_range.1) {
                location_set.insert(idx);
            }
        }
        location_set
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
// Set of road locations where 1 indicates a road is in the set
pub struct RoadSet {
    set: FixedBitSet,
}

// Implement the trait for each type
impl BitSetWrapper for LocationSet {
    const CAPACITY: usize = N_LOCATIONS;

    fn new() -> Self {
        Self {
            set: FixedBitSet::with_capacity(Self::CAPACITY),
        }
    }

    fn as_bitset(&self) -> &FixedBitSet {
        &self.set
    }

    fn as_bitset_mut(&mut self) -> &mut FixedBitSet {
        &mut self.set
    }

    fn from_bitset(set: FixedBitSet) -> Self {
        Self { set }
    }
}

impl BitSetWrapper for BuildLocationSet {
    const CAPACITY: usize = N_BL;

    fn new() -> Self {
        Self {
            set: FixedBitSet::with_capacity(Self::CAPACITY),
        }
    }

    fn as_bitset(&self) -> &FixedBitSet {
        &self.set
    }

    fn as_bitset_mut(&mut self) -> &mut FixedBitSet {
        &mut self.set
    }

    fn from_bitset(set: FixedBitSet) -> Self {
        Self { set }
    }
}

impl BitSetWrapper for RoadSet {
    const CAPACITY: usize = N_ROAD_LOCATIONS;

    fn new() -> Self {
        Self {
            set: FixedBitSet::with_capacity(Self::CAPACITY),
        }
    }

    fn as_bitset(&self) -> &FixedBitSet {
        &self.set
    }

    fn as_bitset_mut(&mut self) -> &mut FixedBitSet {
        &mut self.set
    }

    fn from_bitset(set: FixedBitSet) -> Self {
        Self { set }
    }
}

// Implement Deref to allow direct access to FixedBitSet methods
impl Deref for LocationSet {
    type Target = FixedBitSet;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl DerefMut for LocationSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
    }
}

impl Deref for BuildLocationSet {
    type Target = FixedBitSet;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl DerefMut for BuildLocationSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
    }
}

impl Deref for RoadSet {
    type Target = FixedBitSet;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl DerefMut for RoadSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
    }
}

// Implement Default trait for convenience
impl Default for LocationSet {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BuildLocationSet {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RoadSet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CardType {
    Location(TownName),
    Industry(IndustrySet),
    // If you have explicit Wild cards in the deck (not just as actions) add them:
    WildLocation,
    WildIndustry,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Card {
    pub card_type: CardType,
}

impl Card {
    /// Creates a new card in a const context.
    pub const fn new(card_type: CardType) -> Self {
        Self { card_type }
    }
}

/* Usage Examples for BitSetWrapper types:

The BitSetWrapper trait provides three ways to access functionality:

1. **Trait Methods** - Type-safe operations between same-type sets:
   ```rust
   let mut locations = LocationSet::new();
   locations.insert(5);
   locations.insert(10);

   let mut other_locations = LocationSet::new();
   other_locations.insert(10);
   other_locations.insert(15);

   // Type-safe union (only works with same type)
   locations.union_with(&other_locations);

   // Iterate over set bits
   for bit in locations.ones() {
       println!("Location {} is set", bit);
   }

   // Check operations
   assert!(locations.contains(5));
   assert_eq!(locations.count_ones(), 3);
   ```

2. **Deref to FixedBitSet** - Access all FixedBitSet methods directly:
   ```rust
   let mut roads = RoadSet::new();

   // These work because RoadSet derefs to FixedBitSet
   roads.insert_range(0..5);
   roads.toggle(3);
   let min = roads.minimum();
   let max = roads.maximum();

   // Can even use indexing syntax
   if roads[2] {
       println!("Road 2 is present");
   }
   ```

3. **Direct access via as_bitset()** - When you need the underlying FixedBitSet:
   ```rust
   let buildings = BuildLocationSet::new();
   let bitset_ref: &FixedBitSet = buildings.as_bitset();

   // Useful for interfacing with code that expects FixedBitSet
   some_function_expecting_fixedbitset(buildings.as_bitset());
   ```

**Benefits of this approach:**

- ✅ Type safety: Can't accidentally union a RoadSet with a BuildLocationSet
- ✅ No code duplication: Common methods defined once in the trait
- ✅ Full FixedBitSet API: Deref gives access to all underlying methods
- ✅ Const capacity: Each type knows its own capacity at compile time
- ✅ Zero-cost abstractions: All trait methods are #[inline]
- ✅ Ergonomic: Methods work naturally on each type

**Adding new bitset types:**

To add a new bitset wrapper type:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyNewSet {
    set: FixedBitSet,
}

impl BitSetWrapper for MyNewSet {
    const CAPACITY: usize = MY_CAPACITY;

    fn new() -> Self {
        Self { set: FixedBitSet::with_capacity(Self::CAPACITY) }
    }

    fn as_bitset(&self) -> &FixedBitSet { &self.set }
    fn as_bitset_mut(&mut self) -> &mut FixedBitSet { &mut self.set }
    fn from_bitset(set: FixedBitSet) -> Self { Self { set } }
}

impl Deref for MyNewSet {
    type Target = FixedBitSet;
    fn deref(&self) -> &Self::Target { &self.set }
}

impl DerefMut for MyNewSet {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.set }
}

impl Default for MyNewSet {
    fn default() -> Self { Self::new() }
}
```

That's it! Your new type automatically gets all the common methods.
*/

// Industry Type
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndustryType {
    Coal = 0,
    Iron = 1,
    Beer = 2,
    Goods = 3,
    Pottery = 4,
    Cotton = 5,
}

impl IndustryType {
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => IndustryType::Coal,
            1 => IndustryType::Iron,
            2 => IndustryType::Beer,
            3 => IndustryType::Goods,
            4 => IndustryType::Pottery,
            5 => IndustryType::Cotton,
            _ => panic!("Invalid industry index: {}", idx),
        }
    }

    pub fn as_usize(&self) -> usize {
        match self {
            IndustryType::Coal => 0,
            IndustryType::Iron => 1,
            IndustryType::Beer => 2,
            IndustryType::Goods => 3,
            IndustryType::Pottery => 4,
            IndustryType::Cotton => 5,
        }
    }
    // Coal and Iron are the only market resources
    pub fn is_market_resource(&self) -> bool {
        match self {
            IndustryType::Coal => true,
            IndustryType::Iron => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Era {
    Canal,
    Railroad,
}

// Action Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    BuildBuilding,
    BuildRailroad,
    BuildDoubleRailroad,
    Develop,
    DevelopDouble,
    Sell,
    Loan,
    Scout,
    Pass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextActionChoiceKind {
    ChooseIndustry,
    ChooseCard,
    ChooseBuildLocation,
    ChooseRoad,
    ChooseSecondRoad,
    ChooseSecondIndustry,
    ChooseCoalSource,
    ChooseIronSource,
    ChooseBeerSource,
    ChooseMerchant,
    ChooseSellTargets,
    ChooseFreeDevelopment,
    Confirm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    Coal,
    Iron,
}

// Trade Posts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradePost {
    Shrewbury = NUM_BL as isize,
    Oxford = (NUM_BL + 1) as isize,
    Gloucester = (NUM_BL + 2) as isize,
    Warrington = (NUM_BL + 3) as isize,
    Nottingham = (NUM_BL + 4) as isize,
}

impl TradePost {
    pub const fn as_usize(&self) -> usize {
        *self as usize
    }

    /// 0-based index (0..NUM_TRADE_POSTS) suitable for indexing into
    /// TRADE_POST_TO_BONUS, small FixedBitSets, etc.
    pub const fn to_index(&self) -> usize {
        *self as usize - NUM_BL
    }

    pub fn to_location_name(&self) -> LocationName {
        match self {
            TradePost::Shrewbury => LocationName::Shrewbury,
            TradePost::Oxford => LocationName::Oxford,
            TradePost::Gloucester => LocationName::Gloucester,
            TradePost::Warrington => LocationName::Warrington,
            TradePost::Nottingham => LocationName::Nottingham,
        }
    }
}

// Static data that will be populated by lazy statics
pub static TRADE_POST_ORDERED: [TradePost; NUM_TRADE_POSTS] = [
    TradePost::Shrewbury,
    TradePost::Oxford,
    TradePost::Gloucester,
    TradePost::Warrington,
    TradePost::Nottingham,
];

// Building Type Data
#[derive(Debug, Clone, Copy)]
pub struct BuildingTypeData {
    pub money_cost: u16,
    pub coal_cost: u8,
    pub iron_cost: u8,
    pub beer_needed: u8,
    pub vp_on_flip: u8,
    pub road_vp: u8,
    pub resource_amt: u8,
    pub income: i8,
    pub removed_after_phase1: bool,
    pub can_develop: bool,
    pub num_tiles: u8,
}

impl BuildingTypeData {
    pub fn can_build_in_era(&self, era: Era) -> bool {
        if self.removed_after_phase1 && era == Era::Railroad {
            return false;
        }
        true
    }
}
