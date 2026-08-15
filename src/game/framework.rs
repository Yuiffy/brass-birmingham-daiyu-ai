use crate::actions::{
    BuildOption, DoubleRailroadSecondLinkOption, SellChoice, SellOption, SingleRailroadOption,
};
use crate::board::resources::{BeerSellSource, BreweryBeerSource, ResourceSource};
use crate::board::Board;
use crate::consts::{NUM_TRADE_POSTS, N_BL, N_LOCATIONS, TWO_RAILROAD_PRICE};
use crate::core::types::{
    ActionType, Era as GameEra, IndustryType, NextActionChoiceKind, ResourceType,
};
use crate::merchants::{slot_to_trade_post, MerchantTile, TradePostBonus, TRADE_POST_TO_BONUS};
use fixedbitset::FixedBitSet;
use std::collections::HashMap;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct GameFramework {
    pub board: Board,
    pub current_player: usize, // Player index
    pub action_context: Option<ActionContext>,
    pub replay_mode: bool,
}

// Struct to hold results of initial action validation
#[derive(Debug, Clone, Default)]
pub struct ActionValidationResult {
    pub build_options: Option<Vec<BuildOption>>,
    pub sell_options: Option<Vec<SellOption>>,
    pub dev_options: Option<FixedBitSet>, // Industries player *can* develop (has tiles, tile is developable)
    pub can_loan: bool,                   // Added for loan check
    pub can_scout: bool,                  // Added for scout check
    // Network action options
    pub canal_options: Option<Vec<usize>>, // Vec of road_idx for canal
    pub single_rail_options: Option<Vec<SingleRailroadOption>>,
    pub double_rail_first_link_options: Option<Vec<SingleRailroadOption>>,
    // can_build_double_rail could be a bool here if pre-checked,
    // or derived from double_rail_first_link_options not being empty.
}

#[derive(Debug, Clone)]
pub struct ActionContext {
    pub action_type: ActionType,
    // Build related
    pub selected_industry: Option<IndustryType>,
    pub selected_build_location: Option<usize>,
    pub initial_build_options: Option<Vec<BuildOption>>,
    pub current_filtered_build_options: Vec<BuildOption>,
    // Card related (used by Build, Scout, and as general action cost)
    pub selected_card_idx: Option<usize>,
    // Scout specific
    pub scout_additional_discard_indices: Vec<usize>,
    // Sell related
    pub initial_sell_options: Option<Vec<SellOption>>,
    pub current_filtered_sell_options: Vec<SellOption>,
    pub current_sell_choices: Vec<SellChoice>,
    pub pending_sell_building_loc: Option<usize>,
    pub available_beer_for_pending_sell: Vec<BeerSellSource>,
    pub temp_merchant_beer_consumed_slots: FixedBitSet,
    pub temp_brewery_beer_consumed: HashMap<usize, u8>,
    // Develop related
    pub initial_dev_options: Option<FixedBitSet>,
    pub free_development_choice: Option<IndustryType>, // Also for 2nd develop target
    // Network related (BuildRailroad ActionType covers Canal, Single Rail, Double Rail)
    pub selected_road_idx: Option<usize>, // First/canal/single rail link
    pub selected_second_road_idx: Option<usize>, // Second rail link for double
    pub selected_network_mode: Option<NetworkMode>,
    pub initial_canal_options: Option<Vec<usize>>,
    pub initial_single_rail_options: Option<Vec<SingleRailroadOption>>,
    pub initial_double_rail_first_link_options: Option<Vec<SingleRailroadOption>>,
    // Dynamic lists of available resources for current road choices
    pub available_coal_for_road1: Vec<ResourceSource>,
    pub available_coal_for_road2: Vec<ResourceSource>,
    pub available_beer_for_double_rail: Vec<BreweryBeerSource>,
    // Chosen resources
    pub chosen_coal_sources: Vec<ResourceSource>, // For Build, Rail (can be 1 or 2)
    pub chosen_iron_sources: Vec<ResourceSource>, // For Build, Develop
    pub chosen_beer_sources: Vec<BeerSellSource>, // For Sell (beer for items being sold)
    pub chosen_action_beer_source: Option<BreweryBeerSource>, // For Double Rail *action* beer
    // Other general fields
    pub selected_merchant_tile: Option<MerchantTile>, // If network action involves choosing specific merchant
    pub choices_needed: VecDeque<NextActionChoiceKind>,
    pub available_iron_sources: Vec<ResourceSource>,
    pub available_second_link_options_full_data: Vec<DoubleRailroadSecondLinkOption>, // For BuildDoubleRailroad
    pub available_build_coal_sources: Vec<ResourceSource>,
    pub available_build_iron_sources: Vec<ResourceSource>,
    pub hypothetical_global_conn_after_first_link: Option<Box<[FixedBitSet; N_LOCATIONS]>>,
    pub hypothetical_player_net_after_first_link: Option<FixedBitSet>,
}

impl ActionContext {
    pub fn new(
        action_type: ActionType,
        player_idx: usize,
        validation_result: &ActionValidationResult,
        board: &Board,
    ) -> Self {
        let mut choices_needed = VecDeque::new();
        // Initialize all Option fields to None and Vec fields to empty
        let mut initial_build_options = None;
        let mut current_filtered_build_options = Vec::new();
        let mut initial_sell_options = None;
        let mut current_filtered_sell_options = Vec::new();
        let mut initial_dev_options = None;
        let mut initial_canal_options = None;
        let mut initial_single_rail_options = None;
        let mut initial_double_rail_first_link_options = None;

        // Most actions require a card to be discarded. Prompt for this first
        // if the action isn't Build (where card choice is part of location/industry)
        // or Scout (handles its own main card flow).
        let needs_generic_card_first = match action_type {
            ActionType::Develop
            | ActionType::DevelopDouble
            | ActionType::Loan
            | ActionType::Pass
            | ActionType::BuildRailroad
            | ActionType::BuildDoubleRailroad => !board.players()[player_idx].hand.cards.is_empty(),
            _ => false,
        };

        if needs_generic_card_first {
            choices_needed.push_back(NextActionChoiceKind::ChooseCard);
        }

        match action_type {
            ActionType::BuildBuilding => {
                if let Some(options) = &validation_result.build_options {
                    initial_build_options = Some(options.clone());
                    current_filtered_build_options = options.clone();
                    if !current_filtered_build_options.is_empty() {
                        choices_needed.push_back(NextActionChoiceKind::ChooseIndustry);
                    }
                }
            }
            ActionType::Sell => {
                /* No upfront card choice, card is discarded as generic cost if not chosen first */
                if !board.players()[player_idx].hand.cards.is_empty() && choices_needed.is_empty() {
                    // If no other choice, add card choice first
                    choices_needed.push_back(NextActionChoiceKind::ChooseCard);
                }
                if let Some(options) = &validation_result.sell_options {
                    initial_sell_options = Some(options.clone());
                    current_filtered_sell_options = options.clone();
                    if !current_filtered_sell_options.is_empty() {
                        choices_needed.push_back(NextActionChoiceKind::ChooseSellTargets);
                    }
                }
            }
            ActionType::Develop => {
                if let Some(options) = &validation_result.dev_options {
                    if options.count_ones(..) > 0 {
                        initial_dev_options = Some(options.clone());
                        choices_needed.push_back(NextActionChoiceKind::ChooseIndustry);
                    }
                }
            }
            ActionType::DevelopDouble => {
                if let Some(options) = &validation_result.dev_options {
                    let can_afford_double_dev_iron = board.can_source_iron(player_idx, 2);
                    if options.count_ones(..) >= 2 && can_afford_double_dev_iron {
                        initial_dev_options = Some(options.clone());
                        choices_needed.push_back(NextActionChoiceKind::ChooseIndustry);
                    }
                }
            }
            ActionType::Loan => {
                if validation_result.can_loan && !board.players()[player_idx].hand.cards.is_empty()
                {
                    choices_needed.push_back(NextActionChoiceKind::ChooseCard);
                }
            }
            ActionType::Scout => {
                // Scout handles its first card selection internally as part of its 3 discards
                if validation_result.can_scout {
                    choices_needed.push_back(NextActionChoiceKind::ChooseCard);
                } // This is for the *first* of 3 cards.
            }
            ActionType::Pass => {
                if !board.players()[player_idx].hand.cards.is_empty() {
                    choices_needed.push_back(NextActionChoiceKind::ChooseCard);
                }
            }
            ActionType::BuildRailroad => {
                initial_canal_options = validation_result.canal_options.clone();
                initial_single_rail_options = validation_result.single_rail_options.clone();
                initial_double_rail_first_link_options =
                    validation_result.double_rail_first_link_options.clone();
                if board.era() == GameEra::Canal {
                    if initial_canal_options
                        .as_ref()
                        .map_or(false, |v| !v.is_empty())
                    {
                        choices_needed.push_back(NextActionChoiceKind::ChooseRoad);
                    }
                } else {
                    if initial_single_rail_options
                        .as_ref()
                        .map_or(false, |v| !v.is_empty())
                    {
                        choices_needed.push_back(NextActionChoiceKind::ChooseRoad);
                    }
                }
            }
            ActionType::BuildDoubleRailroad => {
                initial_double_rail_first_link_options =
                    validation_result.double_rail_first_link_options.clone();
                if initial_double_rail_first_link_options
                    .as_ref()
                    .map_or(false, |v| !v.is_empty())
                    && board.era() == GameEra::Railroad
                {
                    choices_needed.push_back(NextActionChoiceKind::ChooseRoad);
                }
            }
        }

        if choices_needed.is_empty() {
            if action_requires_discard_card(action_type)
                && !board.players()[player_idx].hand.cards.is_empty()
            {
                choices_needed.push_back(NextActionChoiceKind::ChooseCard);
            } else {
                choices_needed.push_back(NextActionChoiceKind::Confirm);
            }
        }
        // If ChooseCard was added and it's the only item, and then nothing else was added,
        // this default Confirm is fine as update_next_step will handle after card choice.

        Self {
            action_type,
            selected_industry: None,
            selected_card_idx: None,
            scout_additional_discard_indices: Vec::new(),
            selected_build_location: None,
            selected_road_idx: None,
            selected_second_road_idx: None,
            selected_network_mode: None,
            chosen_coal_sources: vec![],
            chosen_iron_sources: vec![],
            chosen_beer_sources: vec![],
            chosen_action_beer_source: None,
            selected_merchant_tile: None,
            choices_needed,
            current_sell_choices: Vec::new(),
            pending_sell_building_loc: None,
            available_beer_for_pending_sell: Vec::new(),
            free_development_choice: None,
            temp_merchant_beer_consumed_slots: FixedBitSet::with_capacity(NUM_TRADE_POSTS * 2),
            temp_brewery_beer_consumed: HashMap::new(),
            initial_build_options,
            initial_sell_options,
            initial_dev_options,
            initial_canal_options,
            initial_single_rail_options,
            initial_double_rail_first_link_options,
            available_coal_for_road1: Vec::new(),
            available_coal_for_road2: Vec::new(),
            available_beer_for_double_rail: Vec::new(),
            current_filtered_sell_options,
            current_filtered_build_options,
            available_iron_sources: Vec::new(),
            available_second_link_options_full_data: Vec::new(),
            available_build_coal_sources: Vec::new(),
            available_build_iron_sources: Vec::new(),
            hypothetical_global_conn_after_first_link: None,
            hypothetical_player_net_after_first_link: None,
        }
    }
}

impl GameFramework {
    fn coal_sources_for_rail_road(
        options: Option<&Vec<SingleRailroadOption>>,
        road_idx: usize,
    ) -> Vec<ResourceSource> {
        options
            .and_then(|opts| opts.iter().find(|opt| opt.road_idx == road_idx))
            .map(|opt| {
                opt.potential_coal_sources
                    .ones()
                    .map(ResourceSource::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn viable_first_coal_sources_for_double_rail(
        board: &Board,
        player_idx: usize,
        road_idx: usize,
        options: Option<&Vec<SingleRailroadOption>>,
    ) -> Vec<ResourceSource> {
        Self::coal_sources_for_rail_road(options, road_idx)
            .into_iter()
            .filter(|coal_source| {
                let mut money_after_first_link = board.players()[player_idx]
                    .money
                    .saturating_sub(TWO_RAILROAD_PRICE);
                if matches!(coal_source, ResourceSource::Market) {
                    money_after_first_link =
                        money_after_first_link.saturating_sub(board.get_coal_price(1));
                }
                !board
                    .get_options_for_second_rail_link(
                        player_idx,
                        road_idx,
                        coal_source,
                        money_after_first_link,
                    )
                    .is_empty()
            })
            .collect()
    }

    fn viable_first_roads_for_double_rail(
        board: &Board,
        player_idx: usize,
        options: Option<&Vec<SingleRailroadOption>>,
    ) -> Vec<usize> {
        options
            .into_iter()
            .flatten()
            .filter(|option| {
                !Self::viable_first_coal_sources_for_double_rail(
                    board,
                    player_idx,
                    option.road_idx,
                    options,
                )
                .is_empty()
            })
            .map(|option| option.road_idx)
            .collect()
    }

    fn selected_iron_capacity(board: &Board, sources: &[ResourceSource]) -> u8 {
        let mut seen = Vec::<ResourceSource>::new();
        let mut capacity = 0_u8;
        for source in sources {
            if seen.contains(source) {
                continue;
            }
            seen.push(*source);
            match source {
                ResourceSource::Market => return u8::MAX,
                ResourceSource::Building(location) => {
                    if let Some(building) =
                        board.bl_to_building().get(location).filter(|building| {
                            building.industry == IndustryType::Iron && !building.flipped
                        })
                    {
                        capacity = capacity.saturating_add(building.resource_amt);
                    }
                }
            }
        }
        capacity
    }

    fn selected_coal_capacity(board: &Board, sources: &[ResourceSource]) -> u8 {
        let mut seen = Vec::<ResourceSource>::new();
        let mut capacity = 0_u8;
        for source in sources {
            if seen.contains(source) {
                continue;
            }
            seen.push(*source);
            match source {
                ResourceSource::Market => return u8::MAX,
                ResourceSource::Building(location) => {
                    if let Some(building) =
                        board.bl_to_building().get(location).filter(|building| {
                            building.industry == IndustryType::Coal && !building.flipped
                        })
                    {
                        capacity = capacity.saturating_add(building.resource_amt);
                    }
                }
            }
        }
        capacity
    }

    fn available_build_coal_sources_for_selection(
        board: &Board,
        plan: &[ResourceSource],
        selected: &[ResourceSource],
        needed: u8,
    ) -> Vec<ResourceSource> {
        if Self::selected_coal_capacity(board, selected) >= needed {
            return Vec::new();
        }

        let mut buildings = plan
            .iter()
            .copied()
            .filter(|source| matches!(source, ResourceSource::Building(_)))
            .filter(|source| !selected.contains(source))
            .collect::<Vec<_>>();
        if !buildings.is_empty() {
            buildings.sort_unstable();
            buildings.dedup();
            return buildings;
        }

        if plan.contains(&ResourceSource::Market) && !selected.contains(&ResourceSource::Market) {
            vec![ResourceSource::Market]
        } else {
            Vec::new()
        }
    }

    fn available_iron_sources_for_selection(
        board: &Board,
        player_idx: usize,
        selected: &[ResourceSource],
        needed: u8,
    ) -> Vec<ResourceSource> {
        let selected_capacity = Self::selected_iron_capacity(board, selected);
        if selected_capacity >= needed || selected.contains(&ResourceSource::Market) {
            return Vec::new();
        }

        let mut buildings = board
            .iron_locations()
            .ones()
            .filter_map(|location| {
                let source = ResourceSource::Building(location);
                if selected.contains(&source) {
                    return None;
                }
                board
                    .bl_to_building()
                    .get(&location)
                    .filter(|building| !building.flipped && building.resource_amt > 0)
                    .map(|_| source)
            })
            .collect::<Vec<_>>();
        if !buildings.is_empty() {
            buildings.sort_unstable();
            return buildings;
        }

        let market_needed = needed.saturating_sub(selected_capacity);
        if board.players()[player_idx].can_afford(board.get_iron_price(market_needed)) {
            vec![ResourceSource::Market]
        } else {
            Vec::new()
        }
    }

    fn available_beer_units_for_sell(
        board: &Board,
        option: &SellOption,
        consumed_merchants: &FixedBitSet,
        consumed_breweries: &HashMap<usize, u8>,
    ) -> u8 {
        let mut available = 0_u8;
        for source_idx in option.beer_locations.ones() {
            if source_idx >= N_BL {
                let merchant_slot = source_idx - N_BL;
                if board.trade_post_beer().contains(merchant_slot)
                    && !consumed_merchants.contains(merchant_slot)
                {
                    available += 1;
                }
            } else if let Some(building) = board.bl_to_building().get(&source_idx) {
                let reserved = consumed_breweries.get(&source_idx).copied().unwrap_or(0);
                available =
                    available.saturating_add(building.resource_amt.saturating_sub(reserved));
            }
        }
        available
    }

    fn retain_currently_sellable_options(board: &Board, ctx: &mut ActionContext) {
        let consumed_merchants = ctx.temp_merchant_beer_consumed_slots.clone();
        let consumed_breweries = ctx.temp_brewery_beer_consumed.clone();
        ctx.current_filtered_sell_options.retain(|option| {
            let beer_needed = board
                .get_tile_at_loc(option.location)
                .map_or(0, |data| data.beer_needed);
            beer_needed == 0
                || Self::available_beer_units_for_sell(
                    board,
                    option,
                    &consumed_merchants,
                    &consumed_breweries,
                ) >= beer_needed
        });
    }

    fn remove_incomplete_sell_choice(ctx: &mut ActionContext, location: usize) {
        let Some(choice_idx) = ctx
            .current_sell_choices
            .iter()
            .position(|choice| choice.location == location)
        else {
            return;
        };
        let choice = ctx.current_sell_choices.remove(choice_idx);
        for source in choice.beer_sources {
            match source {
                BeerSellSource::Building(source_location) => {
                    if let Some(reserved) = ctx.temp_brewery_beer_consumed.get_mut(&source_location)
                    {
                        *reserved = reserved.saturating_sub(1);
                        if *reserved == 0 {
                            ctx.temp_brewery_beer_consumed.remove(&source_location);
                        }
                    }
                }
                BeerSellSource::TradePost(slot_idx) => {
                    ctx.temp_merchant_beer_consumed_slots.set(slot_idx, false);
                }
            }
        }
    }

    pub fn new(board: Board, current_player: usize) -> Self {
        Self {
            board,
            current_player,
            action_context: None,
            replay_mode: false,
        }
    }

    pub fn compute_valid_options(&self) -> ActionValidationResult {
        let player_idx = self.current_player;
        let mut result = ActionValidationResult::default();

        // Build, Sell, Dev options
        let build_opts = self.board.get_valid_build_options(player_idx);
        if !build_opts.is_empty() {
            result.build_options = Some(build_opts);
        }

        let sell_opts = self.board.get_valid_sell_options(player_idx);
        if !sell_opts.is_empty() {
            result.sell_options = Some(sell_opts);
        }

        result.dev_options = Some(self.board.get_valid_development_options(player_idx));
        result.can_loan = self.board.can_take_loan(player_idx);
        result.can_scout = self.board.can_scout(player_idx);

        // Populate network options based on era
        if self.board.era() == GameEra::Canal {
            let canal_opts = self.board.get_valid_canal_options(player_idx);
            if !canal_opts.is_empty() {
                result.canal_options = Some(canal_opts);
            }
        } else {
            // Rail Era
            let single_rail_opts = self.board.get_valid_single_rail_options(player_idx);
            if !single_rail_opts.is_empty() {
                result.single_rail_options = Some(single_rail_opts);
            }

            // Double rail first link options depend on overall ability to perform double rail
            if self.board.can_double_railroad(player_idx) {
                // Assumes can_double_railroad checks affordability, beer, etc.
                let double_rail_first_opts = self
                    .board
                    .get_valid_double_rail_first_link_options(player_idx);
                if !double_rail_first_opts.is_empty() {
                    result.double_rail_first_link_options = Some(double_rail_first_opts);
                }
            }
        }
        result
    }

    pub fn get_valid_action_types(
        &self,
        validation_result: &ActionValidationResult,
    ) -> Vec<ActionType> {
        let player_idx = self.current_player;
        let mut valid_actions: Vec<ActionType> = Vec::new();
        let has_card_to_discard = !self.board.players()[player_idx].hand.cards.is_empty();

        // In Brass, every root action spends at least one card.
        if !has_card_to_discard {
            return valid_actions;
        }

        if validation_result
            .build_options
            .as_ref()
            .map_or(false, |v| !v.is_empty())
        {
            valid_actions.push(ActionType::BuildBuilding);
        }
        if validation_result
            .sell_options
            .as_ref()
            .map_or(false, |v| !v.is_empty())
        {
            valid_actions.push(ActionType::Sell);
        }
        if let Some(dev_options) = &validation_result.dev_options {
            if dev_options.count_ones(..) > 0 {
                valid_actions.push(ActionType::Develop);
                // DevelopDouble check
                let can_afford_double_dev_iron = self.board.can_source_iron(player_idx, 2);
                if dev_options.count_ones(..) >= 2 && can_afford_double_dev_iron {
                    valid_actions.push(ActionType::DevelopDouble);
                }
            }
        }
        if validation_result.can_loan {
            valid_actions.push(ActionType::Loan);
        }
        if validation_result.can_scout {
            valid_actions.push(ActionType::Scout);
        }

        // Check for Network Action (BuildRailroad is the generic type)
        if validation_result
            .canal_options
            .as_ref()
            .map_or(false, |v| !v.is_empty())
            || validation_result
                .single_rail_options
                .as_ref()
                .map_or(false, |v| !v.is_empty())
            || validation_result
                .double_rail_first_link_options
                .as_ref()
                .map_or(false, |v| !v.is_empty())
        {
            valid_actions.push(ActionType::BuildRailroad);
        }

        valid_actions.push(ActionType::Pass);
        valid_actions
    }

    // Modified to accept ActionValidationResult
    pub fn start_action(
        &mut self,
        action_type: ActionType,
        validation_result: ActionValidationResult,
    ) {
        self.action_context = Some(ActionContext::new(
            action_type,
            self.current_player,
            &validation_result, // Pass by reference
            &self.board,        // Pass board reference
        ));
    }

    pub fn get_next_choices(&self) -> Option<&VecDeque<NextActionChoiceKind>> {
        self.action_context.as_ref().map(|ctx| &ctx.choices_needed)
    }

    pub fn choose_industry(&mut self, industry: IndustryType) {
        if let Some(ctx) = self.action_context.as_mut() {
            let current_choice_kind = ctx.choices_needed.front().cloned();
            match current_choice_kind {
                Some(NextActionChoiceKind::ChooseIndustry) => {
                    ctx.selected_industry = Some(industry);
                }
                Some(NextActionChoiceKind::ChooseSecondIndustry) => {
                    // For DevelopDouble, store the second industry.
                    // Using free_development_choice as a temporary holder for the second industry.
                    if ctx.action_type == ActionType::DevelopDouble {
                        ctx.free_development_choice = Some(industry);
                    } else {
                        eprintln!("Error: ChooseSecondIndustry is only for DevelopDouble action.");
                        return; // Don't update if misused
                    }
                }
                Some(NextActionChoiceKind::ChooseFreeDevelopment) => {
                    // This is for the Sell action's bonus development.
                    ctx.free_development_choice = Some(industry);
                }
                _ => {
                    if self.replay_mode {
                        // Replay compatibility: accept out-of-order industry choices.
                        if ctx.action_type == ActionType::DevelopDouble {
                            if ctx.selected_industry.is_none() {
                                ctx.selected_industry = Some(industry);
                            } else {
                                ctx.free_development_choice = Some(industry);
                            }
                        } else if ctx.action_type == ActionType::Sell {
                            ctx.free_development_choice = Some(industry);
                        } else if ctx.selected_industry.is_none() {
                            ctx.selected_industry = Some(industry);
                        } else if ctx.free_development_choice.is_none() {
                            ctx.free_development_choice = Some(industry);
                        } else {
                            ctx.selected_industry = Some(industry);
                        }
                    } else {
                        eprintln!("Error: choose_industry called when not expected.");
                        return; // Don't update if not the right time
                    }
                }
            }
            self.update_next_step();
        }
    }
    pub fn choose_card(&mut self, card_idx: usize) {
        if let Some(ctx) = self.action_context.as_mut() {
            let hand_len = self.board.players()[self.current_player].hand.cards.len();
            let mut resolved_idx = card_idx;
            if self.replay_mode && resolved_idx >= hand_len {
                if hand_len == 0 {
                    eprintln!("Replay: no cards available for requested index.");
                    return;
                }
                resolved_idx = hand_len - 1;
            }
            match ctx.action_type {
                ActionType::Scout => {
                    if ctx.selected_card_idx.is_none() {
                        if resolved_idx < self.board.players()[self.current_player].hand.cards.len()
                        {
                            ctx.selected_card_idx = Some(resolved_idx);
                        } else {
                            if self.replay_mode {
                                if hand_len > 0 {
                                    ctx.selected_card_idx = Some(hand_len - 1);
                                } else {
                                    eprintln!("Replay: Invalid card index for Scout main discard.");
                                    return;
                                }
                            } else {
                                eprintln!("Error: Invalid card index for Scout main discard.");
                                return;
                            }
                        }
                    } else if ctx.scout_additional_discard_indices.len() < 2 {
                        if resolved_idx < self.board.players()[self.current_player].hand.cards.len()
                            && Some(resolved_idx) != ctx.selected_card_idx
                            && !ctx.scout_additional_discard_indices.contains(&resolved_idx)
                        {
                            ctx.scout_additional_discard_indices.push(resolved_idx);
                        } else {
                            if self.replay_mode {
                                if let Some(fallback) = (0..hand_len).find(|i| {
                                    Some(*i) != ctx.selected_card_idx
                                        && !ctx.scout_additional_discard_indices.contains(i)
                                }) {
                                    ctx.scout_additional_discard_indices.push(fallback);
                                } else {
                                    eprintln!("Replay: Invalid or duplicate card index for Scout additional discard.");
                                    return;
                                }
                            } else {
                                eprintln!("Error: Invalid or duplicate card index for Scout additional discard.");
                                return;
                            }
                        }
                    }
                }
                _ => {
                    ctx.selected_card_idx = Some(resolved_idx);
                }
            }
            self.update_next_step();
        }
    }
    pub fn choose_build_location(&mut self, loc_idx: usize) {
        if let Some(ctx) = self.action_context.as_mut() {
            ctx.selected_build_location = Some(loc_idx);
            self.update_next_step();
        }
    }
    pub fn choose_road(&mut self, road_idx: usize) {
        if let Some(ctx) = self.action_context.as_mut() {
            if ctx.selected_road_idx.is_none() {
                ctx.selected_road_idx = Some(road_idx);
            } else {
                ctx.selected_second_road_idx = Some(road_idx);
            }
            self.update_next_step();
        }
    }

    pub fn choose_sell_target(&mut self, building_loc_idx: usize) {
        if let Some(ctx) = self.action_context.as_mut() {
            ctx.pending_sell_building_loc = Some(building_loc_idx);
            self.update_next_step();
        }
    }

    pub fn choose_beer_source(&mut self, beer_source: BeerSellSource) {
        if let Some(ctx) = self.action_context.as_mut() {
            if let Some(pending_loc) = ctx.pending_sell_building_loc {
                let choice = ctx
                    .current_sell_choices
                    .iter_mut()
                    .find(|sc| sc.location == pending_loc);
                if let Some(existing_choice) = choice {
                    existing_choice.beer_sources.push(beer_source);
                } else {
                    let mut new_choice = SellChoice::new(pending_loc, vec![]);
                    new_choice.beer_sources.push(beer_source);
                    ctx.current_sell_choices.push(new_choice);
                }
                match beer_source {
                    BeerSellSource::Building(loc) => {
                        *ctx.temp_brewery_beer_consumed.entry(loc).or_insert(0) += 1;
                    }
                    BeerSellSource::TradePost(slot_idx) => {
                        ctx.temp_merchant_beer_consumed_slots.insert(slot_idx);
                    }
                }
            }
            self.update_next_step();
        }
    }
    pub fn choose_resource_source(
        &mut self,
        resource_source: ResourceSource,
        resource_type: ResourceType,
    ) {
        // resource_type is a new enum: enum ResourceType { Coal, Iron }
        // This needs to be defined, probably in consts.rs
        if let Some(ctx) = self.action_context.as_mut() {
            match resource_type {
                ResourceType::Coal => ctx.chosen_coal_sources.push(resource_source),
                ResourceType::Iron => ctx.chosen_iron_sources.push(resource_source),
            }
            self.update_next_step();
        }
    }

    pub fn choose_confirm(&mut self) -> Result<(), String> {
        if let Some(ctx) = self.action_context.take() {
            // Common: All actions (except potentially special cases) require a card to be discarded.
            // The specific card might be chosen as part of the action (Build) or any card (Network, Develop, etc.).
            // This discard logic is currently handled in GameRunner after action completion for Pass.
            // For actions that *use* a specific card (Build), ctx.selected_card_idx is vital.
            // For actions that just need *a* discard (Network, Develop, Loan, Scout's main card),
            // selected_card_idx should also be set.
            // Scout has additional discards handled by its board.scout_action.

            // Ensure a card is selected for discard for all actions.
            if ctx.selected_card_idx.is_none() {
                return Err(format!(
                    "action {:?} confirmed without selecting card",
                    ctx.action_type
                ));
            }

            match ctx.action_type {
                ActionType::BuildBuilding => {
                    if let (Some(industry), Some(card_idx), Some(loc)) = (
                        ctx.selected_industry,
                        ctx.selected_card_idx,
                        ctx.selected_build_location,
                    ) {
                        if let Some(options) = ctx.initial_build_options {
                            let expected_level = self.board.players()[self.current_player]
                                .industry_mat
                                .get_lowest_level(industry);
                            let exact = options
                                .iter()
                                .find(|opt| {
                                    opt.industry_type == industry
                                        && opt.build_location_idx == loc
                                        && opt.level == expected_level
                                        && opt.card_used_idx == card_idx
                                })
                                .cloned();

                            // Replay/backward-compat fallback:
                            // saved games persist action choices with card indices, which can drift
                            // after engine changes. If exact card index no longer resolves to a
                            // BuildOption, use a semantically equivalent option for the same
                            // industry/location/level.
                            let build_opt = if let Some(opt) = exact {
                                Some(opt)
                            } else {
                                eprintln!(
                                    "BuildBuilding exact option missing; falling back to semantic match. \
Ind: {:?}, Card: {:?}, Loc: {:?}, Level: {:?}",
                                    Some(industry), Some(card_idx), Some(loc), Some(expected_level)
                                );
                                options.iter()
                                    .find(|opt|
                                        opt.industry_type == industry &&
                                        opt.build_location_idx == loc &&
                                        opt.level == expected_level
                                    )
                                    .cloned()
                                    .or_else(|| {
                                        eprintln!(
                                            "BuildBuilding semantic(level) option missing; \
falling back to industry+location. Ind: {:?}, Loc: {:?}",
                                            Some(industry), Some(loc)
                                        );
                                        options.iter()
                                            .find(|opt|
                                                opt.industry_type == industry &&
                                                opt.build_location_idx == loc
                                            )
                                            .cloned()
                                    })
                                    .or_else(|| {
                                        eprintln!(
                                            "BuildBuilding industry+location option missing; \
falling back to industry-only. Ind: {:?}",
                                            Some(industry)
                                        );
                                        options.iter()
                                            .find(|opt| opt.industry_type == industry)
                                            .cloned()
                                    })
                                    .or_else(|| {
                                        eprintln!(
                                            "BuildBuilding industry option missing; using first available build option for replay compatibility."
                                        );
                                        options.first().cloned()
                                    })
                            };

                            if let Some(build_opt) = build_opt {
                                let chosen_coal = ctx.chosen_coal_sources.clone();
                                let chosen_iron = ctx.chosen_iron_sources.clone();
                                if Self::selected_coal_capacity(&self.board, &chosen_coal)
                                    < build_opt.building_data.coal_cost
                                {
                                    return Err(
                                        "build action has insufficient selected coal".to_string()
                                    );
                                }
                                if Self::selected_iron_capacity(&self.board, &chosen_iron)
                                    < build_opt.building_data.iron_cost
                                {
                                    return Err(
                                        "build action has insufficient selected iron".to_string()
                                    );
                                }
                                self.board
                                    .build_building(
                                        self.current_player,
                                        build_opt,
                                        chosen_coal,
                                        chosen_iron,
                                    )
                                    .map_err(|error| format!("error executing build: {error}"))?;
                            } else {
                                return Err("chosen build combo not found".to_string());
                            }
                        }
                    } else {
                        return Err("missing selections for build action".to_string());
                    }
                }
                ActionType::Sell => {
                    if let Some(card_idx) = ctx.selected_card_idx {
                        self.board.discard_card(self.current_player, card_idx);
                    } else {
                        return Err("sell action missing card selection".to_string());
                    }
                    self.board
                        .sell_all_buildings(
                            self.current_player,
                            ctx.current_sell_choices,
                            ctx.free_development_choice,
                        )
                        .map_err(|e| format!("sell action failed: {}", e))?;
                }
                ActionType::Develop => {
                    if let (Some(industry), Some(card_idx)) =
                        (ctx.selected_industry, ctx.selected_card_idx)
                    {
                        if Self::selected_iron_capacity(&self.board, &ctx.chosen_iron_sources) >= 1
                        {
                            // Develop action itself also requires a card discard
                            self.board.discard_card(self.current_player, card_idx);
                            self.board
                                .develop_action(
                                    self.current_player,
                                    vec![industry],
                                    ctx.chosen_iron_sources,
                                )
                                .map_err(|e| format!("develop action failed: {}", e))?;
                        } else {
                            return Err("develop action missing iron source selection".to_string());
                        }
                    } else {
                        return Err("develop action missing industry/card selection".to_string());
                    }
                }
                ActionType::DevelopDouble => {
                    let mut industries_to_develop = Vec::new();
                    if let Some(industry1) = ctx.selected_industry {
                        industries_to_develop.push(industry1);
                    }
                    if let Some(industry2) = ctx.free_development_choice {
                        industries_to_develop.push(industry2);
                    }

                    if industries_to_develop.len() == 2 {
                        if let Some(card_idx) = ctx.selected_card_idx {
                            if Self::selected_iron_capacity(&self.board, &ctx.chosen_iron_sources)
                                >= 2
                            {
                                self.board.discard_card(self.current_player, card_idx);
                                self.board
                                    .develop_action(
                                        self.current_player,
                                        industries_to_develop,
                                        ctx.chosen_iron_sources,
                                    )
                                    .map_err(|e| format!("develop double action failed: {}", e))?;
                            } else {
                                return Err("develop double missing iron sources".to_string());
                            }
                        } else {
                            return Err("develop double missing card selection".to_string());
                        }
                    } else {
                        return Err("develop double requires two industries".to_string());
                    }
                }
                ActionType::BuildRailroad => {
                    // Covers Canal or Single Rail
                    if let (Some(road_idx), Some(card_idx)) =
                        (ctx.selected_road_idx, ctx.selected_card_idx)
                    {
                        if self.board.era() == GameEra::Canal {
                            self.board
                                .build_canal_action(self.current_player, road_idx, card_idx)
                                .map_err(|e| format!("canal action failed: {}", e))?;
                        } else {
                            // Rail Era - Single Link
                            if !ctx.chosen_coal_sources.is_empty() {
                                self.board
                                    .build_single_rail_action(
                                        self.current_player,
                                        road_idx,
                                        ctx.chosen_coal_sources[0],
                                        card_idx,
                                    )
                                    .map_err(|e| format!("single rail action failed: {}", e))?;
                            } else {
                                return Err("build railroad missing coal source".to_string());
                            }
                        }
                    } else {
                        return Err("build railroad missing road/card".to_string());
                    }
                }
                ActionType::BuildDoubleRailroad => {
                    if let (Some(road1_idx), Some(road2_idx), Some(card_idx), Some(beer_source)) = (
                        ctx.selected_road_idx,
                        ctx.selected_second_road_idx,
                        ctx.selected_card_idx,
                        ctx.chosen_action_beer_source,
                    ) {
                        if ctx.chosen_coal_sources.len() == 2 {
                            self.board
                                .build_double_rail_action(
                                    self.current_player,
                                    road1_idx,
                                    road2_idx,
                                    ctx.chosen_coal_sources[0],
                                    ctx.chosen_coal_sources[1],
                                    beer_source,
                                    card_idx,
                                )
                                .map_err(|e| format!("double rail action failed: {}", e))?;
                        } else {
                            return Err("double rail missing coal sources".to_string());
                        }
                    } else {
                        return Err("double rail missing required selections".to_string());
                    }
                }
                ActionType::Loan => {
                    // Loan action card discard is handled by the loan_action method
                    if let Some(card_idx) = ctx.selected_card_idx {
                        self.board.loan_action(self.current_player, card_idx);
                    } else {
                        return Err("loan action missing card".to_string());
                    }
                }
                ActionType::Scout => {
                    // Scout action handles its 3 discards internally via board.scout_action.
                    // board.scout_action takes the 3 card indices.
                    if let Some(main_card_idx) = ctx.selected_card_idx {
                        if ctx.scout_additional_discard_indices.len() == 2 {
                            let mut all_discards = vec![main_card_idx];
                            all_discards.extend(&ctx.scout_additional_discard_indices);
                            let unique_discards: std::collections::HashSet<usize> =
                                all_discards.iter().cloned().collect();
                            if unique_discards.len() == 3 {
                                self.board.scout_action(
                                    self.current_player,
                                    main_card_idx,
                                    ctx.scout_additional_discard_indices[0],
                                    ctx.scout_additional_discard_indices[1],
                                );
                            } else {
                                return Err(
                                    "scout action discard indices are not unique".to_string()
                                );
                            }
                        } else {
                            return Err("scout action missing additional discards".to_string());
                        }
                    } else {
                        return Err("scout action missing main card".to_string());
                    }
                }
                ActionType::Pass => {
                    if let Some(card_idx) = ctx.selected_card_idx {
                        self.board.discard_card(self.current_player, card_idx);
                    } else {
                        return Err("pass action missing card selection".to_string());
                    }
                }
            }
        }
        Ok(())
    }

    // Add choose_action_beer_source for DoubleRailroad
    pub fn choose_action_beer_source(&mut self, beer_source: BreweryBeerSource) {
        if let Some(ctx) = self.action_context.as_mut() {
            if ctx.action_type == ActionType::BuildDoubleRailroad {
                ctx.chosen_action_beer_source = Some(beer_source);
            } else {
                eprintln!(
                    "Error: choose_action_beer_source called for non-BuildDoubleRailroad action."
                );
                return;
            }
            self.update_next_step();
        }
    }

    fn update_next_step(&mut self) {
        if self.action_context.is_none() {
            return;
        }
        let ctx = self.action_context.as_mut().unwrap();
        let player_idx = self.current_player;
        let board_snapshot = &self.board;
        ctx.choices_needed.clear();

        match ctx.action_type {
            ActionType::BuildBuilding => {
                if ctx.selected_industry.is_none() {
                    if !ctx.current_filtered_build_options.is_empty() {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseIndustry);
                    }
                } else if ctx.selected_card_idx.is_none() {
                    let industry = ctx.selected_industry.unwrap();
                    ctx.current_filtered_build_options
                        .retain(|opt| opt.industry_type == industry);
                    if ctx.current_filtered_build_options.iter().any(|opt| {
                        opt.card_used_idx < board_snapshot.players()[player_idx].hand.cards.len()
                    }) {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseCard);
                    }
                } else if ctx.selected_build_location.is_none() {
                    let card_idx = ctx.selected_card_idx.unwrap();
                    ctx.current_filtered_build_options
                        .retain(|opt| opt.card_used_idx == card_idx);
                    if ctx.current_filtered_build_options.is_empty() {
                        if let (Some(industry), Some(options)) =
                            (ctx.selected_industry, ctx.initial_build_options.as_ref())
                        {
                            // Replay compatibility: card index from persisted event may not map
                            // to the same hand/card after engine changes.
                            ctx.current_filtered_build_options = options
                                .iter()
                                .filter(|opt| opt.industry_type == industry)
                                .cloned()
                                .collect();
                        }
                    }
                    if !ctx.current_filtered_build_options.is_empty() {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseBuildLocation);
                    }
                } else {
                    let chosen_build_opt = ctx.initial_build_options.as_ref().and_then(|options| {
                        options
                            .iter()
                            .find(|opt| {
                                opt.industry_type == ctx.selected_industry.unwrap()
                                    && opt.card_used_idx == ctx.selected_card_idx.unwrap()
                                    && opt.build_location_idx
                                        == ctx.selected_build_location.unwrap()
                                    && opt.level
                                        == board_snapshot.players()[player_idx]
                                            .industry_mat
                                            .get_lowest_level(ctx.selected_industry.unwrap())
                            })
                            .or_else(|| {
                                options.iter().find(|opt| {
                                    opt.industry_type == ctx.selected_industry.unwrap()
                                        && opt.build_location_idx
                                            == ctx.selected_build_location.unwrap()
                                        && opt.level
                                            == board_snapshot.players()[player_idx]
                                                .industry_mat
                                                .get_lowest_level(ctx.selected_industry.unwrap())
                                })
                            })
                            .or_else(|| {
                                options.iter().find(|opt| {
                                    opt.industry_type == ctx.selected_industry.unwrap()
                                        && opt.build_location_idx
                                            == ctx.selected_build_location.unwrap()
                                })
                            })
                            .or_else(|| {
                                options
                                    .iter()
                                    .find(|opt| opt.industry_type == ctx.selected_industry.unwrap())
                            })
                            .or_else(|| options.first())
                    });

                    if let Some(current_opt) = chosen_build_opt {
                        let coal_needed = current_opt.building_data.coal_cost;
                        let iron_needed = current_opt.building_data.iron_cost;

                        let coal_satisfied = coal_needed == 0
                            || Self::selected_coal_capacity(
                                board_snapshot,
                                &ctx.chosen_coal_sources,
                            ) >= coal_needed;
                        let iron_satisfied = iron_needed == 0
                            || Self::selected_iron_capacity(
                                board_snapshot,
                                &ctx.chosen_iron_sources,
                            ) >= iron_needed;

                        if coal_needed > 0 && !coal_satisfied {
                            ctx.available_build_coal_sources =
                                Self::available_build_coal_sources_for_selection(
                                    board_snapshot,
                                    &current_opt.coal_sources,
                                    &ctx.chosen_coal_sources,
                                    coal_needed,
                                );
                            if !ctx.available_build_coal_sources.is_empty() {
                                ctx.choices_needed
                                    .push_back(NextActionChoiceKind::ChooseCoalSource);
                            }
                        } else if iron_needed > 0 && !iron_satisfied {
                            ctx.available_build_iron_sources =
                                Self::available_iron_sources_for_selection(
                                    board_snapshot,
                                    player_idx,
                                    &ctx.chosen_iron_sources,
                                    iron_needed,
                                );
                            if !ctx.available_build_iron_sources.is_empty() {
                                ctx.choices_needed
                                    .push_back(NextActionChoiceKind::ChooseIronSource);
                            }
                        }
                    } else {
                        eprintln!("Error: BuildBuilding - Could not find matching BuildOption for current selections. Ind: {:?}, Card: {:?}, Loc: {:?}", 
                            ctx.selected_industry, ctx.selected_card_idx, ctx.selected_build_location);
                    }
                }
            }
            ActionType::Sell => {
                if ctx.pending_sell_building_loc.is_none() {
                    Self::retain_currently_sellable_options(board_snapshot, ctx);

                    if !ctx.current_filtered_sell_options.is_empty() {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseSellTargets);
                    }
                } else {
                    // A building is pending (ctx.pending_sell_building_loc is Some)
                    let pending_loc = ctx.pending_sell_building_loc.unwrap();
                    let beer_needed_for_pending = self
                        .board
                        .get_tile_at_loc(pending_loc)
                        .map_or(0, |data| data.beer_needed);

                    let mut beer_chosen_for_pending = 0;
                    if let Some(current_choice) = ctx
                        .current_sell_choices
                        .iter()
                        .find(|sc| sc.location == pending_loc)
                    {
                        beer_chosen_for_pending = current_choice.beer_sources.len() as u8;
                        // Assumes 1 source = 1 beer
                    }

                    if beer_chosen_for_pending < beer_needed_for_pending {
                        ctx.available_beer_for_pending_sell.clear();
                        let initial_options = ctx
                            .initial_sell_options
                            .as_ref()
                            .expect("Initial sell options missing");
                        if let Some(original_sell_opt) = initial_options
                            .iter()
                            .find(|so: &&SellOption| so.location == pending_loc)
                        {
                            for beer_source_loc_idx in original_sell_opt.beer_locations.ones() {
                                if beer_source_loc_idx >= N_BL {
                                    // Trade post beer source (merchant)
                                    let merchant_slot_idx = beer_source_loc_idx - N_BL;
                                    let has_beer_in_slot =
                                        self.board.trade_post_beer().contains(merchant_slot_idx);
                                    let is_consumed_temp = ctx
                                        .temp_merchant_beer_consumed_slots
                                        .contains(merchant_slot_idx);

                                    if has_beer_in_slot && !is_consumed_temp {
                                        ctx.available_beer_for_pending_sell
                                            .push(BeerSellSource::TradePost(merchant_slot_idx));
                                    }
                                } else {
                                    // Brewery beer source
                                    if let Some(brewery_building) =
                                        self.board.bl_to_building().get(&beer_source_loc_idx)
                                    {
                                        let already_consumed_temp = ctx
                                            .temp_brewery_beer_consumed
                                            .get(&beer_source_loc_idx)
                                            .copied()
                                            .unwrap_or(0);
                                        if brewery_building.get_resource_amt()
                                            > already_consumed_temp
                                        {
                                            ctx.available_beer_for_pending_sell.push(
                                                BeerSellSource::Building(beer_source_loc_idx),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        if !ctx.available_beer_for_pending_sell.is_empty() {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseBeerSource);
                        } else {
                            Self::remove_incomplete_sell_choice(ctx, pending_loc);
                            ctx.pending_sell_building_loc = None;
                            ctx.current_filtered_sell_options
                                .retain(|opt| opt.location != pending_loc);
                            Self::retain_currently_sellable_options(board_snapshot, ctx);
                            if !ctx.current_filtered_sell_options.is_empty() {
                                ctx.choices_needed
                                    .push_back(NextActionChoiceKind::ChooseSellTargets);
                            }
                        }
                    } else {
                        // Beer requirement met for pending building
                        // Check for free development if a merchant beer was used that gives it
                        let mut chose_dev_bonus_merchant = false;
                        if let Some(choice_for_pending) = ctx
                            .current_sell_choices
                            .iter()
                            .find(|sc| sc.location == pending_loc)
                        {
                            for beer_src in &choice_for_pending.beer_sources {
                                if let BeerSellSource::TradePost(slot_idx) = beer_src {
                                    let trade_post_enum = slot_to_trade_post(*slot_idx);
                                    if TRADE_POST_TO_BONUS[trade_post_enum.to_index()]
                                        == TradePostBonus::FreeDevelopment
                                    {
                                        chose_dev_bonus_merchant = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if chose_dev_bonus_merchant && ctx.free_development_choice.is_none() {
                            // TODO: Populate valid industries for free development
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseFreeDevelopment);
                        }

                        // Ensure a SellChoice entry exists (needed when beer_needed == 0)
                        if !ctx
                            .current_sell_choices
                            .iter()
                            .any(|sc| sc.location == pending_loc)
                        {
                            ctx.current_sell_choices
                                .push(SellChoice::new(pending_loc, vec![]));
                        }

                        ctx.pending_sell_building_loc = None;
                        ctx.current_filtered_sell_options
                            .retain(|opt| opt.location != pending_loc);
                        Self::retain_currently_sellable_options(board_snapshot, ctx);
                        if !ctx.current_filtered_sell_options.is_empty()
                            && ctx.choices_needed.is_empty()
                        {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseSellTargets);
                        }
                    }
                }
            }
            ActionType::Develop => {
                if ctx.selected_industry.is_none() {
                    if let Some(initial_opts) = &ctx.initial_dev_options {
                        if initial_opts.count_ones(..) > 0 {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseIndustry);
                        }
                    }
                } else if Self::selected_iron_capacity(board_snapshot, &ctx.chosen_iron_sources) < 1
                {
                    ctx.available_iron_sources = Self::available_iron_sources_for_selection(
                        board_snapshot,
                        player_idx,
                        &ctx.chosen_iron_sources,
                        1,
                    );
                    if !ctx.available_iron_sources.is_empty() {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseIronSource);
                    } // If no sources, action might fail or auto-confirm to fail.
                }
            }
            ActionType::DevelopDouble => {
                if ctx.selected_industry.is_none() {
                    if let Some(initial_opts) = &ctx.initial_dev_options {
                        if initial_opts.count_ones(..) > 0 {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseIndustry);
                        }
                    }
                } else if ctx.free_development_choice.is_none() {
                    if let Some(initial_opts) = &ctx.initial_dev_options {
                        let mut remaining_options = initial_opts.clone();
                        if let Some(first_choice) = ctx.selected_industry {
                            remaining_options.set(first_choice as usize, false);
                        }
                        if remaining_options.count_ones(..) > 0 {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseSecondIndustry);
                        }
                    }
                } else if Self::selected_iron_capacity(board_snapshot, &ctx.chosen_iron_sources) < 2
                {
                    ctx.available_iron_sources = Self::available_iron_sources_for_selection(
                        board_snapshot,
                        player_idx,
                        &ctx.chosen_iron_sources,
                        2,
                    );
                    if !ctx.available_iron_sources.is_empty() {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseIronSource);
                    } // If no sources, might fail.
                }
            }
            ActionType::BuildRailroad => {
                if ctx.selected_card_idx.is_none()
                    && !board_snapshot.players()[player_idx].hand.cards.is_empty()
                {
                    ctx.choices_needed
                        .push_back(NextActionChoiceKind::ChooseCard);
                } else if board_snapshot.era() == GameEra::Canal {
                    if ctx.selected_road_idx.is_none() {
                        if ctx
                            .initial_canal_options
                            .as_ref()
                            .map_or(false, |v| !v.is_empty())
                        {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseRoad);
                        }
                    }
                } else {
                    // Rail Era - SINGLE RAIL LINK PATH
                    if ctx.selected_road_idx.is_none() {
                        if ctx
                            .initial_single_rail_options
                            .as_ref()
                            .map_or(false, |v| !v.is_empty())
                        {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseRoad);
                        }
                    } else if ctx.chosen_coal_sources.is_empty() {
                        if let Some(road_idx) = ctx.selected_road_idx {
                            ctx.available_coal_for_road1 = Self::coal_sources_for_rail_road(
                                ctx.initial_single_rail_options.as_ref(),
                                road_idx,
                            );
                            if !ctx.available_coal_for_road1.is_empty() {
                                ctx.choices_needed
                                    .push_back(NextActionChoiceKind::ChooseCoalSource);
                            }
                        }
                    }
                }
            }
            ActionType::BuildDoubleRailroad => {
                if ctx.selected_card_idx.is_none()
                    && !board_snapshot.players()[player_idx].hand.cards.is_empty()
                {
                    ctx.choices_needed
                        .push_back(NextActionChoiceKind::ChooseCard);
                } else if ctx.selected_road_idx.is_none() {
                    // Step 1: Choose 1st road
                    if ctx
                        .initial_double_rail_first_link_options
                        .as_ref()
                        .map_or(false, |v| !v.is_empty())
                    {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseRoad);
                    } else { /* No valid first links for double action, will go to Confirm & fail */
                    }
                } else if ctx.chosen_coal_sources.get(0).is_none() {
                    // Step 2: Choose coal for 1st road
                    if let Some(road1_idx) = ctx.selected_road_idx {
                        ctx.available_coal_for_road1 =
                            Self::viable_first_coal_sources_for_double_rail(
                                board_snapshot,
                                player_idx,
                                road1_idx,
                                ctx.initial_double_rail_first_link_options.as_ref(),
                            );
                        if !ctx.available_coal_for_road1.is_empty() {
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseCoalSource);
                        }
                    }
                } else if ctx.selected_second_road_idx.is_none() {
                    // Step 3: Choose 2nd road
                    if let (Some(r1_idx), Some(coal1_src)) = (
                        ctx.selected_road_idx,
                        ctx.chosen_coal_sources.get(0).cloned(),
                    ) {
                        // Calculate money after TWO_RAILROAD_PRICE (15) and cost of first coal (if market)
                        let mut money_for_2nd_coal_check =
                            board_snapshot.players()[player_idx].money;
                        money_for_2nd_coal_check =
                            money_for_2nd_coal_check.saturating_sub(TWO_RAILROAD_PRICE);
                        if let ResourceSource::Market = coal1_src {
                            money_for_2nd_coal_check = money_for_2nd_coal_check
                                .saturating_sub(board_snapshot.get_coal_price(1));
                        }

                        let second_link_options_data = board_snapshot
                            .get_options_for_second_rail_link(
                                player_idx,
                                r1_idx,
                                &coal1_src,
                                money_for_2nd_coal_check,
                            );
                        if !second_link_options_data.is_empty() {
                            ctx.available_second_link_options_full_data = second_link_options_data;
                            ctx.choices_needed
                                .push_back(NextActionChoiceKind::ChooseSecondRoad);
                        } // Else, no valid second links found.
                    }
                } else if ctx.chosen_coal_sources.len() < 2 {
                    // Step 4: Coal for 2nd road (idx 1 in vec)
                    if let Some(sel_road2_idx) = ctx.selected_second_road_idx {
                        if let Some(chosen_second_link_data) = ctx
                            .available_second_link_options_full_data
                            .iter()
                            .find(|opt| opt.second_road_idx == sel_road2_idx)
                        {
                            ctx.available_coal_for_road2 = chosen_second_link_data
                                .potential_coal_sources_for_second_link
                                .clone();
                            if !ctx.available_coal_for_road2.is_empty() {
                                ctx.choices_needed
                                    .push_back(NextActionChoiceKind::ChooseCoalSource);
                            } // Else, no coal for 2nd link from pre-calc.
                        }
                    }
                } else if ctx.chosen_action_beer_source.is_none() {
                    // Step 5: Choose action beer
                    if let Some(sel_road2_idx) = ctx.selected_second_road_idx {
                        if let Some(chosen_second_link_data) = ctx
                            .available_second_link_options_full_data
                            .iter()
                            .find(|opt| opt.second_road_idx == sel_road2_idx)
                        {
                            ctx.available_beer_for_double_rail.clear();
                            ctx.available_beer_for_double_rail.extend(
                                chosen_second_link_data
                                    .potential_beer_sources_for_action
                                    .iter()
                                    .cloned(),
                            );
                            ctx.available_beer_for_double_rail.extend(
                                chosen_second_link_data.own_brewery_sources.iter().cloned(),
                            );
                            if !ctx.available_beer_for_double_rail.is_empty() {
                                ctx.choices_needed
                                    .push_back(NextActionChoiceKind::ChooseBeerSource);
                            }
                        }
                    }
                }
            }
            ActionType::Scout => {
                let hand_size = self.board.players()[self.current_player].hand.cards.len();
                if ctx.selected_card_idx.is_none() {
                    if hand_size > 0 {
                        // Can choose the first card
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseCard);
                    }
                } else if ctx.scout_additional_discard_indices.len() == 0 {
                    // Need to choose 1st additional card from remaining (hand_size - 1) cards
                    if hand_size > 1 {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseCard);
                    }
                } else if ctx.scout_additional_discard_indices.len() == 1 {
                    // Need to choose 2nd additional card from remaining (hand_size - 2) cards
                    if hand_size > 2 {
                        ctx.choices_needed
                            .push_back(NextActionChoiceKind::ChooseCard);
                    }
                }
                // If all three cards selected, Confirm will be added by default logic later
            }
            ActionType::Loan | ActionType::Pass => {
                // These actions should directly go to Confirm if all pre-conditions met.
                // For Loan, can_take_loan is checked before starting action.
                // For Pass, it's always an option.
            }
        };

        if ctx.choices_needed.is_empty() {
            let player_hand_is_empty = board_snapshot.players()[player_idx].hand.cards.is_empty();
            if action_requires_discard_card(ctx.action_type)
                && ctx.selected_card_idx.is_none()
                && !player_hand_is_empty
            {
                ctx.choices_needed
                    .push_back(NextActionChoiceKind::ChooseCard);
            } else {
                ctx.choices_needed.push_back(NextActionChoiceKind::Confirm);
            }
        }
    }
}

/// A normalized description of next legal choices for the active session step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChoiceSet {
    Industry(Vec<IndustryType>),
    Card(Vec<usize>),
    BuildLocation(Vec<usize>),
    Road(Vec<usize>),
    SecondRoad(Vec<usize>),
    CoalSource(Vec<ResourceSource>),
    IronSource(Vec<ResourceSource>),
    BeerSource(Vec<BeerSellSource>),
    ActionBeerSource(Vec<BreweryBeerSource>),
    SellTarget(Vec<usize>),
    FreeDevelopment(Vec<IndustryType>),
    SecondIndustry(Vec<IndustryType>),
    NetworkMode(Vec<NetworkMode>),
    ConfirmOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkMode {
    Single,
    Double,
}

fn action_requires_discard_card(action_type: ActionType) -> bool {
    matches!(
        action_type,
        ActionType::BuildBuilding
            | ActionType::Sell
            | ActionType::Develop
            | ActionType::DevelopDouble
            | ActionType::BuildRailroad
            | ActionType::BuildDoubleRailroad
            | ActionType::Loan
            | ActionType::Scout
            | ActionType::Pass
    )
}

/// A normalized action-choice input consumed by the shared action session API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionChoice {
    Industry(IndustryType),
    Card(usize),
    BuildLocation(usize),
    Road(usize),
    SellTarget(usize),
    CoalSource(ResourceSource),
    IronSource(ResourceSource),
    BeerSource(BeerSellSource),
    ActionBeerSource(BreweryBeerSource),
    FreeDevelopment(IndustryType),
    NetworkMode(NetworkMode),
    Confirm,
    Cancel,
}

#[derive(Debug, Clone, Default)]
pub struct DraftActionState {
    pub projected_money_delta: i32,
    pub consumed_coal_sources: Vec<ResourceSource>,
    pub consumed_iron_sources: Vec<ResourceSource>,
    pub consumed_beer_sources: Vec<BeerSellSource>,
    pub provisional_sell_targets: Vec<usize>,
}

/// Serializable top-level action intent payload that can be sent by UI or AI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionIntent {
    pub action_type: ActionType,
    pub selected_industry: Option<IndustryType>,
    pub selected_second_industry: Option<IndustryType>,
    pub selected_card_idx: Option<usize>,
    pub scout_additional_discard_indices: Vec<usize>,
    pub selected_build_location: Option<usize>,
    pub selected_road_idx: Option<usize>,
    pub selected_second_road_idx: Option<usize>,
    pub chosen_coal_sources: Vec<ResourceSource>,
    pub chosen_iron_sources: Vec<ResourceSource>,
    pub chosen_beer_sources: Vec<BeerSellSource>,
    pub chosen_action_beer_source: Option<BreweryBeerSource>,
    pub selected_network_mode: Option<NetworkMode>,
    pub sell_choices: Vec<SellChoice>,
    pub free_development_choice: Option<IndustryType>,
}

/// Current session state snapshot for client-side orchestration.
#[derive(Debug, Clone)]
pub struct ActionSession {
    pub player_idx: usize,
    pub action_type: ActionType,
    pub next_choices: Vec<NextActionChoiceKind>,
    pub can_confirm: bool,
    pub intent: ActionIntent,
    pub draft: DraftActionState,
}

/// Commit result returned after a confirmed action.
#[derive(Debug, Clone)]
pub struct CommittedActionResult {
    pub action_type: ActionType,
    pub player_idx: usize,
}

/// Between-turn shortfall resolution candidate tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortfallTileChoice {
    pub build_location_idx: usize,
    pub liquidation_value: u16,
}

/// Explicit shortfall resolution session for negative-income handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortfallResolutionSession {
    pub player_idx: usize,
    pub shortfall: u16,
    pub removable_tiles: Vec<ShortfallTileChoice>,
}

impl GameFramework {
    pub fn get_valid_root_actions(&self) -> Vec<ActionType> {
        let validation = self.compute_valid_options();
        self.get_valid_action_types(&validation)
    }

    pub fn start_action_session(&mut self, action_type: ActionType) -> ActionSession {
        let validation: ActionValidationResult = self.compute_valid_options();
        self.start_action(action_type, validation);
        self.current_session()
            .expect("Action session should exist right after start_action")
    }

    pub fn current_session(&self) -> Option<ActionSession> {
        self.action_context.as_ref().map(|ctx| ActionSession {
            player_idx: self.current_player,
            action_type: ctx.action_type,
            next_choices: ctx.choices_needed.iter().copied().collect(),
            can_confirm: self.can_confirm(),
            intent: Self::intent_from_context(ctx),
            draft: Self::draft_from_context(ctx),
        })
    }

    pub fn cancel_action_session(&mut self) {
        self.action_context = None;
    }

    pub fn can_confirm(&self) -> bool {
        self.action_context.as_ref().is_some_and(|ctx| {
            ctx.choices_needed.front().copied() == Some(NextActionChoiceKind::Confirm)
                || (ctx.action_type == ActionType::Sell
                    && ctx.pending_sell_building_loc.is_none()
                    && !ctx.current_sell_choices.is_empty()
                    && ctx.choices_needed.front().copied()
                        == Some(NextActionChoiceKind::ChooseSellTargets))
        })
    }

    pub fn get_next_choice_set(&self) -> Option<ChoiceSet> {
        let ctx = self.action_context.as_ref()?;
        let next = ctx.choices_needed.front().copied()?;
        let set = match next {
            NextActionChoiceKind::ChooseIndustry => {
                if ctx.action_type == ActionType::BuildBuilding {
                    let mut inds = Vec::<IndustryType>::new();
                    for opt in &ctx.current_filtered_build_options {
                        if !inds.contains(&opt.industry_type) {
                            inds.push(opt.industry_type);
                        }
                    }
                    ChoiceSet::Industry(inds)
                } else {
                    ChoiceSet::Industry(Self::industries_from_bitset(
                        ctx.initial_dev_options.as_ref(),
                    ))
                }
            }
            NextActionChoiceKind::ChooseSecondIndustry => ChoiceSet::SecondIndustry(
                Self::industries_from_bitset(ctx.initial_dev_options.as_ref()),
            ),
            NextActionChoiceKind::ChooseFreeDevelopment => {
                let dev_opts = self
                    .board
                    .get_valid_free_development_options(self.current_player);
                ChoiceSet::FreeDevelopment(Self::industries_from_bitset(Some(&dev_opts)))
            }
            NextActionChoiceKind::ChooseCard => {
                if ctx.action_type == ActionType::BuildBuilding {
                    let mut valid_cards = Vec::<usize>::new();
                    for opt in &ctx.current_filtered_build_options {
                        if !valid_cards.contains(&opt.card_used_idx) {
                            valid_cards.push(opt.card_used_idx);
                        }
                    }
                    ChoiceSet::Card(valid_cards)
                } else {
                    let len = self.board.players()[self.current_player].hand.cards.len();
                    ChoiceSet::Card((0..len).collect())
                }
            }
            NextActionChoiceKind::ChooseBuildLocation => {
                let mut locs = Vec::<usize>::new();
                for opt in &ctx.current_filtered_build_options {
                    if !locs.contains(&opt.build_location_idx) {
                        locs.push(opt.build_location_idx);
                    }
                }
                ChoiceSet::BuildLocation(locs)
            }
            NextActionChoiceKind::ChooseRoad => {
                if self.board.era() == GameEra::Railroad
                    && ctx.action_type == ActionType::BuildRailroad
                    && ctx.selected_network_mode.is_none()
                {
                    let mut modes = vec![NetworkMode::Single];
                    if !Self::viable_first_roads_for_double_rail(
                        &self.board,
                        self.current_player,
                        ctx.initial_double_rail_first_link_options.as_ref(),
                    )
                    .is_empty()
                    {
                        modes.push(NetworkMode::Double);
                    }
                    ChoiceSet::NetworkMode(modes)
                } else if self.board.era() == GameEra::Canal {
                    ChoiceSet::Road(ctx.initial_canal_options.clone().unwrap_or_default())
                } else {
                    let options = if ctx.action_type == ActionType::BuildDoubleRailroad {
                        ctx.initial_double_rail_first_link_options.as_ref()
                    } else {
                        ctx.initial_single_rail_options.as_ref()
                    };
                    let roads = if ctx.action_type == ActionType::BuildDoubleRailroad {
                        Self::viable_first_roads_for_double_rail(
                            &self.board,
                            self.current_player,
                            options,
                        )
                    } else {
                        options
                            .map(|entries| entries.iter().map(|entry| entry.road_idx).collect())
                            .unwrap_or_default()
                    };
                    ChoiceSet::Road(roads)
                }
            }
            NextActionChoiceKind::ChooseSecondRoad => ChoiceSet::SecondRoad(
                ctx.available_second_link_options_full_data
                    .iter()
                    .map(|o| o.second_road_idx)
                    .collect(),
            ),
            NextActionChoiceKind::ChooseCoalSource => {
                if ctx.action_type == ActionType::BuildBuilding {
                    ChoiceSet::CoalSource(ctx.available_build_coal_sources.clone())
                } else if ctx.action_type == ActionType::BuildDoubleRailroad
                    && ctx.chosen_coal_sources.len() >= 1
                {
                    ChoiceSet::CoalSource(ctx.available_coal_for_road2.clone())
                } else {
                    ChoiceSet::CoalSource(ctx.available_coal_for_road1.clone())
                }
            }
            NextActionChoiceKind::ChooseIronSource => {
                if ctx.action_type == ActionType::BuildBuilding {
                    ChoiceSet::IronSource(ctx.available_build_iron_sources.clone())
                } else {
                    ChoiceSet::IronSource(ctx.available_iron_sources.clone())
                }
            }
            NextActionChoiceKind::ChooseBeerSource => {
                if ctx.action_type == ActionType::BuildDoubleRailroad {
                    ChoiceSet::ActionBeerSource(ctx.available_beer_for_double_rail.clone())
                } else {
                    ChoiceSet::BeerSource(ctx.available_beer_for_pending_sell.clone())
                }
            }
            NextActionChoiceKind::ChooseSellTargets => ChoiceSet::SellTarget(
                ctx.current_filtered_sell_options
                    .iter()
                    .map(|o| o.location)
                    .collect(),
            ),
            NextActionChoiceKind::ChooseMerchant => ChoiceSet::ConfirmOnly,
            NextActionChoiceKind::Confirm => ChoiceSet::ConfirmOnly,
        };
        Some(set)
    }

    pub fn apply_action_choice(
        &mut self,
        choice: ActionChoice,
    ) -> Result<Option<ActionSession>, String> {
        if self.action_context.is_none() {
            return Err("No active action session".to_string());
        }
        match choice {
            ActionChoice::Industry(industry) => self.choose_industry(industry),
            ActionChoice::Card(idx) => self.choose_card(idx),
            ActionChoice::BuildLocation(loc) => self.choose_build_location(loc),
            ActionChoice::Road(road) => self.choose_road(road),
            ActionChoice::SellTarget(loc) => self.choose_sell_target(loc),
            ActionChoice::CoalSource(src) => self.choose_resource_source(src, ResourceType::Coal),
            ActionChoice::IronSource(src) => self.choose_resource_source(src, ResourceType::Iron),
            ActionChoice::BeerSource(src) => self.choose_beer_source(src),
            ActionChoice::ActionBeerSource(src) => self.choose_action_beer_source(src),
            ActionChoice::FreeDevelopment(industry) => self.choose_industry(industry),
            ActionChoice::NetworkMode(mode) => self.choose_network_mode(mode),
            ActionChoice::Cancel => {
                self.cancel_action_session();
                return Ok(None);
            }
            ActionChoice::Confirm => {
                if !self.can_confirm() {
                    return Err("Action cannot be confirmed yet; more choices required".to_string());
                }
                self.confirm_action_session()?;
                return Ok(None);
            }
        }
        Ok(self.current_session())
    }

    pub fn confirm_action_session(&mut self) -> Result<CommittedActionResult, String> {
        let action_type = match self.action_context.as_ref() {
            Some(ctx) => ctx.action_type,
            None => return Err("No active action session".to_string()),
        };
        let player_idx = self.current_player;

        let valid_actions = self.get_valid_root_actions();
        let root_action = if action_type == ActionType::BuildDoubleRailroad {
            ActionType::BuildRailroad
        } else {
            action_type
        };
        if !valid_actions.contains(&root_action) {
            return Err(format!("Action {:?} is no longer legal", action_type));
        }

        self.choose_confirm()?;
        Ok(CommittedActionResult {
            action_type,
            player_idx,
        })
    }

    pub fn choose_network_mode(&mut self, mode: NetworkMode) {
        if let Some(ctx) = self.action_context.as_mut() {
            ctx.selected_network_mode = Some(mode);
            ctx.action_type = if mode == NetworkMode::Double {
                ActionType::BuildDoubleRailroad
            } else {
                ActionType::BuildRailroad
            };
            self.update_next_step();
        }
    }

    pub fn add_iteration(&mut self) -> Result<Option<ActionSession>, String> {
        if self.action_context.is_none() {
            return Err("No active action session".to_string());
        }
        self.update_next_step();
        Ok(self.current_session())
    }

    pub fn remove_iteration(&mut self, index: usize) -> Result<Option<ActionSession>, String> {
        let Some(ctx) = self.action_context.as_mut() else {
            return Err("No active action session".to_string());
        };
        match ctx.action_type {
            ActionType::Sell => {
                if index < ctx.current_sell_choices.len() {
                    let removed = ctx.current_sell_choices.remove(index);
                    ctx.current_filtered_sell_options
                        .retain(|o| o.location != removed.location);
                }
            }
            ActionType::DevelopDouble => {
                if index == 1 {
                    ctx.free_development_choice = None;
                } else if index == 0 {
                    ctx.selected_industry = None;
                    ctx.free_development_choice = None;
                    ctx.chosen_iron_sources.clear();
                }
            }
            ActionType::BuildDoubleRailroad => {
                if index == 1 {
                    ctx.selected_second_road_idx = None;
                    if ctx.chosen_coal_sources.len() > 1 {
                        ctx.chosen_coal_sources.truncate(1);
                    }
                    ctx.chosen_action_beer_source = None;
                } else if index == 0 {
                    ctx.selected_road_idx = None;
                    ctx.selected_second_road_idx = None;
                    ctx.chosen_coal_sources.clear();
                    ctx.chosen_action_beer_source = None;
                }
            }
            _ => {}
        }
        self.update_next_step();
        Ok(self.current_session())
    }

    pub fn recompute_choices(&mut self) -> Option<ActionSession> {
        if self.action_context.is_none() {
            return None;
        }
        self.update_next_step();
        self.current_session()
    }

    pub fn start_shortfall_resolution_session(
        &self,
        player_idx: usize,
        shortfall: u16,
    ) -> ShortfallResolutionSession {
        use crate::core::static_data::INDUSTRY_MAT;

        let mut removable_tiles = Vec::<ShortfallTileChoice>::new();
        for loc in self.board.state.player_building_mask[player_idx].ones() {
            if let Some(building) = self.board.state.bl_to_building.get(&loc) {
                let original_cost =
                    INDUSTRY_MAT[building.industry as usize][building.level.as_usize()].money_cost;
                removable_tiles.push(ShortfallTileChoice {
                    build_location_idx: loc,
                    liquidation_value: original_cost / 2,
                });
            }
        }
        removable_tiles.sort_by_key(|t| t.liquidation_value);

        ShortfallResolutionSession {
            player_idx,
            shortfall,
            removable_tiles,
        }
    }

    pub fn resolve_shortfall_with_tile_choices(
        &mut self,
        mut session: ShortfallResolutionSession,
        chosen_tile_order: Vec<usize>,
    ) {
        use crate::board::resources::remove_building_from_board;

        self.board.state.players[session.player_idx].money = 0;

        for chosen_loc in chosen_tile_order {
            if session.shortfall == 0 {
                break;
            }
            let Some(tile) = session
                .removable_tiles
                .iter()
                .find(|t| t.build_location_idx == chosen_loc)
            else {
                continue;
            };
            self.board.state.players[session.player_idx].gain_money(tile.liquidation_value);
            session.shortfall = session.shortfall.saturating_sub(tile.liquidation_value);
            remove_building_from_board(&mut self.board.state, chosen_loc);
        }

        if session.shortfall > 0 {
            let vp_loss = session.shortfall;
            let player_state = &mut self.board.state.players[session.player_idx];
            player_state.victory_points = player_state.victory_points.saturating_sub(vp_loss);
            self.board.state.visible_vps[session.player_idx] =
                self.board.state.visible_vps[session.player_idx].saturating_sub(vp_loss);
        }
    }

    fn intent_from_context(ctx: &ActionContext) -> ActionIntent {
        ActionIntent {
            action_type: ctx.action_type,
            selected_industry: ctx.selected_industry,
            selected_second_industry: ctx.free_development_choice,
            selected_card_idx: ctx.selected_card_idx,
            scout_additional_discard_indices: ctx.scout_additional_discard_indices.clone(),
            selected_build_location: ctx.selected_build_location,
            selected_road_idx: ctx.selected_road_idx,
            selected_second_road_idx: ctx.selected_second_road_idx,
            chosen_coal_sources: ctx.chosen_coal_sources.clone(),
            chosen_iron_sources: ctx.chosen_iron_sources.clone(),
            chosen_beer_sources: ctx.chosen_beer_sources.clone(),
            chosen_action_beer_source: ctx.chosen_action_beer_source,
            selected_network_mode: ctx.selected_network_mode,
            sell_choices: ctx.current_sell_choices.clone(),
            free_development_choice: ctx.free_development_choice,
        }
    }

    fn draft_from_context(ctx: &ActionContext) -> DraftActionState {
        DraftActionState {
            projected_money_delta: 0,
            consumed_coal_sources: ctx.chosen_coal_sources.clone(),
            consumed_iron_sources: ctx.chosen_iron_sources.clone(),
            consumed_beer_sources: ctx.chosen_beer_sources.clone(),
            provisional_sell_targets: ctx
                .current_sell_choices
                .iter()
                .map(|s| s.location)
                .collect(),
        }
    }

    fn industries_from_bitset(bitset: Option<&fixedbitset::FixedBitSet>) -> Vec<IndustryType> {
        let mut out = Vec::<IndustryType>::new();
        if let Some(bs) = bitset {
            for idx in bs.ones() {
                out.push(IndustryType::from_usize(idx));
            }
        }
        out
    }
}
