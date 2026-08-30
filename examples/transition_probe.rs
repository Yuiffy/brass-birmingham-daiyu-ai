use fast_brass::actions::SellOption;
use fast_brass::board::resources::BeerSellSource;
use fast_brass::core::locations::LocationName;
use fast_brass::core::static_data::{INDUSTRY_MAT, LINK_LOCATIONS};
use fast_brass::core::types::{ActionType, IndustryType, N_BL, TOTAL_TOWNS};
use fast_brass::game::legal_actions::{enumerate_legal_actions, LegalAction};
use fast_brass::game::rule_ai::{rank_rule_actions, RuleActionScore, RuleDecisionConfig};
use fast_brass::game::runner::GameRunner;
use std::collections::{BTreeMap, HashSet};
use std::env;

#[derive(Debug, Clone)]
struct ForcedActionSpec {
    action_index: usize,
    ranked_index: Option<usize>,
    key: Option<String>,
    root: Option<ActionType>,
    industry: Option<IndustryType>,
}

impl ForcedActionSpec {
    fn from_args(args: &[String]) -> Result<Option<Self>, String> {
        let action_index = parse_usize_arg(args, "force=")?;
        let ranked_index = parse_usize_arg(args, "force_rank=")?;
        let key = named_arg(args, "force_key=").map(str::to_string);
        let root = named_arg(args, "force_root=")
            .map(parse_action_type)
            .transpose()?;
        let industry = named_arg(args, "force_industry=")
            .map(parse_industry)
            .transpose()?;
        let has_selector = ranked_index.is_some()
            || key.as_ref().is_some_and(|value| !value.is_empty())
            || root.is_some()
            || industry.is_some();

        match action_index {
            Some(action_index) if has_selector => Ok(Some(Self {
                action_index,
                ranked_index,
                key,
                root,
                industry,
            })),
            Some(_) => Err(
                "force=<action> requires force_rank, force_key, force_root, or force_industry"
                    .to_string(),
            ),
            None if has_selector => {
                Err("a force selector requires force=<global action index>".to_string())
            }
            None => Ok(None),
        }
    }

    fn select(&self, ranked: &[RuleActionScore]) -> Result<usize, String> {
        if let Some(index) = self.ranked_index {
            let candidate = ranked.get(index).ok_or_else(|| {
                format!(
                    "force_rank={index} is out of range for {} candidates at action {}",
                    ranked.len(),
                    self.action_index
                )
            })?;
            if self.matches(candidate) {
                return Ok(index);
            }
            return Err(format!(
                "candidate at force_rank={index} does not match the other force selectors: {}",
                candidate.action.key()
            ));
        }

        ranked
            .iter()
            .position(|candidate| self.matches(candidate))
            .ok_or_else(|| {
                let available = ranked
                    .iter()
                    .take(16)
                    .map(|candidate| candidate.action.key())
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "no candidate matches the force selectors at action {}; first candidates: {available}",
                    self.action_index
                )
            })
    }

    fn matches(&self, candidate: &RuleActionScore) -> bool {
        self.key
            .as_ref()
            .is_none_or(|key| candidate.action.key() == *key)
            && self
                .root
                .is_none_or(|root| candidate.action.intent.action_type == root)
            && self
                .industry
                .is_none_or(|industry| candidate.action.intent.selected_industry == Some(industry))
    }
}

fn named_arg<'a>(args: &'a [String], prefix: &str) -> Option<&'a str> {
    args.iter().find_map(|value| value.strip_prefix(prefix))
}

fn parse_usize_arg(args: &[String], prefix: &str) -> Result<Option<usize>, String> {
    named_arg(args, prefix)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid {prefix}{value}: {error}"))
        })
        .transpose()
}

fn parse_action_type(value: &str) -> Result<ActionType, String> {
    match value.to_ascii_lowercase().as_str() {
        "build" | "build_building" => Ok(ActionType::BuildBuilding),
        "rail" | "railroad" | "build_railroad" => Ok(ActionType::BuildRailroad),
        "double" | "double_rail" | "build_double_railroad" => Ok(ActionType::BuildDoubleRailroad),
        "develop" => Ok(ActionType::Develop),
        "develop_double" | "double_develop" => Ok(ActionType::DevelopDouble),
        "sell" => Ok(ActionType::Sell),
        "loan" => Ok(ActionType::Loan),
        "scout" => Ok(ActionType::Scout),
        "pass" => Ok(ActionType::Pass),
        _ => Err(format!("unknown force_root={value}")),
    }
}

fn parse_industry(value: &str) -> Result<IndustryType, String> {
    match value.to_ascii_lowercase().as_str() {
        "0" | "coal" => Ok(IndustryType::Coal),
        "1" | "iron" => Ok(IndustryType::Iron),
        "2" | "beer" => Ok(IndustryType::Beer),
        "3" | "goods" => Ok(IndustryType::Goods),
        "4" | "pottery" => Ok(IndustryType::Pottery),
        "5" | "cotton" => Ok(IndustryType::Cotton),
        _ => Err(format!("unknown force_industry={value}")),
    }
}

fn resolve_shortfalls(runner: &mut GameRunner) {
    for session in runner.take_shortfall_sessions() {
        let mut tiles = session.removable_tiles.clone();
        tiles.sort_by_key(|tile| (tile.liquidation_value, tile.build_location_idx));
        runner.resolve_shortfall_with_tiles(
            session,
            tiles
                .into_iter()
                .map(|tile| tile.build_location_idx)
                .collect(),
        );
    }
}

fn advance_to_decision(runner: &mut GameRunner) -> Result<(), String> {
    for _ in 0..256 {
        resolve_shortfalls(runner);
        if runner.is_game_finished() {
            return Ok(());
        }
        if runner.actions_remaining_in_turn == 0 {
            runner.start_turn();
        }
        if !runner.framework.get_valid_root_actions().is_empty() {
            return Ok(());
        }
        runner.end_action_slot();
        if runner.actions_remaining_in_turn == 0 {
            runner.end_turn();
        }
    }
    Err("advance_to_decision exceeded 256 forced slots".to_string())
}

fn player_metrics(
    runner: &GameRunner,
    actor: usize,
) -> (usize, f64, f64, usize, usize, usize, usize) {
    let state = &runner.framework.board.state;
    let mut flagged = 0usize;
    let mut flagged_vp = 0.0;
    let mut flagged_income = 0.0;
    for building in state.bl_to_building.values() {
        if building.owner.as_usize() != actor || building.flipped {
            continue;
        }
        let data = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        if data.removed_after_phase1 {
            flagged += 1;
            flagged_vp += f64::from(data.vp_on_flip);
            flagged_income += f64::from(data.income);
        }
    }
    let builds = runner.framework.board.get_valid_build_options(actor).len();
    let sells = runner.framework.board.get_valid_sell_options(actor).len();
    let develops = runner
        .framework
        .board
        .get_valid_development_options(actor)
        .count_ones(..);
    let networks = if runner.game_phase == fast_brass::game::runner::GamePhase::Canal {
        runner.framework.board.get_valid_canal_options(actor).len()
    } else {
        runner
            .framework
            .board
            .get_valid_single_rail_options(actor)
            .len()
            + runner
                .framework
                .board
                .get_valid_double_rail_first_link_options(actor)
                .len()
    };
    (
        flagged,
        flagged_vp,
        flagged_income,
        builds,
        sells,
        develops,
        networks,
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct RiskState {
    buildings: usize,
    unflipped: usize,
    unflipped_removed: usize,
    unflipped_saleable: usize,
    valid_builds: usize,
    valid_saleable_builds: usize,
    valid_sells: usize,
    valid_networks: usize,
    cash: f64,
    income: f64,
    total_cards: usize,
}

fn is_saleable(industry: IndustryType) -> bool {
    matches!(
        industry,
        IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
    )
}

#[derive(Debug, Clone, Copy)]
struct MacroPlayerSnapshot {
    victory_points: u16,
    potential_era_vp: u16,
    income: i8,
    cash: u16,
    road_value: f64,
}

impl MacroPlayerSnapshot {
    fn progress(self) -> u16 {
        self.victory_points.saturating_add(self.potential_era_vp)
    }
}

#[derive(Debug, Clone)]
struct FinancedSaleCompletion {
    loan_key: String,
    loan_income_loss: i16,
    route_key: String,
    sell_key: String,
    cards_used: usize,
    players: Vec<MacroPlayerSnapshot>,
    sell_option_beer: String,
    chosen_beer: String,
}

fn diagnostic_potential_era_vps(runner: &GameRunner) -> Vec<u16> {
    let state = &runner.framework.board.state;
    let mut potential_vps = vec![0u16; state.players.len()];
    for road_idx in state.built_roads.ones() {
        let owner = (0..state.players.len())
            .find(|player_idx| state.player_road_mask[*player_idx].contains(road_idx));
        let Some(owner_idx) = owner else {
            continue;
        };
        for location_idx in LINK_LOCATIONS[road_idx].locations.ones() {
            for building_idx in LocationName::from_usize(location_idx).to_bl_set().ones() {
                if let Some(building) = state
                    .bl_to_building
                    .get(&building_idx)
                    .filter(|building| building.flipped)
                {
                    let data =
                        &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
                    potential_vps[owner_idx] =
                        potential_vps[owner_idx].saturating_add(u16::from(data.road_vp));
                }
            }
        }
    }
    for building in state
        .bl_to_building
        .values()
        .filter(|building| building.flipped)
    {
        let owner = building.owner.as_usize();
        let data = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        potential_vps[owner] = potential_vps[owner].saturating_add(u16::from(data.vp_on_flip));
    }
    potential_vps
}

fn diagnostic_road_value(runner: &GameRunner, actor: usize) -> f64 {
    let state = &runner.framework.board.state;
    let mut value = 0.0;
    for road_idx in state.player_road_mask[actor].ones() {
        for location_idx in LINK_LOCATIONS[road_idx].locations.ones() {
            let location = LocationName::from_usize(location_idx);
            for building_idx in location.to_bl_set().ones() {
                if let Some(building) = state.bl_to_building.get(&building_idx) {
                    let data =
                        &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
                    value += f64::from(data.road_vp) * if building.flipped { 1.0 } else { 0.25 };
                }
            }
        }
    }
    value
}

fn macro_player_snapshots(runner: &GameRunner) -> Vec<MacroPlayerSnapshot> {
    let potential_vps = diagnostic_potential_era_vps(runner);
    runner
        .framework
        .board
        .state
        .players
        .iter()
        .enumerate()
        .map(|(actor, player)| MacroPlayerSnapshot {
            victory_points: player.victory_points,
            potential_era_vp: potential_vps[actor],
            income: player.get_income_amount(player.income_level),
            cash: player.money,
            road_value: diagnostic_road_value(runner, actor),
        })
        .collect()
}

fn pending_saleable_count(runner: &GameRunner, actor: usize) -> usize {
    runner
        .framework
        .board
        .state
        .bl_to_building
        .values()
        .filter(|building| {
            building.owner.as_usize() == actor
                && !building.flipped
                && is_saleable(building.industry)
        })
        .count()
}

fn diagnostic_merchant_target_mask(
    runner: &GameRunner,
    industry: IndustryType,
) -> [bool; TOTAL_TOWNS] {
    let mut targets = [false; TOTAL_TOWNS];
    runner
        .framework
        .board
        .state
        .trade_post_slots
        .iter()
        .enumerate()
        .for_each(|(slot, merchant)| {
            let Some(merchant) = merchant.as_ref() else {
                return;
            };
            if !merchant.industries.contains(industry.as_usize()) {
                return;
            }
            let location = match slot {
                0 => 22,
                1 | 2 => 23,
                3 | 4 => 24,
                5 | 6 => 25,
                7 | 8 => 26,
                _ => return,
            };
            targets[location] = true;
        });
    targets
}

fn diagnostic_owned_beer_units(runner: &GameRunner, player: usize) -> u16 {
    let state = &runner.framework.board.state;
    state.player_building_mask[player]
        .ones()
        .filter_map(|location| state.bl_to_building.get(&location))
        .filter(|building| {
            building.industry == IndustryType::Beer
                && !building.flipped
                && building.resource_amt > 0
        })
        .map(|building| u16::from(building.resource_amt))
        .sum()
}

/// Diagnostic mirror of `rule_ai::opponent_competes_for_external_beer`.
fn diagnostic_opponent_competes_for_external_beer(
    runner: &GameRunner,
    actor: usize,
    actor_sell_option: &SellOption,
) -> bool {
    let state = &runner.framework.board.state;
    let external_beer_locations = actor_sell_option
        .beer_locations
        .ones()
        .filter(|location| {
            *location >= N_BL
                || state
                    .bl_to_building
                    .get(location)
                    .is_some_and(|building| building.owner.as_usize() != actor)
        })
        .collect::<Vec<_>>();
    if external_beer_locations.is_empty() {
        return false;
    }

    (0..state.players.len())
        .filter(|opponent| *opponent != actor)
        .any(|opponent| {
            let own_beer = diagnostic_owned_beer_units(runner, opponent);
            state.player_building_mask[opponent]
                .ones()
                .any(|product_location| {
                    let Some(building) = state.bl_to_building.get(&product_location) else {
                        return false;
                    };
                    if building.flipped || !is_saleable(building.industry) {
                        return false;
                    }
                    let beer_needed = u16::from(
                        INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()]
                            .beer_needed,
                    );
                    if beer_needed <= own_beer {
                        return false;
                    }

                    let product_town = LocationName::from_bl_idx(product_location);
                    let matching_merchants =
                        diagnostic_merchant_target_mask(runner, building.industry);
                    if !matching_merchants
                        .iter()
                        .enumerate()
                        .any(|(town, matches)| {
                            *matches
                                && state.connectivity.are_towns_connected(
                                    product_town,
                                    LocationName::from_usize(town),
                                )
                        })
                    {
                        return false;
                    }

                    external_beer_locations.iter().copied().any(|source| {
                        if source >= N_BL {
                            let slot = source - N_BL;
                            let Some(merchant) = state
                                .trade_post_slots
                                .get(slot)
                                .and_then(|merchant| merchant.as_ref())
                            else {
                                return false;
                            };
                            let trade_post =
                                fast_brass::market::merchants::slot_to_trade_post(slot)
                                    .to_location_name();
                            state.trade_post_beer.contains(slot)
                                && merchant.industries.contains(building.industry.as_usize())
                                && state
                                    .connectivity
                                    .are_towns_connected(product_town, trade_post)
                        } else {
                            state.bl_to_building.get(&source).is_some_and(|brewery| {
                                brewery.industry == IndustryType::Beer
                                    && !brewery.flipped
                                    && brewery.resource_amt > 0
                                    && (brewery.owner.as_usize() == opponent
                                        || state.connectivity.are_towns_connected(
                                            product_town,
                                            LocationName::from_bl_idx(source),
                                        ))
                            })
                        }
                    })
                })
        })
}

fn describe_external_beer_source(
    runner: &GameRunner,
    actor: usize,
    encoded_location: usize,
) -> Option<String> {
    let state = &runner.framework.board.state;
    if encoded_location >= N_BL {
        let slot = encoded_location - N_BL;
        return Some(format!("merchant{slot}"));
    }
    state
        .bl_to_building
        .get(&encoded_location)
        .filter(|building| building.owner.as_usize() != actor)
        .map(|building| {
            format!(
                "brewery{encoded_location}:p{}:units{}",
                building.owner.as_usize(),
                building.resource_amt
            )
        })
}

fn describe_chosen_beer_source(runner: &GameRunner, source: BeerSellSource) -> String {
    match source {
        BeerSellSource::Building(location) => runner
            .framework
            .board
            .state
            .bl_to_building
            .get(&location)
            .map(|building| {
                format!(
                    "brewery{location}:p{}:units{}",
                    building.owner.as_usize(),
                    building.resource_amt
                )
            })
            .unwrap_or_else(|| format!("brewery{location}:missing")),
        BeerSellSource::TradePost(slot) => format!("merchant{slot}"),
    }
}

fn sell_beer_diagnostics(
    runner: &GameRunner,
    actor: usize,
    sell: &LegalAction,
) -> (String, String) {
    let options = runner.framework.board.get_valid_sell_options(actor);
    let option_descriptions = sell
        .intent
        .sell_choices
        .iter()
        .map(|choice| {
            let Some(option) = options
                .iter()
                .find(|option| option.location == choice.location)
            else {
                return format!("loc{}:option_missing", choice.location);
            };
            let external = option
                .beer_locations
                .ones()
                .filter_map(|source| describe_external_beer_source(runner, actor, source))
                .collect::<Vec<_>>();
            format!(
                "loc{}:external={}:opponent_competes={}",
                choice.location,
                if external.is_empty() {
                    "none".to_string()
                } else {
                    external.join("+")
                },
                diagnostic_opponent_competes_for_external_beer(runner, actor, option),
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    let chosen = sell
        .intent
        .sell_choices
        .iter()
        .map(|choice| {
            let sources = choice
                .beer_sources
                .iter()
                .copied()
                .map(|source| describe_chosen_beer_source(runner, source))
                .collect::<Vec<_>>();
            format!(
                "loc{}:{}",
                choice.location,
                if sources.is_empty() {
                    "none".to_string()
                } else {
                    sources.join("+")
                }
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    (option_descriptions, chosen)
}

fn completion_is_better(
    candidate: &FinancedSaleCompletion,
    current: &FinancedSaleCompletion,
    actor: usize,
) -> bool {
    let candidate_actor = candidate.players[actor];
    let current_actor = current.players[actor];
    candidate_actor
        .progress()
        .cmp(&current_actor.progress())
        .then_with(|| {
            candidate_actor
                .victory_points
                .cmp(&current_actor.victory_points)
        })
        .then_with(|| candidate_actor.income.cmp(&current_actor.income))
        .then_with(|| candidate_actor.cash.cmp(&current_actor.cash))
        .then_with(|| {
            candidate_actor
                .road_value
                .total_cmp(&current_actor.road_value)
        })
        .then_with(|| current.cards_used.cmp(&candidate.cards_used))
        .is_gt()
        || (candidate_actor.progress() == current_actor.progress()
            && candidate_actor.victory_points == current_actor.victory_points
            && candidate_actor.income == current_actor.income
            && candidate_actor.cash == current_actor.cash
            && candidate_actor
                .road_value
                .total_cmp(&current_actor.road_value)
                .is_eq()
            && candidate.cards_used == current.cards_used
            && format!(
                "{}|{}|{}",
                candidate.loan_key, candidate.route_key, candidate.sell_key
            ) < format!(
                "{}|{}|{}",
                current.loan_key, current.route_key, current.sell_key
            ))
}

fn card_invariant_unique(actions: Vec<LegalAction>) -> Vec<LegalAction> {
    let mut seen = HashSet::new();
    actions
        .into_iter()
        .filter(|action| seen.insert(action.card_invariant_key()))
        .collect()
}

fn best_full_sale_after_route(
    runner: &GameRunner,
    actor: usize,
    route_key: &str,
    loan_key: &str,
    loan_income_loss: i16,
    discard_history_before: usize,
) -> Result<(usize, usize, Option<FinancedSaleCompletion>), String> {
    if runner
        .framework
        .board
        .get_valid_sell_options(actor)
        .is_empty()
    {
        return Ok((0, 0, None));
    }
    let sells = card_invariant_unique(enumerate_legal_actions(runner)?)
        .into_iter()
        .filter(|action| action.intent.action_type == ActionType::Sell)
        .collect::<Vec<_>>();
    let sell_attempts = sells.len();
    let mut full_sells = 0usize;
    let mut best = None;
    for sell in sells {
        let (sell_option_beer, chosen_beer) = sell_beer_diagnostics(runner, actor, &sell);
        let mut after_sell = runner.clone();
        sell.apply(&mut after_sell)?;
        if pending_saleable_count(&after_sell, actor) != 0 {
            continue;
        }
        full_sells += 1;
        let candidate = FinancedSaleCompletion {
            loan_key: loan_key.to_string(),
            loan_income_loss,
            route_key: route_key.to_string(),
            sell_key: sell.key(),
            cards_used: after_sell
                .discard_history()
                .len()
                .saturating_sub(discard_history_before),
            players: macro_player_snapshots(&after_sell),
            sell_option_beer,
            chosen_beer,
        };
        if best
            .as_ref()
            .is_none_or(|current| completion_is_better(&candidate, current, actor))
        {
            best = Some(candidate);
        }
    }
    Ok((sell_attempts, full_sells, best))
}

fn merge_completion(
    best: &mut Option<FinancedSaleCompletion>,
    candidate: Option<FinancedSaleCompletion>,
    actor: usize,
) {
    let Some(candidate) = candidate else {
        return;
    };
    if best
        .as_ref()
        .is_none_or(|current| completion_is_better(&candidate, current, actor))
    {
        *best = Some(candidate);
    }
}

fn format_opponent_changes(
    before: &[MacroPlayerSnapshot],
    after: &[MacroPlayerSnapshot],
    actor: usize,
) -> String {
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter(|(player, _)| *player != actor)
        .map(|(player, (before, after))| {
            format!(
                "p{player}:vp{}->{}({:+}),potential{}->{}({:+})",
                before.victory_points,
                after.victory_points,
                i32::from(after.victory_points) - i32::from(before.victory_points),
                before.potential_era_vp,
                after.potential_era_vp,
                i32::from(after.potential_era_vp) - i32::from(before.potential_era_vp),
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn scan_financed_sale_macros(
    runner: &GameRunner,
    actor: usize,
    ranked: &[RuleActionScore],
    max_loan_income_loss: i16,
    action_index: usize,
) -> Result<(), String> {
    let before_players = macro_player_snapshots(runner);
    let discard_history_before = runner.discard_history().len();
    let builds = ranked
        .iter()
        .filter(|candidate| {
            candidate.action.intent.action_type == ActionType::BuildBuilding
                && candidate
                    .action
                    .intent
                    .selected_industry
                    .is_some_and(is_saleable)
        })
        .collect::<Vec<_>>();
    println!(
        "MACRO_SCAN action={action_index} actor={actor} builds={} loan_income_loss_max={max_loan_income_loss} counterfactual=no_opponent_intervention reset=framework.current_player:{actor}+actions_remaining_in_turn:2 objective=actor_vp_plus_potential full_sell=all_actor_pending_products route_limit=1",
        builds.len(),
    );

    for build in builds {
        let build_key = build.action.key();
        let mut after_build = runner.clone();
        build.action.apply(&mut after_build)?;
        if after_build.framework.current_player != actor
            || after_build.actions_remaining_in_turn == 0
        {
            println!(
                "MACRO_NO_COMPLETION action={action_index} build_key={build_key} build_score={:.3} reason=build_did_not_leave_same_turn_action",
                build.score,
            );
            continue;
        }

        let income_before_loan = after_build.framework.board.state.players[actor]
            .get_income_amount(after_build.framework.board.state.players[actor].income_level);
        let loans = card_invariant_unique(enumerate_legal_actions(&after_build)?)
            .into_iter()
            .filter(|action| action.intent.action_type == ActionType::Loan)
            .collect::<Vec<_>>();
        let mut low_cost_loans = 0usize;
        let mut route_attempts = 0usize;
        let mut sell_attempts = 0usize;
        let mut full_sells = 0usize;
        let mut best = None;
        let mut partial_after_loan = None::<(String, i16, usize, Vec<MacroPlayerSnapshot>)>;

        for loan in loans {
            let mut after_loan = after_build.clone();
            loan.apply(&mut after_loan)?;
            let income_after_loan = after_loan.framework.board.state.players[actor]
                .get_income_amount(after_loan.framework.board.state.players[actor].income_level);
            let loan_income_loss = i16::from(income_before_loan) - i16::from(income_after_loan);
            if loan_income_loss > max_loan_income_loss {
                continue;
            }
            if after_loan.is_game_finished()
                || after_loan.has_pending_shortfall()
                || !after_loan.turn_started
            {
                continue;
            }
            low_cost_loans += 1;
            partial_after_loan.get_or_insert_with(|| {
                (
                    loan.key(),
                    loan_income_loss,
                    after_loan
                        .discard_history()
                        .len()
                        .saturating_sub(discard_history_before),
                    macro_player_snapshots(&after_loan),
                )
            });

            // Counterfactual only: skip every opponent turn without changing
            // their state, then give the original actor a normal two-action turn.
            after_loan.framework.current_player = actor;
            after_loan.actions_remaining_in_turn = 2;

            let loan_key = loan.key();
            let (direct_attempts, direct_full, direct_best) = best_full_sale_after_route(
                &after_loan,
                actor,
                "none",
                &loan_key,
                loan_income_loss,
                discard_history_before,
            )?;
            sell_attempts += direct_attempts;
            full_sells += direct_full;
            merge_completion(&mut best, direct_best, actor);

            let routes = card_invariant_unique(enumerate_legal_actions(&after_loan)?)
                .into_iter()
                .filter(|action| {
                    matches!(
                        action.intent.action_type,
                        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
                    )
                })
                .collect::<Vec<_>>();
            for route in routes {
                route_attempts += 1;
                let mut after_route = after_loan.clone();
                route.apply(&mut after_route)?;
                if after_route.framework.current_player != actor
                    || after_route.actions_remaining_in_turn == 0
                {
                    continue;
                }
                let route_key = route.key();
                let (attempts, completed, route_best) = best_full_sale_after_route(
                    &after_route,
                    actor,
                    &route_key,
                    &loan_key,
                    loan_income_loss,
                    discard_history_before,
                )?;
                sell_attempts += attempts;
                full_sells += completed;
                merge_completion(&mut best, route_best, actor);
            }
        }

        let Some(best) = best else {
            if let Some((loan_key, loan_income_loss, cards_used, players)) = partial_after_loan {
                let before_actor = before_players[actor];
                let after_actor = players[actor];
                println!(
                    "MACRO_NO_COMPLETION action={action_index} build_key={build_key} build_score={:.3} loan_key={loan_key} loan_income_loss={loan_income_loss} route_key=none sell_key=none cards_used={cards_used} actor_vp={}->{} actor_potential={}->{} actor_progress={}->{} actor_income={}->{} actor_cash={}->{} actor_road_value={:.2}->{:.2} opponents=[{}] searched=low_cost_loans:{low_cost_loans},routes:{route_attempts},sells:{sell_attempts},full_sells:{full_sells}",
                    build.score,
                    before_actor.victory_points,
                    after_actor.victory_points,
                    before_actor.potential_era_vp,
                    after_actor.potential_era_vp,
                    before_actor.progress(),
                    after_actor.progress(),
                    before_actor.income,
                    after_actor.income,
                    before_actor.cash,
                    after_actor.cash,
                    before_actor.road_value,
                    after_actor.road_value,
                    format_opponent_changes(&before_players, &players, actor),
                );
            } else {
                println!(
                    "MACRO_NO_COMPLETION action={action_index} build_key={build_key} build_score={:.3} loan_key=none loan_income_loss=none route_key=none sell_key=none cards_used=1 low_cost_loans={low_cost_loans} route_attempts={route_attempts} sell_attempts={sell_attempts} full_sells={full_sells}",
                    build.score,
                );
            }
            continue;
        };
        let before_actor = before_players[actor];
        let after_actor = best.players[actor];
        println!(
            "MACRO_RESULT action={action_index} build_key={build_key} build_score={:.3} loan_key={} loan_income_loss={} route_key={} sell_key={} cards_used={} actor_vp={}->{} actor_potential={}->{} actor_progress={}->{} actor_income={}->{} actor_cash={}->{} actor_road_value={:.2}->{:.2} opponents=[{}] searched=low_cost_loans:{low_cost_loans},routes:{route_attempts},sells:{sell_attempts},full_sells:{full_sells}",
            build.score,
            best.loan_key,
            best.loan_income_loss,
            best.route_key,
            best.sell_key,
            best.cards_used,
            before_actor.victory_points,
            after_actor.victory_points,
            before_actor.potential_era_vp,
            after_actor.potential_era_vp,
            before_actor.progress(),
            after_actor.progress(),
            before_actor.income,
            after_actor.income,
            before_actor.cash,
            after_actor.cash,
            before_actor.road_value,
            after_actor.road_value,
            format_opponent_changes(&before_players, &best.players, actor),
        );
        println!(
            "MACRO_BEER action={action_index} build_key={build_key} sell_options=[{}] chosen=[{}]",
            best.sell_option_beer, best.chosen_beer,
        );
    }
    Ok(())
}

fn risk_state(runner: &GameRunner, actor: usize) -> RiskState {
    let state = &runner.framework.board.state;
    let mut result = RiskState {
        cash: f64::from(state.players[actor].money),
        income: f64::from(
            state.players[actor].get_income_amount(state.players[actor].income_level),
        ),
        total_cards: state.deck.cards.len()
            + state
                .players
                .iter()
                .map(|player| player.hand.cards.len())
                .sum::<usize>(),
        ..RiskState::default()
    };
    for building in state.bl_to_building.values() {
        if building.owner.as_usize() != actor {
            continue;
        }
        result.buildings += 1;
        if building.flipped {
            continue;
        }
        result.unflipped += 1;
        let data = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        if data.removed_after_phase1 {
            result.unflipped_removed += 1;
        }
        if is_saleable(building.industry) {
            result.unflipped_saleable += 1;
        }
    }
    result.valid_builds = runner.framework.board.get_valid_build_options(actor).len();
    result.valid_saleable_builds = runner
        .framework
        .board
        .get_valid_build_options(actor)
        .into_iter()
        .filter(|option| is_saleable(option.industry_type))
        .count();
    result.valid_sells = runner.framework.board.get_valid_sell_options(actor).len();
    result.valid_networks = if runner.game_phase == fast_brass::game::runner::GamePhase::Canal {
        runner.framework.board.get_valid_canal_options(actor).len()
    } else {
        runner
            .framework
            .board
            .get_valid_single_rail_options(actor)
            .len()
            + runner
                .framework
                .board
                .get_valid_double_rail_first_link_options(actor)
                .len()
    };
    result
}

/// A state-only transition/liquidation risk estimate used by the temporary
/// comparison policy. It is deliberately outside `rule_ai.rs`: this probe
/// must not change the deployed defaults.
fn transition_risk(
    runner: &GameRunner,
    actor: usize,
    reserve_horizon: f64,
    threshold_rounds: f64,
) -> f64 {
    if runner.game_phase != fast_brass::game::runner::GamePhase::Canal {
        return 0.0;
    }
    let state = risk_state(runner, actor);
    // The guard is intentionally narrow: healthy players with a conversion
    // route should retain the normal policy. It activates for the observed
    // failure mode (large negative income, no ready sale, and assets still
    // unflipped), where a transition can trigger a liquidation cascade.
    if state.income > -6.0 || state.valid_sells > 0 || state.unflipped == 0 {
        return 0.0;
    }
    let players = runner.framework.board.state.players.len().max(2) as f64;
    let rounds = (state.total_cards as f64 / players).ceil();
    let pressure = ((threshold_rounds - rounds) / threshold_rounds).clamp(0.0, 1.0);
    if pressure <= 0.0 {
        return 0.0;
    }
    let debt = (-state.income).max(0.0);
    // Keep enough cash for the first few settlements after the transition.
    // A reserve deficit is the main predictor of the observed liquidation
    // cascade; exposure terms make unflipped Canal tiles less attractive when
    // there is no concrete conversion frontier.
    let reserve_target = debt * reserve_horizon + 4.0;
    let reserve_deficit = (reserve_target - state.cash).max(0.0);
    let stranded_saleable = if state.valid_sells == 0 {
        state.unflipped_saleable as f64
    } else {
        0.0
    };
    let no_conversion = state.valid_sells == 0 && state.valid_saleable_builds == 0;
    let mut loss = reserve_deficit * 2.5;
    loss += state.unflipped_removed as f64 * 2.0;
    loss += stranded_saleable * 1.5;
    if no_conversion && debt >= 6.0 {
        loss += state.unflipped as f64 * 1.25;
    }
    -pressure * loss
}

fn main() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    let seed = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20260852);
    let depth = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let branching = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let risk_mode = args.get(4).cloned().unwrap_or_else(|| "none".to_string());
    let risk_weight = args
        .iter()
        .find_map(|value| value.strip_prefix("weight=")?.parse::<f64>().ok())
        .unwrap_or(1.0);
    let reserve_horizon = args
        .iter()
        .find_map(|value| value.strip_prefix("horizon=")?.parse::<f64>().ok())
        .unwrap_or(2.0);
    let threshold_rounds = args
        .iter()
        .find_map(|value| value.strip_prefix("threshold=")?.parse::<f64>().ok())
        .unwrap_or(4.0);
    let quiet = args.iter().any(|value| value == "quiet");
    let show_frontier = args.iter().any(|value| value == "frontier");
    let scan_product_loops = args.iter().any(|value| value == "scan_product_loops");
    let scan_financed_sales = args.iter().any(|value| value == "scan_financed_sales");
    let max_macro_loan_income_loss = named_arg(&args, "macro_loan_loss_max=")
        .map(|value| {
            value
                .parse::<i16>()
                .map_err(|error| format!("invalid macro_loan_loss_max={value}: {error}"))
        })
        .transpose()?
        .unwrap_or(3);
    if max_macro_loan_income_loss < 0 {
        return Err("macro_loan_loss_max must be non-negative".to_string());
    }
    let inspect_action = args
        .iter()
        .find_map(|value| value.strip_prefix("inspect=")?.parse::<usize>().ok());
    let inspect_next_industry = args
        .iter()
        .find_map(|value| value.strip_prefix("next_industry=")?.parse::<usize>().ok());
    if scan_financed_sales && inspect_action.is_none() {
        return Err("scan_financed_sales requires inspect=<global action index>".to_string());
    }
    let forced_action = ForcedActionSpec::from_args(&args)?;
    let mut forced_action_applied = false;
    let mut runner = GameRunner::new(3, Some(seed));
    let mut config = RuleDecisionConfig::default();
    config.lookahead_depth = depth;
    config.lookahead_branching = branching;
    config.financed_product_cycle_weight = env::args()
        .find_map(|value| {
            value
                .strip_prefix("financed_product_cycle=")?
                .parse::<f64>()
                .ok()
        })
        .unwrap_or(config.financed_product_cycle_weight);
    config.low_cost_loan_runway_weight = env::args()
        .find_map(|value| {
            value
                .strip_prefix("low_cost_loan_runway=")?
                .parse::<f64>()
                .ok()
        })
        .unwrap_or(config.low_cost_loan_runway_weight);
    config.beer_backlog_recovery_weight = env::args()
        .find_map(|value| {
            value
                .strip_prefix("beer_backlog_recovery=")?
                .parse::<f64>()
                .ok()
        })
        .unwrap_or(config.beer_backlog_recovery_weight);
    let mut action = 0usize;
    while !runner.is_game_finished() && action < 512 {
        advance_to_decision(&mut runner)?;
        if runner.is_game_finished() {
            break;
        }
        let actor = runner.framework.current_player;
        let player = &runner.framework.board.state.players[actor];
        let (flagged, flagged_vp, flagged_income, builds, sells, develops, networks) =
            player_metrics(&runner, actor);
        let ranked = rank_rule_actions(&runner, &config)?;
        let baseline_best = ranked.first().ok_or("no candidate")?;
        let baseline_risk = transition_risk(&runner, actor, reserve_horizon, threshold_rounds);
        let policy_chosen_index = if risk_mode == "none" {
            0
        } else {
            ranked
                .iter()
                .enumerate()
                .map(|(index, candidate)| {
                    let mut after_runner = runner.clone();
                    let risk_after = if candidate.action.apply(&mut after_runner).is_ok() {
                        transition_risk(&after_runner, actor, reserve_horizon, threshold_rounds)
                    } else {
                        baseline_risk
                    };
                    let adjusted = candidate.score + risk_weight * (risk_after - baseline_risk);
                    (index, adjusted)
                })
                .max_by(|(left_index, left_score), (right_index, right_score)| {
                    left_score.total_cmp(right_score).then_with(|| {
                        ranked[*right_index]
                            .action
                            .key()
                            .cmp(&ranked[*left_index].action.key())
                    })
                })
                .map(|(index, _)| index)
                .unwrap_or(0)
        };
        let chosen_index = if let Some(spec) = forced_action
            .as_ref()
            .filter(|spec| spec.action_index == action)
        {
            let forced_index = spec.select(&ranked)?;
            let policy_best = &ranked[policy_chosen_index];
            let forced = &ranked[forced_index];
            println!(
                "FORCED_ACTION action={action} p={actor} baseline_type={:?} baseline_key={} baseline_score={:.3} forced_type={:?} forced_key={} forced_score={:.3}",
                policy_best.action.intent.action_type,
                policy_best.action.key(),
                policy_best.score,
                forced.action.intent.action_type,
                forced.action.key(),
                forced.score,
            );
            forced_action_applied = true;
            forced_index
        } else {
            policy_chosen_index
        };
        let best = &ranked[chosen_index];
        if scan_financed_sales && inspect_action == Some(action) {
            scan_financed_sale_macros(&runner, actor, &ranked, max_macro_loan_income_loss, action)?;
        }
        if scan_product_loops {
            let mut best_loop = None::<(f64, String, String, u16, i8)>;
            for candidate in ranked.iter().filter(|candidate| {
                candidate.action.intent.action_type == ActionType::BuildBuilding
                    && matches!(
                        candidate.action.intent.selected_industry,
                        Some(IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery)
                    )
            }) {
                let mut after_build = runner.clone();
                candidate.action.apply(&mut after_build)?;
                if after_build.framework.current_player != actor
                    || after_build.actions_remaining_in_turn == 0
                {
                    continue;
                }
                let after_player = &after_build.framework.board.state.players[actor];
                let after_income = after_player.get_income_amount(after_player.income_level);
                let mut loop_reply = None;
                if !after_build
                    .framework
                    .board
                    .get_valid_sell_options(actor)
                    .is_empty()
                {
                    loop_reply = Some("sell_ready".to_string());
                } else {
                    for reply in
                        enumerate_legal_actions(&after_build)?
                            .into_iter()
                            .filter(|reply| {
                                matches!(
                                    reply.intent.action_type,
                                    ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
                                )
                            })
                    {
                        let mut after_road = after_build.clone();
                        reply.apply(&mut after_road)?;
                        if !after_road
                            .framework
                            .board
                            .get_valid_sell_options(actor)
                            .is_empty()
                        {
                            loop_reply = Some(reply.key());
                            break;
                        }
                    }
                }
                let Some(loop_reply) = loop_reply else {
                    continue;
                };
                let replace = best_loop
                    .as_ref()
                    .is_none_or(|current| candidate.score > current.0);
                if replace {
                    best_loop = Some((
                        candidate.score,
                        candidate.action.key(),
                        loop_reply,
                        after_player.money,
                        after_income,
                    ));
                }
            }
            if let Some((loop_score, build_key, reply_key, after_cash, after_income)) = best_loop {
                println!(
                    "PRODUCT_LOOP action={action} p={actor} chosen={:?} chosen_score={:.3} build_key={build_key} build_score={loop_score:.3} reply_key={reply_key} after_cash={after_cash} after_income={after_income}",
                    best.action.intent.action_type,
                    best.score,
                );
            }
        }
        let mut roots = BTreeMap::<String, f64>::new();
        for candidate in &ranked {
            roots
                .entry(format!("{:?}", candidate.action.intent.action_type))
                .and_modify(|v| *v = v.max(candidate.score))
                .or_insert(candidate.score);
        }
        let total_cards = runner.framework.board.state.deck.cards.len()
            + runner
                .framework
                .board
                .state
                .players
                .iter()
                .map(|p| p.hand.cards.len())
                .sum::<usize>();
        if !quiet && (runner.framework.board.state.deck.cards.len() <= 16 || flagged > 0) {
            println!("probe action={action} p={actor} phase={phase:?} round={round} deck={deck} cards={total_cards} hand={hand} cash={cash} income={income} buildings={buildings} flagged={flagged} flagged_vp={flagged_vp:.0} flagged_income={flagged_income:.0} frontier=build:{builds},sell:{sells},develop:{develops},network:{networks} chosen={:?} score={:.2} roots={roots:?}", best.action.intent.action_type, best.score, phase=runner.game_phase, round=runner.round_in_phase, deck=runner.framework.board.state.deck.cards.len(), hand=player.hand.cards.len(), cash=player.money, income=player.get_income_amount(player.income_level), buildings=state_buildings(&runner, actor));
        }
        if show_frontier
            && matches!(
                best.action.intent.action_type,
                ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
            )
        {
            let before_frontier = risk_state(&runner, actor);
            let mut after_runner = runner.clone();
            if best.action.apply(&mut after_runner).is_ok() {
                let after_frontier = risk_state(&after_runner, actor);
                println!(
                    "FRONTIER action={action} p={actor} key={} before=build:{},saleable:{},sell:{},network:{},unflipped_removed:{},cash:{:.0},income:{:.0} after=build:{},saleable:{},sell:{},network:{},unflipped_removed:{},cash:{:.0},income:{:.0}",
                    best.action.key(),
                    before_frontier.valid_builds,
                    before_frontier.valid_saleable_builds,
                    before_frontier.valid_sells,
                    before_frontier.valid_networks,
                    before_frontier.unflipped_removed,
                    before_frontier.cash,
                    before_frontier.income,
                    after_frontier.valid_builds,
                    after_frontier.valid_saleable_builds,
                    after_frontier.valid_sells,
                    after_frontier.valid_networks,
                    after_frontier.unflipped_removed,
                    after_frontier.cash,
                    after_frontier.income,
                );
            }
        }
        if show_frontier && inspect_action == Some(action) {
            let before_frontier = risk_state(&runner, actor);
            let mut printed_next = false;
            println!(
                "INSPECT action={action} actor={actor} before=build:{},saleable:{},sell:{},network:{},cash:{:.0},income:{:.0}",
                before_frontier.valid_builds,
                before_frontier.valid_saleable_builds,
                before_frontier.valid_sells,
                before_frontier.valid_networks,
                before_frontier.cash,
                before_frontier.income,
            );
            for candidate in ranked.iter() {
                let mut after_runner = runner.clone();
                if let Err(error) = candidate.action.apply(&mut after_runner) {
                    println!(
                        "CAND_ERROR action={action} type={:?} key={} error={error}",
                        candidate.action.intent.action_type,
                        candidate.action.key(),
                    );
                    continue;
                }
                let after_frontier = risk_state(&after_runner, actor);
                println!(
                    "CAND action={action} type={:?} key={} score={:.3} parts={:?} after=build:{},saleable:{},sell:{},network:{},cash:{:.0},income:{:.0}",
                    candidate.action.intent.action_type,
                    candidate.action.key(),
                    candidate.score,
                    candidate.breakdown,
                    after_frontier.valid_builds,
                    after_frontier.valid_saleable_builds,
                    after_frontier.valid_sells,
                    after_frontier.valid_networks,
                    after_frontier.cash,
                    after_frontier.income,
                );
                if !printed_next
                    && inspect_next_industry.is_some_and(|industry| {
                        candidate.action.intent.action_type == ActionType::BuildBuilding
                            && candidate
                                .action
                                .intent
                                .selected_industry
                                .is_some_and(|selected| selected.as_usize() == industry)
                    })
                    && after_runner.framework.current_player == actor
                    && after_runner.actions_remaining_in_turn > 0
                {
                    printed_next = true;
                    let replies = rank_rule_actions(&after_runner, &config)?;
                    if let Some(best_reply) = replies.first() {
                        let mut after_reply = after_runner.clone();
                        best_reply.action.apply(&mut after_reply)?;
                        let reply_player = &after_reply.framework.board.state.players[actor];
                        let reply_income =
                            reply_player.get_income_amount(reply_player.income_level);
                        println!(
                            "NEXT_BEST action={action} after_key={} type={:?} key={} score={:.3} after_cash={} after_income={} after_vp={} after_phase={:?} after_slots={}",
                            candidate.action.key(),
                            best_reply.action.intent.action_type,
                            best_reply.action.key(),
                            best_reply.score,
                            reply_player.money,
                            reply_income,
                            reply_player.victory_points,
                            after_reply.game_phase,
                            after_reply.actions_remaining_in_turn,
                        );
                    }
                    let mut best_road = None::<(usize, f64, String)>;
                    for reply in replies.iter().filter(|reply| {
                        matches!(
                            reply.action.intent.action_type,
                            ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
                        )
                    }) {
                        let mut reply_runner = after_runner.clone();
                        reply.action.apply(&mut reply_runner)?;
                        let sell_count = risk_state(&reply_runner, actor).valid_sells;
                        let replace = best_road.as_ref().is_none_or(|current| {
                            sell_count > current.0
                                || (sell_count == current.0 && reply.score > current.1)
                        });
                        if replace {
                            best_road = Some((sell_count, reply.score, reply.action.key()));
                        }
                    }
                    if let Some((sell_count, score, key)) = best_road {
                        println!(
                            "NEXT_ROAD action={action} after_key={} key={key} score={score:.3} after_sell={sell_count}",
                            candidate.action.key(),
                        );
                    }
                }
            }
        }
        if !quiet
            && runner.game_phase == fast_brass::game::runner::GamePhase::Canal
            && runner.framework.board.state.deck.cards.is_empty()
            && total_cards <= 2
        {
            for owner in 0..runner.framework.board.state.players.len() {
                let mut assets = runner
                    .framework
                    .board
                    .state
                    .bl_to_building
                    .values()
                    .filter(|b| b.owner.as_usize() == owner)
                    .map(|b| {
                        let d = &INDUSTRY_MAT[b.industry.as_usize()][b.level.as_usize()];
                        format!(
                            "loc{}:{:?}L{} flip={} rm={} vp={} inc={} road={}",
                            b.loc,
                            b.industry,
                            b.level.as_u8(),
                            b.flipped,
                            d.removed_after_phase1,
                            d.vp_on_flip,
                            d.income,
                            d.road_vp
                        )
                    })
                    .collect::<Vec<_>>();
                assets.sort();
                println!("PRE_TRANSITION owner={owner}: {assets:?}");
            }
        }
        if !quiet && risk_mode != "none" && chosen_index != 0 {
            let mut after_runner = runner.clone();
            let risk_after = if ranked[chosen_index].action.apply(&mut after_runner).is_ok() {
                transition_risk(&after_runner, actor, reserve_horizon, threshold_rounds)
            } else {
                baseline_risk
            };
            println!("RISK_REORDER action={action} p={actor} baseline={:?}:{:.2} chosen={:?}:{:.2} delta={:.2} risk_before={:.2} risk_after={:.2}", baseline_best.action.intent.action_type, baseline_best.score, best.action.intent.action_type, best.score, best.score + risk_weight * (risk_after - baseline_risk) - baseline_best.score, baseline_risk, risk_after);
        }
        let phase_before = runner.game_phase;
        best.action.apply(&mut runner)?;
        if runner.game_phase != phase_before {
            println!(
                "TRANSITION after_action={action} from={phase_before:?} to={:?}",
                runner.game_phase
            );
            for owner in 0..runner.framework.board.state.players.len() {
                let mut assets = runner
                    .framework
                    .board
                    .state
                    .bl_to_building
                    .values()
                    .filter(|b| b.owner.as_usize() == owner)
                    .map(|b| {
                        let d = &INDUSTRY_MAT[b.industry.as_usize()][b.level.as_usize()];
                        format!(
                            "loc{}:{:?}L{} flip={} rm={} vp={} inc={} road={}",
                            b.loc,
                            b.industry,
                            b.level.as_u8(),
                            b.flipped,
                            d.removed_after_phase1,
                            d.vp_on_flip,
                            d.income,
                            d.road_vp
                        )
                    })
                    .collect::<Vec<_>>();
                assets.sort();
                println!("  assets owner={owner}: {assets:?}");
            }
        }
        action += 1;
    }
    if let Some(spec) = forced_action.as_ref().filter(|_| !forced_action_applied) {
        return Err(format!(
            "forced global action {} was not reached before the game ended",
            spec.action_index
        ));
    }
    resolve_shortfalls(&mut runner);
    for actor in 0..runner.framework.board.state.players.len() {
        let p = &runner.framework.board.state.players[actor];
        println!(
            "final p={actor} vp={} cash={} income={} buildings={}",
            p.victory_points,
            p.money,
            p.get_income_amount(p.income_level),
            state_buildings(&runner, actor)
        );
    }
    Ok(())
}

fn state_buildings(runner: &GameRunner, actor: usize) -> usize {
    runner
        .framework
        .board
        .state
        .bl_to_building
        .values()
        .filter(|b| b.owner.as_usize() == actor)
        .count()
}

#[allow(dead_code)]
fn _industry_name(industry: IndustryType) -> &'static str {
    match industry {
        IndustryType::Coal => "coal",
        IndustryType::Iron => "iron",
        IndustryType::Beer => "beer",
        IndustryType::Cotton => "cotton",
        IndustryType::Goods => "goods",
        IndustryType::Pottery => "pottery",
    }
}

#[allow(dead_code)]
fn _action_name(action: ActionType) -> &'static str {
    match action {
        ActionType::BuildBuilding => "build",
        ActionType::BuildRailroad => "rail",
        ActionType::BuildDoubleRailroad => "double",
        ActionType::Develop => "develop",
        ActionType::DevelopDouble => "develop_double",
        ActionType::Sell => "sell",
        ActionType::Loan => "loan",
        ActionType::Scout => "scout",
        ActionType::Pass => "pass",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_named_force_selectors() {
        let args = [
            "transition_probe",
            "20260830",
            "1",
            "1",
            "none",
            "force=83",
            "force_root=build",
            "force_industry=Cotton",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let spec = ForcedActionSpec::from_args(&args)
            .expect("valid force specification")
            .expect("force specification should be present");
        assert_eq!(spec.action_index, 83);
        assert_eq!(spec.root, Some(ActionType::BuildBuilding));
        assert_eq!(spec.industry, Some(IndustryType::Cotton));
    }

    #[test]
    fn rejects_force_selector_without_action_index() {
        let args = ["transition_probe", "force_root=loan"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();

        assert!(ForcedActionSpec::from_args(&args).is_err());
    }
}
