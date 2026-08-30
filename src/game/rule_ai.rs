//! CPU-only, explainable action selection.
//!
//! This module is intentionally a shallow, CPU-only decision tree rather than
//! a rollout search. It enumerates the already validated legal actions, applies
//! each action to one cloned position, scores the resulting state, and expands
//! only a small same-turn frontier. It never calls a model, samples hidden
//! information, or rolls a game to its terminal state.

use std::collections::{HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::consts::{MAX_MARKET_COAL, MAX_MARKET_IRON, N_BL, ONE_RAILROAD_PRICE, TOTAL_TOWNS};
use crate::core::locations::LocationName;
use crate::core::static_data::{
    BUILD_LOCATION_MASK, INDUSTRY_MAT, LINK_LOCATIONS, LOCATION_TO_ROADS,
};
use crate::core::types::{ActionType, BitSetWrapper, CardType, IndustryType};
use crate::game::framework::ActionIntent;
use crate::game::legal_actions::{
    enumerate_legal_actions, enumerate_legal_actions_for_root, LegalAction,
};
use crate::game::runner::{GamePhase, GameRunner};
use crate::game::search::{
    immediate_effect_between, ImmediateEffect, RootActionEstimate, RootSearchReport,
    RULE_DECISION_TREE_METHOD, RULE_IMMEDIATE_SCORE_SOURCE,
};

/// Tunable weights for the shallow rule tree.  Values are deliberately kept
/// in one struct so a later league/teacher sweep can change policy behavior
/// without changing the evaluator or touching the neural trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RuleDecisionConfig {
    pub recommendation_count: usize,
    pub temperature: f64,
    pub immediate_vp_weight: f64,
    pub potential_vp_weight: f64,
    pub income_weight: f64,
    pub cash_weight: f64,
    pub industry_weight: f64,
    pub network_weight: f64,
    pub resource_weight: f64,
    pub safety_weight: f64,
    pub tempo_weight: f64,
    pub card_weight: f64,
    /// Scale for the discrete action-family bias (sell/build/network/develop).
    /// Keeping this separate makes the hard ordering rules sweepable without
    /// changing the continuous state features.
    pub action_bias_weight: f64,
    /// Scale for cash/income safety and debt-spiral rejection rules.
    pub economic_guard_weight: f64,
    /// Scale for opening and development-frontier priorities.
    pub strategic_priority_weight: f64,
    /// Scale for conversion-first lifecycle gates.
    pub lifecycle_priority_weight: f64,
    /// Reward for a building action whose exact successor exposes a legal
    /// Sell. Kept independent because broad lifecycle scaling also changes
    /// roads, development, recovery, and ready-sale ordering.
    pub sale_frontier_build_weight: f64,
    /// Scale for the established opportunity cost of borrowing against an
    /// already mature positive-income engine.
    pub loan_maturity_weight: f64,
    /// Scale for the extra displayed-income loss above the normal three-point
    /// loan cost. This prices the actual successor rather than an income label.
    pub loan_excess_income_loss_weight: f64,
    /// Reward for a narrowly proven late Railroad loan that loses at most three
    /// displayed income and immediately funds a productive runway action.
    pub low_cost_loan_runway_weight: f64,
    /// Reward for a severe Railroad recovery product build only when one
    /// funded remaining network action provably exposes a legal Sell.
    pub recovery_product_plan_weight: f64,
    /// Scale for pruning late product/loan branches whose remaining cards
    /// cannot complete a legal Build -> network (if needed) -> Sell plan.
    pub conversion_feasibility_weight: f64,
    /// Scale for redirecting a stalled Canal product plan away from another
    /// unconverted Coal/Iron stockpile and back toward its product frontier.
    pub resource_stockpile_weight: f64,
    /// Scale for starting the next provably reachable product cycle after a
    /// very low-score player has converted its only prepared product.
    pub post_sale_product_cycle_weight: f64,
    /// Additional late product-cycle reward when a mature income engine and
    /// a deep remaining hand can support another full conversion.
    pub mature_product_cycle_weight: f64,
    /// Discount applied to a strictly proven Build -> cheap Loan -> next-turn
    /// Route -> Sell macro backed by actor-owned Beer. The leaf Sell score
    /// distinguishes concrete locations instead of rewarding every product.
    pub financed_product_cycle_weight: f64,
    /// Scale for removing premature settlement penalties from a product Build
    /// when its exact same-turn successor is a safe legal Sell.
    pub same_turn_sale_guard_weight: f64,
    /// Penalty for a late product Build whose only route-to-Sell continuation
    /// crosses a turn boundary and depends on Beer contested by an opponent.
    pub contested_external_beer_plan_weight: f64,
    /// Extra penalty for a last-chance Canal action that spends the final
    /// cards without securing VP, income, a flip, or a legal Sell.
    pub late_canal_stall_weight: f64,
    /// Penalty for adding a third unsold product in the late Railroad era
    /// without a legal Sell, economic gain, or secured Beer for the batch.
    pub late_railroad_backlog_weight: f64,
    /// Reward for developing Beer only when several existing products have a
    /// large Beer shortfall and the development unlocks an affordable Brewery.
    pub beer_backlog_recovery_weight: f64,
    /// Penalty for spending one of the final cards on development or another
    /// product without advancing an already stranded product toward a sale.
    pub late_railroad_conversion_stall_weight: f64,
    /// Maximum shallow tree depth. `1` is the immediate evaluator; `2` adds
    /// one same-turn continuation when the first action leaves another slot.
    pub lookahead_depth: usize,
    /// Number of statically strongest root actions to expand at depth two.
    pub lookahead_branching: usize,
    /// Discount applied to the best continuation's incremental score.
    pub lookahead_discount: f64,
}

impl Default for RuleDecisionConfig {
    fn default() -> Self {
        Self {
            recommendation_count: 3,
            // A small temperature preserves a clear best move while keeping
            // alternatives useful for inspection and future data generation.
            temperature: 2.5,
            immediate_vp_weight: 5.0,
            potential_vp_weight: 1.35,
            income_weight: 1.0,
            cash_weight: 0.22,
            industry_weight: 1.25,
            network_weight: 1.15,
            resource_weight: 0.35,
            safety_weight: 1.0,
            tempo_weight: 1.0,
            card_weight: 1.0,
            action_bias_weight: 1.0,
            economic_guard_weight: 1.0,
            strategic_priority_weight: 1.0,
            lifecycle_priority_weight: 1.0,
            // The global sale-frontier reward regressed the human reference
            // and is therefore opt-in until a fixed-seed screen promotes it.
            sale_frontier_build_weight: 0.0,
            // Split the loan opportunity cost between engine maturity and the
            // exact displayed-income loss. This promoted mix preserves useful
            // low-cost borrowing while reducing unusually expensive loans.
            loan_maturity_weight: 0.5,
            loan_excess_income_loss_weight: 0.5,
            // The smallest effective paired-screen value changes only the
            // known 55-VP tail while preserving the human reference. The gate
            // itself excludes mature-income and repeat-loan states.
            low_cost_loan_runway_weight: 14.8,
            recovery_product_plan_weight: 12.0,
            // The broad deck-empty conversion gate improved one last action
            // but introduced a lower 42-VP tail elsewhere, so keep it opt-in.
            conversion_feasibility_weight: 0.0,
            // Promoted by the fixed bad-seed/human-reference gate and a
            // paired 40-game screen: it raises the global floor without
            // suppressing the valid resource-first engine.
            resource_stockpile_weight: 1.0,
            // Promoted after the low-tail and human-reference screens showed
            // that the smallest effective value improves two paired games.
            post_sale_product_cycle_weight: 0.75,
            // The extra reward is gated by both mature income and enough
            // remaining cards; the paired screen changed only the target tail.
            mature_product_cycle_weight: 1.625,
            // Opt-in until the strict four-card macro passes the fixed-seed
            // and paired benchmark gates.
            financed_product_cycle_weight: 0.0,
            // Experimental until the exact Build -> Sell macro passes the
            // fixed-seed and paired gates independently.
            same_turn_sale_guard_weight: 0.0,
            // Avoid a cross-turn sale plan when an opponent can consume the
            // same external Beer before this player acts again.
            contested_external_beer_plan_weight: 3.5,
            // Redirect the final Canal cards toward measurable conversion;
            // the narrow gate avoids healthy-state changes.
            late_canal_stall_weight: 10.5,
            // The smallest effective paired-screen value redirects only an
            // unfunded late backlog; Beer-backed product batches are exempt.
            late_railroad_backlog_weight: 1.01,
            // The smallest effective paired-screen value fixes the observed
            // high-Beer backlog and changes only two beneficial benchmark
            // games. Zero-Beer and already funded batches remain untouched.
            beer_backlog_recovery_weight: 2.18,
            // The smallest screened value removes final-card non-conversion
            // while preserving Beer-funded multi-product batches.
            late_railroad_conversion_stall_weight: 0.9,
            // The interactive policy evaluates one post-action state.  This
            // is the measured fast path for a live game; deeper inspection is
            // available through `deep()` and the benchmark example.
            lookahead_depth: 1,
            lookahead_branching: 1,
            lookahead_discount: 0.65,
        }
    }
}

impl RuleDecisionConfig {
    /// A slower diagnostic profile that expands a small same-turn frontier.
    /// Keep it explicit so opening a game never silently starts a mini-search.
    pub fn deep() -> Self {
        Self {
            lookahead_depth: 2,
            lookahead_branching: 3,
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.recommendation_count == 0 {
            return Err("rule recommendation_count must be at least one".to_string());
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err("rule temperature must be finite and positive".to_string());
        }
        let weights = [
            self.immediate_vp_weight,
            self.potential_vp_weight,
            self.income_weight,
            self.cash_weight,
            self.industry_weight,
            self.network_weight,
            self.resource_weight,
            self.safety_weight,
            self.tempo_weight,
            self.card_weight,
            self.action_bias_weight,
            self.economic_guard_weight,
            self.strategic_priority_weight,
            self.lifecycle_priority_weight,
            self.sale_frontier_build_weight,
            self.loan_maturity_weight,
            self.loan_excess_income_loss_weight,
            self.low_cost_loan_runway_weight,
            self.recovery_product_plan_weight,
            self.conversion_feasibility_weight,
            self.resource_stockpile_weight,
            self.post_sale_product_cycle_weight,
            self.mature_product_cycle_weight,
            self.financed_product_cycle_weight,
            self.same_turn_sale_guard_weight,
            self.contested_external_beer_plan_weight,
            self.late_canal_stall_weight,
            self.late_railroad_backlog_weight,
            self.beer_backlog_recovery_weight,
            self.late_railroad_conversion_stall_weight,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err("rule weights must be finite and non-negative".to_string());
        }
        if self.lookahead_depth > 2 {
            return Err("rule lookahead_depth must be 0, 1, or 2".to_string());
        }
        if self.lookahead_depth > 1 && self.lookahead_branching == 0 {
            return Err("rule lookahead_branching must be positive when depth is 2".to_string());
        }
        if self.lookahead_branching > 64 {
            return Err("rule lookahead_branching must be at most 64".to_string());
        }
        if !self.lookahead_discount.is_finite() || !(0.0..=1.0).contains(&self.lookahead_discount) {
            return Err("rule lookahead_discount must be finite and in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Weighted components of one candidate's score.  These are exposed in the
/// analysis response so a human can see why a rule move ranked above another.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RuleScoreBreakdown {
    pub immediate_vp: f64,
    pub potential_vp: f64,
    pub income: f64,
    pub cash: f64,
    pub industry: f64,
    pub network: f64,
    pub resources: f64,
    pub safety: f64,
    pub tempo: f64,
    pub action_bias: f64,
    pub competitive_pressure: f64,
    /// Penalty/rebate for discarding a card with a valuable future use.
    pub card_value: f64,
    /// Discounted value of the best legal continuation in the same turn.
    pub lookahead: f64,
    pub total: f64,
}

/// A scored legal action, retaining the exact replayable action and its
/// post-action immediate effect.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleActionScore {
    pub action: LegalAction,
    pub score: f64,
    pub breakdown: RuleScoreBreakdown,
    pub immediate_effect: ImmediateEffect,
    pub after_actor_victory_points: f64,
    pub after_actor_victory_point_margin: f64,
}

#[derive(Debug, Clone)]
struct PositionSnapshot {
    victory_points: f64,
    income_level: u8,
    income_amount: i8,
    money: u16,
    buildings: usize,
    flipped_buildings: usize,
    own_roads: usize,
    network_locations: usize,
    connected_trade_posts: usize,
    industry_levels: [u8; 6],
    industry_remaining: [u8; 6],
    industry_quality: f64,
    resource_units: [u32; 3],
    sellable_units: usize,
    sellable_vp: f64,
    sellable_beer_demand: u32,
    unflipped_beer_units: u32,
    road_value: f64,
    /// Sum of the minimum number of new era-legal links needed to reach a
    /// matching merchant for each unflipped sellable building.
    sale_route_distance: f64,
    /// Per-build-location route contribution. Keeping this small fixed array
    /// lets a sale update the route metric without running another search.
    sale_route_by_location: [f64; N_BL],
    available_build_sites: usize,
    available_sell_targets: usize,
    market_coal: u8,
    market_iron: u8,
}

#[cfg(test)]
impl Default for PositionSnapshot {
    fn default() -> Self {
        Self {
            victory_points: 0.0,
            income_level: 0,
            income_amount: 0,
            money: 0,
            buildings: 0,
            flipped_buildings: 0,
            own_roads: 0,
            network_locations: 0,
            connected_trade_posts: 0,
            industry_levels: [0; 6],
            industry_remaining: [0; 6],
            industry_quality: 0.0,
            resource_units: [0; 3],
            sellable_units: 0,
            sellable_vp: 0.0,
            sellable_beer_demand: 0,
            unflipped_beer_units: 0,
            road_value: 0.0,
            sale_route_distance: 0.0,
            sale_route_by_location: [0.0; N_BL],
            available_build_sites: 0,
            available_sell_targets: 0,
            market_coal: 0,
            market_iron: 0,
        }
    }
}

/// Rank every complete legal action with the shallow rule tree.
pub fn rank_rule_actions(
    runner: &GameRunner,
    config: &RuleDecisionConfig,
) -> Result<Vec<RuleActionScore>, String> {
    config.validate()?;
    if runner.is_game_finished() {
        return Err("cannot select a rule action in a finished game".to_string());
    }
    let actor = runner.framework.current_player;
    if actor >= runner.framework.board.state.players.len() {
        return Err(format!("current player {actor} is out of bounds"));
    }
    let actions = enumerate_legal_actions(runner)?;
    if actions.is_empty() {
        return Err("no legal actions available for rule policy".to_string());
    }
    let before = snapshot(runner, actor);
    let card_values = card_future_values(runner, actor);
    let alternatives = actions.len();
    let mut evaluated = Vec::with_capacity(alternatives);

    for action in actions {
        evaluated.push(evaluate_rule_action(
            runner,
            &before,
            alternatives,
            action,
            config,
            &card_values,
        )?);
    }

    // Rank by the immediate evaluator before expanding only a small frontier.
    // This keeps the interactive policy CPU-friendly while allowing a setup
    // action to be recognized when it creates a strong second action.
    sort_evaluated_actions(&mut evaluated);
    if config.lookahead_depth >= 2 {
        // Card-equivalent variants are one strategic branch. Without this
        // grouping a branching budget of three is commonly spent evaluating
        // the same Loan or Build intent three times with different discards.
        let mut seen_intents = HashSet::new();
        let frontier = evaluated
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                seen_intents
                    .insert(candidate.action.card_invariant_key())
                    .then_some(index)
            })
            .take(config.lookahead_branching)
            .collect::<Vec<_>>();
        for index in frontier {
            let continuation =
                best_same_turn_continuation(&evaluated[index].after_runner, actor, config)?;
            let candidate = &mut evaluated[index];
            candidate.breakdown.lookahead = continuation;
            candidate.breakdown.total += continuation;
            candidate.score = candidate.breakdown.total;
        }
    }

    sort_evaluated_actions(&mut evaluated);
    Ok(evaluated
        .into_iter()
        .map(|candidate| RuleActionScore {
            action: candidate.action,
            score: candidate.score,
            breakdown: candidate.breakdown,
            immediate_effect: candidate.effect,
            after_actor_victory_points: candidate.after.victory_points,
            after_actor_victory_point_margin: vp_margin(&candidate.after_runner, actor),
        })
        .collect())
}

/// Internal successor retained only while ranking. Keeping the cloned runner
/// here avoids replaying the root action when the shallow continuation is
/// expanded.
struct EvaluatedRuleAction {
    action: LegalAction,
    after_runner: GameRunner,
    after: PositionSnapshot,
    effect: ImmediateEffect,
    breakdown: RuleScoreBreakdown,
    score: f64,
}

fn evaluate_rule_action(
    runner: &GameRunner,
    before: &PositionSnapshot,
    alternatives: usize,
    action: LegalAction,
    config: &RuleDecisionConfig,
    card_values: &[f64],
) -> Result<EvaluatedRuleAction, String> {
    // No random cards, opponent moves, or terminal rollouts are evaluated.
    let actor = runner.framework.current_player;
    let mut after_runner = runner.clone();
    action.apply(&mut after_runner).map_err(|error| {
        format!(
            "rule policy could not apply legal action {}: {error}",
            action.key()
        )
    })?;
    let after_route = route_data_after_action(runner, &after_runner, actor, before, &action.intent);
    let after = snapshot_with_route_data(&after_runner, actor, after_route);
    let effect = immediate_effect_between(runner, &after_runner);
    let (breakdown, score) = score_action_with_card_values(
        runner,
        &after_runner,
        before,
        &after,
        &effect,
        &action.intent,
        alternatives,
        config,
        card_values,
    );
    Ok(EvaluatedRuleAction {
        action,
        after_runner,
        after,
        effect,
        breakdown,
        score,
    })
}

fn sort_evaluated_actions(actions: &mut [EvaluatedRuleAction]) {
    // Stable deterministic ordering is important for replay and for creating
    // reproducible teacher data. The action key is the final tie-breaker.
    actions.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.action.key().cmp(&right.action.key()))
    });
}

fn best_same_turn_continuation(
    after_runner: &GameRunner,
    actor: usize,
    config: &RuleDecisionConfig,
) -> Result<f64, String> {
    // The first personal turn has one action. A continuation exists only when
    // the exact successor is still the same player with an action slot left.
    if after_runner.is_game_finished()
        || after_runner.framework.current_player != actor
        || after_runner.actions_remaining_in_turn == 0
        || after_runner.framework.action_context.is_some()
        || after_runner.has_pending_shortfall()
    {
        return Ok(0.0);
    }
    let replies = enumerate_legal_actions(after_runner)?;
    if replies.is_empty() {
        return Ok(0.0);
    }
    let reply_count = replies.len();
    let before = snapshot(after_runner, actor);
    let card_values = card_future_values(after_runner, actor);
    let mut best = f64::NEG_INFINITY;
    for reply in replies {
        let mut reply_runner = after_runner.clone();
        reply.apply(&mut reply_runner).map_err(|error| {
            format!(
                "rule policy could not apply continuation {}: {error}",
                reply.key()
            )
        })?;
        let after_route =
            route_data_after_action(after_runner, &reply_runner, actor, &before, &reply.intent);
        let after = snapshot_with_route_data(&reply_runner, actor, after_route);
        let effect = immediate_effect_between(after_runner, &reply_runner);
        let (_, score) = score_action_with_card_values(
            after_runner,
            &reply_runner,
            &before,
            &after,
            &effect,
            &reply.intent,
            reply_count,
            config,
            &card_values,
        );
        best = best.max(score);
    }
    if best.is_finite() {
        // A continuation is a tie-breaker for a setup move, not a hidden
        // second full turn. Capping it prevents a high-VP sale one ply later
        // from overwhelming the root action's liquidity and safety checks.
        Ok(best.clamp(-12.0, 18.0) * config.lookahead_discount)
    } else {
        Ok(0.0)
    }
}

/// Return the best legal action according to the rule tree.
pub fn choose_rule_action(
    runner: &GameRunner,
    config: &RuleDecisionConfig,
) -> Result<LegalAction, String> {
    rank_rule_actions(runner, config)?
        .into_iter()
        .next()
        .map(|candidate| candidate.action)
        .ok_or_else(|| "rule policy returned no candidate".to_string())
}

/// Adapt the rule ranking to the existing analysis/apply pipeline.  The
/// `estimated_shared_win_rate` field is a normalized *relative preference*,
/// not a calibrated win probability; the explicit rule score and breakdown
/// are carried alongside it and the UI labels this mode accordingly.
pub fn rule_decision_report(
    runner: &GameRunner,
    config: &RuleDecisionConfig,
) -> Result<RootSearchReport, String> {
    let ranked = rank_rule_actions(runner, config)?;
    let root_player = runner.framework.current_player;
    let probabilities = softmax(&ranked, config.temperature);
    let total_candidates = ranked.len() as u64;
    let recommendations = ranked
        .iter()
        .enumerate()
        .take(config.recommendation_count)
        .map(|(index, candidate)| RootActionEstimate {
            rank: index + 1,
            action_key: candidate.action.key(),
            action: candidate.action.clone(),
            visits: 1,
            visit_share: 1.0 / total_candidates.max(1) as f64,
            value_source: RULE_IMMEDIATE_SCORE_SOURCE.to_string(),
            value_sample_count: 1,
            estimated_shared_win_rate: probabilities[index],
            estimated_outright_win_rate: None,
            estimated_tied_first_rate: None,
            average_final_victory_points: Some(candidate.after_actor_victory_points),
            average_victory_point_margin: candidate.after_actor_victory_point_margin,
            shared_win_rate_standard_error: None,
            policy_probability: Some(probabilities[index]),
            calibrated_win_rate: None,
            rule_score: Some(candidate.score),
            rule_score_breakdown: Some(candidate.breakdown),
            immediate_effect: candidate.immediate_effect.clone(),
            sample_random_continuation: crate::game::search::SampleRandomContinuation {
                steps: Vec::new(),
                final_victory_points: Vec::new(),
                official_winners: Vec::new(),
            },
        })
        .collect::<Vec<_>>();

    Ok(RootSearchReport {
        method: RULE_DECISION_TREE_METHOD.to_string(),
        value_source: RULE_IMMEDIATE_SCORE_SOURCE.to_string(),
        model_id: None,
        root_model_shared_win_rate: None,
        root_model_victory_point_margin: None,
        root_policy_probabilities: Some(probabilities),
        root_player,
        // These counters represent one deterministic evaluation per legal
        // action, not simulations or terminal games.
        requested_simulations: total_candidates,
        completed_simulations: total_candidates,
        root_action_count: total_candidates as usize,
        evaluated_action_count: total_candidates as usize,
        visited_action_count: total_candidates as usize,
        all_root_actions_evaluated: true,
        max_search_depth: Some(if config.lookahead_depth >= 2 { 2 } else { 1 }),
        neural_leaf_evaluations: None,
        inference_batches: None,
        root_action_model_shared_win_rates: None,
        root_action_model_victory_point_margins: None,
        root_action_model_actor_victory_points: None,
        root_action_model_shared_win_standard_errors: None,
        root_action_model_sample_counts: None,
        recommendations,
    })
}

fn softmax(candidates: &[RuleActionScore], temperature: f64) -> Vec<f64> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let max_score = candidates
        .iter()
        .map(|candidate| candidate.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut values = candidates
        .iter()
        .map(|candidate| ((candidate.score - max_score) / temperature).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / candidates.len() as f64;
        values.fill(uniform);
    } else {
        for value in &mut values {
            *value /= sum;
        }
    }
    values
}

#[cfg(test)]
fn score_action(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
    alternatives: usize,
    config: &RuleDecisionConfig,
) -> (RuleScoreBreakdown, f64) {
    let card_values = card_future_values(runner, runner.framework.current_player);
    score_action_with_card_values(
        runner,
        after_runner,
        before,
        after,
        effect,
        intent,
        alternatives,
        config,
        &card_values,
    )
}

fn score_action_with_card_values(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
    alternatives: usize,
    config: &RuleDecisionConfig,
    card_values: &[f64],
) -> (RuleScoreBreakdown, f64) {
    let actor = runner.framework.current_player;
    let rounds_remaining = estimate_rounds_remaining(runner);
    let phase_is_railroad = matches!(runner.game_phase, GamePhase::Railroad | GamePhase::GameEnd);
    let late_game = rounds_remaining <= 3.0 || runner.framework.board.state.deck.cards.len() <= 12;
    let vp_gap = best_opponent_vp(runner, actor) - before.victory_points;
    let trailing_multiplier = if vp_gap > 0.0 {
        1.0 + (vp_gap / 50.0).clamp(0.0, 0.8)
    } else {
        1.0
    };

    let immediate_vp_delta = actor_delta(&effect.player_victory_points_delta, actor);
    // Potential VP is useful as a sale/road signal, but it is not secured VP.
    // Keep one unusually large phase-transition delta from deciding the whole
    // action.  The explicit lifecycle gates below carry the strategic meaning.
    let potential_vp_delta =
        actor_delta(&effect.player_potential_era_victory_points_delta, actor).clamp(-12.0, 12.0);
    let vp_urgency = if late_game {
        1.35
    } else if phase_is_railroad {
        1.15
    } else {
        1.0
    } * trailing_multiplier;
    let immediate_vp = config.immediate_vp_weight * immediate_vp_delta * vp_urgency;
    let potential_vp = config.potential_vp_weight * potential_vp_delta * vp_urgency;

    let income_amount_delta = after.income_amount as f64 - before.income_amount as f64;
    let income_level_delta = after.income_level as f64 - before.income_level as f64;
    // A loan deliberately trades income for a large cash injection.  Charging
    // the full horizon here would make every useful opening loan look worse
    // than a harmless pass; the action-specific branch below accounts for its
    // real opportunity cost after the unlock bonus is known.
    // Income funds future actions; it is not itself VP.  A bounded horizon
    // keeps a high-income tile from drowning out a ready sale or a route that
    // enables one.  This is deliberately a state-score guard, not a terminal
    // rollout approximation.
    let income_signal = bounded_income_signal(
        income_amount_delta,
        income_level_delta,
        rounds_remaining,
        intent.action_type == ActionType::Loan,
    );
    let loan_maturity_cost = if intent.action_type == ActionType::Loan {
        config.loan_maturity_weight * loan_maturity_surcharge(before.income_amount)
            + config.loan_excess_income_loss_weight
                * loan_excess_income_loss_surcharge(before.income_amount, after.income_amount)
    } else {
        0.0
    };
    let income = config.income_weight
        * (income_signal - loan_maturity_cost)
        * if phase_is_railroad { 0.75 } else { 1.0 };

    let money_delta = after.money as f64 - before.money as f64;
    let cash =
        config.cash_weight * money_delta + liquidity_adjustment(runner, before, after, intent);

    let level_progress = industry_progress(before, after);
    let building_delta = after.buildings as f64 - before.buildings as f64;
    let flipped_delta = after.flipped_buildings as f64 - before.flipped_buildings as f64;
    let quality_delta = after.industry_quality - before.industry_quality;
    let sellable_delta = after.sellable_units as f64 - before.sellable_units as f64;
    let sellable_vp_delta = after.sellable_vp - before.sellable_vp;
    // Mat progress is only a *means* to a stronger build.  The old 1.8
    // multiplier treated every popped tile as immediate VP and caused an
    // endless DevelopDouble loop.  Keep the raw progress small and add a
    // target-aware value in strategic_action_bias.
    let industry_raw = level_progress * 0.22
        + building_delta * 1.2
        + flipped_delta * 0.9
        + quality_delta * 0.08
        + sellable_delta * 0.15
        + sellable_vp_delta.clamp(-8.0, 8.0) * 0.18;
    let industry =
        config.industry_weight * industry_raw * if phase_is_railroad { 0.85 } else { 1.0 };

    // Generic network size is deliberately weak. The useful signal is a real
    // reduction in the number of links needed to reach a matching merchant;
    // otherwise a canal road can look profitable merely because it touches a
    // larger connected component.
    let route_progress = before.sale_route_distance - after.sale_route_distance;
    let network_raw = (after.own_roads as f64 - before.own_roads as f64)
        * if phase_is_railroad { 0.65 } else { 0.25 }
        + (after.network_locations as f64 - before.network_locations as f64)
            * if phase_is_railroad { 0.35 } else { 0.12 }
        + (after.connected_trade_posts as f64 - before.connected_trade_posts as f64)
            * if phase_is_railroad { 1.1 } else { 0.45 }
        + (after.available_build_sites as f64 - before.available_build_sites as f64) * 0.3
        + route_progress * if phase_is_railroad { 2.2 } else { 2.8 }
        + (after.road_value - before.road_value) * if phase_is_railroad { 0.25 } else { 0.08 };
    let network = config.network_weight * network_raw * if phase_is_railroad { 1.35 } else { 0.9 };

    let resource_raw = (after.resource_units[0] as f64 - before.resource_units[0] as f64) * 0.20
        + (after.resource_units[1] as f64 - before.resource_units[1] as f64) * 0.18
        + (after.resource_units[2] as f64 - before.resource_units[2] as f64) * 0.32
        + (after.sellable_units as f64 - before.sellable_units as f64) * 0.45
        + market_access_value(after)
        - market_access_value(before);
    let resources = config.resource_weight * resource_raw;

    let safety = config.safety_weight * safety_score(runner, before, after, intent);
    let action_bias = config.action_bias_weight
        * action_bias_score(
            runner,
            after_runner,
            before,
            after,
            effect,
            intent,
            alternatives,
        )
        + config.economic_guard_weight
            * economic_guard_penalty(runner, after_runner, before, after, effect, intent)
        + config.strategic_priority_weight
            * strategic_priority_adjustment(runner, after_runner, before, after, intent)
        + config.lifecycle_priority_weight
            * (lifecycle_priority_adjustment(runner, after_runner, before, after, effect, intent)
                + recovery_priority_adjustment(
                    runner,
                    after_runner,
                    before,
                    after,
                    effect,
                    intent,
                ))
        + config.sale_frontier_build_weight * sale_frontier_build_signal(before, after, intent);
    let action_bias = action_bias
        + config.recovery_product_plan_weight
            * recovery_product_plan_signal(runner, after_runner, before, after, intent)
        + config.conversion_feasibility_weight
            * conversion_feasibility_adjustment(
                runner,
                after_runner,
                before,
                after,
                effect,
                intent,
            )
        + config.resource_stockpile_weight
            * resource_stockpile_adjustment(runner, before, after, intent)
        + config.post_sale_product_cycle_weight
            * post_sale_product_cycle_signal(runner, after_runner, before, after, intent)
        + config.mature_product_cycle_weight
            * mature_product_cycle_signal(runner, after_runner, before, after, intent)
        + if config.low_cost_loan_runway_weight > 0.0 {
            config.low_cost_loan_runway_weight
                * low_cost_loan_runway_signal(runner, after_runner, before, after, intent)
        } else {
            0.0
        }
        + if config.financed_product_cycle_weight > 0.0 {
            config.financed_product_cycle_weight
                * financed_product_cycle_signal(runner, after_runner, before, after, intent, config)
        } else {
            0.0
        }
        + if config.contested_external_beer_plan_weight > 0.0 {
            config.contested_external_beer_plan_weight
                * contested_external_beer_plan_signal(runner, after_runner, before, after, intent)
        } else {
            0.0
        }
        + config.late_canal_stall_weight
            * late_canal_stall_signal(runner, after_runner, before, after, effect, intent)
        + config.late_railroad_backlog_weight
            * late_railroad_backlog_signal(runner, after_runner, before, after, intent)
        + config.beer_backlog_recovery_weight
            * beer_backlog_recovery_signal(runner, after_runner, before, after, intent)
        + config.late_railroad_conversion_stall_weight
            * late_railroad_conversion_stall_signal(
                runner,
                after_runner,
                before,
                after,
                effect,
                intent,
            )
        + if config.same_turn_sale_guard_weight > 0.0 {
            config.same_turn_sale_guard_weight
                * same_turn_sale_guard_rebate(
                    runner,
                    after_runner,
                    before,
                    after,
                    intent,
                    config.safety_weight,
                    config.economic_guard_weight,
                )
        } else {
            0.0
        };
    let card_value = config.card_weight * card_opportunity_score(runner, intent, card_values);
    let tempo = config.tempo_weight
        * (if effect.turn_count_delta == 0 {
            0.15
        } else if late_game {
            -0.15
        } else {
            0.0
        });

    let competitive_pressure = if vp_gap > 0.0 {
        vp_gap.min(50.0) * 0.02
    } else {
        (-vp_gap).min(50.0) * 0.01
    };
    let total = immediate_vp
        + potential_vp
        + income
        + cash
        + industry
        + network
        + resources
        + safety
        + tempo
        + action_bias
        + competitive_pressure
        + card_value;

    (
        RuleScoreBreakdown {
            immediate_vp,
            potential_vp,
            income,
            cash,
            industry,
            network,
            resources,
            safety,
            tempo,
            action_bias,
            competitive_pressure,
            card_value,
            lookahead: 0.0,
            total,
        },
        total,
    )
}

fn snapshot(runner: &GameRunner, actor: usize) -> PositionSnapshot {
    snapshot_with_route_data(runner, actor, None)
}

#[derive(Debug, Clone, Copy)]
struct SaleRouteData {
    total: f64,
    by_location: [f64; N_BL],
}

fn snapshot_with_route_data(
    runner: &GameRunner,
    actor: usize,
    route_data: Option<SaleRouteData>,
) -> PositionSnapshot {
    let state = &runner.framework.board.state;
    let player = &state.players[actor];
    let mut resource_units = [0u32; 3];
    let mut buildings = 0usize;
    let mut flipped_buildings = 0usize;
    let mut industry_quality = 0.0;
    let mut sellable_units = 0usize;
    let mut sellable_vp = 0.0;
    let mut sellable_beer_demand = 0u32;
    let mut unflipped_beer_units = 0u32;

    // `bl_to_building` is a HashMap. Iterate the fixed build-location domain
    // so floating-point reductions and near-tie ranking do not depend on the
    // process-local hash seed.
    for location in 0..N_BL {
        let Some(building) = state.bl_to_building.get(&location) else {
            continue;
        };
        if building.owner.as_usize() != actor {
            continue;
        }
        buildings += 1;
        let data = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        industry_quality += building.level.as_u8() as f64
            + f64::from(data.income) * 0.1
            + f64::from(data.vp_on_flip) * 0.2
            + f64::from(data.road_vp) * 0.15;
        if building.flipped {
            flipped_buildings += 1;
        } else {
            match building.industry {
                IndustryType::Coal => resource_units[0] += u32::from(building.resource_amt),
                IndustryType::Iron => resource_units[1] += u32::from(building.resource_amt),
                IndustryType::Beer => {
                    resource_units[2] += u32::from(building.resource_amt);
                    unflipped_beer_units =
                        unflipped_beer_units.saturating_add(u32::from(building.resource_amt));
                }
                IndustryType::Goods | IndustryType::Cotton | IndustryType::Pottery => {
                    sellable_units += 1;
                    sellable_vp += f64::from(data.vp_on_flip);
                    sellable_beer_demand =
                        sellable_beer_demand.saturating_add(u32::from(data.beer_needed));
                }
            }
        }
    }

    let network = player.get_locations_in_network(state);
    let connected_trade_posts = network.ones().filter(|location| *location >= 22).count();
    let industry_levels = std::array::from_fn(|index| {
        player
            .industry_mat
            .get_lowest_level(IndustryType::from_usize(index))
            .as_u8()
    });
    let industry_remaining = std::array::from_fn(|index| {
        player
            .industry_mat
            .get_remaining_tiles_at_level(IndustryType::from_usize(index))
    });

    let route_data = route_data.unwrap_or_else(|| sale_route_data(runner, actor));
    PositionSnapshot {
        victory_points: f64::from(player.victory_points),
        income_level: player.income_level,
        income_amount: player.get_income_amount(player.income_level),
        money: player.money,
        buildings,
        flipped_buildings,
        own_roads: state.player_road_mask[actor].ones().count(),
        network_locations: network.ones().count(),
        connected_trade_posts,
        industry_levels,
        industry_remaining,
        industry_quality,
        resource_units,
        sellable_units,
        sellable_vp,
        sellable_beer_demand,
        unflipped_beer_units,
        road_value: own_road_value(runner, actor),
        sale_route_distance: route_data.total,
        sale_route_by_location: route_data.by_location,
        available_build_sites: available_build_sites(runner, actor, &network),
        available_sell_targets: runner.framework.board.get_valid_sell_options(actor).len(),
        market_coal: state.remaining_market_coal,
        market_iron: state.remaining_market_iron,
    }
}

fn available_build_sites(
    runner: &GameRunner,
    actor: usize,
    network: &crate::core::types::LocationSet,
) -> usize {
    let state = &runner.framework.board.state;
    (0..N_BL)
        .filter(|location| {
            if state.bl_to_building.contains_key(location) {
                return false;
            }
            let town = LocationName::from_bl_idx(*location);
            if !network.contains(town.as_usize()) {
                return false;
            }
            BUILD_LOCATION_MASK[*location].ones().any(|industry_idx| {
                let industry = IndustryType::from_usize(industry_idx);
                let tile = state.players[actor]
                    .industry_mat
                    .get_tile_for_industry(industry);
                tile.is_some_and(|data| data.can_build_in_era(state.era))
            })
        })
        .count()
}

fn own_road_value(runner: &GameRunner, actor: usize) -> f64 {
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

/// Compute the remaining route work for the actor's unflipped sellable
/// buildings. Existing roads cost zero; empty links legal in the current era
/// cost one. Distances are computed once per product type with a reverse
/// multi-source 0/1 BFS, then reused for every building on the map.
fn sale_route_data(runner: &GameRunner, actor: usize) -> SaleRouteData {
    let state = &runner.framework.board.state;
    let mut target_masks = [[false; TOTAL_TOWNS]; 3];
    target_masks[0] = merchant_target_mask(state, IndustryType::Cotton);
    target_masks[1] = merchant_target_mask(state, IndustryType::Goods);
    target_masks[2] = merchant_target_mask(state, IndustryType::Pottery);
    let distance_maps = [
        route_distance_map(runner, &target_masks[0]),
        route_distance_map(runner, &target_masks[1]),
        route_distance_map(runner, &target_masks[2]),
    ];

    let mut data = SaleRouteData {
        total: 0.0,
        by_location: [0.0; N_BL],
    };
    // Keep the reduction order stable across independent runners. This value
    // feeds action scores, so HashMap iteration order must not leak into the
    // policy decision.
    for location in 0..N_BL {
        let Some(building) = state.bl_to_building.get(&location) else {
            continue;
        };
        if building.owner.as_usize() != actor || building.flipped {
            continue;
        }
        let target_index = match building.industry {
            IndustryType::Cotton => 0,
            IndustryType::Goods => 1,
            IndustryType::Pottery => 2,
            IndustryType::Coal | IndustryType::Iron | IndustryType::Beer => continue,
        };
        let location = LocationName::from_bl_idx(building.loc as usize).as_usize();
        let distance = distance_maps[target_index][location];
        if distance == usize::MAX {
            continue;
        }
        let tile_data = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
        let urgency =
            1.0 + f64::from(tile_data.vp_on_flip) * 0.12 + f64::from(tile_data.beer_needed) * 0.05;
        let contribution = distance as f64 * urgency;
        data.by_location[building.loc as usize] = contribution;
        data.total += contribution;
    }
    data
}

fn merchant_target_mask(
    state: &crate::board::state::BoardState,
    industry: IndustryType,
) -> [bool; TOTAL_TOWNS] {
    let mut targets = [false; TOTAL_TOWNS];
    state
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
                0 => 22,     // Shrewbury (one slot)
                1 | 2 => 23, // Oxford
                3 | 4 => 24, // Gloucester
                5 | 6 => 25, // Warrington
                7 | 8 => 26, // Nottingham
                _ => return,
            };
            targets[location] = true;
        });
    targets
}

fn route_distance_map(runner: &GameRunner, targets: &[bool; TOTAL_TOWNS]) -> [usize; TOTAL_TOWNS] {
    let state = &runner.framework.board.state;
    let mut distances = [usize::MAX; TOTAL_TOWNS];
    let mut pending = VecDeque::with_capacity(TOTAL_TOWNS);
    for (location, is_target) in targets.iter().copied().enumerate() {
        if is_target {
            distances[location] = 0;
            pending.push_back(location);
        }
    }

    while let Some(location) = pending.pop_front() {
        let distance = distances[location];
        for road_idx in LOCATION_TO_ROADS[location].ones() {
            let link = &LINK_LOCATIONS[road_idx];
            let cost = if state.built_roads.contains(road_idx) {
                0
            } else if link_is_legal_in_phase(link, runner.game_phase) {
                1
            } else {
                continue;
            };
            for destination in link.locations.ones() {
                if destination == location {
                    continue;
                }
                let next_distance = distance + cost;
                if next_distance < distances[destination] {
                    distances[destination] = next_distance;
                    if cost == 0 {
                        pending.push_front(destination);
                    } else {
                        pending.push_back(destination);
                    }
                }
            }
        }
    }
    distances
}

fn route_component_for_building(
    runner: &GameRunner,
    building: &crate::core::building::BuiltBuilding,
) -> f64 {
    if building.flipped
        || !matches!(
            building.industry,
            IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
        )
    {
        return 0.0;
    }
    let target_mask = merchant_target_mask(&runner.framework.board.state, building.industry);
    let location = LocationName::from_bl_idx(building.loc as usize).as_usize();
    let distance = route_distance_map(runner, &target_mask)[location];
    if distance == usize::MAX {
        return 0.0;
    }
    let tile = &INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()];
    let urgency = 1.0 + f64::from(tile.vp_on_flip) * 0.12 + f64::from(tile.beer_needed) * 0.05;
    distance as f64 * urgency
}

/// Reuse route data whenever an action cannot change the road graph or the
/// actor's unflipped sellable buildings. Network actions and era transitions
/// return `None`, asking the snapshot to recompute an exact map.
fn route_data_after_action(
    runner: &GameRunner,
    after_runner: &GameRunner,
    actor: usize,
    before: &PositionSnapshot,
    intent: &ActionIntent,
) -> Option<SaleRouteData> {
    if runner.game_phase != after_runner.game_phase
        || runner.framework.board.state.built_roads
            != after_runner.framework.board.state.built_roads
    {
        return None;
    }

    let mut data = SaleRouteData {
        total: before.sale_route_distance,
        by_location: before.sale_route_by_location,
    };
    match intent.action_type {
        ActionType::Sell => {
            for choice in &intent.sell_choices {
                let location = choice.location;
                if location < N_BL {
                    data.total -= data.by_location[location];
                    data.by_location[location] = 0.0;
                }
            }
        }
        ActionType::BuildBuilding => {
            let Some(location) = intent.selected_build_location else {
                return Some(data);
            };
            let Some(building) = after_runner
                .framework
                .board
                .state
                .bl_to_building
                .get(&location)
            else {
                return Some(data);
            };
            if building.owner.as_usize() == actor && location < N_BL {
                let contribution = route_component_for_building(after_runner, building);
                // Replacing a building at the same location must replace its
                // prior route contribution rather than count both versions.
                data.total -= data.by_location[location];
                data.by_location[location] = contribution;
                data.total += contribution;
            }
        }
        _ => {}
    }
    Some(data)
}

fn link_is_legal_in_phase(link: &crate::core::links::Link, phase: GamePhase) -> bool {
    match phase {
        GamePhase::Canal => link.can_build_canal,
        GamePhase::Railroad => link.can_build_rail,
        // The metric is only used as a tie-breaker after the final action. Keep
        // all links traversable there so a final report remains finite.
        GamePhase::GameEnd => link.can_build_canal || link.can_build_rail,
    }
}

fn industry_progress(before: &PositionSnapshot, after: &PositionSnapshot) -> f64 {
    let mut progress = 0.0;
    for index in 0..6 {
        progress +=
            (after.industry_levels[index] as f64 - before.industry_levels[index] as f64) * 2.5;
        if after.industry_remaining[index] < before.industry_remaining[index] {
            progress +=
                (before.industry_remaining[index] - after.industry_remaining[index]) as f64 * 0.18;
        }
    }
    progress
}

fn liquidity_adjustment(
    runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let action = intent.action_type;
    let before_money = before.money as f64;
    let after_money = after.money as f64;
    let mut adjustment = 0.0;
    if after_money < 3.0 {
        adjustment -= 2.5;
    } else if after_money >= 8.0 && before_money < 8.0 {
        adjustment += 0.5;
    }
    if action == ActionType::Loan {
        // A loan is useful to unlock a concrete next build/network action,
        // but taking it while already liquid is a meaningful opportunity cost.
        if before_money < 8.0 {
            adjustment += 4.0;
        } else if before_money > 14.0 {
            adjustment -= 4.0;
        } else {
            adjustment -= 0.5;
        }
    }
    let _ = runner; // Reserved for future cost-aware liquidity lookahead.
    adjustment
}

/// Estimate the opportunity cost of the cards discarded by an action. The
/// estimate is intentionally bounded: card quality should decide between two
/// otherwise similar discards, never override a real sale or productive build.
fn card_future_values(runner: &GameRunner, actor: usize) -> Vec<f64> {
    runner.framework.board.state.players[actor]
        .hand
        .cards
        .iter()
        .map(|card| card_future_value(runner, &card.card_type))
        .collect()
}

fn card_opportunity_score(runner: &GameRunner, intent: &ActionIntent, card_values: &[f64]) -> f64 {
    let actor = runner.framework.current_player;
    let hand = &runner.framework.board.state.players[actor].hand.cards;
    let mut indices = Vec::with_capacity(3);
    if let Some(index) = intent.selected_card_idx {
        indices.push(index);
    }
    indices.extend(intent.scout_additional_discard_indices.iter().copied());
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        return 0.0;
    }

    let mut score = 0.0;
    for index in indices {
        let Some(card) = hand.get(index) else {
            continue;
        };
        let value = card_values
            .get(index)
            .copied()
            .unwrap_or_else(|| card_future_value(runner, &card.card_type));
        let discard_factor = if intent.action_type == ActionType::Scout {
            0.55
        } else {
            0.75
        };
        score -= value * discard_factor;
        if card_matches_intent(&card.card_type, intent) {
            // Spending a card for the purpose it represents is less costly
            // than throwing away a flexible card to pay a generic action fee.
            score += value * 0.65 + 0.35;
        }
    }
    score.clamp(-8.0, 2.5)
}

fn card_future_value(runner: &GameRunner, card_type: &CardType) -> f64 {
    let actor = runner.framework.current_player;
    let state = &runner.framework.board.state;
    match card_type {
        CardType::WildLocation | CardType::WildIndustry => 4.2,
        CardType::Location(town) => {
            let location = LocationName::from_usize(town.as_usize());
            let network = state.players[actor].get_locations_in_network(state);
            let open_builds = location
                .to_bl_set()
                .ones()
                .filter(|build_location| {
                    !state.bl_to_building.contains_key(build_location)
                        && BUILD_LOCATION_MASK[*build_location]
                            .ones()
                            .any(|industry_idx| {
                                state.players[actor]
                                    .industry_mat
                                    .get_tile_for_industry(IndustryType::from_usize(industry_idx))
                                    .is_some_and(|tile| tile.can_build_in_era(state.era))
                            })
                })
                .count();
            let mut value = if network.contains(location.as_usize()) {
                1.25
            } else {
                0.55
            };
            if open_builds > 0 {
                value += (open_builds as f64).min(2.0) * 0.8;
            }
            value
        }
        CardType::Industry(industries) => industries
            .to_industry_types()
            .into_iter()
            .map(|industry| industry_card_value(runner, industry))
            .fold(0.35, f64::max),
    }
}

fn industry_card_value(runner: &GameRunner, industry: IndustryType) -> f64 {
    let actor = runner.framework.current_player;
    let state = &runner.framework.board.state;
    let mat = &state.players[actor].industry_mat;
    let level = mat.get_lowest_level(industry).as_u8();
    let has_build = runner
        .framework
        .board
        .get_valid_build_options(actor)
        .iter()
        .any(|option| option.industry_type == industry);
    let mut value = 0.7 + f64::from(level.min(3)) * 0.12;
    if has_build {
        value += 1.5;
    }
    if matches!(
        industry,
        IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
    ) {
        value += 0.35;
    }
    value.min(3.8)
}

fn card_matches_intent(card_type: &CardType, intent: &ActionIntent) -> bool {
    if intent.action_type != ActionType::BuildBuilding {
        return false;
    }
    match card_type {
        CardType::WildLocation | CardType::WildIndustry => true,
        CardType::Location(town) => intent.selected_build_location.is_some_and(|location| {
            LocationName::from_bl_idx(location).as_usize() == town.as_usize()
        }),
        CardType::Industry(industries) => intent
            .selected_industry
            .is_some_and(|industry| industries.has_industry(industry)),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ProductiveOptions {
    build: usize,
    /// Legal builds of a product that can be sold (validator includes cards,
    /// resources, and current affordability).
    saleable_builds: usize,
    /// Legal Coal/Iron builds whose full output fits in the market now, so the
    /// tile flips and raises income immediately rather than promising a future
    /// recovery that may never be converted.
    immediate_income_builds: usize,
    sell: usize,
    develop: usize,
    network: usize,
    minimum_build_cost: Option<u16>,
}

impl ProductiveOptions {
    fn count(self) -> usize {
        self.build + self.sell + self.develop + self.network
    }

    fn has_build(self) -> bool {
        self.build > 0
    }

    fn has_conversion_frontier(self) -> bool {
        self.sell > 0 || self.saleable_builds > 0
    }

    fn has_immediate_income_recovery(self) -> bool {
        self.immediate_income_builds > 0
    }
}

/// Return cheap, legality-aware availability signals for one player.  These
/// validators already include cards, resource sources, and affordability, so
/// the rule tree does not need to invent a second approximation of legality.
fn productive_options(runner: &GameRunner, actor: usize) -> ProductiveOptions {
    let board = &runner.framework.board;
    let builds = board.get_valid_build_options(actor);
    let minimum_build_cost = builds.iter().map(|option| option.total_money_cost).min();
    let saleable_builds = builds
        .iter()
        .filter(|option| is_saleable_industry(option.industry_type))
        .count();
    let immediate_income_builds = builds
        .iter()
        .filter(|option| {
            let free_market_space = match option.industry_type {
                IndustryType::Coal => {
                    if !board
                        .state
                        .is_connected_to_trade_post(option.build_location_idx)
                    {
                        return false;
                    }
                    MAX_MARKET_COAL.saturating_sub(board.state.remaining_market_coal)
                }
                IndustryType::Iron => {
                    MAX_MARKET_IRON.saturating_sub(board.state.remaining_market_iron)
                }
                IndustryType::Beer
                | IndustryType::Goods
                | IndustryType::Pottery
                | IndustryType::Cotton => return false,
            };
            option.building_data.income > 0
                && option.building_data.resource_amt > 0
                && free_market_space >= option.building_data.resource_amt
        })
        .count();
    let develop = board.get_valid_development_options(actor).count_ones(..);
    let network = if runner.game_phase == GamePhase::Canal {
        board.get_valid_canal_options(actor).len()
    } else if runner.game_phase == GamePhase::Railroad {
        let single = board.get_valid_single_rail_options(actor).len();
        let double = board.get_valid_double_rail_first_link_options(actor).len();
        single + double
    } else {
        0
    };
    ProductiveOptions {
        build: builds.len(),
        saleable_builds,
        immediate_income_builds,
        sell: board.get_valid_sell_options(actor).len(),
        develop,
        network,
        minimum_build_cost,
    }
}

fn is_saleable_industry(industry: IndustryType) -> bool {
    matches!(
        industry,
        IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
    )
}

/// A compact description of the actionable building frontier.  Comparing
/// this before/after a develop action prevents the policy from spending iron
/// merely because the mat changed while a concrete build was already legal.
fn build_frontier_signature(runner: &GameRunner, actor: usize) -> (u8, usize, usize) {
    let options = runner.framework.board.get_valid_build_options(actor);
    let highest_level = options
        .iter()
        .map(|option| option.level.as_u8())
        .max()
        .unwrap_or(0);
    let saleable_count = options
        .iter()
        .filter(|option| is_saleable_industry(option.industry_type))
        .count();
    (highest_level, saleable_count, options.len())
}

fn development_improves_build_frontier(
    runner: &GameRunner,
    after_runner: &GameRunner,
    actor: usize,
) -> bool {
    let before = build_frontier_signature(runner, actor);
    let after = build_frontier_signature(after_runner, actor);
    after.0 > before.0 || after.1 > before.1 || (before.2 == 0 && after.2 > 0)
}

fn bounded_income_signal(
    income_amount_delta: f64,
    income_level_delta: f64,
    rounds_remaining: f64,
    is_loan: bool,
) -> f64 {
    let horizon = if is_loan {
        rounds_remaining.min(2.5)
    } else {
        rounds_remaining
    };
    (income_amount_delta * horizon).clamp(-10.0, 10.0)
        + (income_level_delta * 0.04).clamp(-0.5, 0.5)
}

/// Borrowing against a mature income engine destroys more accumulated track
/// progress than an opening loan and is harder to rebuild. Keep loans through
/// income +1 unchanged (the observed strong human pattern), then increase the
/// opportunity cost smoothly instead of using a brittle high-income cutoff.
fn loan_maturity_surcharge(before_income: i8) -> f64 {
    let mature_income = f64::from((before_income - 1).max(0));
    0.75 * mature_income.min(3.0)
        + 0.90 * (mature_income - 3.0).clamp(0.0, 5.0)
        + 0.50 * (mature_income - 8.0).clamp(0.0, 10.0)
}

/// A normal loan gives GBP 30 in exchange for a three-point displayed-income
/// reduction. Some positions in the current engine lose more than three. The
/// base income feature already prices the first three points; expose only the
/// excess here so useful low-cost borrowing is unchanged and unusually costly
/// late borrowing can be screened independently.
fn loan_excess_income_loss_surcharge(before_income: i8, after_income: i8) -> f64 {
    let loss = (i16::from(before_income) - i16::from(after_income)).max(0);
    f64::from((loss - 3).clamp(0, 12))
}

#[derive(Debug, Clone, Copy)]
struct LowCostLoanRunwayInputs {
    railroad: bool,
    deck_empty: bool,
    low_score: bool,
    underbuilt_engine: bool,
    first_action: bool,
    no_ready_or_pending_sale: bool,
    before_income: i8,
    displayed_income_loss: i16,
    before_money: u16,
    after_money: u16,
    cards_after: usize,
    productive_follow_up: bool,
}

fn low_cost_loan_runway_is_eligible(input: LowCostLoanRunwayInputs) -> bool {
    input.railroad
        && input.deck_empty
        && input.low_score
        && input.underbuilt_engine
        && input.first_action
        && input.no_ready_or_pending_sale
        && (0..=1).contains(&input.before_income)
        && (0..=3).contains(&input.displayed_income_loss)
        && (15..=20).contains(&input.before_money)
        && input.after_money >= input.before_money.saturating_add(30)
        && input.cards_after >= 5
        && input.productive_follow_up
}

fn low_cost_loan_has_productive_follow_up(
    after_runner: &GameRunner,
    actor: usize,
    pre_loan_money: u16,
) -> bool {
    if after_runner.framework.current_player != actor
        || after_runner.actions_remaining_in_turn != 1
        || after_runner.has_pending_shortfall()
    {
        return false;
    }

    let board = &after_runner.framework.board;
    if board.get_valid_single_rail_options(actor).is_empty() {
        return false;
    }
    let Some(product_cost) = board
        .get_valid_build_options(actor)
        .into_iter()
        .filter(|option| is_saleable_industry(option.industry_type))
        .map(|option| option.total_money_cost)
        .min()
    else {
        return false;
    };
    let player = &board.state.players[actor];
    let displayed_income = player.get_income_amount(player.income_level);
    let settlement_debt = u16::try_from((-displayed_income).max(0)).unwrap_or(u16::MAX);
    // Reserve the worst single-coal market price as well as the link, product,
    // and next settlement. This proves that the loan creates a real two-step
    // runway rather than merely making an already affordable action look rich.
    let required = ONE_RAILROAD_PRICE
        .saturating_add(8)
        .saturating_add(product_cost)
        .saturating_add(settlement_debt)
        .saturating_add(8);
    player.money >= required && pre_loan_money < required
}

fn low_cost_loan_runway_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    if intent.action_type != ActionType::Loan {
        return 0.0;
    }
    let actor = runner.framework.current_player;
    let displayed_income_loss = i16::from(before.income_amount) - i16::from(after.income_amount);
    let cheap_inputs = LowCostLoanRunwayInputs {
        railroad: runner.game_phase == GamePhase::Railroad,
        deck_empty: runner.framework.board.state.deck.cards.is_empty(),
        low_score: before.victory_points < 20.0,
        underbuilt_engine: before.buildings <= 4,
        first_action: after_runner.framework.current_player == actor
            && after_runner.actions_remaining_in_turn == 1,
        no_ready_or_pending_sale: before.available_sell_targets == 0 && before.sellable_units == 0,
        before_income: before.income_amount,
        displayed_income_loss,
        before_money: before.money,
        after_money: after.money,
        cards_after: after_runner.framework.board.state.players[actor]
            .hand
            .cards
            .len(),
        productive_follow_up: true,
    };
    if !low_cost_loan_runway_is_eligible(cheap_inputs) {
        return 0.0;
    }
    f64::from(low_cost_loan_has_productive_follow_up(
        after_runner,
        actor,
        before.money,
    ))
}

fn sale_frontier_build_signal(
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    if intent.action_type != ActionType::BuildBuilding || before.available_sell_targets > 0 {
        return 0.0;
    }
    let new_sell_targets = after
        .available_sell_targets
        .saturating_sub(before.available_sell_targets) as f64;
    if new_sell_targets > 0.0 {
        5.0 + new_sell_targets.min(2.0)
    } else {
        0.0
    }
}

fn recovery_product_plan_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let severe_railroad_recovery = runner.game_phase == GamePhase::Railroad
        && before.victory_points < 35.0
        && before.buildings <= 3
        && before.income_amount <= -5;
    let product_build = intent.action_type == ActionType::BuildBuilding
        && intent.selected_industry.is_some_and(is_saleable_industry)
        && after.sellable_units > before.sellable_units;
    if !severe_railroad_recovery || before.sellable_units > 0 || !product_build {
        return 0.0;
    }

    let actor = runner.framework.current_player;
    if after_runner.framework.current_player != actor || after_runner.actions_remaining_in_turn == 0
    {
        return 0.0;
    }

    let after_player = &after_runner.framework.board.state.players[actor];
    let after_debt =
        u16::try_from((-after_player.get_income_amount(after_player.income_level)).max(0))
            .unwrap_or(u16::MAX);
    if after_runner
        .framework
        .board
        .get_valid_sell_options(actor)
        .len()
        > 0
        && after_player.money >= after_debt
    {
        return 1.0;
    }

    let Ok(replies) = enumerate_legal_actions(after_runner) else {
        return 0.0;
    };
    replies
        .into_iter()
        .filter(|reply| {
            matches!(
                reply.root,
                ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
            )
        })
        .any(|reply| {
            let mut reply_runner = after_runner.clone();
            if reply.apply(&mut reply_runner).is_err() {
                return false;
            }
            let reply_player = &reply_runner.framework.board.state.players[actor];
            let reply_debt =
                u16::try_from((-reply_player.get_income_amount(reply_player.income_level)).max(0))
                    .unwrap_or(u16::MAX);
            reply_player.money >= reply_debt
                && !reply_runner
                    .framework
                    .board
                    .get_valid_sell_options(actor)
                    .is_empty()
        })
        .then_some(1.0)
        .unwrap_or(0.0)
}

/// Minimum additional card actions needed to flip one newly built product
/// from the exact successor board. A ready Sell costs one action. Otherwise
/// the lower bound includes the shortest current-era route plus the Sell; a
/// connected product that is still illegal needs at least one prerequisite
/// (normally Beer) before the Sell.
fn minimum_future_actions_to_sell_product(
    runner: &GameRunner,
    actor: usize,
    build_location: usize,
    industry: IndustryType,
) -> Option<usize> {
    if runner.game_phase == GamePhase::GameEnd {
        return None;
    }
    if runner
        .framework
        .board
        .get_valid_sell_options(actor)
        .iter()
        .any(|option| option.location == build_location)
    {
        return Some(1);
    }

    let targets = merchant_target_mask(&runner.framework.board.state, industry);
    let town = LocationName::from_bl_idx(build_location).as_usize();
    let distance = route_distance_map(runner, &targets)[town];
    if distance == usize::MAX {
        return None;
    }
    if distance == 0 {
        // Connectivity is already present, so another missing sale input such
        // as Beer must be established before spending the Sell card.
        return Some(2);
    }
    let network_actions = match runner.game_phase {
        GamePhase::Canal => distance,
        GamePhase::Railroad => distance.div_ceil(2),
        GamePhase::GameEnd => return None,
    };
    Some(network_actions.saturating_add(1))
}

fn product_build_feasibility_penalty(
    cards_after_build: usize,
    required_future_actions: Option<usize>,
    stacks_unconverted_inventory: bool,
) -> f64 {
    match required_future_actions {
        None => -28.0,
        Some(required) if cards_after_build < required => {
            -28.0 - (required - cards_after_build).min(2) as f64 * 2.0
        }
        Some(_) if stacks_unconverted_inventory => -6.0,
        Some(_) => 0.0,
    }
}

fn nominal_funded_product_build(runner: &GameRunner, actor: usize, money: u16) -> bool {
    runner
        .framework
        .board
        .get_valid_build_options(actor)
        .into_iter()
        .any(|option| {
            is_saleable_industry(option.industry_type)
                && money >= option.total_money_cost.saturating_add(8)
        })
}

/// Prove that a late loan can fund a product and leave enough cards to reach
/// a legal Sell. This executes each already validated build on a cloned board;
/// it does not roll opponents or the game forward.
fn funded_product_conversion_plan(runner: &GameRunner, actor: usize, money: u16) -> bool {
    runner
        .framework
        .board
        .get_valid_build_options(actor)
        .into_iter()
        .filter(|option| {
            is_saleable_industry(option.industry_type)
                && money >= option.total_money_cost.saturating_add(8)
        })
        .any(|option| {
            let build_location = option.build_location_idx;
            let industry = option.industry_type;
            let mut plan = runner.clone();
            if plan.framework.board.execute_build(actor, option).is_err() {
                return false;
            }
            let cards_after_build = plan.framework.board.state.players[actor].hand.cards.len();
            minimum_future_actions_to_sell_product(&plan, actor, build_location, industry)
                .is_some_and(|required| cards_after_build >= required)
        })
}

/// Remove deterministic late-game dead branches from the shallow tree. The
/// gate is Railroad-only and deck-empty, so normal openings and loans taken
/// while future cards will still be drawn retain their existing scores.
fn conversion_feasibility_adjustment(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
) -> f64 {
    if runner.game_phase != GamePhase::Railroad
        || !runner.framework.board.state.deck.cards.is_empty()
    {
        return 0.0;
    }
    let actor = runner.framework.current_player;
    let cards_after_action = after_runner.framework.board.state.players[actor]
        .hand
        .cards
        .len();

    if intent.action_type == ActionType::BuildBuilding
        && intent.selected_industry.is_some_and(is_saleable_industry)
        && effect.flipped_buildings_delta <= 0
    {
        let Some(build_location) = intent.selected_build_location else {
            return 0.0;
        };
        let Some(building) = after_runner
            .framework
            .board
            .state
            .bl_to_building
            .get(&build_location)
        else {
            return 0.0;
        };
        if building.owner.as_usize() != actor || building.flipped {
            return 0.0;
        }
        let required = minimum_future_actions_to_sell_product(
            after_runner,
            actor,
            build_location,
            building.industry,
        );
        let stacks_unconverted_inventory =
            before.sellable_units > 0 && after.available_sell_targets == 0;
        return product_build_feasibility_penalty(
            cards_after_action,
            required,
            stacks_unconverted_inventory,
        );
    }

    if intent.action_type != ActionType::Loan {
        return 0.0;
    }
    if cards_after_action == 0 {
        return -36.0;
    }

    let after_options = productive_options(after_runner, actor);
    if after_options.sell > 0
        || (cards_after_action > 0 && after_options.has_immediate_income_recovery())
    {
        return 0.0;
    }
    if nominal_funded_product_build(after_runner, actor, after.money)
        && !funded_product_conversion_plan(after_runner, actor, after.money)
    {
        return -24.0;
    }
    0.0
}

#[derive(Debug, Clone, Copy)]
struct ResourceStockpileInputs {
    canal: bool,
    low_score: bool,
    action_type: ActionType,
    selected_industry: Option<IndustryType>,
    before_income: i8,
    after_income: i8,
    before_flipped_buildings: usize,
    after_flipped_buildings: usize,
    before_sellable_units: usize,
    before_resource_units: u32,
    after_resource_units: u32,
    product_development_progress: usize,
    support_development_progress: usize,
    has_product_follow_up: bool,
}

fn resource_stockpile_penalty_from_inputs(input: ResourceStockpileInputs) -> f64 {
    if !input.canal
        || !input.low_score
        || input.action_type != ActionType::BuildBuilding
        || !matches!(
            input.selected_industry,
            Some(IndustryType::Coal | IndustryType::Iron)
        )
        || input.before_income >= 0
        || input.after_income > input.before_income
        || input.before_flipped_buildings > 0
        || input.after_flipped_buildings > input.before_flipped_buildings
        || input.before_sellable_units > 0
        || input.before_resource_units == 0
        || input.after_resource_units <= input.before_resource_units
        || input.product_development_progress < 2
        || input.support_development_progress < 2
        || !input.has_product_follow_up
    {
        return 0.0;
    }
    -1.0
}

/// Redirect only a stalled product opening. A healthy resource engine can
/// legitimately build several mines, so this gate requires evidence that the
/// player already spent two mat actions on a product but has converted none
/// of its assets. Beer is excluded because it may be the missing Sell input.
fn resource_stockpile_adjustment(
    runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let resource_index = match intent.selected_industry {
        Some(IndustryType::Coal) => 0,
        Some(IndustryType::Iron) => 1,
        _ => return 0.0,
    };
    if runner.game_phase != GamePhase::Canal
        || intent.action_type != ActionType::BuildBuilding
        || before.income_amount >= 0
        || before.flipped_buildings > 0
        || before.sellable_units > 0
        || before.resource_units[resource_index] == 0
    {
        return 0.0;
    }

    let actor = runner.framework.current_player;
    let state = &runner.framework.board.state;
    let developed_tiles = |industry| {
        let own_buildings = state
            .bl_to_building
            .values()
            .filter(|building| building.owner.as_usize() == actor && building.industry == industry)
            .count();
        mat_removed_count(runner, actor, industry).saturating_sub(own_buildings)
    };
    let product_development_progress = [
        IndustryType::Goods,
        IndustryType::Pottery,
        IndustryType::Cotton,
    ]
    .into_iter()
    .map(developed_tiles)
    .sum::<usize>();
    let support_development_progress = [IndustryType::Coal, IndustryType::Iron, IndustryType::Beer]
        .into_iter()
        .map(developed_tiles)
        .sum::<usize>();
    if product_development_progress < 2 || support_development_progress < 2 {
        return 0.0;
    }
    let board = &runner.framework.board;
    let has_product_build = board
        .get_valid_build_options(actor)
        .iter()
        .any(|option| is_saleable_industry(option.industry_type));
    let has_product_development = board
        .get_valid_development_options(actor)
        .ones()
        .map(IndustryType::from_usize)
        .any(is_saleable_industry);

    resource_stockpile_penalty_from_inputs(ResourceStockpileInputs {
        canal: true,
        low_score: before.victory_points < 20.0,
        action_type: intent.action_type,
        selected_industry: intent.selected_industry,
        before_income: before.income_amount,
        after_income: after.income_amount,
        before_flipped_buildings: before.flipped_buildings,
        after_flipped_buildings: after.flipped_buildings,
        before_sellable_units: before.sellable_units,
        before_resource_units: before.resource_units[resource_index],
        after_resource_units: after.resource_units[resource_index],
        product_development_progress,
        support_development_progress,
        has_product_follow_up: has_product_build || has_product_development,
    })
}

#[derive(Debug, Clone, Copy)]
struct PostSaleProductCycleInputs {
    railroad: bool,
    low_score: bool,
    late_deck: bool,
    action_type: ActionType,
    selected_industry: Option<IndustryType>,
    before_income: i8,
    before_sellable_units: usize,
    after_sellable_units: usize,
    has_converted_product: bool,
    cards_after_build: usize,
    required_future_actions: Option<usize>,
    after_money: u16,
    after_debt: u16,
}

fn post_sale_product_cycle_bonus_from_inputs(input: PostSaleProductCycleInputs) -> f64 {
    let Some(required) = input.required_future_actions else {
        return 0.0;
    };
    if !input.railroad
        || !input.low_score
        || !input.late_deck
        || input.action_type != ActionType::BuildBuilding
        || !input.selected_industry.is_some_and(is_saleable_industry)
        || input.before_income < 0
        || input.before_sellable_units > 0
        || input.after_sellable_units <= input.before_sellable_units
        || !input.has_converted_product
        || required > 3
        || input.cards_after_build < required.saturating_add(1)
        || input.after_money < input.after_debt
    {
        return 0.0;
    }
    1.0
}

/// After a low-score player proves one real product cycle, prefer beginning
/// the next reachable cycle over collecting another small road score. This is
/// late-Railroad only and requires enough remaining cards for the exact
/// successor's route/Sell lower bound plus one financing or Beer action.
fn post_sale_product_cycle_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    if runner.game_phase != GamePhase::Railroad
        || intent.action_type != ActionType::BuildBuilding
        || !intent.selected_industry.is_some_and(is_saleable_industry)
        || before.victory_points >= 15.0
        || before.income_amount < 0
        || before.sellable_units > 0
        || runner.framework.board.state.deck.cards.len() > 2
    {
        return 0.0;
    }

    let actor = runner.framework.current_player;
    let has_converted_product =
        runner
            .framework
            .board
            .state
            .bl_to_building
            .values()
            .any(|building| {
                building.owner.as_usize() == actor
                    && building.flipped
                    && is_saleable_industry(building.industry)
            });
    let Some(build_location) = intent.selected_build_location else {
        return 0.0;
    };
    let Some(building) = after_runner
        .framework
        .board
        .state
        .bl_to_building
        .get(&build_location)
    else {
        return 0.0;
    };
    if building.owner.as_usize() != actor || building.flipped {
        return 0.0;
    }
    let cards_after_build = after_runner.framework.board.state.players[actor]
        .hand
        .cards
        .len();
    let after_debt = u16::try_from((-after.income_amount).max(0)).unwrap_or(u16::MAX);
    let required_future_actions = minimum_future_actions_to_sell_product(
        after_runner,
        actor,
        build_location,
        building.industry,
    );

    post_sale_product_cycle_bonus_from_inputs(PostSaleProductCycleInputs {
        railroad: true,
        low_score: true,
        late_deck: true,
        action_type: intent.action_type,
        selected_industry: intent.selected_industry,
        before_income: before.income_amount,
        before_sellable_units: before.sellable_units,
        after_sellable_units: after.sellable_units,
        has_converted_product,
        cards_after_build,
        required_future_actions,
        after_money: after.money,
        after_debt,
    })
}

#[derive(Debug, Clone, Copy)]
struct MatureProductCycleInputs {
    base_cycle_bonus: f64,
    before_income: i8,
    cards_after_build: usize,
}

fn mature_product_cycle_bonus_from_inputs(input: MatureProductCycleInputs) -> f64 {
    if input.base_cycle_bonus > 0.0 && input.before_income >= 10 && input.cards_after_build >= 5 {
        1.0
    } else {
        0.0
    }
}

fn mature_product_cycle_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let cards_after_build = after_runner.framework.board.state.players[actor]
        .hand
        .cards
        .len();
    if before.income_amount < 10 || cards_after_build < 5 {
        return 0.0;
    }
    mature_product_cycle_bonus_from_inputs(MatureProductCycleInputs {
        base_cycle_bonus: post_sale_product_cycle_signal(
            runner,
            after_runner,
            before,
            after,
            intent,
        ),
        before_income: before.income_amount,
        cards_after_build,
    })
}

#[derive(Debug, Clone, Copy)]
struct FinancedProductCycleInputs {
    railroad: bool,
    deck_empty: bool,
    low_score: bool,
    first_action_product_build: bool,
    no_existing_product: bool,
    no_sell_after_build: bool,
    loan_income_loss: i16,
    cards_after_loan: usize,
}

fn financed_product_cycle_is_eligible(input: FinancedProductCycleInputs) -> bool {
    input.railroad
        && input.deck_empty
        && input.low_score
        && input.first_action_product_build
        && input.no_existing_product
        && input.no_sell_after_build
        && (0..=3).contains(&input.loan_income_loss)
        && input.cards_after_loan >= 2
}

fn advance_to_actor_without_opponent_actions(runner: &mut GameRunner, actor: usize) -> bool {
    let player_count = runner.framework.board.state.players.len();
    // The actor can move from first in the current order to last in the next
    // order, so reaching its next turn may cross two sets of opponent turns.
    for _ in 0..=player_count.saturating_mul(2) {
        if runner.is_game_finished()
            || runner.framework.current_session().is_some()
            || runner.has_pending_shortfall()
        {
            return false;
        }
        if runner.framework.current_player == actor {
            if !runner.turn_started {
                runner.start_turn();
            }
            return runner.framework.current_player == actor
                && runner.turn_started
                && runner.actions_remaining_in_turn == 2;
        }

        if !runner.turn_started {
            runner.start_turn();
        }
        runner.actions_remaining_in_turn = 0;
        runner.end_turn();
        if runner.turn_started {
            return false;
        }
        if !runner.is_game_finished() && !runner.has_pending_shortfall() {
            runner.start_turn();
        }
    }
    false
}

/// Recognize a precise four-card financing sequence without rolling an
/// opponent or the game to completion:
///
/// Build product -> cheap Loan -> next-round Route -> complete Sell using
/// actor-owned Beer.
///
/// The positive signal is the actual Sell successor score. This makes a
/// shorter, higher-value location outrank a superficially cheaper build. Any
/// uncertainty is a failed proof: unlike pruning helpers, this function never
/// treats an enumeration error as evidence for a bonus.
fn financed_product_cycle_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
    config: &RuleDecisionConfig,
) -> f64 {
    let actor = runner.framework.current_player;
    let first_action_product_build = runner.actions_remaining_in_turn == 2
        && after_runner.framework.current_player == actor
        && after_runner.actions_remaining_in_turn == 1
        && intent.action_type == ActionType::BuildBuilding
        && intent.selected_industry.is_some_and(is_saleable_industry)
        && after.sellable_units > before.sellable_units;
    if runner.game_phase != GamePhase::Railroad
        || !runner.framework.board.state.deck.cards.is_empty()
        || before.victory_points >= 20.0
        || !first_action_product_build
        || before.sellable_units > 0
        || before.available_sell_targets > 0
        || after.available_sell_targets > 0
        || after_runner.framework.current_session().is_some()
        || after_runner.has_pending_shortfall()
    {
        return 0.0;
    }
    let Some(build_location) = intent.selected_build_location else {
        return 0.0;
    };

    let loans = match enumerate_legal_actions_for_root(after_runner, ActionType::Loan) {
        Ok(actions) => actions,
        Err(_) => return 0.0,
    };
    let mut seen_loans = HashSet::new();
    let mut best_sell_score = 0.0_f64;
    for loan in loans {
        if !seen_loans.insert(loan.card_invariant_key()) {
            continue;
        }
        let mut after_loan = after_runner.clone();
        if loan.apply(&mut after_loan).is_err() {
            continue;
        }
        let loan_player = &after_loan.framework.board.state.players[actor];
        let loan_income = loan_player.get_income_amount(loan_player.income_level);
        let loan_income_loss = i16::from(after.income_amount) - i16::from(loan_income);
        let inputs = FinancedProductCycleInputs {
            railroad: true,
            deck_empty: true,
            low_score: true,
            first_action_product_build,
            no_existing_product: true,
            no_sell_after_build: true,
            loan_income_loss,
            cards_after_loan: loan_player.hand.cards.len(),
        };
        if !financed_product_cycle_is_eligible(inputs)
            || after_loan.framework.current_session().is_some()
            || after_loan.has_pending_shortfall()
        {
            continue;
        }
        if !advance_to_actor_without_opponent_actions(&mut after_loan, actor) {
            continue;
        }

        let routes = match enumerate_legal_actions_for_root(&after_loan, ActionType::BuildRailroad)
        {
            Ok(actions) => actions,
            Err(_) => continue,
        };
        let mut seen_routes = HashSet::new();
        for route in routes {
            if !seen_routes.insert(route.card_invariant_key()) {
                continue;
            }
            let mut after_route = after_loan.clone();
            if route.apply(&mut after_route).is_err()
                || after_route.framework.current_player != actor
                || after_route.actions_remaining_in_turn != 1
                || !complete_batch_sale_is_legal(&after_route, actor, BatchBeerScope::OwnedOnly)
            {
                continue;
            }
            let sells = match enumerate_legal_actions_for_root(&after_route, ActionType::Sell) {
                Ok(actions) => actions,
                Err(_) => continue,
            };
            let alternatives = sells.len();
            let sell_before = snapshot(&after_route, actor);
            let card_values = card_future_values(&after_route, actor);
            let mut seen_sells = HashSet::new();
            for sell in sells {
                if !seen_sells.insert(sell.card_invariant_key()) {
                    continue;
                }
                let mut after_sell_runner = after_route.clone();
                if sell.apply(&mut after_sell_runner).is_err()
                    || !after_sell_runner
                        .framework
                        .board
                        .state
                        .bl_to_building
                        .get(&build_location)
                        .is_some_and(|building| {
                            building.owner.as_usize() == actor && building.flipped
                        })
                {
                    continue;
                }
                let after_route_data = route_data_after_action(
                    &after_route,
                    &after_sell_runner,
                    actor,
                    &sell_before,
                    &sell.intent,
                );
                let sell_after =
                    snapshot_with_route_data(&after_sell_runner, actor, after_route_data);
                if sell_after.sellable_units != 0 {
                    continue;
                }
                let effect = immediate_effect_between(&after_route, &after_sell_runner);
                let (_, sell_score) = score_action_with_card_values(
                    &after_route,
                    &after_sell_runner,
                    &sell_before,
                    &sell_after,
                    &effect,
                    &sell.intent,
                    alternatives,
                    config,
                    &card_values,
                );
                best_sell_score = best_sell_score.max(sell_score.max(0.0));
            }
        }
    }
    best_sell_score
}

#[derive(Debug, Clone, Copy)]
struct ContestedExternalBeerPlanInputs {
    railroad: bool,
    low_score: bool,
    late_deck: bool,
    first_action_product_build: bool,
    no_sell_after_build: bool,
    route_opens_sell: bool,
    route_ends_turn: bool,
    own_beer_shortfall: bool,
    opponent_competes_for_external_beer: bool,
}

fn contested_external_beer_plan_penalty_from_inputs(input: ContestedExternalBeerPlanInputs) -> f64 {
    if input.railroad
        && input.low_score
        && input.late_deck
        && input.first_action_product_build
        && input.no_sell_after_build
        && input.route_opens_sell
        && input.route_ends_turn
        && input.own_beer_shortfall
        && input.opponent_competes_for_external_beer
    {
        -1.0
    } else {
        0.0
    }
}

fn owned_unflipped_beer_units(state: &crate::board::state::BoardState, player: usize) -> u16 {
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

fn opponent_competes_for_external_beer(
    runner: &GameRunner,
    actor: usize,
    actor_sell_option: &crate::actions::SellOption,
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
            let own_beer = owned_unflipped_beer_units(state, opponent);
            state.player_building_mask[opponent]
                .ones()
                .any(|product_location| {
                    let Some(building) = state.bl_to_building.get(&product_location) else {
                        return false;
                    };
                    if building.flipped || !is_saleable_industry(building.industry) {
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
                    let matching_merchants = merchant_target_mask(state, building.industry);
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
                            let trade_post = crate::market::merchants::slot_to_trade_post(slot)
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

/// Detect a late Railroad product plan whose only immediate route completion
/// leaves its Sell exposed to an opponent turn and to the same external Beer.
/// The signal simulates only the actor's remaining route action; it does not
/// choose or roll out an opponent move.
fn contested_external_beer_plan_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let first_action_product_build = intent.action_type == ActionType::BuildBuilding
        && intent.selected_industry.is_some_and(is_saleable_industry)
        && after.sellable_units > before.sellable_units
        && after_runner.framework.current_player == actor
        && after_runner.actions_remaining_in_turn == 1;
    let late_deck = runner.framework.board.state.deck.cards.len() <= 2;
    if runner.game_phase != GamePhase::Railroad
        || before.victory_points >= 20.0
        || !late_deck
        || !first_action_product_build
    {
        return 0.0;
    }

    let Some(build_location) = intent.selected_build_location else {
        return 0.0;
    };
    let Some(building) = after_runner
        .framework
        .board
        .state
        .bl_to_building
        .get(&build_location)
    else {
        return 0.0;
    };
    if building.owner.as_usize() != actor || building.flipped {
        return 0.0;
    }
    let beer_needed = u16::from(
        INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()].beer_needed,
    );
    let own_beer = owned_unflipped_beer_units(&after_runner.framework.board.state, actor);
    let no_sell_after_build = after_runner
        .framework
        .board
        .get_valid_sell_options(actor)
        .is_empty();
    if beer_needed == 0 || own_beer >= beer_needed || !no_sell_after_build {
        return 0.0;
    }

    let Ok(replies) = enumerate_legal_actions(after_runner) else {
        return 0.0;
    };
    let mut route_opens_sell = false;
    let mut all_open_routes_are_contested = true;
    for reply in replies.into_iter().filter(|reply| {
        matches!(
            reply.root,
            ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
        )
    }) {
        let mut route_runner = after_runner.clone();
        if reply.apply(&mut route_runner).is_err() {
            continue;
        }
        for option in route_runner
            .framework
            .board
            .get_valid_sell_options(actor)
            .into_iter()
            .filter(|option| option.location == build_location)
        {
            route_opens_sell = true;
            if !opponent_competes_for_external_beer(&route_runner, actor, &option) {
                all_open_routes_are_contested = false;
            }
        }
    }

    contested_external_beer_plan_penalty_from_inputs(ContestedExternalBeerPlanInputs {
        railroad: true,
        low_score: true,
        late_deck,
        first_action_product_build,
        no_sell_after_build,
        route_opens_sell,
        route_ends_turn: true,
        own_beer_shortfall: own_beer < beer_needed,
        opponent_competes_for_external_beer: route_opens_sell && all_open_routes_are_contested,
    })
}

#[derive(Debug, Clone, Copy)]
struct SameTurnSaleGuardInputs {
    action_type: ActionType,
    selected_industry: Option<IndustryType>,
    before_sell_targets: usize,
    after_sell_targets: usize,
    same_turn_completion_safe: bool,
    after_income: i8,
    after_money: u16,
    safety_weight: f64,
    economic_guard_weight: f64,
}

fn same_turn_sale_guard_rebate_from_inputs(input: SameTurnSaleGuardInputs) -> f64 {
    if input.action_type != ActionType::BuildBuilding
        || !input.selected_industry.is_some_and(is_saleable_industry)
        || input.before_sell_targets > 0
        || input.after_sell_targets == 0
        || !input.same_turn_completion_safe
    {
        return 0.0;
    }

    -input.safety_weight * settlement_safety_score(input.after_income, input.after_money)
        - input.economic_guard_weight
            * imminent_settlement_guard_penalty(input.after_income, input.after_money)
}

#[derive(Debug, Clone, Copy)]
struct LateCanalStallInputs {
    last_two_card_window: bool,
    low_score: bool,
    severe_debt: bool,
    action_type: ActionType,
    secured_vp: bool,
    actor_flipped_building: bool,
    income_improved: bool,
    new_legal_sell: bool,
    route_progress: bool,
}

fn late_canal_stall_penalty_from_inputs(input: LateCanalStallInputs) -> f64 {
    if !input.last_two_card_window
        || !input.low_score
        || !input.severe_debt
        || input.action_type == ActionType::Pass
        || input.secured_vp
        || input.actor_flipped_building
        || input.income_improved
        || input.new_legal_sell
        || input.route_progress
    {
        0.0
    } else {
        -1.0
    }
}

fn late_canal_stall_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let last_two_card_window = runner.game_phase == GamePhase::Canal
        && after_runner.game_phase == GamePhase::Canal
        && runner.framework.board.state.deck.cards.is_empty()
        && runner.framework.board.state.players[actor].hand.cards.len() <= 2;
    let secured_vp = effect
        .player_victory_points_delta
        .get(actor)
        .copied()
        .unwrap_or(0)
        > 0;
    let before_options = productive_options(runner, actor);
    let after_options = productive_options(after_runner, actor);
    late_canal_stall_penalty_from_inputs(LateCanalStallInputs {
        last_two_card_window,
        low_score: before.victory_points < 35.0,
        severe_debt: before.income_amount <= -8,
        action_type: intent.action_type,
        secured_vp,
        actor_flipped_building: after.flipped_buildings > before.flipped_buildings,
        income_improved: after.income_amount > before.income_amount,
        new_legal_sell: after_options.sell > before_options.sell,
        route_progress: after.sale_route_distance + f64::EPSILON < before.sale_route_distance,
    })
}

#[derive(Debug, Clone, Copy)]
struct LateRailroadBacklogInputs {
    railroad: bool,
    deck_empty: bool,
    low_score: bool,
    cards_before: usize,
    action_type: ActionType,
    selected_industry: Option<IndustryType>,
    before_sellable_units: usize,
    after_sellable_units: usize,
    before_sell_targets: usize,
    after_sell_targets: usize,
    secured_batch_beer: bool,
    actor_flipped_building: bool,
    income_improved: bool,
}

fn late_railroad_backlog_penalty_from_inputs(input: LateRailroadBacklogInputs) -> f64 {
    if input.railroad
        && input.deck_empty
        && input.low_score
        && input.cards_before <= 4
        && input.action_type == ActionType::BuildBuilding
        && input.selected_industry.is_some_and(is_saleable_industry)
        && input.before_sellable_units >= 2
        && input.after_sellable_units > input.before_sellable_units
        && input.before_sell_targets == 0
        && input.after_sell_targets == 0
        && !input.secured_batch_beer
        && !input.actor_flipped_building
        && !input.income_improved
    {
        -1.0
    } else {
        0.0
    }
}

/// Redirect only an unfunded late product backlog. A third product remains a
/// valid action-saving batch when the actor's Beer covers every pending sale;
/// otherwise it is priced when it creates neither a Sell nor economic gain.
fn late_railroad_backlog_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let could_be_penalized = runner.game_phase == GamePhase::Railroad
        && runner.framework.board.state.deck.cards.is_empty()
        && before.victory_points < 35.0
        && runner.framework.board.state.players[actor].hand.cards.len() <= 4
        && intent.action_type == ActionType::BuildBuilding
        && intent.selected_industry.is_some_and(is_saleable_industry)
        && before.sellable_units >= 2
        && after.sellable_units > before.sellable_units
        && before.available_sell_targets == 0
        && after.available_sell_targets == 0
        && after.flipped_buildings <= before.flipped_buildings
        && after.income_amount <= before.income_amount;
    late_railroad_backlog_penalty_from_inputs(LateRailroadBacklogInputs {
        railroad: runner.game_phase == GamePhase::Railroad,
        deck_empty: runner.framework.board.state.deck.cards.is_empty(),
        low_score: before.victory_points < 35.0,
        cards_before: runner.framework.board.state.players[actor].hand.cards.len(),
        action_type: intent.action_type,
        selected_industry: intent.selected_industry,
        before_sellable_units: before.sellable_units,
        after_sellable_units: after.sellable_units,
        before_sell_targets: before.available_sell_targets,
        after_sell_targets: after.available_sell_targets,
        secured_batch_beer: could_be_penalized
            && batch_sale_plan_is_feasible(after_runner, actor, BatchBeerScope::OwnedOnly),
        actor_flipped_building: after.flipped_buildings > before.flipped_buildings,
        income_improved: after.income_amount > before.income_amount,
    })
}

#[derive(Debug, Clone, Copy)]
struct BeerBacklogRecoveryInputs {
    railroad: bool,
    low_score: bool,
    development_action: bool,
    develops_beer: bool,
    existing_unsold_products: usize,
    before_sell_targets: usize,
    beer_shortfall: u32,
    cards_after: usize,
    opens_affordable_brewery: bool,
}

fn beer_backlog_recovery_bonus_from_inputs(input: BeerBacklogRecoveryInputs) -> f64 {
    if input.railroad
        && input.low_score
        && input.development_action
        && input.develops_beer
        && input.existing_unsold_products >= 2
        && input.before_sell_targets == 0
        && input.beer_shortfall >= 3
        && input.cards_after >= 5
        && input.opens_affordable_brewery
    {
        1.0
    } else {
        0.0
    }
}

/// Repair a specific technology mismatch instead of rewarding Beer development
/// generically. The actor must already own a multi-product backlog whose Beer
/// demand cannot be covered, and this exact development must turn an
/// unavailable Brewery build into an affordable one. Zero-Beer Goods and
/// batches that already have enough Beer therefore receive no extra score.
fn beer_backlog_recovery_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let development_action = matches!(
        intent.action_type,
        ActionType::Develop | ActionType::DevelopDouble
    );
    let develops_beer = [intent.selected_industry, intent.selected_second_industry]
        .into_iter()
        .flatten()
        .any(|industry| industry == IndustryType::Beer);
    let beer_shortfall = before
        .sellable_beer_demand
        .saturating_sub(before.unflipped_beer_units);
    let affordable_brewery = |candidate: &GameRunner, money: u16| {
        candidate
            .framework
            .board
            .get_valid_build_options(actor)
            .into_iter()
            .any(|option| {
                option.industry_type == IndustryType::Beer && money >= option.total_money_cost
            })
    };
    let opens_affordable_brewery = development_action
        && develops_beer
        && !affordable_brewery(runner, before.money)
        && affordable_brewery(after_runner, after.money);

    beer_backlog_recovery_bonus_from_inputs(BeerBacklogRecoveryInputs {
        railroad: runner.game_phase == GamePhase::Railroad,
        low_score: before.victory_points < 35.0,
        development_action,
        develops_beer,
        existing_unsold_products: before.sellable_units,
        before_sell_targets: before.available_sell_targets,
        beer_shortfall,
        cards_after: after_runner.framework.board.state.players[actor]
            .hand
            .cards
            .len(),
        opens_affordable_brewery,
    })
}

#[derive(Debug, Clone, Copy)]
struct LateRailroadConversionStallInputs {
    railroad: bool,
    deck_empty: bool,
    low_score: bool,
    cards_before: usize,
    action_type: ActionType,
    selected_industry: Option<IndustryType>,
    existing_unsold_product: bool,
    before_sell_targets: usize,
    after_sell_targets: usize,
    secured_vp: bool,
    actor_flipped_building: bool,
    income_improved: bool,
    route_progress: bool,
    beer_shortfall_reduced: bool,
    batch_sale_feasible: bool,
}

fn late_railroad_conversion_stall_penalty_from_inputs(
    input: LateRailroadConversionStallInputs,
) -> f64 {
    let consumes_setup_card = matches!(
        input.action_type,
        ActionType::Develop | ActionType::DevelopDouble
    ) || (input.action_type == ActionType::BuildBuilding
        && input.selected_industry.is_some_and(is_saleable_industry));
    if input.railroad
        && input.deck_empty
        && input.low_score
        && input.cards_before <= 3
        && consumes_setup_card
        && input.existing_unsold_product
        && input.before_sell_targets == 0
        && input.after_sell_targets == 0
        && !input.secured_vp
        && !input.actor_flipped_building
        && !input.income_improved
        && !input.route_progress
        && !input.beer_shortfall_reduced
        && !input.batch_sale_feasible
    {
        -1.0
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchBeerScope {
    /// A secured batch cannot lose its sale Beer to another player.
    OwnedOnly,
    /// A feasibility proof may use every source accepted by the Sell rules.
    AnyLegal,
}

#[derive(Debug, Clone)]
struct BatchBeerRequirement {
    needed: u8,
    sources: Vec<usize>,
}

fn allocate_batch_beer(
    requirements: &[BatchBeerRequirement],
    requirement_idx: usize,
    capacities: &mut [u8],
) -> bool {
    if requirement_idx == requirements.len() {
        return true;
    }
    let requirement = &requirements[requirement_idx];
    allocate_batch_beer_units(
        requirements,
        requirement_idx,
        requirement.needed,
        0,
        capacities,
    )
}

fn allocate_batch_beer_units(
    requirements: &[BatchBeerRequirement],
    requirement_idx: usize,
    units_left: u8,
    source_start: usize,
    capacities: &mut [u8],
) -> bool {
    if units_left == 0 {
        return allocate_batch_beer(requirements, requirement_idx + 1, capacities);
    }
    let sources = &requirements[requirement_idx].sources;
    for source_position in source_start..sources.len() {
        let source = sources[source_position];
        if capacities.get(source).copied().unwrap_or_default() == 0 {
            continue;
        }
        capacities[source] -= 1;
        if allocate_batch_beer_units(
            requirements,
            requirement_idx,
            units_left - 1,
            source_position,
            capacities,
        ) {
            capacities[source] += 1;
            return true;
        }
        capacities[source] += 1;
    }
    false
}

/// Check whether one legal Sell can flip every pending product. Beer is
/// allocated per product and per source, so a merchant barrel or Brewery cube
/// cannot be counted twice across a multi-product action.
fn complete_batch_sale_is_legal(runner: &GameRunner, actor: usize, scope: BatchBeerScope) -> bool {
    let board = &runner.framework.board;
    let state = &board.state;
    let pending_products = (0..N_BL)
        .filter(|location| {
            state.bl_to_building.get(location).is_some_and(|building| {
                building.owner.as_usize() == actor
                    && !building.flipped
                    && is_saleable_industry(building.industry)
            })
        })
        .collect::<Vec<_>>();
    if pending_products.is_empty() {
        return false;
    }

    let sell_options = board.get_valid_sell_options(actor);
    let free_development_available = board
        .get_valid_free_development_options(actor)
        .count_ones(..)
        > 0;
    let mut capacities = vec![0u8; N_BL + state.trade_post_slots.len()];
    let mut requirements = Vec::with_capacity(pending_products.len());
    for product_location in pending_products {
        let Some(option) = sell_options
            .iter()
            .find(|option| option.location == product_location)
        else {
            return false;
        };
        let building = state
            .bl_to_building
            .get(&product_location)
            .expect("pending product disappeared while checking its Sell option");
        let needed =
            INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()].beer_needed;
        let mut sources = Vec::new();
        for source in option.beer_locations.ones() {
            let capacity = if source < N_BL {
                state
                    .bl_to_building
                    .get(&source)
                    .filter(|beer| {
                        beer.industry == IndustryType::Beer
                            && !beer.flipped
                            && beer.resource_amt > 0
                            && (scope == BatchBeerScope::AnyLegal || beer.owner.as_usize() == actor)
                    })
                    .map_or(0, |beer| beer.resource_amt)
            } else {
                if scope == BatchBeerScope::OwnedOnly {
                    0
                } else {
                    let slot = source - N_BL;
                    let free_development_source = crate::market::merchants::TRADE_POST_TO_BONUS
                        [crate::market::merchants::slot_to_trade_post(slot).to_index()]
                        == crate::market::merchants::TradePostBonus::FreeDevelopment;
                    u8::from(
                        state.trade_post_beer.contains(slot)
                            && (!free_development_source || free_development_available),
                    )
                }
            };
            if capacity > 0 {
                capacities[source] = capacity;
                sources.push(source);
            }
        }
        if needed > 0 && sources.is_empty() {
            return false;
        }
        requirements.push(BatchBeerRequirement { needed, sources });
    }

    requirements.sort_by(|left, right| {
        left.sources
            .len()
            .cmp(&right.sources.len())
            .then_with(|| right.needed.cmp(&left.needed))
    });
    allocate_batch_beer(&requirements, 0, &mut capacities)
}

/// Prove a complete batch sale within the actor's remaining card budget. The
/// setup frontier includes real rail actions and actor-owned Brewery builds,
/// so a multi-product batch is preserved only when the missing Beer can
/// actually be built before the reserved Sell card. The planning clone keeps
/// only this actor's actions adjacent; opponent Beer races are handled by the
/// separate contested-Beer rule.
fn batch_sale_plan_is_feasible(
    after_runner: &GameRunner,
    actor: usize,
    scope: BatchBeerScope,
) -> bool {
    let cards_after = after_runner.framework.board.state.players[actor]
        .hand
        .cards
        .len();
    if cards_after == 0 || after_runner.game_phase == GamePhase::GameEnd {
        return false;
    }
    if complete_batch_sale_is_legal(after_runner, actor, scope) {
        return true;
    }
    if cards_after < 2 || after_runner.game_phase != GamePhase::Railroad {
        return false;
    }
    if after_runner.framework.current_session().is_some() || after_runner.has_pending_shortfall() {
        // Only prune a plan we can prove impossible from a decision boundary.
        return true;
    }

    let mut planning_runner = after_runner.clone();
    planning_runner.framework.current_player = actor;
    planning_runner.actions_remaining_in_turn = cards_after.min(u8::MAX as usize) as u8;
    planning_runner.turn_started = true;
    let mut frontier = vec![planning_runner];
    for _ in 0..cards_after.saturating_sub(1) {
        let mut next_frontier = Vec::new();
        for position in frontier {
            let mut actions =
                match enumerate_legal_actions_for_root(&position, ActionType::BuildRailroad) {
                    Ok(actions) => actions,
                    Err(_) => return true,
                };
            let brewery_actions =
                match enumerate_legal_actions_for_root(&position, ActionType::BuildBuilding) {
                    Ok(actions) => actions,
                    Err(_) => return true,
                };
            actions.extend(brewery_actions.into_iter().filter(|action| {
                action.intent.action_type == ActionType::BuildBuilding
                    && action.intent.selected_industry == Some(IndustryType::Beer)
            }));
            let mut seen_actions = HashSet::new();
            for action in actions {
                let is_route = matches!(
                    action.intent.action_type,
                    ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
                );
                let is_brewery = action.intent.action_type == ActionType::BuildBuilding
                    && action.intent.selected_industry == Some(IndustryType::Beer);
                // A route may discard the only card that can build the missing
                // Brewery. Preserve strategically distinct discard cards in
                // this short (at most four-card) planning frontier.
                if (!is_route && !is_brewery) || !seen_actions.insert(action.key()) {
                    continue;
                }
                let mut successor = position.clone();
                if action.apply(&mut successor).is_err() {
                    continue;
                }
                if complete_batch_sale_is_legal(&successor, actor, scope) {
                    return true;
                }
                next_frontier.push(successor);
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    false
}

/// In the final three-card window, development and extra products must advance
/// an existing sale plan. A Beer-funded multi-product batch is explicitly
/// preserved because one Sell action can flip the whole batch.
fn late_railroad_conversion_stall_signal(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let before_beer_shortfall = before
        .sellable_beer_demand
        .saturating_sub(before.unflipped_beer_units);
    let after_beer_shortfall = after
        .sellable_beer_demand
        .saturating_sub(after.unflipped_beer_units);
    let consumes_setup_card = matches!(
        intent.action_type,
        ActionType::Develop | ActionType::DevelopDouble
    ) || (intent.action_type == ActionType::BuildBuilding
        && intent.selected_industry.is_some_and(is_saleable_industry));
    let could_be_penalized = runner.game_phase == GamePhase::Railroad
        && runner.framework.board.state.deck.cards.is_empty()
        && before.victory_points < 35.0
        && runner.framework.board.state.players[actor].hand.cards.len() <= 3
        && consumes_setup_card
        && before.sellable_units > 0
        && before.available_sell_targets == 0
        && after.available_sell_targets == 0
        && actor_delta(&effect.player_victory_points_delta, actor) <= 0.0
        && after.flipped_buildings <= before.flipped_buildings
        && after.income_amount <= before.income_amount
        && after.sale_route_distance + f64::EPSILON >= before.sale_route_distance
        && after_beer_shortfall >= before_beer_shortfall;
    late_railroad_conversion_stall_penalty_from_inputs(LateRailroadConversionStallInputs {
        railroad: runner.game_phase == GamePhase::Railroad,
        deck_empty: runner.framework.board.state.deck.cards.is_empty(),
        low_score: before.victory_points < 35.0,
        cards_before: runner.framework.board.state.players[actor].hand.cards.len(),
        action_type: intent.action_type,
        selected_industry: intent.selected_industry,
        existing_unsold_product: before.sellable_units > 0,
        before_sell_targets: before.available_sell_targets,
        after_sell_targets: after.available_sell_targets,
        secured_vp: actor_delta(&effect.player_victory_points_delta, actor) > 0.0,
        actor_flipped_building: after.flipped_buildings > before.flipped_buildings,
        income_improved: after.income_amount > before.income_amount,
        route_progress: after.sale_route_distance + f64::EPSILON < before.sale_route_distance,
        beer_shortfall_reduced: after_beer_shortfall < before_beer_shortfall,
        batch_sale_feasible: could_be_penalized
            && batch_sale_plan_is_feasible(after_runner, actor, BatchBeerScope::AnyLegal),
    })
}

fn has_safe_same_turn_sale_completion(
    after_runner: &GameRunner,
    actor: usize,
    build_location: usize,
) -> bool {
    if after_runner.framework.current_player != actor || after_runner.actions_remaining_in_turn == 0
    {
        return false;
    }
    let Ok(replies) = enumerate_legal_actions(after_runner) else {
        return false;
    };
    replies
        .into_iter()
        .filter(|reply| {
            reply.root == ActionType::Sell
                && reply
                    .intent
                    .sell_choices
                    .iter()
                    .any(|choice| choice.location == build_location)
        })
        .any(|reply| {
            let mut completed = after_runner.clone();
            if reply.apply(&mut completed).is_err() {
                return false;
            }
            let phase_changed = completed.game_phase != after_runner.game_phase;
            if !completed.take_shortfall_sessions().is_empty() {
                return false;
            }
            let player = &completed.framework.board.state.players[actor];
            let income = player.get_income_amount(player.income_level);
            let debt = u16::try_from((-income).max(0)).unwrap_or(u16::MAX);
            phase_changed || player.money >= debt
        })
}

/// A Build on the first action of a turn is not settled before the second
/// action. If that exact successor already has a complete legal Sell which
/// safely resolves the temporary deficit, refund only the two premature
/// reserve penalties. The Sell keeps its own normal score and must still win
/// the following decision.
fn same_turn_sale_guard_rebate(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
    safety_weight: f64,
    economic_guard_weight: f64,
) -> f64 {
    let Some(build_location) = intent.selected_build_location else {
        return 0.0;
    };
    let actor = runner.framework.current_player;
    same_turn_sale_guard_rebate_from_inputs(SameTurnSaleGuardInputs {
        action_type: intent.action_type,
        selected_industry: intent.selected_industry,
        before_sell_targets: before.available_sell_targets,
        after_sell_targets: after.available_sell_targets,
        same_turn_completion_safe: has_safe_same_turn_sale_completion(
            after_runner,
            actor,
            build_location,
        ),
        after_income: after.income_amount,
        after_money: after.money,
        safety_weight,
        economic_guard_weight,
    })
}

fn development_is_meaningful(
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> bool {
    let industries = [intent.selected_industry, intent.selected_second_industry];
    industries.into_iter().flatten().any(|industry| {
        let index = industry.as_usize();
        let level_up = after.industry_levels[index] > before.industry_levels[index];
        let remaining_before = before.industry_remaining[index];
        let remaining_after = after.industry_remaining[index];
        level_up
            || (remaining_after < remaining_before
                && (remaining_after <= 1 || remaining_before <= 2))
    })
}

/// Discrete lifecycle gates for the CPU tree.
///
/// The continuous state deltas (cash, income, industry and network) are good
/// tie-breakers, but they can still produce a locally attractive engine move
/// when the player has a concrete conversion available.  These small gates
/// encode the order a strong human normally follows: convert a ready tile,
/// connect an unconverted tile to a merchant, then invest in the next engine
/// step.  They are intentionally independent of terminal scoring.
fn lifecycle_priority_adjustment(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let route_progress = before.sale_route_distance - after.sale_route_distance;
    let new_sell_targets = after
        .available_sell_targets
        .saturating_sub(before.available_sell_targets) as f64;
    let ready_sale = before.available_sell_targets > 0;
    let has_unconverted_sale = before.sellable_units > 0;
    let flipped = effect.flipped_buildings_delta.max(0) as f64;
    let build_options = productive_options(runner, actor);
    let development_opens_frontier =
        development_improves_build_frontier(runner, after_runner, actor);

    match intent.action_type {
        ActionType::Sell => {
            if flipped > 0.0 {
                // A sale is the only action that turns a prepared product into
                // secured progress.  Keep this above any bounded income gain.
                8.0 + flipped * 2.0 + (after.sellable_vp - before.sellable_vp).abs() * 0.05
            } else {
                -3.0
            }
        }
        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad => {
            if ready_sale {
                -5.0
            } else if has_unconverted_sale && (new_sell_targets > 0.0 || route_progress > 0.0) {
                3.5 + new_sell_targets.min(2.0) * 2.0 + route_progress.min(3.0) * 0.8
            } else if before.sellable_units == 0 {
                // A road without a product or a build frontier is usually a
                // sunk action in the canal era.
                if runner.game_phase == GamePhase::Canal {
                    -4.0
                } else {
                    -1.0
                }
            } else {
                -1.5
            }
        }
        ActionType::BuildBuilding => {
            if ready_sale {
                -4.0
            } else if has_unconverted_sale
                && intent.selected_industry == Some(IndustryType::Beer)
                && before.sellable_beer_demand > before.unflipped_beer_units
            {
                // Beer is a conversion resource here, not a generic income
                // tile.  Reward it only while an owned sale actually needs it.
                2.5
            } else if before.buildings == 0 {
                1.5
            } else {
                0.0
            }
        }
        ActionType::Develop | ActionType::DevelopDouble => {
            if ready_sale {
                -4.0
            } else if before.buildings > 0
                && build_options.has_build()
                && !development_opens_frontier
            {
                // Once a player has an industry and a legal build, a mat
                // move that does not unlock a stronger build is usually an
                // avoidable tempo loss. Keep this large enough to beat the
                // bounded income signal from consuming a resource tile.
                -18.0
            } else if development_is_meaningful(before, after, intent) {
                1.5
            } else {
                -2.0
            }
        }
        ActionType::Loan => {
            // Loan scoring has a more detailed affordability check; this gate
            // only prevents borrowing from outranking a ready conversion.
            if ready_sale {
                -6.0
            } else {
                0.0
            }
        }
        ActionType::Scout | ActionType::Pass => {
            if ready_sale {
                -3.0
            } else {
                0.0
            }
        }
    }
}

/// A small set of ordered tree gates sits above the continuous score. These
/// encode the durable human pattern in Brass: establish cash for a productive
/// opening, then spend actions on an industry; do not burn a second action on
/// the same mat unless it actually reaches a useful threshold.
fn strategic_priority_adjustment(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let pre_engine =
        runner.game_phase == GamePhase::Canal && before.buildings == 0 && before.own_roads == 0;

    // A strong opening often spends the first funded turn finishing a
    // saleable mat tier before placing the first tile.  The continuous score
    // sees only a small iron/cash delta here, so make the conversion explicit:
    // reward a meaningful Cotton/Goods/Pottery frontier *only* when the
    // resulting position still has an affordable build.  This is a local tree
    // rule, not a terminal rollout, and it cannot make an endless develop loop
    // attractive because the frontier must actually improve.
    if pre_engine
        && matches!(
            intent.action_type,
            ActionType::Develop | ActionType::DevelopDouble
        )
        && development_improves_build_frontier(runner, after_runner, actor)
    {
        let mut saleable_level_ups = 0usize;
        let mut saleable_near_tiers = 0usize;
        for industry in [intent.selected_industry, intent.selected_second_industry]
            .into_iter()
            .flatten()
        {
            if !is_saleable_industry(industry) {
                continue;
            }
            let index = industry.as_usize();
            if after.industry_levels[index] > before.industry_levels[index] {
                saleable_level_ups += 1;
            } else if after.industry_remaining[index] <= 1
                && after.industry_remaining[index] < before.industry_remaining[index]
            {
                saleable_near_tiers += 1;
            }
        }
        let after_options = productive_options(after_runner, actor);
        let affordable_build = after_options
            .minimum_build_cost
            .is_some_and(|cost| after.money >= cost.saturating_add(2));
        if affordable_build && (saleable_level_ups > 0 || saleable_near_tiers > 0) {
            return 6.0 + saleable_level_ups as f64 * 2.5 + saleable_near_tiers as f64 * 1.5;
        }
    }

    let opening = runner.game_phase == GamePhase::Canal
        && runner.personal_turns_taken.get(actor).copied().unwrap_or(0) == 0
        && before.buildings == 0
        && before.own_roads == 0;
    if !opening {
        return 0.0;
    }

    if !matches!(
        intent.action_type,
        ActionType::Loan | ActionType::Develop | ActionType::DevelopDouble
    ) {
        return 0.0;
    }

    // The first personal turn starts with only one action. If a loan creates
    // enough cash for a legal build, it is normally the highest-leverage
    // opening and should beat a merely available develop action.
    let loan_can_open_build = runner.framework.board.can_take_loan(actor)
        && after_runner
            .framework
            .board
            .get_valid_build_options(actor)
            .iter()
            .any(|option| after.money >= option.total_money_cost.saturating_add(3));
    match intent.action_type {
        ActionType::Loan if loan_can_open_build => 7.0,
        ActionType::Develop | ActionType::DevelopDouble if loan_can_open_build => {
            // Keep the explicit double-develop exception for a genuinely
            // near-complete tier, but do not choose it before financing the
            // engine when the opening has a cash unlock available.
            if intent.action_type == ActionType::DevelopDouble
                && development_is_meaningful(before, after, intent)
                && before.money >= 28
            {
                0.5
            } else {
                -6.0
            }
        }
        _ => 0.0,
    }
}

fn same_turn_repeats_development(runner: &GameRunner, intent: &ActionIntent) -> bool {
    if !matches!(
        intent.action_type,
        ActionType::Develop | ActionType::DevelopDouble
    ) {
        return false;
    }
    let current = [intent.selected_industry, intent.selected_second_industry];
    runner.turn_action_history().iter().any(|previous| {
        if !matches!(
            previous.action_type,
            ActionType::Develop | ActionType::DevelopDouble
        ) {
            return false;
        }
        let prior = [
            previous.selected_industry,
            previous.selected_second_industry,
        ];
        current
            .into_iter()
            .flatten()
            .any(|industry| prior.into_iter().flatten().any(|other| other == industry))
    })
}

#[derive(Debug, Clone, Copy)]
struct TransitionGuardInputs {
    pressure: f64,
    unflipped_removed: usize,
    before_income: i8,
    after_income: i8,
    before_money: u16,
    after_money: u16,
    before_frontier: bool,
    after_frontier: bool,
    route_progress: f64,
    action_type: ActionType,
}

/// Return how close a Canal position is to the era settlement that clears
/// level-one tiles.  Deck size is the primary signal; the shared round counter
/// catches positions where several players have already emptied their hands.
fn canal_transition_pressure(runner: &GameRunner) -> f64 {
    if runner.game_phase != GamePhase::Canal {
        return 0.0;
    }
    let deck_len = runner.framework.board.state.deck.cards.len();
    // Apply the guard while the deck still has enough cards for a corrective
    // action.  Once the deck is empty, the phase boundary is already driven by
    // hand exhaustion; continuing to penalize every late action can suppress
    // legitimate conversion and scoring moves in otherwise healthy games.
    let deck_pressure: f64 = match deck_len {
        5..=7 => 0.85,
        8..=10 => 0.70,
        _ => 0.0,
    };
    let round_pressure: f64 = if deck_pressure <= 0.0 {
        0.0
    } else {
        match runner.round_in_phase {
            0..=4 => 0.0,
            5 => 0.85,
            _ => 1.0,
        }
    };
    deck_pressure.max(round_pressure)
}

fn unflipped_removed_buildings(runner: &GameRunner, actor: usize) -> usize {
    let state = &runner.framework.board.state;
    (0..N_BL)
        .filter(|location| {
            state.bl_to_building.get(location).is_some_and(|building| {
                building.owner.as_usize() == actor
                    && !building.flipped
                    && INDUSTRY_MAT[building.industry.as_usize()][building.level.as_usize()]
                        .removed_after_phase1
            })
        })
        .count()
}

/// Pure portion of the transition guard, kept separate so the threshold and
/// action-family behavior can be regression-tested without constructing a
/// complete board position.
fn transition_guard_penalty_from_inputs(input: TransitionGuardInputs) -> f64 {
    if input.pressure <= 0.0 || input.unflipped_removed == 0 {
        return 0.0;
    }

    let before_debt = f64::from((-input.before_income).max(0));
    let after_debt = f64::from((-input.after_income).max(0));
    let before_reserve = f64::from(input.before_money) - before_debt;
    let after_reserve = f64::from(input.after_money) - after_debt;
    let economically_fragile = (input.before_income < 0 || input.after_income < 0)
        && (before_reserve < 6.0
            || after_reserve < 8.0
            || input.after_money < 12
            || (after_debt >= 6.0 && after_reserve < 12.0));
    if !economically_fragile {
        return 0.0;
    }

    let loses_frontier = input.before_frontier && !input.after_frontier;
    let needs_guard = match input.action_type {
        // A network can destroy an existing conversion route, or be a sunk
        // action when no route existed at all.
        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad => !input.after_frontier,
        // A build/develop/loan may legitimately consume one available option;
        // only guard it when the position had no real conversion route before
        // the action and still has none afterwards.
        ActionType::BuildBuilding
        | ActionType::Develop
        | ActionType::DevelopDouble
        | ActionType::Loan => !input.before_frontier && !input.after_frontier,
        ActionType::Sell | ActionType::Scout | ActionType::Pass => false,
    };
    if !needs_guard {
        return 0.0;
    }

    let exposure = input.unflipped_removed.min(3) as f64;
    let reserve_deficit = (8.0 - after_reserve).max(0.0);
    let base = input.pressure
        * (4.0 + exposure * 2.0 + reserve_deficit * 1.1 + after_debt.min(10.0) * 0.35);
    let action_multiplier = match input.action_type {
        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad => 1.0,
        ActionType::Loan => 0.80,
        ActionType::BuildBuilding => 0.55,
        ActionType::Develop | ActionType::DevelopDouble => 0.40,
        ActionType::Sell | ActionType::Scout | ActionType::Pass => 0.0,
    };
    if action_multiplier == 0.0 {
        return 0.0;
    }

    let mut penalty = -base * action_multiplier;
    if loses_frontier {
        // Losing an existing legal conversion route is worse than merely
        // failing to create one.
        penalty -= base * 0.35;
    }
    if input.route_progress > f64::EPSILON {
        // A route improvement is useful evidence, but without a legal
        // sale/build frontier it is not enough to erase the transition risk.
        penalty *= 0.65;
    }
    penalty.max(-28.0)
}

/// State-dependent Canal/Railroad liquidation guard.  It deliberately uses
/// the validators' actual sale/build frontiers rather than the looser static
/// `available_build_sites` metric.  Healthy positions and network actions that
/// preserve a concrete conversion route therefore retain their normal score.
fn transition_guard_penalty(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    if runner.game_phase != GamePhase::Canal || after_runner.game_phase != GamePhase::Canal {
        return 0.0;
    }
    let pressure = canal_transition_pressure(runner);
    if pressure <= 0.0 {
        return 0.0;
    }
    let actor = runner.framework.current_player;
    let before_options = productive_options(runner, actor);
    let after_options = productive_options(after_runner, actor);
    transition_guard_penalty_from_inputs(TransitionGuardInputs {
        pressure,
        unflipped_removed: unflipped_removed_buildings(runner, actor),
        before_income: before.income_amount,
        after_income: after.income_amount,
        before_money: before.money,
        after_money: after.money,
        before_frontier: before_options.has_conversion_frontier(),
        after_frontier: after_options.has_conversion_frontier(),
        route_progress: before.sale_route_distance - after.sale_route_distance,
        action_type: intent.action_type,
    })
}

#[derive(Debug, Clone, Copy)]
struct RecoveryPriorityInputs {
    late_canal_recovery: bool,
    severe_railroad_recovery: bool,
    action_type: ActionType,
    selected_industry: Option<IndustryType>,
    has_wild_card: bool,
    before_income: i8,
    after_income: i8,
    before_money: u16,
    after_money: u16,
    before_sellable_units: usize,
    before_saleable_builds: usize,
    after_saleable_builds: usize,
    after_sell: usize,
    concrete_conversion: bool,
    selected_build_removed_after_canal: bool,
}

fn recovery_priority_from_inputs(input: RecoveryPriorityInputs) -> f64 {
    if !input.late_canal_recovery && !input.severe_railroad_recovery {
        return 0.0;
    }

    let mut score = 0.0;
    let destroys_last_product_frontier = input.before_saleable_builds > 0
        && input.after_saleable_builds == 0
        && input.after_sell == 0
        && !input.concrete_conversion;

    if input.late_canal_recovery && destroys_last_product_frontier {
        // With no deck cards left, spending the remaining liquidity on a
        // non-converting tile is how the observed 14-VP trajectory lost every
        // funded product option before the era transition.
        score -= 10.0;
    }

    if input.late_canal_recovery
        && input.action_type == ActionType::BuildBuilding
        && input.selected_build_removed_after_canal
        && !input.concrete_conversion
    {
        // A low-score player gains no Railroad engine from an unconverted
        // level-one tile: it spends scarce cash now and is cleared at the era
        // boundary. Leave it available only as a last-resort scored fallback.
        score -= 8.0;
    }

    if !input.severe_railroad_recovery {
        return score;
    }

    if destroys_last_product_frontier {
        score -= 12.0;
    }

    match input.action_type {
        ActionType::Scout => {
            let saleable_gain = input
                .after_saleable_builds
                .saturating_sub(input.before_saleable_builds);
            let debt = u16::try_from((-input.after_income).max(0)).unwrap_or(u16::MAX);
            if !input.has_wild_card
                && saleable_gain >= 3
                && input.after_money >= debt.saturating_add(4)
            {
                // A first Scout is a concrete recovery action when its wild
                // cards turn a nearly dead hand into several funded product
                // builds. Do not reward repeated Scouts once wilds are held.
                score += 10.0 + saleable_gain.min(6) as f64 * 0.8;
            }
        }
        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad => {
            if input.before_sellable_units == 0 && input.after_sell == 0 {
                // Road VP cannot repair a low-score debt spiral by itself.
                // First establish a product or an immediately legal sale.
                let debt = u16::try_from((-input.after_income).max(0)).unwrap_or(u16::MAX);
                let two_settlement_runway = debt.saturating_mul(2).saturating_add(4);
                score -= if !input.concrete_conversion && input.after_money < two_settlement_runway
                {
                    16.0
                } else {
                    8.0
                };
            }
        }
        ActionType::BuildBuilding => {
            if matches!(
                input.selected_industry,
                Some(IndustryType::Coal | IndustryType::Iron | IndustryType::Beer)
            ) && !input.concrete_conversion
            {
                // In emergency recovery, an unflipped resource tile is only
                // another claim on cash. Prefer a funded product loop unless
                // this build flips or raises income now.
                score -= 5.0;
            }
        }
        ActionType::Develop
        | ActionType::DevelopDouble
        | ActionType::Loan
        | ActionType::Sell
        | ActionType::Pass => {}
    }

    // Once debt is already severe, spending cash without improving income or
    // establishing conversion deserves an extra nudge toward a recovery move.
    if input.after_money < input.before_money
        && input.after_income <= input.before_income
        && !input.concrete_conversion
        && matches!(
            input.action_type,
            ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
        )
    {
        score -= 3.0;
    }

    score
}

fn recovery_priority_adjustment(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let low_score = before.victory_points < 35.0;
    let thin_board = before.buildings <= 3;
    let late_canal_recovery = runner.game_phase == GamePhase::Canal
        && runner.framework.board.state.deck.cards.is_empty()
        && low_score
        && thin_board
        && before.income_amount < 0;
    let severe_railroad_recovery = runner.game_phase == GamePhase::Railroad
        && low_score
        && thin_board
        && before.income_amount <= -5;
    if !late_canal_recovery && !severe_railroad_recovery {
        return 0.0;
    }

    let before_options = productive_options(runner, actor);
    let after_options = productive_options(after_runner, actor);
    let route_progress = before.sellable_units > 0
        && after.sale_route_distance + f64::EPSILON < before.sale_route_distance;
    let secured_vp = effect
        .player_victory_points_delta
        .get(actor)
        .copied()
        .unwrap_or(0)
        > 0;
    let concrete_conversion = secured_vp
        || effect.flipped_buildings_delta > 0
        || after.income_amount > before.income_amount
        || after_options.sell > 0
        || route_progress
        // A loan may legitimately fund a Coal/Iron build that flips into the
        // market on the next action. Merely leaving such an option after a road
        // does not mean that the road itself converted anything.
        || (intent.action_type == ActionType::Loan
            && after_options.has_immediate_income_recovery());
    let has_wild_card = runner.framework.board.state.players[actor]
        .hand
        .cards
        .iter()
        .any(|card| {
            matches!(
                card.card_type,
                CardType::WildLocation | CardType::WildIndustry
            )
        });

    recovery_priority_from_inputs(RecoveryPriorityInputs {
        late_canal_recovery,
        severe_railroad_recovery,
        action_type: intent.action_type,
        selected_industry: intent.selected_industry,
        has_wild_card,
        before_income: before.income_amount,
        after_income: after.income_amount,
        before_money: before.money,
        after_money: after.money,
        before_sellable_units: before.sellable_units,
        before_saleable_builds: before_options.saleable_builds,
        after_saleable_builds: after_options.saleable_builds,
        after_sell: after_options.sell,
        concrete_conversion,
        selected_build_removed_after_canal: intent
            .selected_industry
            .is_some_and(|industry| current_tile_data(runner, industry).removed_after_phase1),
    })
}

fn imminent_settlement_guard_penalty(income: i8, money: u16) -> f64 {
    let debt = f64::from((-income).max(0));
    let settlement_reserve = f64::from(money) - debt;
    if income < 0 && settlement_reserve < 0.0 {
        -22.0 - (-settlement_reserve).min(12.0)
    } else if income <= -3 && money < debt as u16 {
        -8.0
    } else {
        0.0
    }
}

/// Hard economic guards are intentionally bounded but large enough to beat a
/// tempting setup score.  A shallow evaluator must reject a move that makes
/// the next income settlement impossible; otherwise it can look attractive
/// while silently liquidating the player's board at round end.
fn economic_guard_penalty(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let debt = f64::from((-after.income_amount).max(0));
    let mut penalty = imminent_settlement_guard_penalty(after.income_amount, after.money);

    // A second loan at a negative income level is almost always a debt spiral.
    // Permit it only when the loan changes the legal action frontier and leaves
    // a concrete build or sale that can repay the debt.
    if intent.action_type == ActionType::Loan {
        let before_options = productive_options(runner, actor);
        let after_options = productive_options(after_runner, actor);
        let unlocks_build = !before_options.has_build() && after_options.has_build();
        let unlocks_sale = before_options.sell == 0 && after_options.sell > 0;
        let funded_saleable_build = after_runner
            .framework
            .board
            .get_valid_build_options(actor)
            .into_iter()
            .any(|option| {
                is_saleable_industry(option.industry_type)
                    && after.money >= option.total_money_cost.saturating_add(8)
            });
        let route_conversion = before.sellable_units > 0
            && after.sale_route_distance + f64::EPSILON < before.sale_route_distance
            && after.money
                >= u16::try_from((-after.income_amount).max(0))
                    .unwrap_or(u16::MAX)
                    .saturating_add(8);
        let late_canal_debt = runner.game_phase == GamePhase::Canal
            && runner.framework.board.state.deck.cards.is_empty()
            && before.income_amount < 0;
        let funded_conversion = after_options.sell > 0 || funded_saleable_build || route_conversion;
        if late_canal_debt
            && after_options.sell == 0
            && funded_saleable_build
            && !after_options.has_immediate_income_recovery()
            && !route_conversion
        {
            // A product tile that is merely affordable is weaker evidence
            // than a legal sale or a market flip.  Keep it as a possible plan,
            // but charge the repeated late loan enough to beat the observed
            // build-without-selling debt spirals without forcing Pass loops.
            penalty -= 8.0;
        }
        let has_concrete_use = funded_conversion;
        // A loan normally lowers income below zero.  That fact alone is not a
        // reason to reject it: the standard Brass opening is often to borrow
        // while the first build is already legal, then use the larger cash
        // buffer for two or three high-value tiles.  Charge debt only when the
        // cash does not open a real investment frontier.
        let opening_investment = before.buildings == 0
            && after.money >= 30
            && after_options.has_build()
            && after_options
                .minimum_build_cost
                .is_some_and(|cost| after.money >= cost.saturating_add(8));
        if after.income_amount < 0 && !opening_investment && !funded_conversion {
            penalty -= if unlocks_build || unlocks_sale {
                12.0
            } else {
                30.0
            };
        }
        if !has_concrete_use {
            penalty -= 24.0;
        }
        if before.income_amount < 0 {
            penalty -= if funded_conversion {
                5.0 + f64::from((-before.income_amount).min(6)) * 0.5
            } else {
                26.0
            };
        }
        if after.income_amount <= -6 {
            penalty -= if funded_conversion { 3.0 } else { 18.0 };
        }
        if let Some(minimum_cost) = after_options.minimum_build_cost {
            let post_build_reserve = after.money.saturating_sub(minimum_cost) as f64 - debt;
            if post_build_reserve < 0.0 {
                penalty -= 10.0;
            }
        }
    }

    // Developing without reaching a new tier is a low-value card and iron
    // spend.  Keep the opening exception for a near-complete mat tier, but do
    // not let the same action loop indefinitely after the tier is exhausted.
    if matches!(
        intent.action_type,
        ActionType::Develop | ActionType::DevelopDouble
    ) && !development_is_meaningful(before, after, intent)
    {
        penalty -= if before.buildings == 0 && runner.personal_turns_taken[actor] == 0 {
            3.0
        } else {
            11.0
        };
    }

    if same_turn_repeats_development(runner, intent) {
        penalty -= if development_is_meaningful(before, after, intent) {
            1.5
        } else {
            8.0
        };
    }

    // Canal links are only worth their cash when they change the production
    // frontier.  Railroad links retain a broader end-of-era network value.
    if runner.game_phase == GamePhase::Canal
        && matches!(
            intent.action_type,
            ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
        )
    {
        let new_sell_targets = after
            .available_sell_targets
            .saturating_sub(before.available_sell_targets);
        let new_build_sites = after
            .available_build_sites
            .saturating_sub(before.available_build_sites);
        let new_trade_posts = after
            .connected_trade_posts
            .saturating_sub(before.connected_trade_posts);
        let route_progress = after.sale_route_distance + f64::EPSILON < before.sale_route_distance;
        if new_sell_targets == 0 && new_build_sites == 0 && new_trade_posts == 0 && !route_progress
        {
            // The raw network delta counts every newly reachable town, which
            // can be a large number even when the link has no sale/build
            // consequence.  Make the guard decisive against that signal.
            penalty -= 24.0;
        }
    }

    // A non-sale action that consumes the last liquid cash while income is
    // negative should lose to Pass or a sale, even if its static network or
    // industry value is high.
    if after.money == 0
        && after.income_amount < 0
        && intent.action_type != ActionType::Sell
        && effect
            .player_victory_points_delta
            .get(actor)
            .copied()
            .unwrap_or(0)
            <= 0
    {
        penalty -= 16.0;
    }
    penalty += transition_guard_penalty(runner, after_runner, before, after, intent);
    penalty
}

fn settlement_safety_score(income: i8, money: u16) -> f64 {
    let debt = f64::from((-income).max(0));
    let reserve = f64::from(money) - debt;
    if reserve < 0.0 {
        -18.0
    } else if reserve < 2.0 {
        -5.0
    } else if reserve < 5.0 {
        -1.5
    } else {
        0.0
    }
}

fn safety_score(
    runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let mut score = settlement_safety_score(after.income_amount, after.money);

    if intent.action_type == ActionType::Loan && before.income_level <= 14 {
        score -= 3.0;
    }

    let has_industry = before.buildings > 0;
    let is_network = matches!(
        intent.action_type,
        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad
    );
    if is_network && !has_industry && before.own_roads == 0 {
        // A road without a first industry consumes scarce cash and creates no
        // personal scoring engine.  This guard is intentionally strong.
        score -= 10.0;
    }

    if runner.game_phase == GamePhase::Canal && after.money == 0 && after.income_amount < 0 {
        score -= 5.0;
    }
    score
}

fn action_bias_score(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    effect: &ImmediateEffect,
    intent: &ActionIntent,
    alternatives: usize,
) -> f64 {
    let rounds_remaining = estimate_rounds_remaining(runner);
    let late_game = rounds_remaining <= 3.0 || runner.framework.board.state.deck.cards.len() <= 12;
    // `round_in_phase` is shared by all seats, so it labels several later
    // turns as an opening.  Personal turns are the meaningful lifecycle
    // boundary for a player deciding whether to establish its engine.
    let actor = runner.framework.current_player;
    let opening = runner.game_phase == GamePhase::Canal
        && runner.personal_turns_taken.get(actor).copied().unwrap_or(0) == 0
        && before.buildings == 0
        && before.own_roads == 0;
    let new_sell_targets = after
        .available_sell_targets
        .saturating_sub(before.available_sell_targets) as f64;
    let new_build_sites = after
        .available_build_sites
        .saturating_sub(before.available_build_sites) as f64;
    let has_productive_building = before.buildings > 0;
    let selected_industry = intent.selected_industry;

    match intent.action_type {
        ActionType::Pass => {
            if alternatives > 1 {
                -8.0
            } else {
                -1.0
            }
        }
        ActionType::Sell => {
            // A legal sale is the main VP conversion primitive.  It should
            // dominate setup actions whenever the actor has a ready target.
            let flipped = effect.flipped_buildings_delta.max(0) as f64;
            let targets = intent.sell_choices.len().max(1) as f64;
            7.0 + 3.0 * flipped
                + 0.65 * targets
                + if late_game || runner.game_phase == GamePhase::Railroad {
                    1.5
                } else {
                    0.8
                }
        }
        ActionType::BuildBuilding => {
            let industry = selected_industry.unwrap_or(IndustryType::Goods);
            let tile = current_tile_data(runner, industry);
            let mut score = build_industry_bias(
                runner,
                after_runner,
                before,
                after,
                industry,
                tile,
                rounds_remaining,
                new_sell_targets,
                intent.selected_build_location,
            );
            if before.buildings == 0 {
                // The first tile starts the whole scoring engine.  Prefer a
                // saleable industry, but retain a smaller bonus for a needed
                // resource tile when no saleable build is legal.
                score += if matches!(
                    industry,
                    IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
                ) {
                    3.0
                } else {
                    1.0
                };
            } else if after.buildings > before.buildings {
                score += 0.8;
            }
            score
        }
        ActionType::BuildRailroad | ActionType::BuildDoubleRailroad => {
            let road_count = intent
                .selected_road_idx
                .into_iter()
                .chain(intent.selected_second_road_idx)
                .count() as f64;
            let mut score = if runner.game_phase == GamePhase::Railroad {
                1.4 + 0.55 * road_count
            } else {
                0.25
            };
            score += 6.5 * new_sell_targets;
            score += 0.65 * new_build_sites;
            score += (after.network_locations as f64 - before.network_locations as f64) * 0.45;
            score +=
                (after.connected_trade_posts as f64 - before.connected_trade_posts as f64) * 1.1;
            score += (before.sale_route_distance - after.sale_route_distance) * 1.8;
            // Road VP is useful in the railroad era, but is a weak signal in
            // the canal era unless it also shortens a sale route.
            score += (after.road_value - before.road_value)
                * if runner.game_phase == GamePhase::Railroad {
                    0.25
                } else {
                    0.04
                };
            if !has_productive_building {
                score -= 8.0;
            } else if runner.game_phase == GamePhase::Canal
                && before.sellable_units > 0
                && new_sell_targets == 0.0
            {
                score -= 1.25;
            } else if runner.game_phase == GamePhase::Canal && before.sellable_units == 0 {
                score -= 0.75;
            }
            if intent.action_type == ActionType::BuildDoubleRailroad {
                if after.money >= 12 {
                    score += 0.8;
                } else {
                    score -= 1.5;
                }
            }
            score
        }
        ActionType::Develop | ActionType::DevelopDouble => {
            let mut score = development_plan_value(runner, after_runner, before, after, intent);
            if opening && intent.action_type == ActionType::DevelopDouble {
                // Human openings commonly borrow first, then use one or two
                // double-develops to unlock Cotton/Goods II before building.
                // This branch is capped by mat progress in
                // development_plan_value, so it cannot loop forever.
                if development_is_meaningful(before, after, intent) && before.money >= 30 {
                    score += 4.5;
                } else if development_is_meaningful(before, after, intent) {
                    score += 1.5;
                } else {
                    score -= 2.5;
                }
            }
            if before.sellable_units > 0 && before.available_sell_targets == 0 {
                score -= 1.0;
            }
            if before.money < 8 {
                score -= 2.0;
            }
            if late_game {
                score -= 1.0;
            }
            score
        }
        ActionType::Loan => loan_plan_value(runner, after_runner, before, after, opening),
        ActionType::Scout => {
            // Scout is a hand-repair action, not a source of points.  Keep it
            // as a fallback only when no productive root is available.
            if !has_productive_building && before.money < 8 {
                0.5
            } else {
                -2.0
            }
        }
    }
}

fn current_tile_data(
    runner: &GameRunner,
    industry: IndustryType,
) -> &'static crate::core::types::BuildingTypeData {
    let player = &runner.framework.board.state.players[runner.framework.current_player];
    player
        .industry_mat
        .get_tile_for_industry(industry)
        .unwrap_or_else(|| {
            &INDUSTRY_MAT[industry.as_usize()][INDUSTRY_MAT[industry.as_usize()].len() - 1]
        })
}

fn build_industry_bias(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    industry: IndustryType,
    tile: &'static crate::core::types::BuildingTypeData,
    rounds_remaining: f64,
    new_sell_targets: f64,
    build_location: Option<usize>,
) -> f64 {
    let phase = runner.game_phase;
    let mut flip_probability = match industry {
        IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery => 0.58,
        IndustryType::Beer => 0.45,
        IndustryType::Coal | IndustryType::Iron => 0.42,
    };
    if new_sell_targets > 0.0 {
        flip_probability = 0.95;
    } else if matches!(
        industry,
        IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery
    ) && after_runner
        .framework
        .board
        .get_valid_sell_options(runner.framework.current_player)
        .len()
        > 0
    {
        flip_probability = 0.82;
    }

    let own_touching_road = intent_road_touches_location(
        after_runner,
        runner.framework.current_player,
        build_location,
    );
    let expected_vp = f64::from(tile.vp_on_flip) * flip_probability
        + if own_touching_road {
            f64::from(tile.road_vp) * flip_probability * 0.5
        } else {
            0.0
        };
    let expected_income =
        f64::from(tile.income.max(0)) * flip_probability * (rounds_remaining / 5.0) * 0.35;
    let cash_cost = (before.money as f64 - after.money as f64).max(0.0) * 0.16;
    let mut score = expected_vp + expected_income - cash_cost;

    score += match industry {
        IndustryType::Cotton => 2.4,
        IndustryType::Goods => 2.0,
        IndustryType::Pottery => 1.7,
        IndustryType::Beer => {
            let demand_gap =
                before.sellable_beer_demand as f64 - before.unflipped_beer_units as f64;
            if demand_gap > 0.0 {
                2.0 + demand_gap.min(3.0) * 0.35
            } else {
                -0.6
            }
        }
        IndustryType::Coal => {
            if before.resource_units[0] < 2 {
                1.2
            } else {
                0.1
            }
        }
        IndustryType::Iron => {
            if before.resource_units[1] < 2 {
                1.0
            } else {
                0.1
            }
        }
    };

    if phase == GamePhase::Railroad {
        let level = runner.framework.board.state.players[runner.framework.current_player]
            .industry_mat
            .get_lowest_level(industry)
            .as_u8();
        if level == 0 {
            score -= 1.5;
        } else {
            score += f64::from(level) * 0.35;
        }
    }
    score
}

fn development_plan_value(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    intent: &ActionIntent,
) -> f64 {
    let actor = runner.framework.current_player;
    let mut industries = Vec::new();
    if let Some(industry) = intent.selected_industry {
        industries.push(industry);
    }
    if let Some(industry) = intent.selected_second_industry {
        industries.push(industry);
    }
    if industries.is_empty() {
        return -4.0;
    }
    let industry_count = industries.len();
    let mut level_ups = 0usize;
    let mut removed_total = 0.0;
    let mut score = if intent.action_type == ActionType::DevelopDouble {
        0.35
    } else {
        0.0
    };
    for industry in industries {
        let index = industry.as_usize();
        let old_level = before.industry_levels[index] as usize;
        let new_level = after.industry_levels[index] as usize;
        let old_remaining = before.industry_remaining[index] as f64;
        let new_remaining = after.industry_remaining[index] as f64;
        let removed = (old_remaining - new_remaining).max(0.0);
        removed_total += removed;
        let current_tiles = f64::from(INDUSTRY_MAT[index][old_level].num_tiles.max(1));
        score += 0.65 * (removed / current_tiles);
        if new_level > old_level {
            level_ups += 1;
            let target = &INDUSTRY_MAT[index][new_level];
            let previous = &INDUSTRY_MAT[index][old_level];
            score += 2.8
                + (f64::from(target.vp_on_flip) - f64::from(previous.vp_on_flip)) * 0.22
                + f64::from((target.income - previous.income).max(0)) * 0.12;
        }
        score += match industry {
            IndustryType::Cotton => 1.05,
            IndustryType::Goods => 0.85,
            IndustryType::Pottery => 0.65,
            IndustryType::Beer => 0.25,
            IndustryType::Coal | IndustryType::Iron => {
                let resource_index = match industry {
                    IndustryType::Coal => 0,
                    IndustryType::Iron => 1,
                    _ => unreachable!("resource index requested for non-resource industry"),
                };
                if before.resource_units[resource_index] >= 3 {
                    -0.65
                } else {
                    0.15
                }
            }
        };
        // Spending iron on a tile that neither advances the mat nor reaches
        // its last remaining slot is usually dominated by a build or a sale.
        // Keep a small allowance for the final one/two-tile push that unlocks
        // the next tier.
        if new_level == old_level && removed == 0.0 {
            score -= 1.35;
        } else if new_level == old_level && new_remaining > 1.0 {
            score -= 0.65;
        }
        let removed_total_before = mat_removed_count(runner, actor, industry);
        let removed_total_after = mat_removed_count(after_runner, actor, industry);
        if removed_total_after >= removed_total_before && removed_total_after >= 4 {
            score -= 0.25;
        }
    }
    // Development is a setup action; it should not beat a ready sale.
    if before.available_sell_targets > 0 {
        score -= 5.0;
    }
    if industry_count >= 2 && level_ups == 0 && removed_total < 2.0 {
        score -= 2.5;
    }
    if before.income_amount < 0 {
        score -= 2.0 + f64::from((-before.income_amount).min(6)) * 0.5;
    }
    if after.money < 6 {
        score -= 1.5;
    }
    score
}

fn loan_plan_value(
    runner: &GameRunner,
    after_runner: &GameRunner,
    before: &PositionSnapshot,
    after: &PositionSnapshot,
    opening: bool,
) -> f64 {
    let actor = runner.framework.current_player;
    let before_options = productive_options(runner, actor);
    let after_options = productive_options(after_runner, actor);
    let unlocks_build = !before_options.has_build() && after_options.has_build();
    let unlocks_sell = before_options.sell == 0 && after_options.sell > 0;
    let unlocks_network = before_options.network == 0 && after_options.network > 0;
    let funded_saleable_build_after = after_runner
        .framework
        .board
        .get_valid_build_options(actor)
        .into_iter()
        .any(|option| {
            is_saleable_industry(option.industry_type)
                && after.money >= option.total_money_cost.saturating_add(8)
        });
    let route_conversion_after = before.sellable_units > 0
        && after.sale_route_distance + f64::EPSILON < before.sale_route_distance
        && after.money
            >= u16::try_from((-after.income_amount).max(0))
                .unwrap_or(u16::MAX)
                .saturating_add(8);
    let funded_conversion_after =
        after_options.sell > 0 || funded_saleable_build_after || route_conversion_after;
    let mut score = if opening {
        // Borrowing on the first personal turn is a legitimate engine start,
        // but only when the cash actually opens a productive frontier.
        if unlocks_build || unlocks_sell || unlocks_network {
            7.5
        } else if after_options.has_build()
            && after.money >= 30
            && after_options
                .minimum_build_cost
                .is_some_and(|cost| after.money >= cost.saturating_add(8))
        {
            // The first build may already be affordable before borrowing.  In
            // that case the value is the *second* affordable build next turn,
            // not an unlock in the validator's boolean frontier.
            13.5
        } else {
            -3.0
        }
    } else if before.money < 8 {
        if unlocks_build || unlocks_sell {
            4.0
        } else {
            -4.0
        }
    } else if before.money < 14 {
        2.0
    } else {
        -4.0
    };
    let before_builds = runner.framework.board.get_valid_build_options(actor).len();
    let after_builds = after_runner
        .framework
        .board
        .get_valid_build_options(actor)
        .len();
    if before_builds == 0 && after_builds > 0 {
        score += 4.0;
    }
    if before.buildings == 0 && after.money >= 35 {
        score += 1.5;
    }
    // Borrowing is sound when it buys a concrete tile with a reserve for the
    // next settlement.  This covers the human pattern of taking several
    // purposeful loans while income is temporarily negative, while leaving
    // unfunded repeat loans on the existing debt-spiral path.
    if funded_conversion_after {
        if before.money < 14 {
            score += 4.5;
        } else if before.money < 20 {
            score += 2.0;
        }
        if after_options.saleable_builds > 0 {
            score += 2.0;
        }
    }
    if before.income_level <= 14 {
        score -= 1.5;
    }
    if before.income_amount < 0 {
        if funded_conversion_after {
            score -= 4.0 + f64::from((-before.income_amount).min(6)) * 0.5;
        } else {
            // Borrowing to fund a non-saleable resource/road while already in
            // debt is the failure mode that produces the observed late-game
            // loan loop.  Make it lose to a neutral fallback unless it has a
            // concrete conversion path.
            score -= 18.0 + f64::from((-before.income_amount).min(6)) * 1.5;
        }
    }
    if after.income_amount <= -6 {
        score -= if funded_conversion_after { 3.0 } else { 14.0 };
    }
    if after_options.count() == 0 {
        score -= 16.0;
    }
    if after.money > 50 {
        score -= 1.0;
    }
    score
}

fn mat_removed_count(runner: &GameRunner, actor: usize, industry: IndustryType) -> usize {
    let mat = &runner.framework.board.state.players[actor].industry_mat;
    let level = mat.get_lowest_level(industry).as_usize();
    let remaining = usize::from(mat.get_remaining_tiles_at_level(industry));
    let mut consumed = 0usize;
    for prior_level in 0..level {
        consumed += usize::from(INDUSTRY_MAT[industry.as_usize()][prior_level].num_tiles);
    }
    consumed
        + usize::from(INDUSTRY_MAT[industry.as_usize()][level].num_tiles).saturating_sub(remaining)
}

fn intent_road_touches_location(
    runner: &GameRunner,
    actor: usize,
    build_location: Option<usize>,
) -> bool {
    let Some(build_location) = build_location else {
        return false;
    };
    let town = LocationName::from_bl_idx(build_location);
    runner.framework.board.state.player_road_mask[actor]
        .ones()
        .any(|road_idx| LINK_LOCATIONS[road_idx].locations.contains(town.as_usize()))
}

fn market_access_value(snapshot: &PositionSnapshot) -> f64 {
    let coal = f64::from(snapshot.market_coal) / f64::from(MAX_MARKET_COAL);
    let iron = f64::from(snapshot.market_iron) / f64::from(MAX_MARKET_IRON);
    // A larger remaining market is a little more valuable because future
    // builds can buy resources cheaply.  The actor's own capacity remains the
    // dominant resource signal above.
    (coal + iron) * 0.15 + f64::from(snapshot.resource_units[2]) * 0.02
}

fn estimate_rounds_remaining(runner: &GameRunner) -> f64 {
    let state = &runner.framework.board.state;
    let players = state.players.len().max(2) as f64;
    let cards_in_hands = state
        .players
        .iter()
        .map(|player| player.hand.cards.len())
        .sum::<usize>();
    let cards = state.deck.cards.len() + cards_in_hands;
    // Each complete round consumes roughly one card per player.  Clamp the
    // estimate so income does not swamp immediate VP in an endgame position.
    ((cards as f64 / players).ceil()).clamp(1.0, 12.0)
}

fn best_opponent_vp(runner: &GameRunner, actor: usize) -> f64 {
    runner
        .framework
        .board
        .state
        .players
        .iter()
        .enumerate()
        .filter_map(|(index, player)| (index != actor).then_some(f64::from(player.victory_points)))
        .fold(0.0, f64::max)
}

fn vp_margin(runner: &GameRunner, actor: usize) -> f64 {
    let own = f64::from(runner.framework.board.state.players[actor].victory_points);
    own - best_opponent_vp(runner, actor)
}

fn actor_delta(values: &[i32], actor: usize) -> f64 {
    values.get(actor).copied().unwrap_or_default() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_intent(action_type: ActionType) -> ActionIntent {
        ActionIntent {
            action_type,
            selected_industry: None,
            selected_second_industry: None,
            selected_card_idx: None,
            scout_additional_discard_indices: Vec::new(),
            selected_build_location: None,
            selected_road_idx: None,
            selected_second_road_idx: None,
            chosen_coal_sources: Vec::new(),
            chosen_iron_sources: Vec::new(),
            chosen_beer_sources: Vec::new(),
            chosen_action_beer_source: None,
            selected_network_mode: None,
            sell_choices: Vec::new(),
            free_development_choice: None,
        }
    }

    #[test]
    fn initial_position_ranks_all_legal_actions_without_rollout() {
        let runner = GameRunner::new(2, Some(91_201));
        let ranked = rank_rule_actions(&runner, &RuleDecisionConfig::default()).unwrap();
        assert!(!ranked.is_empty());
        assert!(ranked.windows(2).all(|pair| {
            pair[0].score > pair[1].score
                || (pair[0].score == pair[1].score && pair[0].action.key() <= pair[1].action.key())
        }));
        assert!(ranked
            .iter()
            .all(|candidate| candidate.breakdown.total.is_finite()));
    }

    #[test]
    fn network_before_first_build_is_penalized() {
        let runner = GameRunner::new(2, Some(91_202));
        let actor = runner.framework.current_player;
        let before = snapshot(&runner, actor);
        let mut after_runner = runner.clone();
        after_runner.framework.board.state.player_road_mask[actor].insert(0);
        after_runner.framework.board.state.built_roads.insert(0);
        after_runner.framework.board.state.connectivity.add_road(0);
        let after = snapshot(&after_runner, actor);
        let effect = immediate_effect_between(&runner, &after_runner);
        let (network_breakdown, _) = score_action(
            &runner,
            &after_runner,
            &before,
            &after,
            &effect,
            &ActionIntent {
                action_type: ActionType::BuildRailroad,
                selected_industry: None,
                selected_second_industry: None,
                selected_card_idx: None,
                scout_additional_discard_indices: Vec::new(),
                selected_build_location: None,
                selected_road_idx: Some(0),
                selected_second_road_idx: None,
                chosen_coal_sources: Vec::new(),
                chosen_iron_sources: Vec::new(),
                chosen_beer_sources: Vec::new(),
                chosen_action_beer_source: None,
                selected_network_mode: None,
                sell_choices: Vec::new(),
                free_development_choice: None,
            },
            2,
            &RuleDecisionConfig::default(),
        );
        assert!(network_breakdown.safety < -5.0);
    }

    #[test]
    fn railroad_weights_network_and_vp_more_than_canal() {
        let runner = GameRunner::new(2, Some(91_203));
        let actor = runner.framework.current_player;
        let before = snapshot(&runner, actor);
        let mut canal_after = runner.clone();
        let mut rail_after = runner.clone();
        canal_after.game_phase = GamePhase::Canal;
        rail_after.game_phase = GamePhase::Railroad;
        let canal = snapshot(&canal_after, actor);
        let rail = snapshot(&rail_after, actor);
        let effect_canal = immediate_effect_between(&runner, &canal_after);
        let effect_rail = immediate_effect_between(&runner, &rail_after);
        let intent = ActionIntent {
            action_type: ActionType::BuildRailroad,
            selected_industry: None,
            selected_second_industry: None,
            selected_card_idx: None,
            scout_additional_discard_indices: Vec::new(),
            selected_build_location: None,
            selected_road_idx: Some(0),
            selected_second_road_idx: None,
            chosen_coal_sources: Vec::new(),
            chosen_iron_sources: Vec::new(),
            chosen_beer_sources: Vec::new(),
            chosen_action_beer_source: None,
            selected_network_mode: None,
            sell_choices: Vec::new(),
            free_development_choice: None,
        };
        let (canal_score, _) = score_action(
            &canal_after,
            &canal_after,
            &before,
            &canal,
            &effect_canal,
            &intent,
            2,
            &RuleDecisionConfig::default(),
        );
        let (rail_score, _) = score_action(
            &rail_after,
            &rail_after,
            &before,
            &rail,
            &effect_rail,
            &intent,
            2,
            &RuleDecisionConfig::default(),
        );
        assert!(rail_score.network >= canal_score.network);
        assert!(rail_score.immediate_vp >= canal_score.immediate_vp);
    }

    #[test]
    fn loan_is_not_preferred_when_cash_is_already_high() {
        let mut runner = GameRunner::new(2, Some(91_204));
        let actor = runner.framework.current_player;
        runner.framework.board.state.players[actor].money = 30;
        let ranked = rank_rule_actions(&runner, &RuleDecisionConfig::default()).unwrap();
        if let Some(loan) = ranked
            .iter()
            .find(|candidate| candidate.action.intent.action_type == ActionType::Loan)
        {
            let best_non_loan = ranked
                .iter()
                .find(|candidate| candidate.action.intent.action_type != ActionType::Loan)
                .expect("there should be a non-loan action");
            assert!(loan.score < best_non_loan.score);
        }
    }

    #[test]
    fn income_signal_is_bounded_so_setup_cannot_drown_out_vp() {
        let signal = bounded_income_signal(500.0, 500.0, 12.0, false);
        assert!(signal <= 10.5, "income signal was not capped: {signal}");
        let loan_signal = bounded_income_signal(500.0, 500.0, 12.0, true);
        assert!(
            loan_signal <= 10.5,
            "loan income signal was not capped: {loan_signal}"
        );
    }

    #[test]
    fn ready_sale_gets_a_lifecycle_priority_over_setup() {
        let runner = GameRunner::new(2, Some(91_205));
        let mut before = PositionSnapshot::default();
        before.available_sell_targets = 1;
        let after = before.clone();
        let mut effect = immediate_effect_between(&runner, &runner);
        effect.flipped_buildings_delta = 1;

        let sale = lifecycle_priority_adjustment(
            &runner,
            &runner,
            &before,
            &after,
            &effect,
            &test_intent(ActionType::Sell),
        );
        let develop = lifecycle_priority_adjustment(
            &runner,
            &runner,
            &before,
            &after,
            &effect,
            &test_intent(ActionType::Develop),
        );
        assert!(sale > develop);
        assert!(sale >= 10.0, "ready sale priority too small: {sale}");
    }

    #[test]
    fn sale_frontier_build_reward_is_independent_from_lifecycle_priority() {
        let runner = GameRunner::new(2, Some(91_208));
        let before = PositionSnapshot::default();
        let mut after = before.clone();
        after.available_sell_targets = 1;
        let effect = immediate_effect_between(&runner, &runner);
        let mut intent = test_intent(ActionType::BuildBuilding);
        intent.selected_industry = Some(IndustryType::Goods);

        let lifecycle =
            lifecycle_priority_adjustment(&runner, &runner, &before, &after, &effect, &intent);
        let frontier = sale_frontier_build_signal(&before, &after, &intent);
        assert_eq!(lifecycle, 1.5);
        assert_eq!(frontier, 6.0);
        assert!(lifecycle + frontier >= 7.5);
    }

    #[test]
    fn network_without_an_industry_is_penalized_by_lifecycle_gate() {
        let runner = GameRunner::new(2, Some(91_206));
        let before = PositionSnapshot::default();
        let after = before.clone();
        let effect = immediate_effect_between(&runner, &runner);
        let network = lifecycle_priority_adjustment(
            &runner,
            &runner,
            &before,
            &after,
            &effect,
            &test_intent(ActionType::BuildRailroad),
        );
        assert!(
            network <= -4.0,
            "network gate was too permissive: {network}"
        );
    }

    #[test]
    fn opening_loan_priority_requires_a_real_build_frontier() {
        let runner = GameRunner::new(2, Some(91_207));
        let actor = runner.framework.current_player;
        let before = snapshot(&runner, actor);
        let loan = enumerate_legal_actions(&runner)
            .unwrap()
            .into_iter()
            .find(|action| action.intent.action_type == ActionType::Loan)
            .expect("fresh position should expose a loan action");
        let mut after_runner = runner.clone();
        loan.apply(&mut after_runner).unwrap();
        let after = snapshot(&after_runner, actor);
        let priority =
            strategic_priority_adjustment(&runner, &after_runner, &before, &after, &loan.intent);
        assert!(
            priority >= 7.0,
            "opening loan was not recognized: {priority}"
        );
    }

    #[test]
    fn transition_guard_penalizes_network_without_real_conversion_frontier() {
        let penalty = transition_guard_penalty_from_inputs(TransitionGuardInputs {
            pressure: 0.70,
            unflipped_removed: 1,
            before_income: -3,
            after_income: -3,
            before_money: 9,
            after_money: 6,
            before_frontier: false,
            after_frontier: false,
            route_progress: 0.0,
            action_type: ActionType::BuildRailroad,
        });
        assert!(
            penalty <= -8.0,
            "transition-risk network penalty was too weak: {penalty}"
        );
    }

    #[test]
    fn transition_guard_spares_action_that_keeps_conversion_frontier() {
        let penalty = transition_guard_penalty_from_inputs(TransitionGuardInputs {
            pressure: 1.0,
            unflipped_removed: 2,
            before_income: -6,
            after_income: -6,
            before_money: 17,
            after_money: 14,
            before_frontier: true,
            after_frontier: true,
            route_progress: 0.0,
            action_type: ActionType::BuildRailroad,
        });
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn transition_guard_ignores_healthy_or_unexposed_positions() {
        let healthy = transition_guard_penalty_from_inputs(TransitionGuardInputs {
            pressure: 1.0,
            unflipped_removed: 2,
            before_income: 4,
            after_income: 4,
            before_money: 10,
            after_money: 7,
            before_frontier: false,
            after_frontier: false,
            route_progress: 0.0,
            action_type: ActionType::BuildRailroad,
        });
        let unexposed = transition_guard_penalty_from_inputs(TransitionGuardInputs {
            pressure: 1.0,
            unflipped_removed: 0,
            before_income: -6,
            after_income: -6,
            before_money: 10,
            after_money: 7,
            before_frontier: false,
            after_frontier: false,
            route_progress: 0.0,
            action_type: ActionType::BuildRailroad,
        });
        assert_eq!(healthy, 0.0);
        assert_eq!(unexposed, 0.0);
    }

    fn recovery_inputs(action_type: ActionType) -> RecoveryPriorityInputs {
        RecoveryPriorityInputs {
            late_canal_recovery: false,
            severe_railroad_recovery: false,
            action_type,
            selected_industry: None,
            has_wild_card: false,
            before_income: -8,
            after_income: -8,
            before_money: 30,
            after_money: 30,
            before_sellable_units: 0,
            before_saleable_builds: 2,
            after_saleable_builds: 2,
            after_sell: 0,
            concrete_conversion: false,
            selected_build_removed_after_canal: false,
        }
    }

    #[test]
    fn late_canal_recovery_rejects_destroying_last_product_frontier() {
        let mut input = recovery_inputs(ActionType::BuildBuilding);
        input.late_canal_recovery = true;
        input.after_money = 7;
        input.after_saleable_builds = 0;
        input.selected_industry = Some(IndustryType::Beer);
        let score = recovery_priority_from_inputs(input);
        assert!(
            score <= -10.0,
            "frontier destruction was not rejected: {score}"
        );
    }

    #[test]
    fn late_canal_recovery_rejects_unconverted_build_that_will_be_removed() {
        let mut input = recovery_inputs(ActionType::BuildBuilding);
        input.late_canal_recovery = true;
        input.selected_build_removed_after_canal = true;
        input.selected_industry = Some(IndustryType::Beer);
        let score = recovery_priority_from_inputs(input);
        assert!(
            score <= -8.0,
            "removable unconverted build was not rejected: {score}"
        );

        input.concrete_conversion = true;
        assert_eq!(recovery_priority_from_inputs(input), 0.0);
    }

    #[test]
    fn severe_railroad_recovery_rewards_first_scout_that_opens_products() {
        let mut input = recovery_inputs(ActionType::Scout);
        input.severe_railroad_recovery = true;
        input.after_saleable_builds = 7;
        let score = recovery_priority_from_inputs(input);
        assert!(score >= 14.0, "recovery Scout bonus was too small: {score}");

        input.has_wild_card = true;
        assert_eq!(recovery_priority_from_inputs(input), 0.0);
    }

    #[test]
    fn severe_railroad_recovery_rejects_road_before_product() {
        let mut input = recovery_inputs(ActionType::BuildRailroad);
        input.severe_railroad_recovery = true;
        input.after_money = 24;
        input.after_saleable_builds = 0;
        let score = recovery_priority_from_inputs(input);
        assert!(
            score <= -20.0,
            "unproductive road penalty was too small: {score}"
        );
    }

    #[test]
    fn severe_railroad_recovery_requires_two_settlement_road_runway() {
        let mut input = recovery_inputs(ActionType::BuildRailroad);
        input.severe_railroad_recovery = true;
        input.before_saleable_builds = 8;
        input.after_saleable_builds = 4;
        input.before_money = 22;
        input.after_money = 16;
        let score = recovery_priority_from_inputs(input);
        assert!(score <= -19.0, "thin-runway road was not pruned: {score}");

        input.after_money = 24;
        let funded_score = recovery_priority_from_inputs(input);
        assert!(
            funded_score > score,
            "funded road did not retain the softer fallback: {funded_score}"
        );
    }

    #[test]
    fn loan_maturity_surcharge_starts_after_income_one_and_is_monotone() {
        for income in [-10, -6, -3, 0, 1] {
            assert_eq!(loan_maturity_surcharge(income), 0.0);
        }
        assert_eq!(loan_maturity_surcharge(2), 0.75);
        assert_eq!(loan_maturity_surcharge(4), 2.25);
        assert!((loan_maturity_surcharge(9) - 6.75).abs() < 1e-9);
        assert!((loan_maturity_surcharge(19) - 11.75).abs() < 1e-9);
        assert_eq!(loan_maturity_surcharge(30), 11.75);

        let mut previous = 0.0;
        for income in -10..=30 {
            let current = loan_maturity_surcharge(income);
            assert!(current >= previous);
            assert!(current <= 11.75);
            previous = current;
        }
    }

    #[test]
    fn loan_excess_income_loss_only_prices_loss_beyond_three() {
        for (before, after) in [(0, -3), (5, 2), (-6, -9), (3, 3), (-3, 0)] {
            assert_eq!(loan_excess_income_loss_surcharge(before, after), 0.0);
        }
        assert_eq!(loan_excess_income_loss_surcharge(4, -1), 2.0);
        assert_eq!(loan_excess_income_loss_surcharge(10, 2), 5.0);
        assert_eq!(loan_excess_income_loss_surcharge(30, -10), 12.0);

        let mut previous = 0.0;
        for loss in 0..=30 {
            let current = loan_excess_income_loss_surcharge(20, 20 - loss);
            assert!(current >= previous);
            assert!(current <= 12.0);
            previous = current;
        }
    }

    fn low_cost_loan_runway_inputs() -> LowCostLoanRunwayInputs {
        LowCostLoanRunwayInputs {
            railroad: true,
            deck_empty: true,
            low_score: true,
            underbuilt_engine: true,
            first_action: true,
            no_ready_or_pending_sale: true,
            before_income: 1,
            displayed_income_loss: 3,
            before_money: 18,
            after_money: 48,
            cards_after: 7,
            productive_follow_up: true,
        }
    }

    #[test]
    fn low_cost_loan_runway_requires_every_narrow_gate() {
        let baseline = low_cost_loan_runway_inputs();
        assert!(low_cost_loan_runway_is_eligible(baseline));
        for input in [
            LowCostLoanRunwayInputs {
                railroad: false,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                deck_empty: false,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                low_score: false,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                underbuilt_engine: false,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                first_action: false,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                no_ready_or_pending_sale: false,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                before_income: 2,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                displayed_income_loss: 4,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                before_money: 21,
                after_money: 51,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                after_money: 47,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                cards_after: 4,
                ..baseline
            },
            LowCostLoanRunwayInputs {
                productive_follow_up: false,
                ..baseline
            },
        ] {
            assert!(!low_cost_loan_runway_is_eligible(input), "{input:?}");
        }
    }

    #[test]
    fn endgame_product_budget_prunes_certainly_unconvertible_builds() {
        assert_eq!(product_build_feasibility_penalty(0, Some(1), false), -30.0);
        assert_eq!(product_build_feasibility_penalty(1, Some(2), false), -30.0);
        assert_eq!(product_build_feasibility_penalty(4, None, false), -28.0);
    }

    #[test]
    fn endgame_product_budget_keeps_exactly_funded_plan_but_prices_stacking() {
        assert_eq!(product_build_feasibility_penalty(2, Some(2), false), 0.0);
        assert_eq!(product_build_feasibility_penalty(2, Some(2), true), -6.0);
        assert_eq!(product_build_feasibility_penalty(3, Some(2), false), 0.0);
    }

    fn stockpile_inputs() -> ResourceStockpileInputs {
        ResourceStockpileInputs {
            canal: true,
            low_score: true,
            action_type: ActionType::BuildBuilding,
            selected_industry: Some(IndustryType::Coal),
            before_income: -3,
            after_income: -3,
            before_flipped_buildings: 0,
            after_flipped_buildings: 0,
            before_sellable_units: 0,
            before_resource_units: 2,
            after_resource_units: 4,
            product_development_progress: 2,
            support_development_progress: 2,
            has_product_follow_up: true,
        }
    }

    #[test]
    fn stalled_product_opening_prices_another_unconverted_resource_stockpile() {
        assert_eq!(
            resource_stockpile_penalty_from_inputs(stockpile_inputs()),
            -1.0
        );
    }

    #[test]
    fn resource_stockpile_gate_preserves_healthy_and_converting_builds() {
        let baseline = stockpile_inputs();
        for input in [
            ResourceStockpileInputs {
                product_development_progress: 1,
                ..baseline
            },
            ResourceStockpileInputs {
                support_development_progress: 1,
                ..baseline
            },
            ResourceStockpileInputs {
                before_flipped_buildings: 1,
                after_flipped_buildings: 1,
                ..baseline
            },
            ResourceStockpileInputs {
                after_income: -2,
                ..baseline
            },
            ResourceStockpileInputs {
                before_resource_units: 0,
                ..baseline
            },
            ResourceStockpileInputs {
                selected_industry: Some(IndustryType::Beer),
                ..baseline
            },
            ResourceStockpileInputs {
                has_product_follow_up: false,
                ..baseline
            },
        ] {
            assert_eq!(resource_stockpile_penalty_from_inputs(input), 0.0);
        }
    }

    fn post_sale_cycle_inputs() -> PostSaleProductCycleInputs {
        PostSaleProductCycleInputs {
            railroad: true,
            low_score: true,
            late_deck: true,
            action_type: ActionType::BuildBuilding,
            selected_industry: Some(IndustryType::Goods),
            before_income: 3,
            before_sellable_units: 0,
            after_sellable_units: 1,
            has_converted_product: true,
            cards_after_build: 4,
            required_future_actions: Some(3),
            after_money: 4,
            after_debt: 0,
        }
    }

    #[test]
    fn post_sale_cycle_rewards_exactly_reachable_product_with_one_action_reserve() {
        assert_eq!(
            post_sale_product_cycle_bonus_from_inputs(post_sale_cycle_inputs()),
            1.0
        );
    }

    #[test]
    fn post_sale_cycle_requires_real_prior_conversion_and_action_budget() {
        let baseline = post_sale_cycle_inputs();
        for input in [
            PostSaleProductCycleInputs {
                has_converted_product: false,
                ..baseline
            },
            PostSaleProductCycleInputs {
                cards_after_build: 3,
                ..baseline
            },
            PostSaleProductCycleInputs {
                required_future_actions: Some(4),
                ..baseline
            },
            PostSaleProductCycleInputs {
                before_sellable_units: 1,
                after_sellable_units: 2,
                ..baseline
            },
            PostSaleProductCycleInputs {
                before_income: -1,
                ..baseline
            },
        ] {
            assert_eq!(post_sale_product_cycle_bonus_from_inputs(input), 0.0);
        }
    }

    #[test]
    fn mature_product_cycle_requires_income_and_card_runway() {
        let baseline = MatureProductCycleInputs {
            base_cycle_bonus: 1.0,
            before_income: 10,
            cards_after_build: 5,
        };
        assert_eq!(mature_product_cycle_bonus_from_inputs(baseline), 1.0);
        for input in [
            MatureProductCycleInputs {
                base_cycle_bonus: 0.0,
                ..baseline
            },
            MatureProductCycleInputs {
                before_income: 9,
                ..baseline
            },
            MatureProductCycleInputs {
                cards_after_build: 4,
                ..baseline
            },
        ] {
            assert_eq!(mature_product_cycle_bonus_from_inputs(input), 0.0);
        }
    }

    fn financed_product_cycle_inputs() -> FinancedProductCycleInputs {
        FinancedProductCycleInputs {
            railroad: true,
            deck_empty: true,
            low_score: true,
            first_action_product_build: true,
            no_existing_product: true,
            no_sell_after_build: true,
            loan_income_loss: 3,
            cards_after_loan: 6,
        }
    }

    #[test]
    fn financed_product_cycle_accepts_only_a_cheap_funded_window() {
        assert!(financed_product_cycle_is_eligible(
            financed_product_cycle_inputs()
        ));
        assert!(!financed_product_cycle_is_eligible(
            FinancedProductCycleInputs {
                loan_income_loss: 4,
                ..financed_product_cycle_inputs()
            }
        ));
        assert!(!financed_product_cycle_is_eligible(
            FinancedProductCycleInputs {
                cards_after_loan: 1,
                ..financed_product_cycle_inputs()
            }
        ));
    }

    #[test]
    fn financed_product_cycle_requires_every_narrow_gate() {
        let baseline = financed_product_cycle_inputs();
        for input in [
            FinancedProductCycleInputs {
                railroad: false,
                ..baseline
            },
            FinancedProductCycleInputs {
                deck_empty: false,
                ..baseline
            },
            FinancedProductCycleInputs {
                low_score: false,
                ..baseline
            },
            FinancedProductCycleInputs {
                first_action_product_build: false,
                ..baseline
            },
            FinancedProductCycleInputs {
                no_existing_product: false,
                ..baseline
            },
            FinancedProductCycleInputs {
                no_sell_after_build: false,
                ..baseline
            },
        ] {
            assert!(!financed_product_cycle_is_eligible(input));
        }
    }

    #[test]
    fn financed_product_cycle_accepts_an_already_opened_future_turn() {
        let mut runner = GameRunner::new(3, Some(91_239));
        for _ in 0..3 {
            runner.start_turn();
            runner.actions_remaining_in_turn = 0;
            runner.end_turn();
        }

        runner.start_turn();
        let actor = runner.framework.current_player;
        let money_before = runner.framework.board.state.players[actor].money;
        assert_eq!(runner.actions_remaining_in_turn, 2);
        assert!(advance_to_actor_without_opponent_actions(
            &mut runner,
            actor
        ));
        assert_eq!(runner.actions_remaining_in_turn, 2);
        assert_eq!(
            runner.framework.board.state.players[actor].money,
            money_before
        );
    }

    #[test]
    fn financed_product_cycle_crosses_two_turn_orders_when_actor_moves_last() {
        let mut runner = GameRunner::new(3, Some(91_238));
        runner.framework.board.state.turn_order = vec![0, 1, 2];
        runner.framework.current_player = 0;
        runner.start_turn();
        runner.framework.board.state.players[0].spent_this_turn = 20;
        runner.actions_remaining_in_turn = 0;
        runner.end_turn();
        runner.start_turn();

        assert_eq!(runner.framework.current_player, 1);
        assert!(advance_to_actor_without_opponent_actions(&mut runner, 0));
        assert_eq!(runner.framework.current_player, 0);
        assert_eq!(runner.actions_remaining_in_turn, 2);
    }

    fn contested_external_beer_inputs() -> ContestedExternalBeerPlanInputs {
        ContestedExternalBeerPlanInputs {
            railroad: true,
            low_score: true,
            late_deck: true,
            first_action_product_build: true,
            no_sell_after_build: true,
            route_opens_sell: true,
            route_ends_turn: true,
            own_beer_shortfall: true,
            opponent_competes_for_external_beer: true,
        }
    }

    #[test]
    fn contested_external_beer_penalizes_only_exposed_cross_turn_plan() {
        assert_eq!(
            contested_external_beer_plan_penalty_from_inputs(contested_external_beer_inputs()),
            -1.0
        );

        let baseline = contested_external_beer_inputs();
        for input in [
            ContestedExternalBeerPlanInputs {
                railroad: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                low_score: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                late_deck: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                first_action_product_build: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                no_sell_after_build: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                route_opens_sell: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                route_ends_turn: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                own_beer_shortfall: false,
                ..baseline
            },
            ContestedExternalBeerPlanInputs {
                opponent_competes_for_external_beer: false,
                ..baseline
            },
        ] {
            assert_eq!(contested_external_beer_plan_penalty_from_inputs(input), 0.0);
        }
    }

    fn same_turn_sale_guard_inputs() -> SameTurnSaleGuardInputs {
        SameTurnSaleGuardInputs {
            action_type: ActionType::BuildBuilding,
            selected_industry: Some(IndustryType::Goods),
            before_sell_targets: 0,
            after_sell_targets: 1,
            same_turn_completion_safe: true,
            after_income: -8,
            after_money: 4,
            safety_weight: 1.0,
            economic_guard_weight: 1.0,
        }
    }

    #[test]
    fn same_turn_sale_guard_refunds_only_temporary_settlement_penalties() {
        assert_eq!(settlement_safety_score(-8, 4), -18.0);
        assert_eq!(imminent_settlement_guard_penalty(-8, 4), -26.0);
        assert_eq!(
            same_turn_sale_guard_rebate_from_inputs(same_turn_sale_guard_inputs()),
            44.0
        );

        let scaled = SameTurnSaleGuardInputs {
            safety_weight: 0.5,
            economic_guard_weight: 0.25,
            ..same_turn_sale_guard_inputs()
        };
        assert_eq!(same_turn_sale_guard_rebate_from_inputs(scaled), 15.5);
    }

    #[test]
    fn same_turn_sale_guard_requires_new_product_sale_and_safe_reply() {
        let baseline = same_turn_sale_guard_inputs();
        for input in [
            SameTurnSaleGuardInputs {
                action_type: ActionType::BuildRailroad,
                ..baseline
            },
            SameTurnSaleGuardInputs {
                selected_industry: Some(IndustryType::Coal),
                ..baseline
            },
            SameTurnSaleGuardInputs {
                before_sell_targets: 1,
                ..baseline
            },
            SameTurnSaleGuardInputs {
                after_sell_targets: 0,
                ..baseline
            },
            SameTurnSaleGuardInputs {
                same_turn_completion_safe: false,
                ..baseline
            },
        ] {
            assert_eq!(same_turn_sale_guard_rebate_from_inputs(input), 0.0);
        }
    }

    fn late_canal_stall_inputs() -> LateCanalStallInputs {
        LateCanalStallInputs {
            last_two_card_window: true,
            low_score: true,
            severe_debt: true,
            action_type: ActionType::BuildBuilding,
            secured_vp: false,
            actor_flipped_building: false,
            income_improved: false,
            new_legal_sell: false,
            route_progress: false,
        }
    }

    #[test]
    fn late_canal_stall_penalizes_only_last_chance_stagnation() {
        assert_eq!(
            late_canal_stall_penalty_from_inputs(late_canal_stall_inputs()),
            -1.0
        );

        let baseline = late_canal_stall_inputs();
        for input in [
            LateCanalStallInputs {
                last_two_card_window: false,
                ..baseline
            },
            LateCanalStallInputs {
                low_score: false,
                ..baseline
            },
            LateCanalStallInputs {
                severe_debt: false,
                ..baseline
            },
            LateCanalStallInputs {
                action_type: ActionType::Pass,
                ..baseline
            },
        ] {
            assert_eq!(late_canal_stall_penalty_from_inputs(input), 0.0);
        }
    }

    #[test]
    fn late_canal_stall_requires_actor_owned_frontier_progress() {
        let baseline = late_canal_stall_inputs();
        for input in [
            LateCanalStallInputs {
                secured_vp: true,
                ..baseline
            },
            LateCanalStallInputs {
                actor_flipped_building: true,
                ..baseline
            },
            LateCanalStallInputs {
                income_improved: true,
                ..baseline
            },
            LateCanalStallInputs {
                new_legal_sell: true,
                ..baseline
            },
            LateCanalStallInputs {
                route_progress: true,
                ..baseline
            },
        ] {
            assert_eq!(late_canal_stall_penalty_from_inputs(input), 0.0);
        }

        // An opponent-only flip leaves the actor-owned flag false.
        assert_eq!(late_canal_stall_penalty_from_inputs(baseline), -1.0);
    }

    fn late_railroad_backlog_inputs() -> LateRailroadBacklogInputs {
        LateRailroadBacklogInputs {
            railroad: true,
            deck_empty: true,
            low_score: true,
            cards_before: 4,
            action_type: ActionType::BuildBuilding,
            selected_industry: Some(IndustryType::Goods),
            before_sellable_units: 2,
            after_sellable_units: 3,
            before_sell_targets: 0,
            after_sell_targets: 0,
            secured_batch_beer: false,
            actor_flipped_building: false,
            income_improved: false,
        }
    }

    #[test]
    fn late_railroad_backlog_penalizes_third_unconverted_product() {
        assert_eq!(
            late_railroad_backlog_penalty_from_inputs(late_railroad_backlog_inputs()),
            -1.0
        );
    }

    #[test]
    fn late_railroad_backlog_preserves_early_batches_and_real_progress() {
        let baseline = late_railroad_backlog_inputs();
        for input in [
            LateRailroadBacklogInputs {
                railroad: false,
                ..baseline
            },
            LateRailroadBacklogInputs {
                deck_empty: false,
                ..baseline
            },
            LateRailroadBacklogInputs {
                low_score: false,
                ..baseline
            },
            LateRailroadBacklogInputs {
                cards_before: 5,
                ..baseline
            },
            LateRailroadBacklogInputs {
                selected_industry: Some(IndustryType::Coal),
                ..baseline
            },
            LateRailroadBacklogInputs {
                before_sellable_units: 1,
                after_sellable_units: 2,
                ..baseline
            },
            LateRailroadBacklogInputs {
                after_sell_targets: 1,
                ..baseline
            },
            LateRailroadBacklogInputs {
                secured_batch_beer: true,
                ..baseline
            },
            LateRailroadBacklogInputs {
                actor_flipped_building: true,
                ..baseline
            },
            LateRailroadBacklogInputs {
                income_improved: true,
                ..baseline
            },
        ] {
            assert_eq!(late_railroad_backlog_penalty_from_inputs(input), 0.0);
        }
    }

    fn late_railroad_conversion_stall_inputs() -> LateRailroadConversionStallInputs {
        LateRailroadConversionStallInputs {
            railroad: true,
            deck_empty: true,
            low_score: true,
            cards_before: 3,
            action_type: ActionType::DevelopDouble,
            selected_industry: Some(IndustryType::Goods),
            existing_unsold_product: true,
            before_sell_targets: 0,
            after_sell_targets: 0,
            secured_vp: false,
            actor_flipped_building: false,
            income_improved: false,
            route_progress: false,
            beer_shortfall_reduced: false,
            batch_sale_feasible: false,
        }
    }

    #[test]
    fn late_railroad_conversion_stall_penalizes_dead_develop_and_product_build() {
        let develop = late_railroad_conversion_stall_inputs();
        assert_eq!(
            late_railroad_conversion_stall_penalty_from_inputs(develop),
            -1.0
        );
        assert_eq!(
            late_railroad_conversion_stall_penalty_from_inputs(LateRailroadConversionStallInputs {
                action_type: ActionType::BuildBuilding,
                ..develop
            }),
            -1.0
        );
    }

    #[test]
    fn late_railroad_conversion_stall_preserves_funded_goods_batch() {
        let baseline = late_railroad_conversion_stall_inputs();
        for input in [
            LateRailroadConversionStallInputs {
                batch_sale_feasible: true,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                after_sell_targets: 3,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                beer_shortfall_reduced: true,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                route_progress: true,
                ..baseline
            },
        ] {
            assert_eq!(
                late_railroad_conversion_stall_penalty_from_inputs(input),
                0.0
            );
        }
    }

    #[test]
    fn late_railroad_conversion_stall_is_late_and_narrow() {
        let baseline = late_railroad_conversion_stall_inputs();
        for input in [
            LateRailroadConversionStallInputs {
                railroad: false,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                deck_empty: false,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                low_score: false,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                cards_before: 4,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                existing_unsold_product: false,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                before_sell_targets: 1,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                action_type: ActionType::BuildRailroad,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                action_type: ActionType::BuildBuilding,
                selected_industry: Some(IndustryType::Beer),
                ..baseline
            },
            LateRailroadConversionStallInputs {
                secured_vp: true,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                actor_flipped_building: true,
                ..baseline
            },
            LateRailroadConversionStallInputs {
                income_improved: true,
                ..baseline
            },
        ] {
            assert_eq!(
                late_railroad_conversion_stall_penalty_from_inputs(input),
                0.0
            );
        }
    }

    fn add_batch_test_building(
        runner: &mut GameRunner,
        player: usize,
        location: usize,
        industry: IndustryType,
        level: crate::core::types::IndustryLevel,
    ) {
        let state = &mut runner.framework.board.state;
        state.bl_to_building.insert(
            location,
            crate::core::building::BuiltBuilding::build(
                industry,
                level,
                location as u8,
                crate::core::player::PlayerId::from_usize(player),
            ),
        );
        state.build_locations_occupied.insert(location);
        state.player_building_mask[player].insert(location);
        match industry {
            IndustryType::Coal => state.coal_locations.insert(location),
            IndustryType::Iron => state.iron_locations.insert(location),
            IndustryType::Beer => state.beer_locations.insert(location),
            IndustryType::Cotton | IndustryType::Goods | IndustryType::Pottery => {}
        }
    }

    fn set_batch_test_hand(runner: &mut GameRunner, player: usize, cards: usize) {
        runner.framework.board.state.players[player].hand.cards = (0..cards)
            .map(|_| {
                crate::core::types::Card::new(CardType::Industry(
                    crate::core::types::IndustrySet::new_from_industry_types(&[
                        IndustryType::Goods,
                    ]),
                ))
            })
            .collect();
    }

    #[test]
    fn complete_batch_sale_counts_three_goods_beer_once_per_source() {
        use crate::core::types::IndustryLevel;
        use crate::market::merchants::{MerchantTile, MerchantTileType};

        let mut runner = GameRunner::new(2, Some(91_240));
        let actor = runner.framework.current_player;
        let state = &mut runner.framework.board.state;
        state.place_link(actor, 31);
        for slot in &mut state.trade_post_slots {
            *slot = None;
        }
        state.trade_post_slots[1] = Some(MerchantTile::from_type(MerchantTileType::Goods));
        state.trade_post_beer.clear();

        // Goods II + III + V need 1 + 0 + 2 Beer. One Sell can flip all three.
        add_batch_test_building(
            &mut runner,
            actor,
            36,
            IndustryType::Goods,
            IndustryLevel::II,
        );
        add_batch_test_building(
            &mut runner,
            actor,
            37,
            IndustryType::Goods,
            IndustryLevel::III,
        );
        add_batch_test_building(
            &mut runner,
            actor,
            39,
            IndustryType::Goods,
            IndustryLevel::V,
        );
        for location in [3, 15, 47] {
            add_batch_test_building(
                &mut runner,
                actor,
                location,
                IndustryType::Beer,
                IndustryLevel::II,
            );
        }

        assert!(complete_batch_sale_is_legal(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));

        runner
            .framework
            .board
            .state
            .bl_to_building
            .get_mut(&47)
            .expect("test Brewery should exist")
            .resource_amt = 0;
        assert!(!complete_batch_sale_is_legal(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));

        // The Oxford merchant barrel supplies exactly the third unit. It may
        // be allocated once, not once per Goods tile.
        runner.framework.board.state.trade_post_beer.insert(1);
        assert!(complete_batch_sale_is_legal(
            &runner,
            actor,
            BatchBeerScope::AnyLegal
        ));
        runner
            .framework
            .board
            .state
            .bl_to_building
            .get_mut(&15)
            .expect("test Brewery should exist")
            .resource_amt = 0;
        assert!(!complete_batch_sale_is_legal(
            &runner,
            actor,
            BatchBeerScope::AnyLegal
        ));
    }

    fn beer_backlog_recovery_inputs() -> BeerBacklogRecoveryInputs {
        BeerBacklogRecoveryInputs {
            railroad: true,
            low_score: true,
            development_action: true,
            develops_beer: true,
            existing_unsold_products: 2,
            before_sell_targets: 0,
            beer_shortfall: 4,
            cards_after: 7,
            opens_affordable_brewery: true,
        }
    }

    #[test]
    fn beer_backlog_recovery_requires_every_narrow_gate() {
        let baseline = beer_backlog_recovery_inputs();
        assert_eq!(beer_backlog_recovery_bonus_from_inputs(baseline), 1.0);
        for input in [
            BeerBacklogRecoveryInputs {
                railroad: false,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                low_score: false,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                development_action: false,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                develops_beer: false,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                existing_unsold_products: 1,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                before_sell_targets: 1,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                beer_shortfall: 2,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                cards_after: 4,
                ..baseline
            },
            BeerBacklogRecoveryInputs {
                opens_affordable_brewery: false,
                ..baseline
            },
        ] {
            assert_eq!(beer_backlog_recovery_bonus_from_inputs(input), 0.0);
        }
    }

    #[test]
    fn batch_plan_can_build_missing_beer_before_selling_three_goods() {
        use crate::core::types::{Card, Era, IndustryLevel};
        use crate::market::merchants::{MerchantTile, MerchantTileType};

        let mut runner = GameRunner::new(2, Some(91_243));
        let actor = runner.framework.current_player;
        runner.game_phase = GamePhase::Railroad;
        runner.framework.board.state.era = Era::Railroad;
        runner.framework.board.state.deck.cards.clear();
        runner.framework.board.state.players[actor].money = 100;
        runner.framework.board.state.players[actor].hand.cards =
            (0..3).map(|_| Card::new(CardType::WildLocation)).collect();
        while runner.framework.board.state.players[actor]
            .industry_mat
            .get_lowest_level(IndustryType::Beer)
            == IndustryLevel::I
        {
            runner.framework.board.state.players[actor]
                .industry_mat
                .pop_tile(IndustryType::Beer);
        }
        runner.framework.board.state.place_link(actor, 31);
        for slot in &mut runner.framework.board.state.trade_post_slots {
            *slot = None;
        }
        runner.framework.board.state.trade_post_slots[1] =
            Some(MerchantTile::from_type(MerchantTileType::Goods));
        runner.framework.board.state.trade_post_beer.clear();

        // Goods II + III + V need three Beer. One owned Brewery exists, and
        // two more real Brewery actions plus one Sell fit exactly in the hand.
        for (location, level) in [
            (36, IndustryLevel::II),
            (37, IndustryLevel::III),
            (39, IndustryLevel::V),
        ] {
            add_batch_test_building(&mut runner, actor, location, IndustryType::Goods, level);
        }
        add_batch_test_building(&mut runner, actor, 3, IndustryType::Beer, IndustryLevel::II);

        assert!(!complete_batch_sale_is_legal(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));
        assert!(batch_sale_plan_is_feasible(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));

        runner.framework.board.state.players[actor]
            .hand
            .cards
            .truncate(2);
        assert!(!batch_sale_plan_is_feasible(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));
    }

    #[test]
    fn batch_plan_preserves_required_brewery_card_across_route_discard() {
        use crate::core::types::{Card, Era, IndustryLevel, IndustrySet};
        use crate::market::merchants::{MerchantTile, MerchantTileType};

        let mut runner = GameRunner::new(2, Some(91_244));
        let actor = runner.framework.current_player;
        runner.game_phase = GamePhase::Railroad;
        runner.framework.board.state.era = Era::Railroad;
        runner.framework.board.state.deck.cards.clear();
        runner.framework.board.state.players[actor].money = 100;
        let card = |industry| {
            Card::new(CardType::Industry(IndustrySet::new_from_industry_types(&[
                industry,
            ])))
        };
        runner.framework.board.state.players[actor].hand.cards = vec![
            card(IndustryType::Beer),
            card(IndustryType::Coal),
            card(IndustryType::Goods),
        ];
        while runner.framework.board.state.players[actor]
            .industry_mat
            .get_lowest_level(IndustryType::Beer)
            == IndustryLevel::I
        {
            runner.framework.board.state.players[actor]
                .industry_mat
                .pop_tile(IndustryType::Beer);
        }
        for slot in &mut runner.framework.board.state.trade_post_slots {
            *slot = None;
        }
        runner.framework.board.state.trade_post_slots[1] =
            Some(MerchantTile::from_type(MerchantTileType::Goods));
        runner.framework.board.state.trade_post_beer.clear();
        runner.framework.board.state.place_link(actor, 31);
        add_batch_test_building(
            &mut runner,
            actor,
            36,
            IndustryType::Goods,
            IndustryLevel::II,
        );
        add_batch_test_building(&mut runner, actor, 5, IndustryType::Coal, IndustryLevel::II);

        assert!(!complete_batch_sale_is_legal(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));
        // The only completion is Route r16 (discard Coal), Brewery b47
        // (discard Beer), then Sell. Discard-invariant route pruning used to
        // retain the Beer discard and lose this valid completion.
        assert!(batch_sale_plan_is_feasible(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));
    }

    #[test]
    fn batch_plan_requires_joint_routes_not_maximum_individual_distance() {
        use crate::core::types::{Era, IndustryLevel};
        use crate::market::merchants::{MerchantTile, MerchantTileType};

        let mut runner = GameRunner::new(2, Some(91_241));
        let actor = runner.framework.current_player;
        runner.game_phase = GamePhase::Railroad;
        runner.framework.board.state.era = Era::Railroad;
        runner.framework.board.state.deck.cards.clear();
        runner.framework.board.state.players[actor].money = 100;
        set_batch_test_hand(&mut runner, actor, 2);
        for slot in &mut runner.framework.board.state.trade_post_slots {
            *slot = None;
        }
        runner.framework.board.state.trade_post_slots[1] =
            Some(MerchantTile::from_type(MerchantTileType::Goods));
        runner.framework.board.state.trade_post_slots[5] =
            Some(MerchantTile::from_type(MerchantTileType::Goods));
        runner.framework.board.state.trade_post_beer.clear();

        // Each zero-Beer Goods is one link from its own merchant, but the two
        // links are on separate branches. With no Brewery, double rail is not
        // legal, so two cards cannot cover both links plus the Sell.
        add_batch_test_building(
            &mut runner,
            actor,
            12,
            IndustryType::Goods,
            IndustryLevel::III,
        );
        add_batch_test_building(
            &mut runner,
            actor,
            36,
            IndustryType::Goods,
            IndustryLevel::III,
        );
        add_batch_test_building(
            &mut runner,
            actor,
            13,
            IndustryType::Coal,
            IndustryLevel::II,
        );
        add_batch_test_building(
            &mut runner,
            actor,
            38,
            IndustryType::Coal,
            IndustryLevel::II,
        );
        assert!(!batch_sale_plan_is_feasible(
            &runner,
            actor,
            BatchBeerScope::AnyLegal
        ));
    }

    #[test]
    fn batch_plan_preserves_three_goods_behind_one_shared_route() {
        use crate::core::types::{Era, IndustryLevel};
        use crate::market::merchants::{MerchantTile, MerchantTileType};

        let mut runner = GameRunner::new(2, Some(91_242));
        let actor = runner.framework.current_player;
        runner.game_phase = GamePhase::Railroad;
        runner.framework.board.state.era = Era::Railroad;
        runner.framework.board.state.deck.cards.clear();
        runner.framework.board.state.players[actor].money = 100;
        set_batch_test_hand(&mut runner, actor, 2);
        for slot in &mut runner.framework.board.state.trade_post_slots {
            *slot = None;
        }
        runner.framework.board.state.trade_post_slots[1] =
            Some(MerchantTile::from_type(MerchantTileType::Goods));
        runner.framework.board.state.trade_post_beer.clear();

        for location in [36, 37, 39] {
            add_batch_test_building(
                &mut runner,
                actor,
                location,
                IndustryType::Goods,
                IndustryLevel::III,
            );
        }
        add_batch_test_building(
            &mut runner,
            actor,
            38,
            IndustryType::Coal,
            IndustryLevel::II,
        );
        assert!(batch_sale_plan_is_feasible(
            &runner,
            actor,
            BatchBeerScope::OwnedOnly
        ));
    }

    #[test]
    fn sale_frontier_build_signal_is_narrow_and_opt_in() {
        let mut before = PositionSnapshot::default();
        let mut after = before.clone();
        after.available_sell_targets = 2;
        let build = test_intent(ActionType::BuildBuilding);
        assert_eq!(sale_frontier_build_signal(&before, &after, &build), 7.0);
        assert_eq!(
            sale_frontier_build_signal(&before, &after, &test_intent(ActionType::BuildRailroad)),
            0.0
        );

        before.available_sell_targets = 1;
        after.available_sell_targets = 2;
        assert_eq!(sale_frontier_build_signal(&before, &after, &build), 0.0);
        assert_eq!(
            RuleDecisionConfig::default().sale_frontier_build_weight,
            0.0
        );
    }

    #[test]
    fn default_loan_cost_blends_maturity_with_actual_excess_loss() {
        let config = RuleDecisionConfig::default();
        assert_eq!(config.loan_maturity_weight, 0.5);
        assert_eq!(config.loan_excess_income_loss_weight, 0.5);
        assert_eq!(config.low_cost_loan_runway_weight, 14.8);
        assert_eq!(config.recovery_product_plan_weight, 12.0);
        assert_eq!(config.conversion_feasibility_weight, 0.0);
        assert_eq!(config.resource_stockpile_weight, 1.0);
        assert_eq!(config.post_sale_product_cycle_weight, 0.75);
        assert_eq!(config.mature_product_cycle_weight, 1.625);
        assert_eq!(config.financed_product_cycle_weight, 0.0);
        assert_eq!(config.same_turn_sale_guard_weight, 0.0);
        assert_eq!(config.contested_external_beer_plan_weight, 3.5);
        assert_eq!(config.late_canal_stall_weight, 10.5);
        assert_eq!(config.late_railroad_backlog_weight, 1.01);
        assert_eq!(config.beer_backlog_recovery_weight, 2.18);
        assert_eq!(config.late_railroad_conversion_stall_weight, 0.9);
    }

    #[test]
    fn recovery_priority_is_inactive_for_healthy_positions() {
        let input = recovery_inputs(ActionType::Scout);
        assert_eq!(recovery_priority_from_inputs(input), 0.0);
    }
}
