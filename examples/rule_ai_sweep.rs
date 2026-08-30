//! Fast, paired parameter screening for the CPU rule tree.
//!
//! Every candidate sees the same game seeds.  The command is deliberately
//! small and deterministic: it is a screening tool, not a promotion gate.
//! Example:
//!
//!   cargo run --release --example rule_ai_sweep -- 2 3 20260830 axis=action values=0.5,0.75,1,1.25

use fast_brass::game::rule_ai::{rank_rule_actions, RuleDecisionConfig};
use fast_brass::game::runner::GameRunner;
use fast_brass::game::search::official_winners;
use serde_json::json;
use std::collections::BTreeMap;
use std::env;
use std::time::Instant;

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

#[derive(Default)]
struct CandidateResult {
    scores: Vec<u16>,
    game_minima: Vec<u16>,
    game_maxima: Vec<u16>,
    game_top_margins: Vec<u16>,
    actions: u64,
    elapsed_us: u128,
    action_counts: BTreeMap<String, u64>,
}

fn play_game(
    players: usize,
    seed: u64,
    config: &RuleDecisionConfig,
) -> Result<(Vec<u16>, u64, u128, BTreeMap<String, u64>), String> {
    let mut runner = GameRunner::new(players, Some(seed));
    let started = Instant::now();
    let mut actions = 0_u64;
    let mut counts = BTreeMap::<String, u64>::new();
    while !runner.is_game_finished() {
        advance_to_decision(&mut runner)?;
        if runner.is_game_finished() {
            break;
        }
        let ranked = rank_rule_actions(&runner, config)?;
        let best = ranked
            .first()
            .ok_or_else(|| "rule policy returned no action".to_string())?;
        let root = format!("{:?}", best.action.intent.action_type);
        *counts.entry(root).or_default() += 1;
        best.action.apply(&mut runner)?;
        actions += 1;
        if actions > 512 {
            return Err("rule game exceeded 512 actions".to_string());
        }
    }
    resolve_shortfalls(&mut runner);
    // Validate the terminal state and rulebook tie-break path even though all
    // seats use the same policy in this screening command.
    let _ = official_winners(&runner)?;
    let scores = runner
        .framework
        .board
        .state
        .players
        .iter()
        .map(|player| player.victory_points)
        .collect::<Vec<_>>();
    Ok((scores, actions, started.elapsed().as_micros(), counts))
}

fn set_axis(config: &mut RuleDecisionConfig, axis: &str, value: f64) -> Result<(), String> {
    match axis {
        "action" | "action_bias" => config.action_bias_weight = value,
        "economic" | "economic_guard" => config.economic_guard_weight = value,
        "strategic" | "strategic_priority" => config.strategic_priority_weight = value,
        "lifecycle" | "lifecycle_priority" => config.lifecycle_priority_weight = value,
        "sale_frontier" | "sale_frontier_build" => config.sale_frontier_build_weight = value,
        "loan_maturity" => config.loan_maturity_weight = value,
        "loan_excess_loss" | "loan_excess_income_loss" => {
            config.loan_excess_income_loss_weight = value
        }
        "low_cost_loan_runway" | "loan_runway" => config.low_cost_loan_runway_weight = value,
        "recovery_product" | "recovery_product_plan" => {
            config.recovery_product_plan_weight = value
        }
        "conversion" | "conversion_feasibility" => {
            config.conversion_feasibility_weight = value
        }
        "resource_stockpile" | "stockpile" => config.resource_stockpile_weight = value,
        "post_sale_product_cycle" | "product_cycle" => {
            config.post_sale_product_cycle_weight = value
        }
        "mature_product_cycle" | "mature_cycle" => config.mature_product_cycle_weight = value,
        "financed_product_cycle" | "financed_cycle" => {
            config.financed_product_cycle_weight = value
        }
        "same_turn_sale_guard" | "sale_guard" => config.same_turn_sale_guard_weight = value,
        "contested_external_beer_plan" | "contested_beer" => {
            config.contested_external_beer_plan_weight = value
        }
        "late_canal_stall" | "canal_stall" => config.late_canal_stall_weight = value,
        "late_railroad_backlog" | "product_backlog" => {
            config.late_railroad_backlog_weight = value
        }
        "beer_backlog_recovery" | "beer_recovery" => {
            config.beer_backlog_recovery_weight = value
        }
        "late_railroad_conversion_stall" | "conversion_stall" => {
            config.late_railroad_conversion_stall_weight = value
        }
        "immediate_vp" => config.immediate_vp_weight = value,
        "potential_vp" => config.potential_vp_weight = value,
        "income" => config.income_weight = value,
        "network" => config.network_weight = value,
        "industry" => config.industry_weight = value,
        "cash" => config.cash_weight = value,
        "resource" => config.resource_weight = value,
        "safety" => config.safety_weight = value,
        "card" => config.card_weight = value,
        "tempo" => config.tempo_weight = value,
        "lookahead_discount" => config.lookahead_discount = value,
        _ => {
            return Err(format!(
                "unknown axis {axis}; use action, economic, strategic, lifecycle, sale_frontier, loan_maturity, loan_excess_loss, low_cost_loan_runway, recovery_product, conversion_feasibility, resource_stockpile, post_sale_product_cycle, mature_product_cycle, financed_product_cycle, same_turn_sale_guard, contested_external_beer_plan, late_canal_stall, late_railroad_backlog, beer_backlog_recovery, late_railroad_conversion_stall, potential_vp, safety, lookahead_discount, or a continuous weight"
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_axis_exposes_all_experimental_weights() {
        let mut config = RuleDecisionConfig::default();
        for (axis, value) in [
            ("potential_vp", 1.7),
            ("safety", 0.8),
            ("lookahead_discount", 0.4),
            ("sale_frontier", 0.5),
            ("loan_maturity", 0.75),
            ("loan_excess_loss", 2.0),
            ("low_cost_loan_runway", 15.0),
            ("recovery_product", 16.0),
            ("conversion_feasibility", 0.75),
            ("resource_stockpile", 4.0),
            ("post_sale_product_cycle", 2.0),
            ("mature_product_cycle", 1.75),
            ("financed_product_cycle", 0.3),
            ("same_turn_sale_guard", 1.0),
            ("contested_external_beer_plan", 3.5),
            ("late_canal_stall", 12.0),
            ("late_railroad_backlog", 1.25),
            ("beer_backlog_recovery", 2.25),
            ("late_railroad_conversion_stall", 0.9),
        ] {
            set_axis(&mut config, axis, value).unwrap();
        }
        assert_eq!(config.potential_vp_weight, 1.7);
        assert_eq!(config.safety_weight, 0.8);
        assert_eq!(config.lookahead_discount, 0.4);
        assert_eq!(config.sale_frontier_build_weight, 0.5);
        assert_eq!(config.loan_maturity_weight, 0.75);
        assert_eq!(config.loan_excess_income_loss_weight, 2.0);
        assert_eq!(config.low_cost_loan_runway_weight, 15.0);
        assert_eq!(config.recovery_product_plan_weight, 16.0);
        assert_eq!(config.conversion_feasibility_weight, 0.75);
        assert_eq!(config.resource_stockpile_weight, 4.0);
        assert_eq!(config.post_sale_product_cycle_weight, 2.0);
        assert_eq!(config.mature_product_cycle_weight, 1.75);
        assert_eq!(config.financed_product_cycle_weight, 0.3);
        assert_eq!(config.same_turn_sale_guard_weight, 1.0);
        assert_eq!(config.contested_external_beer_plan_weight, 3.5);
        assert_eq!(config.late_canal_stall_weight, 12.0);
        assert_eq!(config.late_railroad_backlog_weight, 1.25);
        assert_eq!(config.beer_backlog_recovery_weight, 2.25);
        assert_eq!(config.late_railroad_conversion_stall_weight, 0.9);
    }

    #[test]
    fn set_axis_rejects_unknown_axis() {
        let mut config = RuleDecisionConfig::default();
        let error = set_axis(&mut config, "does_not_exist", 1.0).unwrap_err();
        assert!(error.contains("unknown axis"));
    }
}

fn arg_value<'a>(args: &'a [String], prefix: &str) -> Option<&'a str> {
    args.iter().find_map(|value| value.strip_prefix(prefix))
}

fn parse_values(args: &[String]) -> Result<Vec<f64>, String> {
    let raw = arg_value(args, "values=").unwrap_or("0.5,0.75,1.0,1.25,1.5");
    let values = raw
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|error| format!("invalid sweep value {value:?}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("values must contain finite non-negative numbers".to_string());
    }
    Ok(values)
}

fn main() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    let games = args
        .get(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2);
    let players = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let base_seed = args
        .get(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2_026_0830);
    let axis = arg_value(&args, "axis=").unwrap_or("action");
    let values = parse_values(&args)?;
    let depth = arg_value(&args, "depth=")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid depth: {error}"))?
        .unwrap_or(1);
    let branching = arg_value(&args, "branching=")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid branching: {error}"))?
        .unwrap_or(1);
    if games == 0 || !(2..=4).contains(&players) {
        return Err("games must be positive and players must be 2, 3, or 4".to_string());
    }

    let started = Instant::now();
    for value in values {
        let mut config = RuleDecisionConfig::default();
        config.lookahead_depth = depth;
        config.lookahead_branching = branching;
        set_axis(&mut config, axis, value)?;
        let mut result = CandidateResult::default();
        for game in 0..games {
            let (scores, actions, elapsed_us, counts) =
                play_game(players, base_seed + game, &config)?;
            result
                .game_minima
                .push(scores.iter().copied().min().unwrap_or_default());
            result
                .game_maxima
                .push(scores.iter().copied().max().unwrap_or_default());
            let mut ordered_scores = scores.clone();
            ordered_scores.sort_unstable_by(|left, right| right.cmp(left));
            let top = ordered_scores.first().copied().unwrap_or_default();
            let runner_up = ordered_scores.get(1).copied().unwrap_or(top);
            result.game_top_margins.push(top.saturating_sub(runner_up));
            result.scores.extend(scores);
            result.actions += actions;
            result.elapsed_us += elapsed_us;
            for (action, count) in counts {
                *result.action_counts.entry(action).or_default() += count;
            }
        }
        let count = result.scores.len().max(1) as f64;
        let mean = result
            .scores
            .iter()
            .map(|score| f64::from(*score))
            .sum::<f64>()
            / count;
        let min = result.scores.iter().copied().min().unwrap_or_default();
        let max = result.scores.iter().copied().max().unwrap_or_default();
        let zero = result.scores.iter().filter(|score| **score == 0).count();
        let mean_game_min = result
            .game_minima
            .iter()
            .map(|score| f64::from(*score))
            .sum::<f64>()
            / result.game_minima.len().max(1) as f64;
        let worst_game_min = result.game_minima.iter().copied().min().unwrap_or_default();
        let mean_top_vp = result
            .game_maxima
            .iter()
            .map(|score| f64::from(*score))
            .sum::<f64>()
            / result.game_maxima.len().max(1) as f64;
        let mean_top_margin = result
            .game_top_margins
            .iter()
            .map(|margin| f64::from(*margin))
            .sum::<f64>()
            / result.game_top_margins.len().max(1) as f64;
        // The objective is only a screening sort key: robustness matters, but
        // the fixed-seed VP mean remains the dominant term.
        let objective = mean + mean_game_min * 0.25 - zero as f64 / count * 20.0;
        println!(
            "{}",
            json!({
                "axis": axis,
                "value": value,
                "games": games,
                "players": players,
                "base_seed": base_seed,
                "depth": depth,
                "branching": branching,
                "vp": result.scores,
                "mean_vp": mean,
                "min_vp": min,
                "max_vp": max,
                "game_min_vp": result.game_minima,
                "game_max_vp": result.game_maxima,
                "game_top_margin_vp": result.game_top_margins,
                "mean_game_min_vp": mean_game_min,
                "mean_top_vp": mean_top_vp,
                "mean_top_margin_vp": mean_top_margin,
                "worst_game_min_vp": worst_game_min,
                "zero_score_players": zero,
                "actions": result.actions,
                "config": config,
                "actions_per_second": result.actions as f64
                    / (result.elapsed_us as f64 / 1_000_000.0).max(f64::MIN_POSITIVE),
                "action_counts": result.action_counts,
                "screening_objective": objective,
            })
        );
    }
    eprintln!(
        "sweep_elapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
