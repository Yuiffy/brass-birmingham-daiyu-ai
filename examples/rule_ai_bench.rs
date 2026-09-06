#![recursion_limit = "256"]

use fast_brass::game::rule_ai::{rank_rule_actions, RuleDecisionConfig};
use fast_brass::game::runner::GameRunner;
use fast_brass::game::search::official_winners;
use serde_json::json;
use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
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

fn profile_name(depth: usize, branching: usize) -> &'static str {
    match (depth, branching) {
        (2, 8) => "live_default",
        (2, 16) => "deep_diagnostic",
        _ => "custom",
    }
}

fn optional_f64_arg(args: &[String], prefix: &str, default: f64) -> Result<f64, String> {
    args.iter()
        .find_map(|value| value.strip_prefix(prefix))
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|error| format!("invalid {prefix}{value}: {error}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_default_profile_matches_rule_config_default() {
        let config = RuleDecisionConfig::default();
        assert_eq!(config.lookahead_depth, 2);
        assert_eq!(config.lookahead_branching, 8);
        assert_eq!(
            profile_name(config.lookahead_depth, config.lookahead_branching),
            "live_default"
        );
    }

    #[test]
    fn non_default_profiles_are_explicitly_labeled() {
        assert_eq!(profile_name(2, 16), "deep_diagnostic");
        assert_eq!(profile_name(2, 1), "custom");
    }

    #[test]
    fn experimental_weights_are_parsed_independently() {
        let args = vec![
            "bench".to_string(),
            "sale_frontier=0.5".to_string(),
            "loan_maturity=0.75".to_string(),
            "loan_excess_loss=2".to_string(),
            "recovery_product=16".to_string(),
            "conversion_feasibility=0.75".to_string(),
            "resource_stockpile=4".to_string(),
            "post_sale_product_cycle=2".to_string(),
            "mature_product_cycle=1.75".to_string(),
            "financed_product_cycle=0.3".to_string(),
            "same_turn_sale_guard=1".to_string(),
            "contested_external_beer_plan=3.5".to_string(),
            "late_canal_stall=12".to_string(),
            "late_railroad_backlog=1.25".to_string(),
            "beer_backlog_recovery=2.25".to_string(),
            "late_railroad_conversion_stall=0.9".to_string(),
        ];
        assert_eq!(optional_f64_arg(&args, "sale_frontier=", 0.0).unwrap(), 0.5);
        assert_eq!(
            optional_f64_arg(&args, "loan_maturity=", 1.0).unwrap(),
            0.75
        );
        assert_eq!(
            optional_f64_arg(&args, "loan_excess_loss=", 0.0).unwrap(),
            2.0
        );
        assert_eq!(
            optional_f64_arg(&args, "recovery_product=", 0.0).unwrap(),
            16.0
        );
        assert_eq!(
            optional_f64_arg(&args, "conversion_feasibility=", 0.0).unwrap(),
            0.75
        );
        assert_eq!(
            optional_f64_arg(&args, "resource_stockpile=", 0.0).unwrap(),
            4.0
        );
        assert_eq!(
            optional_f64_arg(&args, "post_sale_product_cycle=", 0.0).unwrap(),
            2.0
        );
        assert_eq!(
            optional_f64_arg(&args, "mature_product_cycle=", 0.0).unwrap(),
            1.75
        );
        assert_eq!(
            optional_f64_arg(&args, "financed_product_cycle=", 0.0).unwrap(),
            0.3
        );
        assert_eq!(
            optional_f64_arg(&args, "same_turn_sale_guard=", 0.0).unwrap(),
            1.0
        );
        assert_eq!(
            optional_f64_arg(&args, "contested_external_beer_plan=", 0.0).unwrap(),
            3.5
        );
        assert_eq!(
            optional_f64_arg(&args, "late_canal_stall=", 0.0).unwrap(),
            12.0
        );
        assert_eq!(
            optional_f64_arg(&args, "late_railroad_backlog=", 0.0).unwrap(),
            1.25
        );
        assert_eq!(
            optional_f64_arg(&args, "beer_backlog_recovery=", 0.0).unwrap(),
            2.25
        );
        assert_eq!(
            optional_f64_arg(&args, "late_railroad_conversion_stall=", 0.0).unwrap(),
            0.9
        );
    }
}

fn play_game(
    players: usize,
    seed: u64,
    trace: bool,
    trace_limit: u64,
    detail: bool,
    config: &RuleDecisionConfig,
) -> Result<
    (
        Vec<u16>,
        u64,
        u128,
        u128,
        u64,
        Vec<BTreeMap<String, u32>>,
        BTreeMap<i16, u32>,
    ),
    String,
> {
    let mut runner = GameRunner::new(players, Some(seed));
    let started = Instant::now();
    let mut actions = 0_u64;
    let mut rank_micros = 0_u128;
    let mut legal_action_total = 0_u64;
    let mut action_counts = vec![BTreeMap::<String, u32>::new(); players];
    let mut loan_income_loss_counts = BTreeMap::<i16, u32>::new();
    while !runner.is_game_finished() {
        advance_to_decision(&mut runner)?;
        if runner.is_game_finished() {
            break;
        }
        let rank_started = Instant::now();
        let ranked = rank_rule_actions(&runner, config)?;
        rank_micros += rank_started.elapsed().as_micros();
        legal_action_total += ranked.len() as u64;
        let best = ranked
            .first()
            .ok_or_else(|| "rule policy returned no action".to_string())?;
        if trace && actions < trace_limit {
            println!(
                "trace action={} player={} phase={:?} round={} deck={} hand={} vp={} income={} cash={} root={:?} key={} score={:.2} parts={:?}",
                actions,
                runner.framework.current_player,
                runner.game_phase,
                runner.round_in_phase,
                runner.framework.board.state.deck.cards.len(),
                runner.framework.board.state.players[runner.framework.current_player].hand.cards.len(),
                runner.framework.board.state.players[runner.framework.current_player].victory_points,
                runner.framework.board.state.players[runner.framework.current_player].get_income_amount(
                    runner.framework.board.state.players[runner.framework.current_player].income_level,
                ),
                runner.framework.board.state.players[runner.framework.current_player].money,
                best.action.intent.action_type,
                best.action.key(),
                best.score,
                best.breakdown,
            );
            println!(
                "  alternatives={:?}",
                ranked
                    .iter()
                    .take(5)
                    .map(|candidate| {
                        (
                            format!("{:?}", candidate.action.intent.action_type),
                            candidate.action.key(),
                            candidate.score,
                            candidate.breakdown.industry,
                            candidate.breakdown.income,
                            candidate.breakdown.cash,
                        )
                    })
                    .collect::<Vec<_>>()
            );
            let mut by_root = BTreeMap::<String, f64>::new();
            for candidate in &ranked {
                let root = format!("{:?}", candidate.action.intent.action_type);
                by_root
                    .entry(root)
                    .and_modify(|value| *value = value.max(candidate.score))
                    .or_insert(candidate.score);
            }
            println!("  best_by_root={by_root:?}");
            if detail && actions == 0 {
                for candidate in ranked.iter().filter(|candidate| {
                    matches!(
                        candidate.action.intent.action_type,
                        fast_brass::core::types::ActionType::Loan
                    )
                }) {
                    println!(
                        "  loan_detail key={} score={:.3} breakdown={:?} after_vp={} after_margin={:.1}",
                        candidate.action.key(),
                        candidate.score,
                        candidate.breakdown,
                        candidate.after_actor_victory_points,
                        candidate.after_actor_victory_point_margin
                    );
                }
            }
        }
        let actor = runner.framework.current_player;
        let before_player = &runner.framework.board.state.players[actor];
        let before_income = before_player.get_income_amount(before_player.income_level);
        let is_loan = best.action.intent.action_type == fast_brass::core::types::ActionType::Loan;
        let root_name = format!("{:?}", best.action.intent.action_type);
        *action_counts[actor].entry(root_name).or_default() += 1;
        best.action.apply(&mut runner)?;
        if is_loan {
            let after_player = &runner.framework.board.state.players[actor];
            let after_income = after_player.get_income_amount(after_player.income_level);
            let displayed_loss = i16::from(before_income) - i16::from(after_income);
            *loan_income_loss_counts.entry(displayed_loss).or_default() += 1;
        }
        actions += 1;
        if actions > 512 {
            return Err("rule game exceeded 512 actions".to_string());
        }
    }
    resolve_shortfalls(&mut runner);
    let winners = official_winners(&runner)?;
    let vps = runner
        .framework
        .board
        .state
        .players
        .iter()
        .map(|player| player.victory_points)
        .collect::<Vec<_>>();
    let elapsed = started.elapsed().as_micros();
    if trace {
        for (player_idx, player) in runner.framework.board.state.players.iter().enumerate() {
            let income = player.get_income_amount(player.income_level);
            let buildings = runner
                .framework
                .board
                .state
                .bl_to_building
                .values()
                .filter(|building| building.owner.as_usize() == player_idx)
                .count();
            let flipped = runner
                .framework
                .board
                .state
                .bl_to_building
                .values()
                .filter(|building| building.owner.as_usize() == player_idx && building.flipped)
                .count();
            let roads = runner.framework.board.state.player_road_mask[player_idx]
                .ones()
                .count();
            let mut industries = BTreeMap::<String, u32>::new();
            for building in runner
                .framework
                .board
                .state
                .bl_to_building
                .values()
                .filter(|building| building.owner.as_usize() == player_idx)
            {
                let name = format!("{:?}", building.industry);
                *industries.entry(name).or_default() += 1;
            }
            println!(
                "final player={} vp={} income={} cash={} buildings={} flipped={} roads={} industries={:?} actions={:?}",
                player_idx,
                player.victory_points,
                income,
                player.money,
                buildings,
                flipped,
                roads,
                industries,
                action_counts[player_idx]
            );
        }
    }
    black_box(winners);
    Ok((
        vps,
        actions,
        elapsed,
        rank_micros,
        legal_action_total,
        action_counts,
        loan_income_loss_counts,
    ))
}

fn main() -> Result<(), String> {
    let games = env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(4);
    let players = env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let base_seed = env::args()
        .nth(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2026_0830);
    let args = env::args().collect::<Vec<_>>();
    let trace = args.iter().any(|value| value == "trace");
    let trace_limit = if args.iter().any(|value| value == "trace_all") {
        512
    } else {
        40
    };
    let detail = args.iter().any(|value| value == "detail");
    let mut config = if args.iter().any(|value| value == "legacy") {
        RuleDecisionConfig::legacy()
    } else {
        RuleDecisionConfig::default()
    };
    let lookahead_depth = args
        .iter()
        .find_map(|value| value.strip_prefix("depth=")?.parse::<usize>().ok())
        .unwrap_or(config.lookahead_depth);
    let lookahead_branching = args
        .iter()
        .find_map(|value| value.strip_prefix("branching=")?.parse::<usize>().ok())
        .unwrap_or(config.lookahead_branching);
    config.lookahead_depth = lookahead_depth;
    config.lookahead_branching = lookahead_branching;
    config.sale_frontier_build_weight =
        optional_f64_arg(&args, "sale_frontier=", config.sale_frontier_build_weight)?;
    config.loan_maturity_weight =
        optional_f64_arg(&args, "loan_maturity=", config.loan_maturity_weight)?;
    config.loan_excess_income_loss_weight = optional_f64_arg(
        &args,
        "loan_excess_loss=",
        config.loan_excess_income_loss_weight,
    )?;
    config.low_cost_loan_runway_weight = optional_f64_arg(
        &args,
        "low_cost_loan_runway=",
        config.low_cost_loan_runway_weight,
    )?;
    config.recovery_product_plan_weight = optional_f64_arg(
        &args,
        "recovery_product=",
        config.recovery_product_plan_weight,
    )?;
    config.conversion_feasibility_weight = optional_f64_arg(
        &args,
        "conversion_feasibility=",
        config.conversion_feasibility_weight,
    )?;
    config.resource_stockpile_weight = optional_f64_arg(
        &args,
        "resource_stockpile=",
        config.resource_stockpile_weight,
    )?;
    config.post_sale_product_cycle_weight = optional_f64_arg(
        &args,
        "post_sale_product_cycle=",
        config.post_sale_product_cycle_weight,
    )?;
    config.mature_product_cycle_weight = optional_f64_arg(
        &args,
        "mature_product_cycle=",
        config.mature_product_cycle_weight,
    )?;
    config.financed_product_cycle_weight = optional_f64_arg(
        &args,
        "financed_product_cycle=",
        config.financed_product_cycle_weight,
    )?;
    config.same_turn_sale_guard_weight = optional_f64_arg(
        &args,
        "same_turn_sale_guard=",
        config.same_turn_sale_guard_weight,
    )?;
    config.contested_external_beer_plan_weight = optional_f64_arg(
        &args,
        "contested_external_beer_plan=",
        config.contested_external_beer_plan_weight,
    )?;
    config.late_canal_stall_weight =
        optional_f64_arg(&args, "late_canal_stall=", config.late_canal_stall_weight)?;
    config.late_railroad_backlog_weight = optional_f64_arg(
        &args,
        "late_railroad_backlog=",
        config.late_railroad_backlog_weight,
    )?;
    config.beer_backlog_recovery_weight = optional_f64_arg(
        &args,
        "beer_backlog_recovery=",
        config.beer_backlog_recovery_weight,
    )?;
    config.late_railroad_conversion_stall_weight = optional_f64_arg(
        &args,
        "late_railroad_conversion_stall=",
        config.late_railroad_conversion_stall_weight,
    )?;
    let profile = profile_name(lookahead_depth, lookahead_branching);

    let mut all_vps = Vec::new();
    let mut total_actions = 0_u64;
    let mut total_rank_micros = 0_u128;
    let mut total_legal_actions = 0_u64;
    let mut total_action_counts = BTreeMap::<String, u64>::new();
    let mut total_loan_income_loss_counts = BTreeMap::<i16, u64>::new();
    let mut all_player_vps = Vec::new();
    let started = Instant::now();
    for index in 0..games {
        let (
            vps,
            actions,
            micros,
            rank_micros,
            legal_actions,
            action_counts,
            loan_income_loss_counts,
        ) = play_game(
            players,
            base_seed + index,
            trace,
            trace_limit,
            detail,
            &config,
        )?;
        total_actions += actions;
        total_rank_micros += rank_micros;
        total_legal_actions += legal_actions;
        for player_counts in &action_counts {
            for (action, count) in player_counts {
                *total_action_counts.entry(action.clone()).or_default() += u64::from(*count);
            }
        }
        for (loss, count) in &loan_income_loss_counts {
            *total_loan_income_loss_counts.entry(*loss).or_default() += u64::from(*count);
        }
        all_player_vps.extend(vps.iter().copied());
        all_vps.push(vps);
        println!(
            "{}",
            json!({"game": index, "actions": actions, "elapsed_us": micros, "rank_elapsed_us": rank_micros, "legal_actions": legal_actions, "vp": all_vps.last(), "action_counts": action_counts, "loan_income_loss_counts": loan_income_loss_counts})
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    let total_player_scores = all_player_vps.len().max(1) as f64;
    let mean_vp = all_player_vps.iter().map(|vp| f64::from(*vp)).sum::<f64>() / total_player_scores;
    let min_vp = all_player_vps.iter().copied().min().unwrap_or_default();
    let zero_vp_players = all_player_vps.iter().filter(|vp| **vp == 0).count();
    let mean_game_min_vp = all_vps
        .iter()
        .filter_map(|scores| scores.iter().copied().min())
        .map(f64::from)
        .sum::<f64>()
        / games.max(1) as f64;
    let high_cost_loan_actions = total_loan_income_loss_counts
        .iter()
        .filter(|(loss, _)| **loss > 3)
        .map(|(_, count)| *count)
        .sum::<u64>();
    let loan_income_loss_total = total_loan_income_loss_counts
        .iter()
        .map(|(loss, count)| f64::from(*loss) * *count as f64)
        .sum::<f64>();
    let loan_actions = total_action_counts.get("Loan").copied().unwrap_or(0);
    println!(
        "{}",
        json!({
            "engine": if config.economic_evaluation { fast_brass::game::search::RULE_ECONOMIC_METHOD } else { "fast_brass_rule_decision_tree" },
            "economic_evaluation": config.economic_evaluation,
            "players": players,
            "games": games,
            "seconds": elapsed,
            "games_per_second": games as f64 / elapsed.max(f64::MIN_POSITIVE),
            "actions": total_actions,
            "actions_per_second": total_actions as f64 / elapsed.max(f64::MIN_POSITIVE),
            "rank_elapsed_seconds": total_rank_micros as f64 / 1_000_000.0,
            "mean_legal_actions": total_legal_actions as f64 / total_actions.max(1) as f64,
            "mean_player_vp": mean_vp,
            "mean_game_min_vp": mean_game_min_vp,
            "min_player_vp": min_vp,
            "zero_vp_players": zero_vp_players,
            "action_counts": total_action_counts,
            "sale_actions": total_action_counts.get("Sell").copied().unwrap_or(0),
            "network_actions": total_action_counts.get("BuildRailroad").copied().unwrap_or(0)
                + total_action_counts.get("BuildDoubleRailroad").copied().unwrap_or(0),
            "build_actions": total_action_counts.get("BuildBuilding").copied().unwrap_or(0),
            "develop_actions": total_action_counts.get("Develop").copied().unwrap_or(0)
                + total_action_counts.get("DevelopDouble").copied().unwrap_or(0),
            "loan_actions": loan_actions,
            "loan_income_loss_counts": total_loan_income_loss_counts,
            "high_cost_loan_actions": high_cost_loan_actions,
            "mean_loan_income_loss": loan_income_loss_total / loan_actions.max(1) as f64,
            "profile": profile,
            "lookahead_depth": config.lookahead_depth,
            "lookahead_branching": config.lookahead_branching,
            "sale_frontier_build_weight": config.sale_frontier_build_weight,
            "loan_maturity_weight": config.loan_maturity_weight,
            "loan_excess_income_loss_weight": config.loan_excess_income_loss_weight,
            "recovery_product_plan_weight": config.recovery_product_plan_weight,
            "conversion_feasibility_weight": config.conversion_feasibility_weight,
            "resource_stockpile_weight": config.resource_stockpile_weight,
            "post_sale_product_cycle_weight": config.post_sale_product_cycle_weight,
            "mature_product_cycle_weight": config.mature_product_cycle_weight,
            "financed_product_cycle_weight": config.financed_product_cycle_weight,
            "same_turn_sale_guard_weight": config.same_turn_sale_guard_weight,
            "contested_external_beer_plan_weight": config.contested_external_beer_plan_weight,
            "late_canal_stall_weight": config.late_canal_stall_weight,
            "late_railroad_backlog_weight": config.late_railroad_backlog_weight,
            "beer_backlog_recovery_weight": config.beer_backlog_recovery_weight,
            "late_railroad_conversion_stall_weight": config.late_railroad_conversion_stall_weight,
            "vp": all_vps,
            "rollouts": 0,
            "neural_inference": false,
        })
    );
    Ok(())
}
