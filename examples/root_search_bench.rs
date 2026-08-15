use std::env;
use std::time::Instant;

use fast_brass::game::runner::GameRunner;
use fast_brass::game::search::{search_top_actions, RootSearchConfig};
use serde_json::json;

fn main() -> Result<(), String> {
    let simulations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000);
    let player_count = env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);
    let seed = env::args()
        .nth(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(20_260_815);

    let runner = GameRunner::new(player_count, Some(seed));
    let config = RootSearchConfig {
        simulations,
        recommendation_count: 3,
        seed: seed ^ 0xa076_1d64_78bd_642f,
        ..RootSearchConfig::default()
    };
    let started = Instant::now();
    let report = search_top_actions(&runner, &config)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let recommendations = report
        .recommendations
        .iter()
        .map(|candidate| {
            json!({
                "rank": candidate.rank,
                "action_key": candidate.action_key,
                "visits": candidate.visits,
                "visit_share": candidate.visit_share,
                "estimated_shared_win_rate": candidate.estimated_shared_win_rate,
                "standard_error": candidate.shared_win_rate_standard_error,
                "average_final_victory_points": candidate.average_final_victory_points,
                "average_victory_point_margin": candidate.average_victory_point_margin,
                "policy_probability": candidate.policy_probability,
                "calibrated_win_rate": candidate.calibrated_win_rate,
            })
        })
        .collect::<Vec<_>>();

    println!(
        "{}",
        json!({
            "engine": report.method,
            "players": player_count,
            "simulations": report.completed_simulations,
            "root_actions": report.root_action_count,
            "evaluated_actions": report.evaluated_action_count,
            "seconds": elapsed_seconds,
            "simulations_per_second": report.completed_simulations as f64 / elapsed_seconds,
            "recommendations": recommendations,
        })
    );
    Ok(())
}
