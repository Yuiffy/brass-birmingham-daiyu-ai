//! Small reproducible native-engine integration screen, independent of JS results.
use fast_brass::game::trained_ai::{trained_decision_report, MODEL_ID};
use fast_brass::game::{
    enumerate_legal_actions, rank_rule_actions, GameRunner, RuleDecisionConfig,
};
use serde_json::json;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

fn prepare(r: &mut GameRunner) {
    for session in r.take_shortfall_sessions() {
        let choices = session
            .removable_tiles
            .iter()
            .map(|c| c.build_location_idx)
            .collect();
        r.resolve_shortfall_with_tiles(session, choices);
    }
    if !r.is_game_finished() {
        r.start_turn();
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let games = args.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(2);
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("native-trained-screen.jsonl");
    let mut output = BufWriter::new(File::create(path)?);
    writeln!(
        output,
        "{}",
        json!({"protocol":"native-trained-integration-screen-v1","gamesPerPlayerCountAndPolicy":games,"model":MODEL_ID,"seeds":"730001 + game index","lineups":["trained-selfplay","rule-selfplay"],"observation":"own-hand-public-board-sampled-hidden-cards","benchmarkScope":"Rust engine, distinct from JS economy-v2"})
    )?;
    output.flush()?;
    for n in 2..=4 {
        for game in 0..games {
            for trained in [true, false] {
                let seed = 730001 + game;
                let started = Instant::now();
                let mut r = GameRunner::new(n, Some(seed));
                let mut actions = Vec::new();
                let mut latency = Vec::new();
                while !r.is_game_finished() {
                    prepare(&mut r);
                    if r.is_game_finished() {
                        break;
                    }
                    let decision = Instant::now();
                    let action = if trained {
                        trained_decision_report(&r, 1)?
                            .recommendations
                            .remove(0)
                            .action
                    } else {
                        rank_rule_actions(&r, &RuleDecisionConfig::default())?
                            .remove(0)
                            .action
                    };
                    latency.push(decision.elapsed().as_millis() as u64);
                    actions.push(action.key());
                    action.apply(&mut r)?;
                    if actions.len() > 400 {
                        return Err("game exceeded 400 actions".into());
                    }
                }
                // Reconstruct from legal keys without running the policy, auditing both eras.
                let mut replay = GameRunner::new(n, Some(seed));
                for key in &actions {
                    prepare(&mut replay);
                    let action = enumerate_legal_actions(&replay)?
                        .into_iter()
                        .find(|a| &a.key() == key)
                        .ok_or("recorded action not legal on replay")?;
                    action.apply(&mut replay)?;
                }
                prepare(&mut replay);
                let scores = r
                    .framework
                    .board
                    .state
                    .players
                    .iter()
                    .map(|p| (p.victory_points, p.money, p.income_level))
                    .collect::<Vec<_>>();
                let replay_scores = replay
                    .framework
                    .board
                    .state
                    .players
                    .iter()
                    .map(|p| (p.victory_points, p.money, p.income_level))
                    .collect::<Vec<_>>();
                assert!(replay.is_game_finished());
                assert_eq!(scores, replay_scores);
                latency.sort_unstable();
                writeln!(
                    output,
                    "{}",
                    json!({"players":n,"seed":seed,"policy":if trained{"trained-selfplay"}else{"rule-selfplay"},"terminal":true,"scores":scores,"actions":actions,"replayVerified":true,"elapsedMs":started.elapsed().as_millis(),"decisionP95Ms":latency[latency.len()*95/100],"decisionMaxMs":latency.last()})
                )?;
                output.flush()?;
                eprintln!(
                    "{n}p seed {seed} {} {:?} ({:.1}s)",
                    if trained { "trained" } else { "rule" },
                    scores,
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    Ok(())
}
