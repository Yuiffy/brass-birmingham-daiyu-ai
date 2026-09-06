//! Seat-rotated policy matches, with one observation per seed.

use fast_brass::game::rule_ai::{rank_rule_actions, RuleDecisionConfig};
use fast_brass::game::runner::GameRunner;
use fast_brass::game::search::official_winners;
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Serialize)]
struct GameResult {
    seed: u64,
    candidate_seat: usize,
    scores: Vec<u16>,
    final_income_levels: Vec<i8>,
    final_money: Vec<u16>,
    win_share: f64,
    margin: f64,
    actions: usize,
    candidate_actions: BTreeMap<String, usize>,
    actions_by_era: BTreeMap<String, usize>,
    shortfalls: usize,
    canal_vp: u16,
    final_cash: u16,
    final_income: i8,
    unflipped_tiles: usize,
    candidate_millis: f64,
    legacy_millis: f64,
}

fn play(
    players: usize,
    seed: u64,
    seat: usize,
    config: &RuleDecisionConfig,
    opponent: &RuleDecisionConfig,
    trace: bool,
) -> Result<GameResult, String> {
    let mut runner = GameRunner::new(players, Some(seed));
    let mut actions = 0;
    let mut candidate_actions = BTreeMap::new();
    let mut actions_by_era = BTreeMap::new();
    let mut shortfalls = 0;
    let mut canal_vp = None;
    let mut candidate_millis = 0.0;
    let mut legacy_millis = 0.0;
    while !runner.is_game_finished() {
        for session in runner.take_shortfall_sessions() {
            if session.player_idx == seat {
                shortfalls += 1;
            }
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
        runner.start_turn();
        if runner.framework.get_valid_root_actions().is_empty() {
            runner.end_action_slot();
            if runner.actions_remaining_in_turn == 0 {
                runner.end_turn();
            }
            actions += 1;
            if actions > 512 {
                return Err(format!("seed {seed}: forced-action limit"));
            }
            continue;
        }
        let candidate_turn = runner.framework.current_player == seat;
        let started = Instant::now();
        let ranked = rank_rule_actions(&runner, if candidate_turn { config } else { opponent })?;
        let best = ranked.first().ok_or("no ranked action")?;
        let action = &best.action;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        if candidate_turn {
            candidate_millis += elapsed;
            *candidate_actions
                .entry(format!("{:?}", action.intent.action_type))
                .or_default() += 1;
            *actions_by_era
                .entry(format!(
                    "{:?}/{:?}",
                    runner.game_phase, action.intent.action_type
                ))
                .or_default() += 1;
            if trace {
                let player = &runner.framework.board.state.players[seat];
                eprintln!("trace seed={seed} seat={seat} action={actions} phase={:?} cash={} income={} vp={} hand={} deck={} key={} score={:.3} parts={:?} alternatives={:?}", runner.game_phase, player.money, player.get_income_amount(player.income_level), player.victory_points, player.hand.cards.len(), runner.framework.board.state.deck.cards.len(), action.key(), best.score, best.breakdown, ranked.iter().take(5).map(|a| (a.action.key(), a.score)).collect::<Vec<_>>());
            }
        } else {
            legacy_millis += elapsed;
        }
        action.apply(&mut runner)?;
        if runner.game_phase != fast_brass::game::runner::GamePhase::Canal && canal_vp.is_none() {
            canal_vp = Some(runner.framework.board.state.players[seat].victory_points);
        }
        actions += 1;
        if actions > 512 {
            return Err(format!("seed {seed}: action limit"));
        }
    }
    let winners = official_winners(&runner)?;
    let scores = runner
        .framework
        .board
        .state
        .players
        .iter()
        .map(|p| p.victory_points)
        .collect::<Vec<_>>();
    let best_opponent = scores
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != seat)
        .map(|(_, vp)| *vp)
        .max()
        .unwrap();
    Ok(GameResult {
        seed,
        candidate_seat: seat,
        win_share: if winners.contains(&seat) {
            1.0 / winners.len() as f64
        } else {
            0.0
        },
        margin: f64::from(scores[seat]) - f64::from(best_opponent),
        scores,
        final_income_levels: runner
            .framework
            .board
            .state
            .players
            .iter()
            .map(|p| p.get_income_amount(p.income_level))
            .collect(),
        final_money: runner
            .framework
            .board
            .state
            .players
            .iter()
            .map(|p| p.money)
            .collect(),
        actions,
        candidate_actions,
        actions_by_era,
        shortfalls,
        canal_vp: canal_vp.unwrap_or(0),
        final_cash: runner.framework.board.state.players[seat].money,
        final_income: runner.framework.board.state.players[seat]
            .get_income_amount(runner.framework.board.state.players[seat].income_level),
        unflipped_tiles: runner
            .framework
            .board
            .state
            .bl_to_building
            .values()
            .filter(|b| b.owner.as_usize() == seat && !b.flipped)
            .count(),
        candidate_millis,
        legacy_millis,
    })
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match args.iter().position(|a| a == name) {
        Some(i) => args
            .get(i + 1)
            .ok_or_else(|| format!("missing {name}"))?
            .parse()
            .map_err(|_| format!("invalid {name}")),
        None => Ok(default),
    }
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let rounds = arg(&args, "--rounds", 8usize)?;
    let players = arg(&args, "--players", 3usize)?;
    let seed = arg(&args, "--seed", 2026090600u64)?;
    let workers = arg(&args, "--workers", 4usize)?;
    let profile = arg(&args, "--candidate", "economy".to_string())?;
    let opponent_profile = arg(&args, "--opponent", "economy-v1".to_string())?;
    let trace = args.iter().any(|arg| arg == "--trace");
    let seat = arg(&args, "--seat", players)?;
    if rounds == 0 || !(2..=4).contains(&players) || !(1..=8).contains(&workers) || seat > players {
        return Err("require rounds >= 1, players 2..4, workers 1..8, and a valid seat".into());
    }
    let seats = if seat < players { 1 } else { players };
    let make_config = |name: &str| -> Result<RuleDecisionConfig, String> {
        match name {
            "economy" => Ok(RuleDecisionConfig::default()),
            "economy-v1" => Ok(RuleDecisionConfig::economic_v1()),
            "legacy" => Ok(RuleDecisionConfig::legacy()),
            _ => Err("policy must be economy, economy-v1, or legacy".into()),
        }
    };
    let mut config = make_config(&profile)?;
    let opponent = make_config(&opponent_profile)?;
    config.lookahead_depth = arg(&args, "--depth", config.lookahead_depth)?;
    config.lookahead_branching = arg(&args, "--branching", config.lookahead_branching)?;
    let tasks = Arc::new(AtomicUsize::new(0));
    let results = Arc::new(Mutex::new(Vec::new()));
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let tasks = &tasks;
            let results = &results;
            let config = &config;
            let opponent = &opponent;
            scope.spawn(move || loop {
                let i = tasks.fetch_add(1, Ordering::Relaxed);
                if i >= rounds * seats {
                    break;
                }
                let result = play(
                    players,
                    seed + (i / seats) as u64,
                    if seats == 1 { seat } else { i % players },
                    config,
                    opponent,
                    trace,
                );
                if let Ok(game) = &result {
                    eprintln!(
                        "seed={} seat={} vp={:?} margin={}",
                        game.seed, game.candidate_seat, game.scores, game.margin
                    );
                }
                results.lock().unwrap().push(result);
            });
        }
    });
    let mut games = results
        .lock()
        .unwrap()
        .drain(..)
        .collect::<Result<Vec<_>, _>>()?;
    games.sort_by_key(|game| (game.seed, game.candidate_seat));
    let seed_margins = games
        .chunks_exact(seats)
        .map(|group| group.iter().map(|g| g.margin).sum::<f64>() / seats as f64)
        .collect::<Vec<_>>();
    let mean_margin = seed_margins.iter().sum::<f64>() / rounds as f64;
    let variance = seed_margins
        .iter()
        .map(|m| (m - mean_margin).powi(2))
        .sum::<f64>()
        / (rounds - 1).max(1) as f64;
    // Conservative two-sided Student-t critical values, rounded upwards by df.
    let critical = match rounds - 1 {
        1 => 12.707,
        2 => 4.303,
        3 => 3.183,
        4 => 2.777,
        5..=6 => 2.571,
        7..=9 => 2.365,
        10..=14 => 2.229,
        15..=19 => 2.132,
        20..=29 => 2.086,
        _ => 2.043,
    };
    let half_width = critical * (variance / rounds as f64).sqrt();
    let count = games.len() as f64;
    let report = json!({
        "candidate": profile, "config": config, "opponent": opponent_profile,
        "rules": "official_rules_corrections_2026_09_06",
        "players": players, "rounds": rounds, "base_seed": seed, "workers": workers,
        "games": games, "seconds": started.elapsed().as_secs_f64(),
        "mean_candidate_vp": games.iter().map(|g| f64::from(g.scores[g.candidate_seat])).sum::<f64>() / count,
        "mean_opponent_vp": games.iter().map(|g| g.scores.iter().enumerate().filter(|(i, _)| *i != g.candidate_seat).map(|(_, v)| f64::from(*v)).sum::<f64>() / (players - 1) as f64).sum::<f64>() / count,
        "win_share": games.iter().map(|g| g.win_share).sum::<f64>() / count,
        "min_candidate_vp": games.iter().map(|g| g.scores[g.candidate_seat]).min(),
        "candidate_under_100": games.iter().filter(|g| g.scores[g.candidate_seat] < 100).count(),
        "candidate_under_50": games.iter().filter(|g| g.scores[g.candidate_seat] < 50).count(),
        "candidate_shortfalls": games.iter().map(|g| g.shortfalls).sum::<usize>(),
        "mean_canal_vp": games.iter().map(|g| f64::from(g.canal_vp)).sum::<f64>() / count,
        "mean_vp_margin": mean_margin,
        "seed_group_margin_ci95": (rounds > 1).then_some([mean_margin - half_width, mean_margin + half_width]),
        "seed_margins": seed_margins,
        "candidate_ms_per_action": games.iter().map(|g| g.candidate_millis).sum::<f64>() / games.iter().map(|g| g.candidate_actions.values().sum::<usize>()).sum::<usize>() as f64,
        "legacy_ms_per_action": games.iter().map(|g| g.legacy_millis).sum::<f64>() / games.iter().map(|g| g.actions - g.candidate_actions.values().sum::<usize>()).sum::<usize>() as f64,
    });
    let output = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(i) = args.iter().position(|a| a == "--output") {
        let path = args.get(i + 1).ok_or("missing --output path")?;
        std::fs::write(path, &output).map_err(|e| e.to_string())?;
    }
    println!("{output}");
    Ok(())
}
