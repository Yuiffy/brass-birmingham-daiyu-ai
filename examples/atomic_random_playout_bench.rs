use fast_brass::game::legal_actions::enumerate_legal_actions;
use fast_brass::game::runner::GameRunner;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn play_random_game(player_count: usize, seed: u64) -> Result<u64, String> {
    let mut runner = GameRunner::new(player_count, Some(seed));
    let mut rng = StdRng::seed_from_u64(seed ^ 0xa076_1d64_78bd_642f);
    let mut action_count = 0_u64;

    while !runner.is_game_finished() {
        if runner.has_pending_shortfall() {
            for session in runner.take_shortfall_sessions() {
                let locations = session
                    .removable_tiles
                    .iter()
                    .map(|tile| tile.build_location_idx)
                    .collect();
                runner.resolve_shortfall_with_tiles(session, locations);
            }
        }

        let actions = enumerate_legal_actions(&runner)?;
        let action = actions.choose(&mut rng).cloned().ok_or_else(|| {
            format!(
                "no legal action at seed {seed}, player {}, turn {}, phase {:?}",
                runner.framework.current_player, runner.turn_count, runner.game_phase
            )
        })?;
        action.apply(&mut runner)?;
        action_count += 1;
    }

    Ok(action_count)
}

fn main() -> Result<(), String> {
    let games = env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(10);
    let player_count = env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);

    let started = Instant::now();
    let mut total_actions = 0_u64;
    for seed in 0..games {
        total_actions += black_box(play_random_game(player_count, seed + 20_000)?);
    }
    let elapsed = started.elapsed().as_secs_f64();

    println!(
        "{{\"engine\":\"fast_brass_atomic_rust\",\"players\":{player_count},\"episodes\":{games},\"seconds\":{elapsed:.6},\"episodes_per_second\":{:.3},\"actions\":{total_actions},\"actions_per_second\":{:.0}}}",
        games as f64 / elapsed,
        total_actions as f64 / elapsed,
    );
    Ok(())
}
