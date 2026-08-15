use fast_brass::game::framework::{ActionChoice, ChoiceSet};
use fast_brass::game::runner::GameRunner;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::env;
use std::hint::black_box;
use std::time::Instant;

#[derive(Default)]
struct GameMetrics {
    actions: u64,
    choices: u64,
}

fn pick<T: Clone>(values: &[T], rng: &mut StdRng, label: &str) -> Result<T, String> {
    values
        .choose(rng)
        .cloned()
        .ok_or_else(|| format!("empty choice set: {label}"))
}

fn random_choice(
    set: ChoiceSet,
    used_card_indices: &[usize],
    rng: &mut StdRng,
) -> Result<Option<ActionChoice>, String> {
    let choice = match set {
        ChoiceSet::Industry(values) => ActionChoice::Industry(pick(&values, rng, "industry")?),
        ChoiceSet::Card(values) => {
            let available: Vec<usize> = values
                .into_iter()
                .filter(|index| !used_card_indices.contains(index))
                .collect();
            ActionChoice::Card(pick(&available, rng, "card")?)
        }
        ChoiceSet::BuildLocation(values) => {
            ActionChoice::BuildLocation(pick(&values, rng, "build location")?)
        }
        ChoiceSet::Road(values) | ChoiceSet::SecondRoad(values) => {
            ActionChoice::Road(pick(&values, rng, "road")?)
        }
        ChoiceSet::CoalSource(values) => {
            ActionChoice::CoalSource(pick(&values, rng, "coal source")?)
        }
        ChoiceSet::IronSource(values) => {
            ActionChoice::IronSource(pick(&values, rng, "iron source")?)
        }
        ChoiceSet::BeerSource(values) => {
            ActionChoice::BeerSource(pick(&values, rng, "beer source")?)
        }
        ChoiceSet::ActionBeerSource(values) => {
            ActionChoice::ActionBeerSource(pick(&values, rng, "action beer source")?)
        }
        ChoiceSet::SellTarget(values) => {
            ActionChoice::SellTarget(pick(&values, rng, "sell target")?)
        }
        ChoiceSet::FreeDevelopment(values) => {
            ActionChoice::FreeDevelopment(pick(&values, rng, "free development")?)
        }
        ChoiceSet::SecondIndustry(values) => {
            ActionChoice::Industry(pick(&values, rng, "second industry")?)
        }
        ChoiceSet::NetworkMode(values) => {
            ActionChoice::NetworkMode(pick(&values, rng, "network mode")?)
        }
        ChoiceSet::ConfirmOnly => return Ok(None),
    };
    Ok(Some(choice))
}

fn play_random_game(
    player_count: usize,
    seed: u64,
    max_actions: Option<u64>,
) -> Result<GameMetrics, String> {
    let mut runner = GameRunner::new(player_count, Some(seed));
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut metrics = GameMetrics::default();

    while !runner.is_game_finished() {
        let roots = runner.start_turn();
        if roots.is_empty() {
            let state = runner.get_game_state();
            let hand_sizes: Vec<usize> = state
                .players
                .iter()
                .map(|player| player.hand.cards.len())
                .collect();
            return Err(format!(
                "empty root action set at seed {seed}, phase {:?}, round {}, turn {}, player {}, actions remaining {}, hands {:?}",
                state.phase,
                state.round_in_phase,
                state.turn_count,
                state.current_player,
                state.actions_remaining_in_turn,
                hand_sizes,
            ));
        }
        let root = pick(&roots, &mut rng, "root action")?;
        runner.start_action(root);

        let mut choice_guard = 0_u32;
        let mut used_card_indices = Vec::<usize>::new();
        let mut choice_trace = Vec::<String>::new();
        loop {
            choice_guard += 1;
            if choice_guard > 1_000 {
                return Err(format!(
                    "choice loop exceeded guard at seed {seed}, turn {}, action {:?}",
                    runner.turn_count, root
                ));
            }

            let set = runner
                .framework
                .get_next_choice_set()
                .ok_or_else(|| format!("missing choice set for {:?}", root))?;
            let set_debug = format!("{set:?}");
            match random_choice(set, &used_card_indices, &mut rng)? {
                Some(choice) => {
                    choice_trace.push(format!("{set_debug} -> {choice:?}"));
                    let beer_building = match &choice {
                        ActionChoice::BeerSource(
                            fast_brass::board::resources::BeerSellSource::Building(location),
                        ) => Some(*location),
                        _ => None,
                    };
                    if let ActionChoice::Card(index) = &choice {
                        used_card_indices.push(*index);
                    }
                    runner.apply_choice(choice);
                    if let Some(location) = beer_building {
                        let resource = runner
                            .framework
                            .board
                            .state
                            .bl_to_building
                            .get(&location)
                            .map(|building| building.resource_amt);
                        let context = runner.framework.action_context.as_ref();
                        choice_trace.push(format!(
                            "post-beer building={location} resource={resource:?} pending={:?} temp={:?} next={:?}",
                            context.and_then(|ctx| ctx.pending_sell_building_loc),
                            context.and_then(|ctx| ctx.temp_brewery_beer_consumed.get(&location)),
                            runner.framework.get_next_choice_set(),
                        ));
                    }
                    metrics.choices += 1;
                }
                None => {
                    choice_trace.push(set_debug);
                    if !runner.framework.can_confirm() {
                        return Err(format!(
                            "ConfirmOnly exposed before confirm at seed {seed}, turn {}, action {:?}",
                            runner.turn_count, root
                        ));
                    }
                    let session = runner.framework.current_session();
                    runner.confirm_action().map_err(|error| {
                        format!(
                            "confirm failed at seed {seed}, turn {}, action {:?}: {error}; session={session:?}; trace={choice_trace:?}",
                            runner.turn_count, root,
                        )
                    })?;
                    metrics.actions += 1;
                    if max_actions.is_some_and(|limit| metrics.actions >= limit) {
                        return Ok(metrics);
                    }
                    break;
                }
            }
        }

        if runner.actions_remaining_in_turn == 0 {
            runner.end_turn();
            for session in runner.take_shortfall_sessions() {
                let locations = session
                    .removable_tiles
                    .iter()
                    .map(|tile| tile.build_location_idx)
                    .collect();
                runner.resolve_shortfall_with_tiles(session, locations);
            }
        }
    }

    Ok(metrics)
}

fn main() -> Result<(), String> {
    let games = env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(100);
    let player_count = env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    let max_actions = env::args()
        .nth(3)
        .and_then(|value| value.parse::<u64>().ok());

    for seed in 0..3 {
        black_box(play_random_game(player_count, seed, max_actions)?);
    }

    let started = Instant::now();
    let mut total_actions = 0_u64;
    let mut total_choices = 0_u64;
    for seed in 0..games {
        let metrics = play_random_game(player_count, seed + 10_000, max_actions)?;
        total_actions += metrics.actions;
        total_choices += metrics.choices;
    }
    let elapsed = started.elapsed().as_secs_f64();

    println!(
        "{{\"engine\":\"fast_brass_rust\",\"players\":{player_count},\"episodes\":{games},\"max_actions_per_episode\":{},\"seconds\":{elapsed:.6},\"episodes_per_second\":{:.3},\"actions\":{total_actions},\"actions_per_second\":{:.0},\"staged_choices\":{total_choices}}}",
        max_actions.map_or_else(|| "null".to_string(), |value| value.to_string()),
        games as f64 / elapsed,
        total_actions as f64 / elapsed,
    );
    Ok(())
}
