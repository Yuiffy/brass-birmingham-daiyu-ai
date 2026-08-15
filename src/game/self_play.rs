use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::game::legal_actions::{enumerate_legal_actions, LegalAction};
use crate::game::runner::{GamePhase, GameRunner};
use crate::game::search::{official_winners, search_top_actions, RootSearchConfig};
use crate::game::training::{
    encode_training_action, encode_training_state, training_feature_schema, TrainingFeatureSchema,
    ACTION_FEATURE_DIM, STATE_FEATURE_DIM, TRAINING_FEATURE_VERSION,
};

pub const SELF_PLAY_FORMAT: &str = "fast_brass_self_play_jsonl";
pub const SELF_PLAY_FORMAT_VERSION: u32 = 1;

const GAME_SEED_STREAM: u64 = 0x6761_6d65_5f73_6565;
const SEARCH_SEED_STREAM: u64 = 0x7365_6172_6368_5f73;
const SELECTION_SEED_STREAM: u64 = 0x7365_6c65_6374_5f73;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfPlayConfig {
    pub games: u32,
    pub num_players: usize,
    pub base_seed: u64,
    pub simulations_per_decision: u64,
    pub exploration_constant: f64,
    pub max_rollout_actions: u32,
    pub max_game_actions: u32,
    pub sell_stop_probability: f64,
    pub engine_revision: String,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            games: 1,
            num_players: 2,
            base_seed: 20_260_815,
            simulations_per_decision: 800,
            exploration_constant: std::f64::consts::SQRT_2,
            max_rollout_actions: 256,
            max_game_actions: 256,
            sell_stop_probability: 0.35,
            engine_revision: "unknown".to_string(),
        }
    }
}

impl SelfPlayConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.games == 0 {
            return Err("games must be at least one".to_string());
        }
        if !(2..=4).contains(&self.num_players) {
            return Err("num_players must be between 2 and 4".to_string());
        }
        if self.simulations_per_decision == 0 || self.simulations_per_decision > 1_000_000 {
            return Err("simulations_per_decision must be between 1 and 1000000".to_string());
        }
        if !self.exploration_constant.is_finite() || self.exploration_constant < 0.0 {
            return Err("exploration_constant must be finite and non-negative".to_string());
        }
        if self.max_rollout_actions == 0 {
            return Err("max_rollout_actions must be at least one".to_string());
        }
        if self.max_game_actions == 0 {
            return Err("max_game_actions must be at least one".to_string());
        }
        if !self.sell_stop_probability.is_finite()
            || !(0.0..=1.0).contains(&self.sell_stop_probability)
        {
            return Err("sell_stop_probability must be between zero and one".to_string());
        }
        if self.engine_revision.trim().is_empty() {
            return Err("engine_revision must not be empty".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfPlayShardHeader {
    pub record_type: &'static str,
    pub format: &'static str,
    pub format_version: u32,
    pub crate_version: &'static str,
    pub engine_revision: String,
    pub feature_schema: TrainingFeatureSchema,
    pub games: u32,
    pub num_players: usize,
    pub base_seed: u64,
    pub simulations_per_decision: u64,
    pub exploration_constant: f64,
    pub max_rollout_actions: u32,
    pub max_game_actions: u32,
    pub sell_stop_probability: f64,
    pub action_selection: &'static str,
    pub shortfall_resolution: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfPlayActionTarget {
    pub index: usize,
    pub key: String,
    pub feature_indices: Vec<u16>,
    pub visits: u64,
    pub policy_target: f32,
    pub estimated_shared_win_rate: Option<f32>,
    pub average_victory_point_margin: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfPlayValueTarget {
    pub shared_win: f32,
    pub victory_point_margin: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfPlayPositionRecord {
    pub record_type: &'static str,
    pub format_version: u32,
    pub feature_version: u32,
    pub game_index: u32,
    pub position_index: u32,
    pub game_seed: u64,
    pub search_seed: u64,
    pub selection_seed: u64,
    pub actor: usize,
    pub phase: &'static str,
    pub turn_count: u32,
    pub round_in_phase: u32,
    pub actions_remaining_in_turn: u8,
    pub forced_action_slots_before: u32,
    pub state_features: Vec<f32>,
    pub legal_actions: Vec<SelfPlayActionTarget>,
    pub selected_action_index: usize,
    pub selected_action_key: String,
    pub value_target: SelfPlayValueTarget,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfPlayGameRecord {
    pub record_type: &'static str,
    pub format_version: u32,
    pub game_index: u32,
    pub game_seed: u64,
    pub positions: usize,
    pub official_winners: Vec<usize>,
    pub shared_win_values: Vec<f32>,
    pub placements: Vec<usize>,
    pub finish_order: Vec<usize>,
    pub victory_points: Vec<u16>,
    pub victory_point_margins: Vec<i32>,
    pub income_levels: Vec<u8>,
    pub money: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeneratedSelfPlayGame {
    pub outcome: SelfPlayGameRecord,
    pub positions: Vec<SelfPlayPositionRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelfPlayExportSummary {
    pub games_written: u32,
    pub positions_written: usize,
}

pub fn self_play_shard_header(config: &SelfPlayConfig) -> SelfPlayShardHeader {
    SelfPlayShardHeader {
        record_type: "metadata",
        format: SELF_PLAY_FORMAT,
        format_version: SELF_PLAY_FORMAT_VERSION,
        crate_version: env!("CARGO_PKG_VERSION"),
        engine_revision: config.engine_revision.clone(),
        feature_schema: training_feature_schema(),
        games: config.games,
        num_players: config.num_players,
        base_seed: config.base_seed,
        simulations_per_decision: config.simulations_per_decision,
        exploration_constant: config.exploration_constant,
        max_rollout_actions: config.max_rollout_actions,
        max_game_actions: config.max_game_actions,
        sell_stop_probability: config.sell_stop_probability,
        action_selection: "sample_from_root_visit_counts",
        shortfall_resolution: "ascending_liquidation_value_then_location",
    }
}

pub fn generate_self_play_game(
    config: &SelfPlayConfig,
    game_index: u32,
) -> Result<GeneratedSelfPlayGame, String> {
    config.validate()?;
    if game_index >= config.games {
        return Err(format!(
            "game_index {game_index} is outside configured game count {}",
            config.games
        ));
    }
    let game_seed = derive_stream_seed(config.base_seed, GAME_SEED_STREAM, game_index as u64);
    let runner = GameRunner::new(config.num_players, Some(game_seed));
    generate_self_play_game_from_runner(config, game_index, game_seed, runner)
}

pub fn export_self_play_shard<F>(
    output_path: &Path,
    config: &SelfPlayConfig,
    mut on_game_complete: F,
) -> Result<SelfPlayExportSummary, String>
where
    F: FnMut(&SelfPlayGameRecord),
{
    config.validate()?;
    if output_path.exists() {
        return Err(format!(
            "refusing to overwrite existing shard {}",
            output_path.display()
        ));
    }
    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "failed to create shard directory {}: {error}",
            parent.display()
        )
    })?;

    let partial_path = partial_output_path(output_path)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial_path)
        .map_err(|error| {
            format!(
                "failed to create partial shard {}: {error}",
                partial_path.display()
            )
        })?;
    let mut writer = BufWriter::new(file);

    let export_result: Result<SelfPlayExportSummary, String> = (|| {
        write_json_line(&mut writer, &self_play_shard_header(config))?;
        let mut positions_written = 0usize;
        for game_index in 0..config.games {
            let game = generate_self_play_game(config, game_index)?;
            write_json_line(&mut writer, &game.outcome)?;
            for position in &game.positions {
                write_json_line(&mut writer, position)?;
            }
            positions_written += game.positions.len();
            on_game_complete(&game.outcome);
        }
        writer
            .flush()
            .map_err(|error| format!("failed to flush self-play shard: {error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("failed to sync self-play shard: {error}"))?;
        Ok(SelfPlayExportSummary {
            games_written: config.games,
            positions_written,
        })
    })();

    let summary = match export_result {
        Ok(summary) => summary,
        Err(error) => {
            drop(writer);
            return Err(format!(
                "{error}; incomplete output remains at {}",
                partial_path.display()
            ));
        }
    };
    drop(writer);
    fs::rename(&partial_path, output_path).map_err(|error| {
        format!(
            "failed to finalize shard {} from {}: {error}",
            output_path.display(),
            partial_path.display()
        )
    })?;
    Ok(summary)
}

fn generate_self_play_game_from_runner(
    config: &SelfPlayConfig,
    game_index: u32,
    game_seed: u64,
    mut runner: GameRunner,
) -> Result<GeneratedSelfPlayGame, String> {
    let mut positions = Vec::<SelfPlayPositionRecord>::new();

    while !runner.is_game_finished() {
        let forced_action_slots_before = advance_to_next_decision(&mut runner)?;
        if runner.is_game_finished() {
            break;
        }
        if positions.len() >= config.max_game_actions as usize {
            return Err(format!(
                "self-play game {game_index} exceeded {} decisions at turn {}",
                config.max_game_actions, runner.turn_count
            ));
        }

        let position_index = positions.len() as u32;
        let actor = runner.framework.current_player;
        let state_features = encode_training_state(&runner, actor)?;
        debug_assert_eq!(state_features.len(), STATE_FEATURE_DIM);
        let legal_actions = enumerate_legal_actions(&runner)?;
        if legal_actions.is_empty() {
            return Err(format!(
                "self-play game {game_index} reached an empty decision at turn {}",
                runner.turn_count
            ));
        }

        let search_seed = derive_stream_seed(game_seed, SEARCH_SEED_STREAM, position_index as u64);
        let selection_seed =
            derive_stream_seed(game_seed, SELECTION_SEED_STREAM, position_index as u64);
        let search_config = RootSearchConfig {
            simulations: config.simulations_per_decision,
            exploration_constant: config.exploration_constant,
            seed: search_seed,
            recommendation_count: legal_actions.len(),
            max_rollout_actions: config.max_rollout_actions,
            sample_continuation_length: 0,
            sell_stop_probability: config.sell_stop_probability,
        };
        let report = search_top_actions(&runner, &search_config)?;
        if report.completed_simulations != config.simulations_per_decision {
            return Err(format!(
                "search completed {} of {} requested simulations",
                report.completed_simulations, config.simulations_per_decision
            ));
        }
        if report.root_action_count != legal_actions.len() {
            return Err(format!(
                "search enumerated {} root actions but exporter enumerated {}",
                report.root_action_count,
                legal_actions.len()
            ));
        }

        let action_targets = build_action_targets(&runner, actor, &legal_actions, &report)?;
        let selected_action_index = sample_action_index(&action_targets, selection_seed)?;
        let selected_action_key = legal_actions[selected_action_index].key();

        positions.push(SelfPlayPositionRecord {
            record_type: "position",
            format_version: SELF_PLAY_FORMAT_VERSION,
            feature_version: TRAINING_FEATURE_VERSION,
            game_index,
            position_index,
            game_seed,
            search_seed,
            selection_seed,
            actor,
            phase: phase_name(runner.game_phase),
            turn_count: runner.turn_count,
            round_in_phase: runner.round_in_phase,
            actions_remaining_in_turn: runner.actions_remaining_in_turn,
            forced_action_slots_before,
            state_features,
            legal_actions: action_targets,
            selected_action_index,
            selected_action_key,
            value_target: SelfPlayValueTarget {
                shared_win: 0.0,
                victory_point_margin: 0,
            },
        });

        legal_actions[selected_action_index]
            .apply(&mut runner)
            .map_err(|error| {
                format!(
                    "selected action {} failed in self-play game {game_index}: {error}",
                    legal_actions[selected_action_index].key()
                )
            })?;
    }

    resolve_shortfalls_deterministically(&mut runner);
    if !runner.is_game_finished() {
        return Err(format!(
            "self-play game {game_index} ended generation before GameEnd"
        ));
    }

    let outcome = build_game_outcome(&runner, game_index, game_seed, positions.len())?;
    for position in &mut positions {
        position.value_target = SelfPlayValueTarget {
            shared_win: outcome.shared_win_values[position.actor],
            victory_point_margin: outcome.victory_point_margins[position.actor],
        };
    }

    Ok(GeneratedSelfPlayGame { outcome, positions })
}

fn build_action_targets(
    runner: &GameRunner,
    actor: usize,
    legal_actions: &[LegalAction],
    report: &crate::game::search::RootSearchReport,
) -> Result<Vec<SelfPlayActionTarget>, String> {
    let mut estimates = HashMap::with_capacity(report.recommendations.len());
    for estimate in &report.recommendations {
        if estimates
            .insert(estimate.action_key.as_str(), estimate)
            .is_some()
        {
            return Err(format!(
                "search returned duplicate action key {}",
                estimate.action_key
            ));
        }
    }

    let mut targets = Vec::with_capacity(legal_actions.len());
    let mut visit_sum = 0u64;
    for (index, action) in legal_actions.iter().enumerate() {
        let key = action.key();
        let estimate = estimates.get(key.as_str()).copied();
        let visits = estimate.map_or(0, |value| value.visits);
        visit_sum += visits;
        let feature_indices = encode_training_action(runner, actor, action)?;
        if feature_indices
            .iter()
            .any(|feature| *feature as usize >= ACTION_FEATURE_DIM)
        {
            return Err(format!("action {key} contains an out-of-range feature"));
        }
        targets.push(SelfPlayActionTarget {
            index,
            key,
            feature_indices,
            visits,
            policy_target: visits as f32 / report.completed_simulations as f32,
            estimated_shared_win_rate: estimate.map(|value| value.estimated_shared_win_rate as f32),
            average_victory_point_margin: estimate
                .map(|value| value.average_victory_point_margin as f32),
        });
    }
    if visit_sum != report.completed_simulations {
        return Err(format!(
            "stable action targets contain {visit_sum} visits, expected {}",
            report.completed_simulations
        ));
    }
    Ok(targets)
}

fn sample_action_index(targets: &[SelfPlayActionTarget], seed: u64) -> Result<usize, String> {
    let total_visits = targets.iter().map(|target| target.visits).sum::<u64>();
    if total_visits == 0 {
        return Err("cannot sample an action from zero total visits".to_string());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut draw = rng.gen_range(0..total_visits);
    for target in targets {
        if draw < target.visits {
            return Ok(target.index);
        }
        draw -= target.visits;
    }
    Err("visit sampling failed to select an action".to_string())
}

fn advance_to_next_decision(runner: &mut GameRunner) -> Result<u32, String> {
    let mut forced_action_slots = 0u32;
    for _ in 0..256 {
        resolve_shortfalls_deterministically(runner);
        if runner.is_game_finished() {
            return Ok(forced_action_slots);
        }
        if runner.actions_remaining_in_turn == 0 {
            let _ = runner.start_turn();
        }
        if !runner.framework.get_valid_root_actions().is_empty() {
            return Ok(forced_action_slots);
        }

        runner.end_action_slot();
        forced_action_slots += 1;
        if runner.actions_remaining_in_turn == 0 {
            runner.end_turn();
        }
    }
    Err("advance_to_next_decision exceeded 256 forced action slots".to_string())
}

fn resolve_shortfalls_deterministically(runner: &mut GameRunner) {
    for session in runner.take_shortfall_sessions() {
        let mut tiles = session.removable_tiles.clone();
        tiles.sort_by_key(|tile| (tile.liquidation_value, tile.build_location_idx));
        let chosen_tile_order = tiles
            .into_iter()
            .map(|tile| tile.build_location_idx)
            .collect();
        runner.resolve_shortfall_with_tiles(session, chosen_tile_order);
    }
}

fn build_game_outcome(
    runner: &GameRunner,
    game_index: u32,
    game_seed: u64,
    positions: usize,
) -> Result<SelfPlayGameRecord, String> {
    let state = &runner.framework.board.state;
    let winners = official_winners(runner)?;
    let winner_credit = 1.0 / winners.len() as f32;
    let shared_win_values = (0..state.players.len())
        .map(|player_idx| {
            if winners.contains(&player_idx) {
                winner_credit
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let ranking_keys = state
        .players
        .iter()
        .map(|player| (player.victory_points, player.income_level, player.money))
        .collect::<Vec<_>>();
    let placements = ranking_keys
        .iter()
        .map(|key| 1 + ranking_keys.iter().filter(|other| *other > key).count())
        .collect::<Vec<_>>();
    let mut finish_order = (0..state.players.len()).collect::<Vec<_>>();
    finish_order.sort_by(|left, right| {
        ranking_keys[*right]
            .cmp(&ranking_keys[*left])
            .then_with(|| left.cmp(right))
    });
    let victory_points = state
        .players
        .iter()
        .map(|player| player.victory_points)
        .collect::<Vec<_>>();
    let victory_point_margins = victory_points
        .iter()
        .enumerate()
        .map(|(player_idx, score)| {
            let best_opponent = victory_points
                .iter()
                .enumerate()
                .filter_map(|(other_idx, other)| (other_idx != player_idx).then_some(*other))
                .max()
                .unwrap_or(0);
            *score as i32 - best_opponent as i32
        })
        .collect::<Vec<_>>();

    Ok(SelfPlayGameRecord {
        record_type: "game",
        format_version: SELF_PLAY_FORMAT_VERSION,
        game_index,
        game_seed,
        positions,
        official_winners: winners,
        shared_win_values,
        placements,
        finish_order,
        victory_points,
        victory_point_margins,
        income_levels: state
            .players
            .iter()
            .map(|player| player.income_level)
            .collect(),
        money: state.players.iter().map(|player| player.money).collect(),
    })
}

fn derive_stream_seed(base_seed: u64, stream: u64, index: u64) -> u64 {
    splitmix64(base_seed ^ stream ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn phase_name(phase: GamePhase) -> &'static str {
    match phase {
        GamePhase::Canal => "canal",
        GamePhase::Railroad => "railroad",
        GamePhase::GameEnd => "game_end",
    }
}

fn partial_output_path(output_path: &Path) -> Result<PathBuf, String> {
    let file_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid output filename {}", output_path.display()))?;
    Ok(output_path.with_file_name(format!(".{file_name}.partial-{}", std::process::id())))
}

fn write_json_line<W: Write, T: Serialize>(writer: &mut W, value: &T) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, value)
        .map_err(|error| format!("failed to encode self-play record: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("failed to write self-play record: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Era;

    fn compact_rail_endgame() -> GameRunner {
        let mut runner = GameRunner::new(2, Some(81_001));
        runner.game_phase = GamePhase::Railroad;
        runner.framework.board.state.era = Era::Railroad;
        runner.framework.board.state.deck.cards.clear();
        runner.framework.board.state.discard_pile.clear();
        runner.actions_remaining_in_turn = 0;
        runner.personal_turns_taken.fill(0);
        for player in &mut runner.framework.board.state.players {
            player.hand.cards.truncate(1);
            player.money = 0;
            player.income_level = 0;
            player.victory_points = 0;
            player.spent_this_turn = 0;
        }
        runner
    }

    fn test_config() -> SelfPlayConfig {
        SelfPlayConfig {
            games: 1,
            num_players: 2,
            base_seed: 91_001,
            simulations_per_decision: 4,
            max_rollout_actions: 8,
            max_game_actions: 8,
            engine_revision: "test-revision".to_string(),
            ..SelfPlayConfig::default()
        }
    }

    #[test]
    fn compact_self_play_is_deterministic_and_has_complete_targets() {
        let config = test_config();
        let first = generate_self_play_game_from_runner(&config, 0, 81_001, compact_rail_endgame())
            .unwrap();
        let second =
            generate_self_play_game_from_runner(&config, 0, 81_001, compact_rail_endgame())
                .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.positions.len(), 2);
        assert_eq!(first.outcome.shared_win_values, vec![0.5, 0.5]);
        for position in &first.positions {
            assert_eq!(position.state_features.len(), STATE_FEATURE_DIM);
            assert_eq!(
                position
                    .legal_actions
                    .iter()
                    .map(|action| action.visits)
                    .sum::<u64>(),
                config.simulations_per_decision
            );
            assert!(position
                .legal_actions
                .iter()
                .enumerate()
                .all(|(index, action)| action.index == index));
            assert!(position.legal_actions[position.selected_action_index].visits > 0);
            assert_eq!(position.value_target.shared_win, 0.5);
            assert_eq!(position.value_target.victory_point_margin, 0);
        }
    }

    #[test]
    fn header_locks_format_feature_and_engine_versions() {
        let config = test_config();
        let header = self_play_shard_header(&config);
        assert_eq!(header.format, SELF_PLAY_FORMAT);
        assert_eq!(header.format_version, SELF_PLAY_FORMAT_VERSION);
        assert_eq!(header.feature_schema.version, TRAINING_FEATURE_VERSION);
        assert_eq!(header.feature_schema.state_dim, STATE_FEATURE_DIM);
        assert_eq!(header.feature_schema.action_dim, ACTION_FEATURE_DIM);
        assert_eq!(header.engine_revision, "test-revision");
    }
}
