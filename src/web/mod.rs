pub mod analysis;
mod browser;
mod model_inference;
pub mod serialize;

use axum::{
    extract::State,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use rusqlite::{params, Connection};
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tower_http::{compression::CompressionLayer, decompression::RequestDecompressionLayer};

use crate::board::resources::{BeerSellSource, BreweryBeerSource, ResourceSource};
use crate::core::types::*;
use crate::game::framework::{ActionChoice, ChoiceSet, NetworkMode};
use crate::game::rule_ai::{rule_decision_report, RuleDecisionConfig};
use crate::game::runner::{GameRunner, ReplayTurnCheckpoint};
use crate::game::search::{
    search_top_actions, BatchedNeuralPuctConfig, BatchedNeuralPuctSearch, RootSearchConfig,
    RootSearchReport,
};

use analysis::{explain_analysis_question_for_observer, serialize_analysis_for_observer};
use model_inference::ModelInferenceClient;
use serialize::*;

pub struct ServerState {
    pub runner: Option<GameRunner>,
    pub active_game_id: Option<i64>,
    pub observer_player: Option<usize>,
    pub db_path: String,
    inference_client: Option<ModelInferenceClient>,
    revision: u64,
    last_analysis: Option<CachedAnalysis>,
    analysis_session: Option<NeuralAnalysisSession>,
}

#[derive(Clone)]
struct CachedAnalysis {
    revision: u64,
    runner: GameRunner,
    report: RootSearchReport,
}

struct NeuralAnalysisSession {
    revision: u64,
    config: RootSearchConfig,
    model_id: String,
    search: BatchedNeuralPuctSearch,
}

impl ServerState {
    fn mark_position_changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.last_analysis = None;
        self.analysis_session = None;
    }
}

pub type SharedState = Arc<Mutex<ServerState>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistEvent {
    StartTurn,
    StartAction {
        action_type: String,
    },
    ApplyChoice {
        choice_kind: String,
        value: serde_json::Value,
    },
    ConfirmAction,
    CancelAction,
    UndoLastAction,
    EndTurn,
    ResolveShortfall {
        player_idx: usize,
        chosen_tile_order: Vec<usize>,
    },
    ReplayAnalysis {
        position_key: String,
        player_idx: usize,
        state: serde_json::Value,
        analysis: serde_json::Value,
    },
    ReplayMove {
        position_key: String,
        player_idx: usize,
        action_type: String,
        action_key: Option<String>,
        selections: Vec<String>,
        before_state: serde_json::Value,
        after_state: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
struct GameListItem {
    id: i64,
    created_at: i64,
    round_in_phase: u32,
    era: String,
    num_players: usize,
    seed: u64,
}

#[derive(Embed)]
#[folder = "src/public/"]
#[prefix = "/"]
struct Assets;

#[cfg(feature = "cloud-ui")]
#[derive(Embed)]
#[folder = "ui/build/"]
struct UiAssets;

pub async fn start_server(port: u16) {
    let db_path = std::env::var("FAST_BRASS_DB_PATH").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("games.sqlite3");
        p.to_string_lossy().to_string()
    });
    let _ = init_db(&db_path);
    let inference_client = ModelInferenceClient::from_env()
        .unwrap_or_else(|error| panic!("invalid model inference configuration: {error}"));
    if let Some(client) = &inference_client {
        println!("Model-guided analysis enabled via {}", client.endpoint());
    }
    let state: SharedState = Arc::new(Mutex::new(ServerState {
        runner: None,
        active_game_id: None,
        observer_player: None,
        db_path,
        inference_client,
        revision: 0,
        last_analysis: None,
        analysis_session: None,
    }));

    #[cfg(not(feature = "cloud-ui"))]
    let app = Router::new()
        .route(
            "/api/browser_request",
            browser_request_route(),
        )
        .route("/api/new_game", post(api_new_game))
        .route("/api/games", get(api_games))
        .route("/api/load_game", post(api_load_game))
        .route("/api/replay", post(api_replay))
        .route("/api/state", get(api_state))
        .route("/api/set_observer", post(api_set_observer))
        .route("/api/industry_data", get(api_industry_data))
        .route("/api/analyze", post(api_analyze))
        .route("/api/explain", post(api_explain))
        .route(
            "/api/apply_analyzed_action",
            post(api_apply_analyzed_action),
        )
        .route("/api/resolve_shortfalls", post(api_resolve_shortfalls))
        .route("/api/start_turn", post(api_start_turn))
        .route("/api/start_action", post(api_start_action))
        .route("/api/apply_choice", post(api_apply_choice))
        .route("/api/confirm_action", post(api_confirm_action))
        .route("/api/undo_last_action", post(api_undo_last_action))
        .route("/api/cancel_action", post(api_cancel_action))
        .route("/api/end_turn", post(api_end_turn))
        .with_state(state)
        .layer(axum::extract::DefaultBodyLimit::max(4 * 1024 * 1024))
        .fallback(static_handler);

    #[cfg(feature = "cloud-ui")]
    let app = Router::new()
        .route(
            "/api/browser_request",
            browser_request_route(),
        )
        .route("/api/industry_data", get(api_industry_data))
        .with_state(state)
        .layer(axum::extract::DefaultBodyLimit::max(4 * 1024 * 1024))
        .fallback(static_handler);

    let addr = format!("0.0.0.0:{}", port);
    println!("Starting Brass Birmingham at http://localhost:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn browser_request_route() -> axum::routing::MethodRouter<SharedState> {
    post(browser::request)
        .layer::<_, std::convert::Infallible>(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer::<_, std::convert::Infallible>(RequestDecompressionLayer::new())
        .layer(CompressionLayer::new())
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn init_db(path: &str) -> Result<(), String> {
    let conn = Connection::open(path).map_err(|e| format!("db open: {}", e))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            num_players INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            action_log TEXT NOT NULL,
            round_in_phase INTEGER NOT NULL,
            era TEXT NOT NULL
        );",
    )
    .map_err(|e| format!("db schema: {}", e))?;
    Ok(())
}

fn with_actions_if_available(
    runner: &GameRunner,
    observer_player: Option<usize>,
    gs: &mut FullGameState,
) {
    if observer_player_index(runner, observer_player) == runner.framework.current_player
        && runner.framework.current_session().is_none()
        && runner.actions_remaining_in_turn > 0
    {
        let actions = runner.framework.get_valid_root_actions();
        gs.available_actions = Some(actions.iter().map(|a| action_type_str(*a)).collect());
    }
}

fn serialize_state_for_observer(
    runner: &GameRunner,
    observer_player: Option<usize>,
) -> FullGameState {
    let mut game_state = serialize_game_state(runner, observer_player);
    with_actions_if_available(runner, observer_player, &mut game_state);
    game_state
}

fn append_event_and_update_meta(
    db_path: &str,
    game_id: i64,
    event: PersistEvent,
    runner: &GameRunner,
) -> Result<(), String> {
    append_events_and_update_meta(db_path, game_id, &[event], runner)
}

fn append_events_and_update_meta(
    db_path: &str,
    game_id: i64,
    new_events: &[PersistEvent],
    runner: &GameRunner,
) -> Result<(), String> {
    let conn = Connection::open(db_path).map_err(|e| format!("db open: {}", e))?;
    let action_log: String = conn
        .query_row(
            "SELECT action_log FROM games WHERE id = ?1",
            params![game_id],
            |r| r.get(0),
        )
        .map_err(|e| format!("db load log: {}", e))?;
    let mut events: Vec<PersistEvent> =
        serde_json::from_str(&action_log).map_err(|e| format!("decode log: {}", e))?;
    events.extend_from_slice(new_events);
    let encoded = serde_json::to_string(&events).map_err(|e| format!("encode log: {}", e))?;
    conn.execute(
        "UPDATE games
         SET action_log = ?1, updated_at = ?2, round_in_phase = ?3, era = ?4
         WHERE id = ?5",
        params![
            encoded,
            now_unix(),
            runner.round_in_phase as i64,
            format!("{:?}", runner.framework.board.state.era),
            game_id
        ],
    )
    .map_err(|e| format!("db update: {}", e))?;
    Ok(())
}

fn replay_events(runner: &mut GameRunner, events: &[PersistEvent]) -> Result<(), String> {
    runner.framework.replay_mode = true;
    let mut turn_checkpoint: Option<ReplayTurnCheckpoint> = None;
    let mut replay_aborted = false;
    for ev in events {
        if replay_aborted {
            break;
        }
        match ev {
            PersistEvent::StartTurn => {
                let _ = runner.start_turn();
                turn_checkpoint = Some(runner.checkpoint_replay_turn());
            }
            PersistEvent::StartAction { action_type } => {
                let action = action_type_from_str(action_type)
                    .ok_or_else(|| format!("Unknown action {}", action_type))?;
                let _ = runner.start_action(action);
            }
            PersistEvent::ApplyChoice { choice_kind, value } => {
                let choice = parse_choice(choice_kind, value)?;
                let _ = runner.apply_choice(choice);
            }
            PersistEvent::ConfirmAction => {
                let mut confirm_result = runner.confirm_action();
                if let Err(e) = &confirm_result {
                    if e.contains("without selecting card") {
                        if let Some(ChoiceSet::Card(opts)) = runner.framework.get_next_choice_set()
                        {
                            if let Some(card_idx) = opts.first() {
                                let _ = runner.apply_choice(ActionChoice::Card(*card_idx));
                                confirm_result = runner.confirm_action();
                                if confirm_result.is_ok() {
                                    eprintln!(
                                        "Replay compatibility: auto-selected card {} before confirm.",
                                        card_idx
                                    );
                                }
                            }
                        }
                    }
                }
                if let Err(e) = confirm_result {
                    eprintln!(
                        "Replay confirm failed: {}. Rolling back to turn start and stopping replay.",
                        e
                    );
                    if let Some(cp) = turn_checkpoint.take() {
                        runner.restore_replay_turn(cp);
                    } else {
                        runner.framework.cancel_action_session();
                    }
                    replay_aborted = true;
                }
            }
            PersistEvent::CancelAction => {
                runner.framework.cancel_action_session();
            }
            PersistEvent::UndoLastAction => {
                if let Err(e) = runner.undo_last_confirmed_action() {
                    eprintln!("Replay undo failed: {}. Skipping.", e);
                }
            }
            PersistEvent::EndTurn => {
                runner.end_turn_for_replay();
                turn_checkpoint = None;
            }
            PersistEvent::ResolveShortfall {
                player_idx,
                chosen_tile_order,
            } => {
                let session_idx = runner
                    .pending_shortfall_sessions
                    .iter()
                    .position(|session| session.player_idx == *player_idx)
                    .ok_or_else(|| {
                        format!("No pending shortfall for player {player_idx} during replay")
                    })?;
                let session = runner.pending_shortfall_sessions.remove(session_idx);
                runner.resolve_shortfall_with_tiles(session, chosen_tile_order.clone());
            }
            PersistEvent::ReplayAnalysis { .. } | PersistEvent::ReplayMove { .. } => {}
        }
    }
    runner.framework.replay_mode = false;
    Ok(())
}

fn replay_position_key(runner: &GameRunner) -> String {
    format!(
        "{:?}:{}:{}:{}:{}:{}",
        runner.game_phase,
        runner.round_in_phase,
        runner.turn_count,
        runner.framework.current_player,
        runner.actions_remaining_in_turn,
        runner.framework.current_session().is_some()
    )
}

#[derive(Debug, Clone, Serialize)]
struct ReplayMoveJson {
    player_idx: usize,
    action_type: String,
    action_key: Option<String>,
    selections: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReplayPositionJson {
    index: usize,
    position_key: String,
    player_idx: usize,
    state: serde_json::Value,
    analysis: Option<serde_json::Value>,
    #[serde(rename = "move")]
    move_data: Option<ReplayMoveJson>,
}

fn replay_history_from_events(
    events: &[PersistEvent],
) -> (Vec<ReplayPositionJson>, Option<serde_json::Value>) {
    let mut positions = Vec::new();
    let mut position_indices = HashMap::<String, usize>::new();
    let mut final_state = None;

    for event in events {
        match event {
            PersistEvent::ReplayAnalysis {
                position_key,
                player_idx,
                state,
                analysis,
            } => {
                let index = *position_indices
                    .entry(position_key.clone())
                    .or_insert_with(|| {
                        let index = positions.len();
                        positions.push(ReplayPositionJson {
                            index,
                            position_key: position_key.clone(),
                            player_idx: *player_idx,
                            state: state.clone(),
                            analysis: None,
                            move_data: None,
                        });
                        index
                    });
                let position = &mut positions[index];
                position.player_idx = *player_idx;
                position.state = state.clone();
                position.analysis = Some(analysis.clone());
            }
            PersistEvent::ReplayMove {
                position_key,
                player_idx,
                action_type,
                action_key,
                selections,
                before_state,
                after_state,
            } => {
                let index = *position_indices
                    .entry(position_key.clone())
                    .or_insert_with(|| {
                        let index = positions.len();
                        positions.push(ReplayPositionJson {
                            index,
                            position_key: position_key.clone(),
                            player_idx: *player_idx,
                            state: before_state.clone(),
                            analysis: None,
                            move_data: None,
                        });
                        index
                    });
                let position = &mut positions[index];
                position.player_idx = *player_idx;
                position.move_data = Some(ReplayMoveJson {
                    player_idx: *player_idx,
                    action_type: action_type.clone(),
                    action_key: action_key.clone(),
                    selections: selections.clone(),
                });
                final_state = Some(after_state.clone());
            }
            _ => {}
        }
    }

    for (index, position) in positions.iter_mut().enumerate() {
        position.index = index;
    }
    (positions, final_state)
}

async fn static_handler(uri: Uri) -> impl IntoResponse {
    let path = uri.path().to_string();
    let path = if path == "/" || path.is_empty() {
        "/index.html".to_string()
    } else {
        path
    };

    let mime = match path.rsplit('.').next() {
        Some("html") => "text/html",
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    };

    #[cfg(not(feature = "cloud-ui"))]
    let asset = Assets::get(&path);
    #[cfg(feature = "cloud-ui")]
    let asset = UiAssets::get(path.trim_start_matches('/'));
    match asset {
        Some(asset) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime)
            .body(axum::body::Body::from(asset.data.to_vec()))
            .unwrap()
            .into_response(),
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(axum::body::Body::from("Not Found"))
            .unwrap()
            .into_response(),
    }
}

#[derive(Deserialize)]
struct NewGameRequest {
    num_players: usize,
    seed: Option<u64>,
}

async fn api_new_game(
    State(state): State<SharedState>,
    Json(req): Json<NewGameRequest>,
) -> Json<serde_json::Value> {
    let runner = GameRunner::new(req.num_players, req.seed);
    let mut guard = state.lock().await;

    let conn = match Connection::open(&guard.db_path) {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("db open: {}", e)}))
        }
    };
    let created = now_unix();
    let seed = runner.framework.board.state.seed as i64;
    let era = format!("{:?}", runner.framework.board.state.era);
    if let Err(e) = conn.execute(
        "INSERT INTO games (created_at, updated_at, num_players, seed, action_log, round_in_phase, era)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![created, created, req.num_players as i64, seed, "[]", 0_i64, era],
    ) {
        return Json(serde_json::json!({"ok": false, "error": format!("db insert: {}", e)}));
    }
    let game_id = conn.last_insert_rowid();

    guard.active_game_id = Some(game_id);
    guard.runner = Some(runner);
    guard.observer_player = None;
    guard.mark_position_changed();
    let runner = guard.runner.as_ref().unwrap();
    let gs = serialize_state_for_observer(runner, guard.observer_player);
    Json(serde_json::json!({"ok": true, "state": gs}))
}

async fn api_games(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let guard = state.lock().await;
    let conn = match Connection::open(&guard.db_path) {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("db open: {}", e)}))
        }
    };
    let mut stmt = match conn.prepare(
        "SELECT id, created_at, round_in_phase, era, num_players, seed
         FROM games ORDER BY updated_at DESC",
    ) {
        Ok(s) => s,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("db query: {}", e)}))
        }
    };
    let rows = match stmt.query_map([], |r| {
        Ok(GameListItem {
            id: r.get(0)?,
            created_at: r.get::<_, i64>(1)?,
            round_in_phase: r.get::<_, i64>(2)? as u32,
            era: r.get::<_, String>(3)?,
            num_players: r.get::<_, i64>(4)? as usize,
            seed: r.get::<_, i64>(5)? as u64,
        })
    }) {
        Ok(r) => r,
        Err(e) => return Json(serde_json::json!({"ok": false, "error": format!("db map: {}", e)})),
    };
    let games: Vec<GameListItem> = rows.filter_map(Result::ok).collect();
    Json(serde_json::json!({"ok": true, "games": games}))
}

#[derive(Deserialize)]
struct LoadGameRequest {
    game_id: i64,
}

async fn api_load_game(
    State(state): State<SharedState>,
    Json(req): Json<LoadGameRequest>,
) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let conn = match Connection::open(&guard.db_path) {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("db open: {}", e)}))
        }
    };
    let row = conn.query_row(
        "SELECT num_players, seed, action_log FROM games WHERE id = ?1",
        params![req.game_id],
        |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
            ))
        },
    );
    let (num_players, seed, action_log) = match row {
        Ok(v) => v,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("load game: {}", e)}))
        }
    };
    let events: Vec<PersistEvent> = match serde_json::from_str(&action_log) {
        Ok(v) => v,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("decode log: {}", e)}))
        }
    };
    let mut runner = GameRunner::new(num_players as usize, Some(seed as u64));
    if let Err(e) = replay_events(&mut runner, &events) {
        return Json(serde_json::json!({"ok": false, "error": format!("replay: {}", e)}));
    }
    guard.active_game_id = Some(req.game_id);
    guard.runner = Some(runner);
    guard.observer_player = None;
    guard.mark_position_changed();
    let runner = guard.runner.as_ref().unwrap();
    let gs = serialize_state_for_observer(runner, guard.observer_player);
    Json(serde_json::json!({"ok": true, "state": gs}))
}

async fn api_replay(
    State(state): State<SharedState>,
    Json(req): Json<LoadGameRequest>,
) -> Json<serde_json::Value> {
    let guard = state.lock().await;
    let conn = match Connection::open(&guard.db_path) {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!("db open: {}", e)}))
        }
    };
    let action_log: String = match conn.query_row(
        "SELECT action_log FROM games WHERE id = ?1",
        params![req.game_id],
        |row| row.get(0),
    ) {
        Ok(value) => value,
        Err(e) => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("load replay: {}", e)
            }))
        }
    };
    let events: Vec<PersistEvent> = match serde_json::from_str(&action_log) {
        Ok(value) => value,
        Err(e) => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("decode replay: {}", e)
            }))
        }
    };
    let (positions, final_state) = replay_history_from_events(&events);
    Json(serde_json::json!({
        "ok": true,
        "game_id": req.game_id,
        "positions": positions,
        "final_state": final_state,
    }))
}

async fn api_state(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let guard = state.lock().await;
    match guard.runner.as_ref() {
        Some(runner) => {
            let gs = serialize_state_for_observer(runner, guard.observer_player);
            Json(serde_json::json!({"ok": true, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

#[derive(Deserialize)]
struct SetObserverRequest {
    player_index: Option<usize>,
}

async fn api_set_observer(
    State(state): State<SharedState>,
    Json(req): Json<SetObserverRequest>,
) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let Some(player_count) = guard
        .runner
        .as_ref()
        .map(|runner| runner.framework.board.state.players.len())
    else {
        return Json(serde_json::json!({"ok": false, "error": "No game in progress"}));
    };
    if let Some(player_index) = req.player_index {
        if player_index >= player_count {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!(
                    "observer player index {player_index} is out of range for {player_count} players"
                )
            }));
        }
    }

    let observer_changed = guard.observer_player != req.player_index;
    guard.observer_player = req.player_index;
    if observer_changed {
        // Analysis includes private-card identities when the observer is the acting player.
        // Treat an observer change as a new view revision so cached actions cannot leak or apply
        // against a different seat's hand.
        guard.mark_position_changed();
    }
    let runner = guard.runner.as_ref().expect("runner was checked above");
    let game_state = serialize_state_for_observer(runner, guard.observer_player);
    Json(serde_json::json!({
        "ok": true,
        "observer_player": guard.observer_player,
        "revision": guard.revision,
        "state": game_state
    }))
}

async fn api_industry_data() -> Json<serde_json::Value> {
    Json(serialize::serialize_all_industry_data())
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    #[serde(skip)]
    skip_replay_recording: bool,
    simulations: Option<u64>,
    progress_to: Option<u64>,
    top_n: Option<usize>,
    seed: Option<u64>,
    mode: Option<String>,
}

const WEB_NEURAL_DETERMINIZATIONS: usize = 4;
const WEB_NEURAL_BATCH_SIZE: usize = 64;

async fn create_model_guided_search(
    client: &ModelInferenceClient,
    runner: &GameRunner,
    config: &RootSearchConfig,
) -> Result<(BatchedNeuralPuctSearch, String), String> {
    let policy = client.evaluate(runner).await?;
    let model_id = policy.model_id.clone();
    let search = BatchedNeuralPuctSearch::new(
        runner,
        BatchedNeuralPuctConfig {
            search: config.clone(),
            determinizations: WEB_NEURAL_DETERMINIZATIONS,
            score_utility_weight: 0.0,
            final_vp_utility_weight: 0.0,
            group_card_choices: false,
        },
        policy,
    )?;
    Ok((search, model_id))
}

async fn advance_model_guided_search(
    client: &ModelInferenceClient,
    search: &mut BatchedNeuralPuctSearch,
    model_id: &str,
    target_simulations: u64,
) -> Result<RootSearchReport, String> {
    while search.completed_simulations() < target_simulations {
        let leaves =
            search.next_inference_batch_until(WEB_NEURAL_BATCH_SIZE, target_simulations)?;
        if leaves.is_empty() {
            if search.completed_simulations() >= target_simulations {
                break;
            }
            return Err("neural search returned an empty incomplete leaf batch".to_string());
        }
        let evaluations = client.evaluate_leaf_batch(&leaves, model_id).await?;
        search.submit_inference_batch(evaluations)?;
    }
    search.report_at_current_progress()
}

async fn api_analyze(
    State(state): State<SharedState>,
    Json(req): Json<AnalyzeRequest>,
) -> Json<serde_json::Value> {
    // The embedded teacher uses native transitions and needs no inference service.
    let mode = req.mode.as_deref().unwrap_or("trained");
    let use_trained_mode = mode == "trained";
    let use_rule_mode = match mode {
        "rule" => true,
        "trained" | "auto" | "search" | "neural" => false,
        _ => {
            return Json(serde_json::json!({
                "ok": false,
                "error": "mode must be one of trained, auto, search, neural, or rule"
            }))
        }
    };
    let simulations = req.simulations.unwrap_or(2_000);
    if !(1..=50_000).contains(&simulations) {
        return Json(serde_json::json!({
            "ok": false,
            "error": "simulations must be between 1 and 50000"
        }));
    }
    let progress_to = req.progress_to.unwrap_or(simulations);
    if !(1..=simulations).contains(&progress_to) {
        return Json(serde_json::json!({
            "ok": false,
            "error": "progress_to must be between 1 and simulations"
        }));
    }
    let top_n = req.top_n.unwrap_or(3);
    if !(1..=10).contains(&top_n) {
        return Json(serde_json::json!({
            "ok": false,
            "error": "top_n must be between 1 and 10"
        }));
    }

    let (runner, revision, inference_client, previous_session) = {
        let mut guard = state.lock().await;
        let Some(runner) = guard.runner.as_ref() else {
            return Json(serde_json::json!({"ok": false, "error": "No game in progress"}));
        };
        (
            runner.clone(),
            guard.revision,
            guard.inference_client.clone(),
            guard.analysis_session.take(),
        )
    };
    let search_seed = req.seed.unwrap_or_else(|| {
        runner.framework.board.state.seed ^ revision.rotate_left(17) ^ simulations.rotate_left(31)
    });
    let config = RootSearchConfig {
        simulations,
        recommendation_count: top_n,
        seed: search_seed,
        ..RootSearchConfig::default()
    };
    let started = Instant::now();
    let mut previous_session = previous_session;
    let mut retained_session = None;
    let report_result = if use_trained_mode {
        let trained_runner = runner.clone();
        match tokio::task::spawn_blocking(move || {
            crate::game::trained_ai::trained_decision_report(&trained_runner, top_n)
        })
        .await
        {
            Ok(result) => result,
            Err(error) => Err(format!("trained analysis task failed: {error}")),
        }
    } else if use_rule_mode {
        let rule_runner = runner.clone();
        let rule_config = RuleDecisionConfig {
            recommendation_count: top_n,
            ..RuleDecisionConfig::default()
        };
        match tokio::task::spawn_blocking(move || rule_decision_report(&rule_runner, &rule_config))
            .await
        {
            Ok(result) => result,
            Err(error) => Err(format!("rule analysis task failed: {error}")),
        }
    } else {
        match inference_client {
            Some(client) => {
                let mut session = match previous_session
                    .take()
                    .filter(|session| session.revision == revision && session.config == config)
                {
                    Some(session) => session,
                    None => {
                        let (search, model_id) =
                            match create_model_guided_search(&client, &runner, &config).await {
                                Ok(value) => value,
                                Err(error) => {
                                    return Json(serde_json::json!({"ok": false, "error": error}))
                                }
                            };
                        NeuralAnalysisSession {
                            revision,
                            config: config.clone(),
                            model_id,
                            search,
                        }
                    }
                };
                let result = advance_model_guided_search(
                    &client,
                    &mut session.search,
                    &session.model_id,
                    progress_to,
                )
                .await;
                if result.is_ok() && !session.search.is_complete() {
                    retained_session = Some(session);
                }
                result
            }
            None => {
                let mut stage_config = config.clone();
                stage_config.simulations = progress_to;
                let search_runner = runner.clone();
                match tokio::task::spawn_blocking(move || {
                    search_top_actions(&search_runner, &stage_config)
                })
                .await
                {
                    Ok(result) => result.map(|mut report| {
                        report.requested_simulations = simulations;
                        report
                    }),
                    Err(error) => Err(format!("analysis task failed: {error}")),
                }
            }
        }
    };
    let report = match report_result {
        Ok(report) => report,
        Err(error) => {
            return Json(serde_json::json!({"ok": false, "error": error}));
        }
    };
    let elapsed_ms = started.elapsed().as_millis();

    let mut guard = state.lock().await;
    if guard.revision != revision {
        return Json(serde_json::json!({
            "ok": false,
            "error": "The position changed while analysis was running. Analyze the current position again."
        }));
    }
    guard.analysis_session = retained_session;
    let reveal_private_cards =
        observer_player_index(&runner, guard.observer_player) == report.root_player;
    let analysis = serialize_analysis_for_observer(
        &report,
        &runner,
        revision,
        elapsed_ms,
        reveal_private_cards,
    );
    if progress_to == simulations && !req.skip_replay_recording {
        if let Some(game_id) = guard.active_game_id {
            match (
                serde_json::to_value(serialize_state_for_observer(&runner, guard.observer_player)),
                serde_json::to_value(&analysis),
            ) {
                (Ok(state), Ok(analysis_value)) => {
                    let event = PersistEvent::ReplayAnalysis {
                        position_key: replay_position_key(&runner),
                        player_idx: runner.framework.current_player,
                        state,
                        analysis: analysis_value,
                    };
                    if let Err(error) =
                        append_event_and_update_meta(&guard.db_path, game_id, event, &runner)
                    {
                        eprintln!("Replay analysis persistence failed: {error}");
                    }
                }
                (Err(error), _) | (_, Err(error)) => {
                    eprintln!("Replay analysis serialization failed: {error}");
                }
            }
        }
    }
    guard.last_analysis = Some(CachedAnalysis {
        revision,
        runner,
        report,
    });
    Json(serde_json::json!({"ok": true, "analysis": analysis}))
}

#[derive(Deserialize)]
struct ExplainRequest {
    revision: u64,
    action_key: String,
    question: String,
}

async fn api_explain(
    State(state): State<SharedState>,
    Json(req): Json<ExplainRequest>,
) -> Json<serde_json::Value> {
    let question = req.question.trim();
    if question.is_empty() || question.chars().count() > 500 {
        return Json(serde_json::json!({
            "ok": false,
            "error": "question must contain between 1 and 500 characters"
        }));
    }
    let (cached, reveal_private_cards) = {
        let guard = state.lock().await;
        if guard.revision != req.revision {
            return Json(serde_json::json!({
                "ok": false,
                "error": "The position changed. Analyze it again before asking a question."
            }));
        }
        let Some(cached) = guard.last_analysis.as_ref() else {
            return Json(serde_json::json!({
                "ok": false,
                "error": "No analysis is cached for this position"
            }));
        };
        if cached.revision != req.revision {
            return Json(serde_json::json!({
                "ok": false,
                "error": "The cached analysis is stale"
            }));
        }
        let reveal_private_cards = observer_player_index(&cached.runner, guard.observer_player)
            == cached.report.root_player;
        (cached.clone(), reveal_private_cards)
    };

    match explain_analysis_question_for_observer(
        &cached.report,
        &cached.runner,
        cached.revision,
        &req.action_key,
        question,
        reveal_private_cards,
    ) {
        Ok(explanation) => Json(serde_json::json!({"ok": true, "explanation": explanation})),
        Err(error) => Json(serde_json::json!({"ok": false, "error": error})),
    }
}

#[derive(Deserialize)]
struct ApplyAnalyzedActionRequest {
    revision: u64,
    action_key: String,
}

async fn api_apply_analyzed_action(
    State(state): State<SharedState>,
    Json(req): Json<ApplyAnalyzedActionRequest>,
) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    if guard.revision != req.revision {
        return Json(serde_json::json!({
            "ok": false,
            "error": "The position changed. Analyze it again before applying a recommendation."
        }));
    }
    let Some(cached) = guard.last_analysis.as_ref() else {
        return Json(serde_json::json!({
            "ok": false,
            "error": "No analysis is cached for this position"
        }));
    };
    let Some(action) = cached
        .report
        .recommendations
        .iter()
        .find(|candidate| candidate.action_key == req.action_key)
        .map(|candidate| candidate.action.clone())
    else {
        return Json(serde_json::json!({
            "ok": false,
            "error": "The requested recommendation is not in the cached analysis"
        }));
    };

    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    let replay_position = replay_position_key(&cached.runner);
    let replay_player = cached.runner.framework.current_player;
    let replay_before_state = match serde_json::to_value(serialize_state_for_observer(
        &cached.runner,
        observer_player,
    )) {
        Ok(state) => state,
        Err(error) => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("serialize replay position: {}", error)
            }))
        }
    };
    let runner = guard
        .runner
        .as_mut()
        .expect("cached analysis requires a runner");
    let before_runner = runner.clone();
    let before_turn_count = runner.turn_count;
    let mut events = Vec::new();
    if runner.actions_remaining_in_turn == 0 {
        events.push(PersistEvent::StartTurn);
    }
    events.push(PersistEvent::StartAction {
        action_type: action_type_str(action.root).to_string(),
    });
    for choice in &action.choices {
        match persist_event_for_choice(choice) {
            Ok(Some(event)) => events.push(event),
            Ok(None) => {}
            Err(error) => return Json(serde_json::json!({"ok": false, "error": error})),
        }
    }
    events.push(PersistEvent::ConfirmAction);

    if let Err(error) = action.apply(runner) {
        return Json(serde_json::json!({
            "ok": false,
            "error": format!("recommendation became stale: {error}")
        }));
    }
    if runner.turn_count > before_turn_count {
        events.push(PersistEvent::EndTurn);
        if !runner.is_game_finished() && runner.actions_remaining_in_turn > 0 {
            events.push(PersistEvent::StartTurn);
        }
    }
    let replay_after_state =
        match serde_json::to_value(serialize_state_for_observer(runner, observer_player)) {
            Ok(state) => state,
            Err(error) => {
                *runner = before_runner;
                return Json(serde_json::json!({
                    "ok": false,
                    "error": format!("serialize replay result: {}", error)
                }));
            }
        };
    events.push(PersistEvent::ReplayMove {
        position_key: replay_position,
        player_idx: replay_player,
        action_type: action_type_str(action.root).to_string(),
        action_key: Some(req.action_key.clone()),
        selections: Vec::new(),
        before_state: replay_before_state,
        after_state: replay_after_state,
    });
    if let Some(game_id) = active_game_id {
        if let Err(error) = append_events_and_update_meta(&db_path, game_id, &events, runner) {
            *runner = before_runner;
            return Json(serde_json::json!({"ok": false, "error": error}));
        }
    }

    let game_state = serialize_state_for_observer(runner, observer_player);
    guard.mark_position_changed();
    let revision = guard.revision;
    Json(serde_json::json!({
        "ok": true,
        "revision": revision,
        "applied_action_key": req.action_key,
        "state": game_state
    }))
}

async fn api_resolve_shortfalls(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    let Some(runner) = guard.runner.as_mut() else {
        return Json(serde_json::json!({"ok": false, "error": "No game in progress"}));
    };
    if !runner.has_pending_shortfall() {
        let game_state = serialize_state_for_observer(runner, observer_player);
        return Json(serde_json::json!({
            "ok": true,
            "resolved": [],
            "state": game_state
        }));
    }

    let before_runner = runner.clone();
    let mut events = Vec::new();
    let mut resolved = Vec::new();
    for session in runner.take_shortfall_sessions() {
        let chosen_tile_order = deterministic_shortfall_tile_order(&session);
        events.push(PersistEvent::ResolveShortfall {
            player_idx: session.player_idx,
            chosen_tile_order: chosen_tile_order.clone(),
        });
        resolved.push(serde_json::json!({
            "player_idx": session.player_idx,
            "shortfall": session.shortfall,
            "chosen_tile_order": chosen_tile_order,
        }));
        runner.resolve_shortfall_with_tiles(session, chosen_tile_order);
    }
    if !runner.is_game_finished() && runner.actions_remaining_in_turn == 0 {
        let _ = runner.start_turn();
        events.push(PersistEvent::StartTurn);
    }
    if let Some(game_id) = active_game_id {
        if let Err(error) = append_events_and_update_meta(&db_path, game_id, &events, runner) {
            *runner = before_runner;
            return Json(serde_json::json!({"ok": false, "error": error}));
        }
    }

    let game_state = serialize_state_for_observer(runner, observer_player);
    guard.mark_position_changed();
    Json(serde_json::json!({
        "ok": true,
        "resolved": resolved,
        "revision": guard.revision,
        "state": game_state
    }))
}

fn deterministic_shortfall_tile_order(
    session: &crate::game::framework::ShortfallResolutionSession,
) -> Vec<usize> {
    let mut tiles = session.removable_tiles.clone();
    tiles.sort_by_key(|tile| (tile.liquidation_value, tile.build_location_idx));
    tiles
        .into_iter()
        .map(|tile| tile.build_location_idx)
        .collect()
}

async fn api_start_turn(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            if runner.is_game_finished() {
                return Json(serde_json::json!({"ok": false, "error": "Game is already finished"}));
            }
            if runner.framework.current_session().is_some() {
                return Json(serde_json::json!({
                    "ok": false,
                    "error": "Finish or cancel the active action before starting a turn"
                }));
            }
            if runner.has_pending_shortfall() {
                return Json(serde_json::json!({
                    "ok": false,
                    "error": "Resolve income shortfall before starting a turn"
                }));
            }
            let before_runner = runner.clone();
            let was_started = runner.turn_started;
            let _ = runner.start_turn();
            let gs = serialize_state_for_observer(runner, observer_player);
            if !was_started {
                if let Some(game_id) = active_game_id {
                    if let Err(error) = append_event_and_update_meta(
                        &db_path,
                        game_id,
                        PersistEvent::StartTurn,
                        runner,
                    ) {
                        *runner = before_runner;
                        return Json(serde_json::json!({"ok": false, "error": error}));
                    }
                }
                guard.mark_position_changed();
            }
            Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

#[derive(Deserialize)]
struct ActionRequest {
    action_type: String,
}

async fn api_start_action(
    State(state): State<SharedState>,
    Json(req): Json<ActionRequest>,
) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            let action = match action_type_from_str(&req.action_type) {
                Some(a) => a,
                None => {
                    return Json(
                        serde_json::json!({"ok": false, "error": format!("Unknown action: {}", req.action_type)}),
                    )
                }
            };
            let before_runner = runner.clone();
            let _choice_set = match runner.try_start_action(action) {
                Ok(choice_set) => choice_set,
                Err(error) => {
                    return Json(serde_json::json!({"ok": false, "error": error}));
                }
            };
            let gs = serialize_state_for_observer(runner, observer_player);
            if let Some(game_id) = active_game_id {
                if let Err(error) = append_event_and_update_meta(
                    &db_path,
                    game_id,
                    PersistEvent::StartAction {
                        action_type: req.action_type.clone(),
                    },
                    runner,
                ) {
                    *runner = before_runner;
                    return Json(serde_json::json!({"ok": false, "error": error}));
                }
            }
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

#[derive(Deserialize)]
struct ChoiceRequest {
    choice_kind: String,
    value: serde_json::Value,
}

async fn api_apply_choice(
    State(state): State<SharedState>,
    Json(req): Json<ChoiceRequest>,
) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            let choice = match parse_choice(&req.choice_kind, &req.value) {
                Ok(c) => c,
                Err(e) => return Json(serde_json::json!({"ok": false, "error": e})),
            };
            let before_runner = runner.clone();
            if let Err(error) = runner.try_apply_choice(choice) {
                return Json(serde_json::json!({"ok": false, "error": error}));
            }
            let gs = serialize_state_for_observer(runner, observer_player);
            if let Some(game_id) = active_game_id {
                if let Err(error) = append_event_and_update_meta(
                    &db_path,
                    game_id,
                    PersistEvent::ApplyChoice {
                        choice_kind: req.choice_kind.clone(),
                        value: req.value.clone(),
                    },
                    runner,
                ) {
                    *runner = before_runner;
                    return Json(serde_json::json!({"ok": false, "error": error}));
                }
            }
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_confirm_action(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            let before_runner = runner.clone();
            let replay_position = replay_position_key(runner);
            let replay_player = runner.framework.current_player;
            let replay_action_type = runner
                .framework
                .current_session()
                .map(|session| action_type_str(session.action_type).to_string());
            let replay_before_state =
                match serde_json::to_value(serialize_state_for_observer(runner, observer_player)) {
                    Ok(state) => state,
                    Err(error) => {
                        *runner = before_runner.clone();
                        return Json(serde_json::json!({
                            "ok": false,
                            "error": format!("serialize replay position: {}", error)
                        }));
                    }
                };
            match runner.confirm_action() {
                Ok(()) => {
                    let gs = serialize_state_for_observer(runner, observer_player);
                    let selections = gs
                        .turn_action_history
                        .last()
                        .map(|action| action.selections.clone())
                        .unwrap_or_default();
                    let replay_after_state = match serde_json::to_value(&gs) {
                        Ok(state) => state,
                        Err(error) => {
                            *runner = before_runner.clone();
                            return Json(serde_json::json!({
                                "ok": false,
                                "error": format!("serialize replay result: {}", error)
                            }));
                        }
                    };
                    if let Some(game_id) = active_game_id {
                        let events = vec![
                            PersistEvent::ConfirmAction,
                            PersistEvent::ReplayMove {
                                position_key: replay_position,
                                player_idx: replay_player,
                                action_type: replay_action_type
                                    .unwrap_or_else(|| "Unknown".to_string()),
                                action_key: None,
                                selections,
                                before_state: replay_before_state,
                                after_state: replay_after_state,
                            },
                        ];
                        if let Err(error) =
                            append_events_and_update_meta(&db_path, game_id, &events, runner)
                        {
                            *runner = before_runner.clone();
                            return Json(serde_json::json!({"ok": false, "error": error}));
                        }
                    }
                    guard.mark_position_changed();
                    Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
                }
                Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
            }
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_cancel_action(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            if runner.framework.current_session().is_none() {
                return Json(serde_json::json!({
                    "ok": false,
                    "error": "No active action session to cancel"
                }));
            }
            let before_runner = runner.clone();
            runner.framework.cancel_action_session();
            let gs = serialize_state_for_observer(runner, observer_player);
            if let Some(game_id) = active_game_id {
                if let Err(error) = append_event_and_update_meta(
                    &db_path,
                    game_id,
                    PersistEvent::CancelAction,
                    runner,
                ) {
                    *runner = before_runner;
                    return Json(serde_json::json!({"ok": false, "error": error}));
                }
            }
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_undo_last_action(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            let before_runner = runner.clone();
            match runner.undo_last_confirmed_action() {
                Ok(()) => {
                    let _ = runner.start_turn();
                    let gs = serialize_state_for_observer(runner, observer_player);
                    if let Some(game_id) = active_game_id {
                        if let Err(error) = append_event_and_update_meta(
                            &db_path,
                            game_id,
                            PersistEvent::UndoLastAction,
                            runner,
                        ) {
                            *runner = before_runner;
                            return Json(serde_json::json!({"ok": false, "error": error}));
                        }
                    }
                    guard.mark_position_changed();
                    Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
                }
                Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
            }
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_end_turn(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    let observer_player = guard.observer_player;
    match guard.runner.as_mut() {
        Some(runner) => {
            let before_runner = runner.clone();
            if let Err(error) = runner.try_end_turn() {
                return Json(serde_json::json!({"ok": false, "error": error}));
            }
            let gs = serialize_state_for_observer(runner, observer_player);
            if let Some(game_id) = active_game_id {
                if let Err(error) =
                    append_event_and_update_meta(&db_path, game_id, PersistEvent::EndTurn, runner)
                {
                    *runner = before_runner;
                    return Json(serde_json::json!({"ok": false, "error": error}));
                }
            }
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "revision": guard.revision, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

fn parse_choice(kind: &str, value: &serde_json::Value) -> Result<ActionChoice, String> {
    match kind {
        "industry" | "free_development" | "second_industry" => {
            let s = value.as_str().ok_or("Expected string")?;
            let ind = match s {
                "Coal" => IndustryType::Coal,
                "Iron" => IndustryType::Iron,
                "Beer" => IndustryType::Beer,
                "Goods" => IndustryType::Goods,
                "Pottery" => IndustryType::Pottery,
                "Cotton" => IndustryType::Cotton,
                _ => return Err(format!("Unknown industry: {}", s)),
            };
            if kind == "free_development" || kind == "second_industry" {
                Ok(ActionChoice::FreeDevelopment(ind))
            } else {
                Ok(ActionChoice::Industry(ind))
            }
        }
        "card" => {
            let idx = value.as_u64().ok_or("Expected number")? as usize;
            Ok(ActionChoice::Card(idx))
        }
        "build_location" => {
            let loc = value.as_u64().ok_or("Expected number")? as usize;
            Ok(ActionChoice::BuildLocation(loc))
        }
        "road" | "second_road" => {
            let r = value.as_u64().ok_or("Expected number")? as usize;
            Ok(ActionChoice::Road(r))
        }
        "coal_source" | "iron_source" => {
            let src = parse_resource_source(value)?;
            if kind == "coal_source" {
                Ok(ActionChoice::CoalSource(src))
            } else {
                Ok(ActionChoice::IronSource(src))
            }
        }
        "beer_source" => {
            let src = parse_beer_sell_source(value)?;
            Ok(ActionChoice::BeerSource(src))
        }
        "action_beer_source" => {
            let src = parse_brewery_beer_source(value)?;
            Ok(ActionChoice::ActionBeerSource(src))
        }
        "sell_target" => {
            let loc = value.as_u64().ok_or("Expected number")? as usize;
            Ok(ActionChoice::SellTarget(loc))
        }
        "network_mode" => {
            let s = value.as_str().ok_or("Expected string")?;
            let mode = match s {
                "Single" => NetworkMode::Single,
                "Double" => NetworkMode::Double,
                _ => return Err(format!("Unknown network mode: {}", s)),
            };
            Ok(ActionChoice::NetworkMode(mode))
        }
        "confirm" => Ok(ActionChoice::Confirm),
        _ => Err(format!("Unknown choice kind: {}", kind)),
    }
}

fn persist_event_for_choice(choice: &ActionChoice) -> Result<Option<PersistEvent>, String> {
    let (choice_kind, value) = match choice {
        ActionChoice::Industry(industry) => {
            ("industry", serde_json::json!(industry_str(*industry)))
        }
        ActionChoice::Card(card_idx) => ("card", serde_json::json!(card_idx)),
        ActionChoice::BuildLocation(location) => ("build_location", serde_json::json!(location)),
        ActionChoice::Road(road_idx) => ("road", serde_json::json!(road_idx)),
        ActionChoice::SellTarget(location) => ("sell_target", serde_json::json!(location)),
        ActionChoice::CoalSource(source) => ("coal_source", resource_source_json(*source)),
        ActionChoice::IronSource(source) => ("iron_source", resource_source_json(*source)),
        ActionChoice::BeerSource(source) => ("beer_source", beer_source_json(*source)),
        ActionChoice::ActionBeerSource(source) => {
            ("action_beer_source", action_beer_source_json(*source))
        }
        ActionChoice::FreeDevelopment(industry) => (
            "free_development",
            serde_json::json!(industry_str(*industry)),
        ),
        ActionChoice::NetworkMode(mode) => (
            "network_mode",
            serde_json::json!(match mode {
                NetworkMode::Single => "Single",
                NetworkMode::Double => "Double",
            }),
        ),
        ActionChoice::Confirm => return Ok(None),
        ActionChoice::Cancel => {
            return Err("a legal recommendation cannot contain Cancel".to_string())
        }
    };
    Ok(Some(PersistEvent::ApplyChoice {
        choice_kind: choice_kind.to_string(),
        value,
    }))
}

fn resource_source_json(source: ResourceSource) -> serde_json::Value {
    match source {
        ResourceSource::Building(location) => serde_json::json!({"Building": location}),
        ResourceSource::Market => serde_json::json!("Market"),
    }
}

fn beer_source_json(source: BeerSellSource) -> serde_json::Value {
    match source {
        BeerSellSource::Building(location) => serde_json::json!({"Building": location}),
        BeerSellSource::TradePost(slot) => serde_json::json!({"TradePost": slot}),
    }
}

fn action_beer_source_json(source: BreweryBeerSource) -> serde_json::Value {
    match source {
        BreweryBeerSource::OwnBrewery(location) => {
            serde_json::json!({"OwnBrewery": location})
        }
        BreweryBeerSource::OpponentBrewery(location) => {
            serde_json::json!({"OpponentBrewery": location})
        }
    }
}

fn parse_resource_source(v: &serde_json::Value) -> Result<ResourceSource, String> {
    if v.as_str() == Some("Market") {
        return Ok(ResourceSource::Market);
    }
    if let Some(obj) = v.as_object() {
        if let Some(loc) = obj.get("Building") {
            let loc = loc.as_u64().ok_or("Expected number for Building loc")? as usize;
            return Ok(ResourceSource::Building(loc));
        }
    }
    Err("Invalid resource source".to_string())
}

fn parse_beer_sell_source(v: &serde_json::Value) -> Result<BeerSellSource, String> {
    if let Some(obj) = v.as_object() {
        if let Some(loc) = obj.get("Building") {
            return Ok(BeerSellSource::Building(
                loc.as_u64().ok_or("Expected number")? as usize,
            ));
        }
        if let Some(slot) = obj.get("TradePost") {
            return Ok(BeerSellSource::TradePost(
                slot.as_u64().ok_or("Expected number")? as usize,
            ));
        }
    }
    Err("Invalid beer source".to_string())
}

fn parse_brewery_beer_source(v: &serde_json::Value) -> Result<BreweryBeerSource, String> {
    if let Some(obj) = v.as_object() {
        if let Some(loc) = obj.get("OwnBrewery") {
            return Ok(BreweryBeerSource::OwnBrewery(
                loc.as_u64().ok_or("Expected number")? as usize,
            ));
        }
        if let Some(loc) = obj.get("OpponentBrewery") {
            return Ok(BreweryBeerSource::OpponentBrewery(
                loc.as_u64().ok_or("Expected number")? as usize,
            ));
        }
    }
    Err("Invalid brewery beer source".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::framework::{ShortfallResolutionSession, ShortfallTileChoice};

    #[test]
    fn deterministic_shortfall_order_uses_value_then_location() {
        let session = ShortfallResolutionSession {
            player_idx: 0,
            shortfall: 5,
            removable_tiles: vec![
                ShortfallTileChoice {
                    build_location_idx: 8,
                    liquidation_value: 4,
                },
                ShortfallTileChoice {
                    build_location_idx: 3,
                    liquidation_value: 2,
                },
                ShortfallTileChoice {
                    build_location_idx: 1,
                    liquidation_value: 2,
                },
            ],
        };

        assert_eq!(deterministic_shortfall_tile_order(&session), vec![1, 3, 8]);
    }

    #[test]
    fn replay_applies_persisted_shortfall_resolution() {
        let mut runner = GameRunner::new(2, Some(91_700));
        runner.framework.board.state.players[0].victory_points = 10;
        runner.framework.board.state.visible_vps[0] = 10;
        runner
            .pending_shortfall_sessions
            .push(ShortfallResolutionSession {
                player_idx: 0,
                shortfall: 4,
                removable_tiles: Vec::new(),
            });
        let events = vec![PersistEvent::ResolveShortfall {
            player_idx: 0,
            chosen_tile_order: Vec::new(),
        }];

        replay_events(&mut runner, &events).unwrap();

        assert!(runner.pending_shortfall_sessions.is_empty());
        assert_eq!(runner.framework.board.state.players[0].victory_points, 6);
        assert_eq!(runner.framework.board.state.visible_vps[0], 6);
    }

    #[test]
    fn replay_history_joins_analysis_to_the_recorded_move() {
        let position_key = "Canal:0:0:1:1:false".to_string();
        let before_state = serde_json::json!({"current_player": 1});
        let after_state = serde_json::json!({"current_player": 0});
        let analysis = serde_json::json!({
            "recommendations": [
                {"rank": 1, "action_key": "pass|c0,confirm"},
                {"rank": 2, "action_key": "loan|c1,confirm"}
            ]
        });
        let events = vec![
            PersistEvent::ReplayAnalysis {
                position_key: position_key.clone(),
                player_idx: 1,
                state: before_state.clone(),
                analysis: analysis.clone(),
            },
            PersistEvent::ReplayMove {
                position_key,
                player_idx: 1,
                action_type: "Pass".to_string(),
                action_key: Some("pass|c0,confirm".to_string()),
                selections: Vec::new(),
                before_state,
                after_state: after_state.clone(),
            },
        ];

        let (positions, final_state) = replay_history_from_events(&events);

        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].player_idx, 1);
        assert_eq!(positions[0].analysis.as_ref(), Some(&analysis));
        assert_eq!(
            positions[0]
                .move_data
                .as_ref()
                .and_then(|movement| movement.action_key.as_deref()),
            Some("pass|c0,confirm")
        );
        assert_eq!(final_state, Some(after_state));
    }
}
