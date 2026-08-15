pub mod analysis;
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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

use crate::board::resources::{BeerSellSource, BreweryBeerSource, ResourceSource};
use crate::core::types::*;
use crate::game::framework::{ActionChoice, ChoiceSet, NetworkMode};
use crate::game::runner::{GameRunner, ReplayTurnCheckpoint};
use crate::game::search::{
    search_top_actions, search_top_actions_with_policy, BatchedNeuralPuctConfig,
    BatchedNeuralPuctSearch, RootSearchConfig, RootSearchReport,
};

use analysis::{explain_analysis_question, serialize_analysis};
use model_inference::ModelInferenceClient;
use serialize::*;

pub struct ServerState {
    pub runner: Option<GameRunner>,
    pub active_game_id: Option<i64>,
    pub db_path: String,
    inference_client: Option<ModelInferenceClient>,
    revision: u64,
    last_analysis: Option<CachedAnalysis>,
}

#[derive(Clone)]
struct CachedAnalysis {
    revision: u64,
    runner: GameRunner,
    report: RootSearchReport,
}

impl ServerState {
    fn mark_position_changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.last_analysis = None;
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
        db_path,
        inference_client,
        revision: 0,
        last_analysis: None,
    }));

    let app = Router::new()
        .route("/api/new_game", post(api_new_game))
        .route("/api/games", get(api_games))
        .route("/api/load_game", post(api_load_game))
        .route("/api/state", get(api_state))
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
        .fallback(static_handler);

    let addr = format!("0.0.0.0:{}", port);
    println!("Starting Brass Birmingham at http://localhost:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
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

fn with_actions_if_available(runner: &GameRunner, gs: &mut FullGameState) {
    if runner.framework.current_session().is_none() && runner.actions_remaining_in_turn > 0 {
        let actions = runner.framework.get_valid_root_actions();
        gs.available_actions = Some(actions.iter().map(|a| action_type_str(*a)).collect());
    }
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
                runner.end_turn();
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
        }
    }
    runner.framework.replay_mode = false;
    Ok(())
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

    match Assets::get(&path) {
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
    guard.mark_position_changed();
    let runner = guard.runner.as_ref().unwrap();
    let mut gs = serialize_game_state(runner);
    with_actions_if_available(runner, &mut gs);
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
    guard.mark_position_changed();
    let runner = guard.runner.as_ref().unwrap();
    let mut gs = serialize_game_state(runner);
    with_actions_if_available(runner, &mut gs);
    Json(serde_json::json!({"ok": true, "state": gs}))
}

async fn api_state(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let guard = state.lock().await;
    match guard.runner.as_ref() {
        Some(runner) => {
            let mut gs = serialize_game_state(runner);
            with_actions_if_available(runner, &mut gs);
            Json(serde_json::json!({"ok": true, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_industry_data() -> Json<serde_json::Value> {
    Json(serialize::serialize_all_industry_data())
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    simulations: Option<u64>,
    top_n: Option<usize>,
    seed: Option<u64>,
}

const WEB_NEURAL_DETERMINIZATIONS: usize = 4;
const WEB_NEURAL_BATCH_SIZE: usize = 64;

async fn run_model_guided_search(
    client: &ModelInferenceClient,
    runner: &GameRunner,
    config: &RootSearchConfig,
) -> Result<RootSearchReport, String> {
    let policy = client.evaluate(runner).await?;
    if runner.framework.board.state.players.len() != 2 {
        let search_runner = runner.clone();
        let search_config = config.clone();
        return tokio::task::spawn_blocking(move || {
            search_top_actions_with_policy(&search_runner, &search_config, &policy)
        })
        .await
        .map_err(|error| format!("analysis task failed: {error}"))?;
    }

    let model_id = policy.model_id.clone();
    let mut search = BatchedNeuralPuctSearch::new(
        runner,
        BatchedNeuralPuctConfig {
            search: config.clone(),
            determinizations: WEB_NEURAL_DETERMINIZATIONS,
        },
        policy,
    )?;
    while !search.is_complete() {
        let leaves = search.next_inference_batch(WEB_NEURAL_BATCH_SIZE)?;
        if leaves.is_empty() {
            if search.is_complete() {
                break;
            }
            return Err("neural search returned an empty incomplete leaf batch".to_string());
        }
        let evaluations = client.evaluate_leaf_batch(&leaves, &model_id).await?;
        search.submit_inference_batch(evaluations)?;
    }
    search.finish_report()
}

async fn api_analyze(
    State(state): State<SharedState>,
    Json(req): Json<AnalyzeRequest>,
) -> Json<serde_json::Value> {
    let simulations = req.simulations.unwrap_or(2_000);
    if !(1..=50_000).contains(&simulations) {
        return Json(serde_json::json!({
            "ok": false,
            "error": "simulations must be between 1 and 50000"
        }));
    }
    let top_n = req.top_n.unwrap_or(3);
    if !(1..=10).contains(&top_n) {
        return Json(serde_json::json!({
            "ok": false,
            "error": "top_n must be between 1 and 10"
        }));
    }

    let (runner, revision, inference_client) = {
        let guard = state.lock().await;
        let Some(runner) = guard.runner.as_ref() else {
            return Json(serde_json::json!({"ok": false, "error": "No game in progress"}));
        };
        (
            runner.clone(),
            guard.revision,
            guard.inference_client.clone(),
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
    let report_result = match inference_client {
        Some(client) => run_model_guided_search(&client, &runner, &config).await,
        None => {
            let search_runner = runner.clone();
            tokio::task::spawn_blocking(move || search_top_actions(&search_runner, &config))
                .await
                .map_err(|error| format!("analysis task failed: {error}"))
                .and_then(|report| report)
        }
    };
    let report = match report_result {
        Ok(report) => report,
        Err(error) => {
            return Json(serde_json::json!({"ok": false, "error": error}));
        }
    };
    let elapsed_ms = started.elapsed().as_millis();
    let analysis = serialize_analysis(&report, &runner, revision, elapsed_ms);

    let mut guard = state.lock().await;
    if guard.revision != revision {
        return Json(serde_json::json!({
            "ok": false,
            "error": "The position changed while analysis was running. Analyze the current position again."
        }));
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
    let cached = {
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
        cached.clone()
    };

    match explain_analysis_question(
        &cached.report,
        &cached.runner,
        cached.revision,
        &req.action_key,
        question,
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
    if let Some(game_id) = active_game_id {
        if let Err(error) = append_events_and_update_meta(&db_path, game_id, &events, runner) {
            *runner = before_runner;
            return Json(serde_json::json!({"ok": false, "error": error}));
        }
    }

    let mut game_state = serialize_game_state(runner);
    with_actions_if_available(runner, &mut game_state);
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
    let Some(runner) = guard.runner.as_mut() else {
        return Json(serde_json::json!({"ok": false, "error": "No game in progress"}));
    };
    if !runner.has_pending_shortfall() {
        let mut game_state = serialize_game_state(runner);
        with_actions_if_available(runner, &mut game_state);
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

    let mut game_state = serialize_game_state(runner);
    with_actions_if_available(runner, &mut game_state);
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
    match guard.runner.as_mut() {
        Some(runner) => {
            let actions = runner.start_turn();
            let action_strs: Vec<&str> = actions.iter().map(|a| action_type_str(*a)).collect();
            let mut gs = serialize_game_state(runner);
            gs.available_actions = Some(action_strs);
            if let Some(game_id) = active_game_id {
                let _ = append_event_and_update_meta(
                    &db_path,
                    game_id,
                    PersistEvent::StartTurn,
                    runner,
                );
            }
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "state": gs}))
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
            let _choice_set = runner.start_action(action);
            let mut gs = serialize_game_state(runner);
            if let Some(game_id) = active_game_id {
                let _ = append_event_and_update_meta(
                    &db_path,
                    game_id,
                    PersistEvent::StartAction {
                        action_type: req.action_type.clone(),
                    },
                    runner,
                );
            }
            with_actions_if_available(runner, &mut gs);
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "state": gs}))
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
    match guard.runner.as_mut() {
        Some(runner) => {
            let choice = match parse_choice(&req.choice_kind, &req.value) {
                Ok(c) => c,
                Err(e) => return Json(serde_json::json!({"ok": false, "error": e})),
            };
            match runner.apply_choice(choice) {
                _ => {
                    let mut gs = serialize_game_state(runner);
                    if let Some(game_id) = active_game_id {
                        let _ = append_event_and_update_meta(
                            &db_path,
                            game_id,
                            PersistEvent::ApplyChoice {
                                choice_kind: req.choice_kind.clone(),
                                value: req.value.clone(),
                            },
                            runner,
                        );
                    }
                    with_actions_if_available(runner, &mut gs);
                    guard.mark_position_changed();
                    Json(serde_json::json!({"ok": true, "state": gs}))
                }
            }
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_confirm_action(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    match guard.runner.as_mut() {
        Some(runner) => match runner.confirm_action() {
            Ok(()) => {
                let mut gs = serialize_game_state(runner);
                if let Some(game_id) = active_game_id {
                    let _ = append_event_and_update_meta(
                        &db_path,
                        game_id,
                        PersistEvent::ConfirmAction,
                        runner,
                    );
                }
                with_actions_if_available(runner, &mut gs);
                guard.mark_position_changed();
                Json(serde_json::json!({"ok": true, "state": gs}))
            }
            Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
        },
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_cancel_action(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    match guard.runner.as_mut() {
        Some(runner) => {
            runner.framework.cancel_action_session();
            let mut gs = serialize_game_state(runner);
            if let Some(game_id) = active_game_id {
                let _ = append_event_and_update_meta(
                    &db_path,
                    game_id,
                    PersistEvent::CancelAction,
                    runner,
                );
            }
            with_actions_if_available(runner, &mut gs);
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "state": gs}))
        }
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_undo_last_action(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    match guard.runner.as_mut() {
        Some(runner) => match runner.undo_last_confirmed_action() {
            Ok(()) => {
                let actions = runner.start_turn();
                let action_strs: Vec<&str> = actions.iter().map(|a| action_type_str(*a)).collect();
                let mut gs = serialize_game_state(runner);
                gs.available_actions = Some(action_strs);
                if let Some(game_id) = active_game_id {
                    let _ = append_event_and_update_meta(
                        &db_path,
                        game_id,
                        PersistEvent::UndoLastAction,
                        runner,
                    );
                }
                guard.mark_position_changed();
                Json(serde_json::json!({"ok": true, "state": gs}))
            }
            Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
        },
        None => Json(serde_json::json!({"ok": false, "error": "No game in progress"})),
    }
}

async fn api_end_turn(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut guard = state.lock().await;
    let db_path = guard.db_path.clone();
    let active_game_id = guard.active_game_id;
    match guard.runner.as_mut() {
        Some(runner) => {
            runner.end_turn();
            let mut gs = serialize_game_state(runner);
            if let Some(game_id) = active_game_id {
                let _ =
                    append_event_and_update_meta(&db_path, game_id, PersistEvent::EndTurn, runner);
            }
            with_actions_if_available(runner, &mut gs);
            guard.mark_position_changed();
            Json(serde_json::json!({"ok": true, "state": gs}))
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
}
