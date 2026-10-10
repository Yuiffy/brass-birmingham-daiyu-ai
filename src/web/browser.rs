//! A browser owns its saved action logs. Each cloud request reconstructs an
//! isolated runner, so scale-to-zero and concurrent visitors cannot mix games.
use super::*;

#[derive(Default, Serialize, Deserialize)]
pub(super) struct Session {
    #[serde(default)]
    games: Vec<SavedGame>,
    active_game_id: Option<i64>,
    observer_player: Option<usize>,
    #[serde(default)]
    revision: u64,
    analysis_request: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
struct SavedGame {
    id: i64,
    created_at: i64,
    updated_at: i64,
    num_players: usize,
    // Keep all 64 seed bits when JSON passes through JavaScript.
    seed: String,
    action_log: String,
    round_in_phase: u32,
    era: String,
}

#[derive(Deserialize)]
pub(super) struct Request {
    endpoint: String,
    #[serde(default)]
    body: serde_json::Value,
    #[serde(default)]
    session: Session,
}

struct TemporaryDatabase(PathBuf);
impl Drop for TemporaryDatabase {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(super) async fn request(Json(req): Json<Request>) -> Json<serde_json::Value> {
    match tokio::spawn(execute(req)).await {
        Ok(Ok(value)) => Json(value),
        Ok(Err(error)) => Json(serde_json::json!({"ok": false, "error": error})),
        Err(_) => Json(serde_json::json!({"ok": false, "error": "Invalid saved game request"})),
    }
}

async fn execute(req: Request) -> Result<serde_json::Value, String> {
    if req.session.games.len() > 100 {
        return Err("Too many saved games in this browser".into());
    }
    if req.endpoint == "new_game" {
        let parsed: NewGameRequest =
            serde_json::from_value(req.body.clone()).map_err(|error| error.to_string())?;
        if !(2..=4).contains(&parsed.num_players) {
            return Err("Choose between 2 and 4 players".into());
        }
    }
    let temporary = TemporaryDatabase(std::env::temp_dir().join(format!(
        "brass-browser-{:016x}.sqlite3",
        rand::random::<u64>()
    )));
    let db_path = temporary.0.to_string_lossy().to_string();
    init_db(&db_path)?;
    {
        let mut conn = Connection::open(&db_path).map_err(|error| error.to_string())?;
        let transaction = conn.transaction().map_err(|error| error.to_string())?;
        for game in &req.session.games {
            if !(2..=4).contains(&game.num_players) || game.id <= 0 {
                return Err("Invalid saved game".into());
            }
            let seed = game
                .seed
                .parse::<u64>()
                .map_err(|error| error.to_string())?;
            let events: Vec<PersistEvent> =
                serde_json::from_str(&game.action_log).map_err(|error| error.to_string())?;
            if events.len() > 10_000 {
                return Err("Saved game action log is too large".into());
            }
            transaction.execute(
                "INSERT INTO games (id, created_at, updated_at, num_players, seed, action_log, round_in_phase, era)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![game.id, game.created_at, game.updated_at, game.num_players as i64,
                    seed as i64, game.action_log, game.round_in_phase as i64, game.era],
            ).map_err(|error| error.to_string())?;
        }
        transaction.commit().map_err(|error| error.to_string())?;
    }
    let state = Arc::new(Mutex::new(ServerState {
        runner: None,
        active_game_id: None,
        observer_player: None,
        db_path: db_path.clone(),
        inference_client: None,
        revision: 0,
        last_analysis: None,
        analysis_session: None,
    }));
    if let Some(game_id) = req.session.active_game_id {
        let restored = api_load_game(State(state.clone()), Json(LoadGameRequest { game_id }))
            .await
            .0;
        if restored["ok"] != true {
            return Err(restored["error"]
                .as_str()
                .unwrap_or("Could not restore game")
                .into());
        }
    }
    {
        let mut guard = state.lock().await;
        if let Some(observer) = req.session.observer_player {
            if guard
                .runner
                .as_ref()
                .is_none_or(|runner| observer >= runner.framework.board.state.players.len())
            {
                return Err("Invalid observer seat".into());
            }
        }
        guard.observer_player = req.session.observer_player;
        guard.revision = req.session.revision;
    }
    // Analysis caches are process-local. Recompute the same requested report
    // only for operations that need it, retaining the revision stale check.
    if matches!(req.endpoint.as_str(), "apply_analyzed_action" | "explain") {
        if let Some(body) = &req.session.analysis_request {
            let mut analysis_request: AnalyzeRequest =
                serde_json::from_value(body.clone()).map_err(|error| error.to_string())?;
            // Restoring a process-local cache must not duplicate saved replay frames.
            analysis_request.skip_replay_recording = true;
            let analysis = api_analyze(State(state.clone()), Json(analysis_request))
                .await
                .0;
            if analysis["ok"] != true {
                return Ok(analysis);
            }
        }
    }
    macro_rules! with_body {
        ($handler:ident) => {{
            let body =
                serde_json::from_value(req.body.clone()).map_err(|error| error.to_string())?;
            $handler(State(state.clone()), Json(body)).await.0
        }};
    }
    macro_rules! without_body {
        ($handler:ident) => {
            $handler(State(state.clone())).await.0
        };
    }
    let mut response = match req.endpoint.as_str() {
        "new_game" => with_body!(api_new_game),
        "games" => without_body!(api_games),
        "load_game" => with_body!(api_load_game),
        "replay" => with_body!(api_replay),
        "state" => without_body!(api_state),
        "set_observer" => with_body!(api_set_observer),
        "analyze" => with_body!(api_analyze),
        "explain" => with_body!(api_explain),
        "apply_analyzed_action" => with_body!(api_apply_analyzed_action),
        "resolve_shortfalls" => without_body!(api_resolve_shortfalls),
        "start_turn" => without_body!(api_start_turn),
        "start_action" => with_body!(api_start_action),
        "apply_choice" => with_body!(api_apply_choice),
        "confirm_action" => without_body!(api_confirm_action),
        "cancel_action" => without_body!(api_cancel_action),
        "undo_last_action" => without_body!(api_undo_last_action),
        "end_turn" => without_body!(api_end_turn),
        _ => return Err("Unknown game endpoint".into()),
    };
    if response["ok"] != true {
        return Ok(response);
    }
    let games = {
        let conn = Connection::open(&db_path).map_err(|error| error.to_string())?;
        let mut statement = conn.prepare(
        "SELECT id, created_at, updated_at, num_players, seed, action_log, round_in_phase, era FROM games ORDER BY id"
    ).map_err(|error| error.to_string())?;
        let games = statement
            .query_map([], |row| {
                Ok(SavedGame {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    updated_at: row.get(2)?,
                    num_players: row.get::<_, i64>(3)? as usize,
                    seed: (row.get::<_, i64>(4)? as u64).to_string(),
                    action_log: row.get(5)?,
                    round_in_phase: row.get::<_, i64>(6)? as u32,
                    era: row.get(7)?,
                })
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        games
    };
    let guard = state.lock().await;
    let analysis_request = if req.endpoint == "analyze" {
        Some(req.body)
    } else if guard.revision == req.session.revision {
        req.session.analysis_request
    } else {
        None
    };
    response["browser_session"] = serde_json::to_value(Session {
        games,
        active_game_id: guard.active_game_id,
        observer_player: guard.observer_player,
        revision: guard.revision,
        analysis_request,
    })
    .map_err(|error| error.to_string())?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    async fn call(session: &mut Value, endpoint: &str, body: Value) -> Value {
        let response = request(Json(
            serde_json::from_value(json!({
                "session": session, "endpoint": endpoint, "body": body
            }))
            .unwrap(),
        ))
        .await
        .0;
        assert_eq!(response["ok"], true, "{endpoint}: {response}");
        *session = response["browser_session"].clone();
        response
    }

    #[tokio::test]
    async fn trained_recommendations_remain_identical_after_restoring_midgame() {
        let original: Value =
            serde_json::from_str(include_str!("fixtures/trained-session.json")).unwrap();
        let mut expected = None;
        for _ in 0..20 {
            let mut session = original.clone();
            let result = call(
                &mut session,
                "analyze",
                json!({"simulations": 1, "top_n": 3}),
            )
            .await;
            let recommendations = result["analysis"]["recommendations"].clone();
            if let Some(expected) = &expected {
                assert_eq!(&recommendations, expected);
            } else {
                expected = Some(recommendations);
            }
            let analysis_frames = |session: &Value| {
                let events: Vec<Value> =
                    serde_json::from_str(session["games"][0]["action_log"].as_str().unwrap())
                        .unwrap();
                events
                    .iter()
                    .filter(|e| e["kind"] == "replay_analysis")
                    .count()
            };
            let frames_before = analysis_frames(&session);
            call(
                &mut session,
                "apply_analyzed_action",
                json!({
                    "revision": result["analysis"]["revision"],
                    "action_key": result["analysis"]["recommendations"][0]["action_key"]
                }),
            )
            .await;
            assert_eq!(
                analysis_frames(&session),
                frames_before,
                "cache restoration duplicated replay analysis"
            );
        }
    }

    #[tokio::test]
    async fn restores_pending_choices_confirmed_actions_and_undo() {
        let mut session = json!({});
        call(
            &mut session,
            "new_game",
            json!({"num_players": 2, "seed": 42}),
        )
        .await;
        let opening = call(&mut session, "start_turn", Value::Null).await;
        let player = opening["state"]["current_player"].as_u64().unwrap() as usize;
        let money = opening["state"]["players"][player]["money"]
            .as_i64()
            .unwrap();
        call(&mut session, "start_action", json!({"action_type": "Loan"})).await;
        let pending = call(&mut session, "state", Value::Null).await;
        let card = pending["state"]["choice_set"]["options"][0]["value"].clone();
        call(
            &mut session,
            "apply_choice",
            json!({"choice_kind": "card", "value": card}),
        )
        .await;
        let confirmed = call(&mut session, "confirm_action", Value::Null).await;
        assert_eq!(confirmed["state"]["players"][player]["money"], money + 30);
        let restored = call(&mut session, "state", Value::Null).await;
        assert_eq!(confirmed["state"], restored["state"]);
        let undone = call(&mut session, "undo_last_action", Value::Null).await;
        assert_eq!(undone["state"]["players"][player]["money"], money);
    }

    #[tokio::test]
    async fn analysis_survives_request_boundaries_and_rejects_stale_revision() {
        let mut session = json!({});
        call(
            &mut session,
            "new_game",
            json!({"num_players": 2, "seed": 1234}),
        )
        .await;
        call(&mut session, "start_turn", Value::Null).await;
        let analysis = call(&mut session, "analyze", json!({"simulations": 1})).await;
        assert_eq!(
            analysis["analysis"]["method"],
            crate::game::trained_ai::METHOD
        );
        assert_eq!(
            analysis["analysis"]["model_id"],
            crate::game::trained_ai::MODEL_ID
        );
        let explanation = call(
            &mut session,
            "explain",
            json!({
                "revision": analysis["analysis"]["revision"],
                "action_key": analysis["analysis"]["recommendations"][0]["action_key"],
                "question": "这个评分可靠吗？"
            }),
        )
        .await;
        assert!(explanation["explanation"]["caveat"]
            .as_str()
            .unwrap()
            .contains("不是胜率"));
        let body = json!({"revision": analysis["analysis"]["revision"],
            "action_key": analysis["analysis"]["recommendations"][0]["action_key"]});
        call(&mut session, "apply_analyzed_action", body.clone()).await;
        let rejected = request(Json(
            serde_json::from_value(json!({
                "session": session, "endpoint": "apply_analyzed_action", "body": body
            }))
            .unwrap(),
        ))
        .await
        .0;
        assert_eq!(rejected["ok"], false);
        assert!(rejected["error"]
            .as_str()
            .unwrap()
            .contains("position changed"));
    }

    #[tokio::test]
    async fn concurrent_browsers_and_reload_keep_independent_saved_games() {
        let mut first = json!({});
        let mut second = json!({});
        let (a, b) = tokio::join!(
            call(&mut first, "new_game", json!({"num_players": 2, "seed": 1})),
            call(
                &mut second,
                "new_game",
                json!({"num_players": 4, "seed": u64::MAX})
            )
        );
        assert_eq!(a["state"]["players"].as_array().unwrap().len(), 2);
        assert_eq!(b["state"]["players"].as_array().unwrap().len(), 4);
        assert_eq!(second["games"][0]["seed"], u64::MAX.to_string());
        call(&mut first, "new_game", json!({"num_players": 3, "seed": 2})).await;
        let games = call(&mut first, "games", Value::Null).await;
        assert_eq!(games["games"].as_array().unwrap().len(), 2);
        let resumed = call(&mut first, "load_game", json!({"game_id": 1})).await;
        assert_eq!(resumed["state"], a["state"]);
        let untouched = call(&mut second, "state", Value::Null).await;
        assert_eq!(untouched["state"], b["state"]);
    }

    #[tokio::test]
    async fn restores_through_income_card_refill_and_both_eras() {
        let mut session = json!({});
        let mut response = call(
            &mut session,
            "new_game",
            json!({"num_players": 2, "seed": 1234}),
        )
        .await;
        let mut saw_rail = false;
        for _ in 0..250 {
            if response["state"]["game_over"] == true {
                assert!(saw_rail);
                return;
            }
            response = call(&mut session, "start_turn", Value::Null).await;
            saw_rail |= response["state"]["era"] == "Railroad";
            while response["state"]["actions_remaining"].as_u64().unwrap() > 0 {
                let choice =
                    call(&mut session, "start_action", json!({"action_type": "Pass"})).await;
                let card = choice["state"]["choice_set"]["options"][0]["value"].clone();
                call(
                    &mut session,
                    "apply_choice",
                    json!({"choice_kind": "card", "value": card}),
                )
                .await;
                response = call(&mut session, "confirm_action", Value::Null).await;
                let restored = call(&mut session, "state", Value::Null).await;
                assert_eq!(response["state"], restored["state"]);
            }
            response = call(&mut session, "end_turn", Value::Null).await;
        }
        panic!("game did not finish within the expected turn count");
    }
}
