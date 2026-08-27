use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::actions::{BuildOption, SellOption, SingleRailroadOption};
use crate::board::resources::{BeerSellSource, BreweryBeerSource, ResourceSource};
use crate::consts::{
    MAX_MARKET_COAL, MAX_MARKET_IRON, NUM_TRADE_POSTS, N_BL, N_LOCATIONS, N_PLAYERS,
    N_ROAD_LOCATIONS, TWO_RAILROAD_PRICE,
};
use crate::core::locations::LocationName;
use crate::core::static_data::{BUILD_LOCATION_MASK, INDUSTRY_MAT, LINK_LOCATIONS};
use crate::core::types::{ActionType, BitSetWrapper, Era, IndustryType};
use crate::game::framework::{ActionChoice, ChoiceSet, NetworkMode, ShortfallResolutionSession};
use crate::game::hidden_information::determinize_hidden_information;
use crate::game::legal_actions::{enumerate_legal_actions, LegalAction};
use crate::game::runner::{GamePhase, GameRunner};
use crate::game::search::{
    official_winners, search_top_actions, search_top_actions_with_policy,
    search_top_actions_with_policy_and_action_values, BatchedNeuralPuctConfig,
    BatchedNeuralPuctSearch, NeuralLeafEvaluation, RootActionValueEvaluation, RootPolicyEvaluation,
    RootSearchConfig, RootSearchReport,
};
use crate::game::training::{
    card_type_index, encode_root_action_successor_batch, encode_training_action,
    encode_training_state, training_feature_schema, ACTION_FEATURE_DIM, TRAINING_FEATURE_VERSION,
};
use crate::market::merchants::MerchantTileType;

const ROOT_ACTION_COUNT: usize = 8;
const CARD_TYPE_DIM: usize = 29;
const MAX_HAND_MASK_DIM: usize = 64;
const NETWORK_MODE_DIM: usize = 2;
const COAL_SOURCE_DIM: usize = N_BL + 2; // build locations + market + stop token
const IRON_SOURCE_DIM: usize = N_BL + 2; // build locations + market + stop token
const BEER_SOURCE_DIM: usize = N_BL + (NUM_TRADE_POSTS * 2) + 1; // breweries + merchant slots + stop
const ACTION_BEER_SOURCE_DIM: usize = (N_BL * 2) + 1; // own + opponent breweries + stop
const SELL_TARGET_DIM: usize = N_BL + 1; // build locations + stop
const SHORTFALL_TILE_DIM: usize = N_BL + 1; // build locations + stop

const ROOT_BUILD_BUILDING: usize = 0;
const ROOT_BUILD_RAILROAD: usize = 1;
const ROOT_DEVELOP: usize = 2;
const ROOT_DEVELOP_DOUBLE: usize = 3;
const ROOT_SELL: usize = 4;
const ROOT_LOAN: usize = 5;
const ROOT_SCOUT: usize = 6;
const ROOT_PASS: usize = 7;

const STOP_INDEX_COAL: usize = COAL_SOURCE_DIM - 1;
const STOP_INDEX_IRON: usize = IRON_SOURCE_DIM - 1;
const STOP_INDEX_BEER: usize = BEER_SOURCE_DIM - 1;
const STOP_INDEX_ACTION_BEER: usize = ACTION_BEER_SOURCE_DIM - 1;
const STOP_INDEX_SELL_TARGET: usize = SELL_TARGET_DIM - 1;
const STOP_INDEX_SHORTFALL: usize = SHORTFALL_TILE_DIM - 1;

#[derive(Clone, Default)]
struct CompositeActionPayload {
    root_action: Option<usize>,
    card_values: Vec<usize>,
    industry_values: Vec<usize>,
    second_industry_values: Vec<usize>,
    build_location_values: Vec<usize>,
    network_mode_values: Vec<usize>,
    road_values: Vec<usize>,
    second_road_values: Vec<usize>,
    coal_source_values: Vec<usize>,
    iron_source_values: Vec<usize>,
    beer_source_values: Vec<usize>,
    action_beer_source_values: Vec<usize>,
    sell_target_values: Vec<usize>,
    shortfall_tile_order: Vec<usize>,

    card_idx: usize,
    industry_idx: usize,
    second_industry_idx: usize,
    build_location_idx: usize,
    network_mode_idx: usize,
    road_idx: usize,
    second_road_idx: usize,
    coal_source_idx: usize,
    iron_source_idx: usize,
    beer_source_idx: usize,
    action_beer_source_idx: usize,
    sell_target_idx: usize,
    used_card_choices: HashSet<usize>,
}

impl CompositeActionPayload {
    fn from_py(action: &Bound<'_, PyDict>) -> PyResult<Self> {
        let root_action = extract_optional_usize(action, "root_action")?;
        Ok(Self {
            root_action,
            card_values: extract_usize_list(action, "card")?,
            industry_values: extract_usize_list(action, "industry")?,
            second_industry_values: extract_usize_list(action, "second_industry")?,
            build_location_values: extract_usize_list(action, "build_location")?,
            network_mode_values: extract_usize_list(action, "network_mode")?,
            road_values: extract_usize_list(action, "road")?,
            second_road_values: extract_usize_list(action, "second_road")?,
            coal_source_values: extract_usize_list(action, "coal_sources")?,
            iron_source_values: extract_usize_list(action, "iron_sources")?,
            beer_source_values: extract_usize_list(action, "beer_sources")?,
            action_beer_source_values: extract_usize_list(action, "action_beer_source")?,
            sell_target_values: extract_usize_list(action, "sell_targets")?,
            shortfall_tile_order: extract_usize_list(action, "shortfall_tile_order")?,
            card_idx: 0,
            industry_idx: 0,
            second_industry_idx: 0,
            build_location_idx: 0,
            network_mode_idx: 0,
            road_idx: 0,
            second_road_idx: 0,
            coal_source_idx: 0,
            iron_source_idx: 0,
            beer_source_idx: 0,
            action_beer_source_idx: 0,
            sell_target_idx: 0,
            used_card_choices: HashSet::new(),
        })
    }

    fn next_card(&mut self) -> Option<usize> {
        next_from_list(&self.card_values, &mut self.card_idx)
    }

    fn next_industry(&mut self) -> Option<usize> {
        next_from_list(&self.industry_values, &mut self.industry_idx)
    }

    fn next_second_industry(&mut self) -> Option<usize> {
        next_from_list(&self.second_industry_values, &mut self.second_industry_idx)
    }

    fn next_build_location(&mut self) -> Option<usize> {
        next_from_list(&self.build_location_values, &mut self.build_location_idx)
    }

    fn next_network_mode(&mut self) -> Option<usize> {
        next_from_list(&self.network_mode_values, &mut self.network_mode_idx)
    }

    fn next_road(&mut self) -> Option<usize> {
        next_from_list(&self.road_values, &mut self.road_idx)
    }

    fn next_second_road(&mut self) -> Option<usize> {
        next_from_list(&self.second_road_values, &mut self.second_road_idx)
    }

    fn next_coal_source(&mut self) -> Option<usize> {
        next_from_list(&self.coal_source_values, &mut self.coal_source_idx)
    }

    fn next_iron_source(&mut self) -> Option<usize> {
        next_from_list(&self.iron_source_values, &mut self.iron_source_idx)
    }

    fn next_beer_source(&mut self) -> Option<usize> {
        next_from_list(&self.beer_source_values, &mut self.beer_source_idx)
    }

    fn next_action_beer_source(&mut self) -> Option<usize> {
        next_from_list(
            &self.action_beer_source_values,
            &mut self.action_beer_source_idx,
        )
    }

    fn next_sell_target(&mut self) -> Option<usize> {
        next_from_list(&self.sell_target_values, &mut self.sell_target_idx)
    }

    fn peek_sell_target(&self) -> Option<usize> {
        self.sell_target_values.get(self.sell_target_idx).copied()
    }

    fn select_unique_card_from_options(&mut self, options: &[usize]) -> PyResult<usize> {
        let requested_idx = self
            .next_card()
            .ok_or_else(|| PyValueError::new_err("missing required card choice"))?;
        if !options.contains(&requested_idx) {
            return Err(PyValueError::new_err(format!(
                "card choice {requested_idx} is not legal; expected one of {options:?}"
            )));
        }
        if !self.used_card_choices.insert(requested_idx) {
            return Err(PyValueError::new_err(format!(
                "card choice {requested_idx} was used more than once"
            )));
        }
        Ok(requested_idx)
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct BrassSavedState {
    runner: GameRunner,
    current_shortfall: Option<ShortfallResolutionSession>,
}

#[pyclass(unsendable)]
pub struct BrassRLGame {
    runner: GameRunner,
    current_shortfall: Option<ShortfallResolutionSession>,
    legal_action_cache: Vec<LegalAction>,
}

#[pyclass(unsendable)]
pub struct BrassNeuralSearch {
    search: BatchedNeuralPuctSearch,
    legal_actions: Vec<LegalAction>,
    forced_advances: usize,
}

#[pymethods]
impl BrassNeuralSearch {
    fn is_complete(&self) -> bool {
        self.search.is_complete()
    }

    fn completed_simulations(&self) -> u64 {
        self.search.completed_simulations()
    }

    fn pending_evaluations(&self) -> usize {
        self.search.pending_evaluations()
    }

    #[pyo3(signature = (max_batch_size=32))]
    fn next_inference_batch(
        &mut self,
        py: Python<'_>,
        max_batch_size: usize,
    ) -> PyResult<PyObject> {
        let requests = self
            .search
            .next_inference_batch(max_batch_size)
            .map_err(PyValueError::new_err)?;
        let positions = PyList::empty(py);
        for request in requests {
            let position = PyDict::new(py);
            position.set_item("request_id", request.request_id)?;
            position.set_item("depth", request.depth)?;
            position.set_item("evaluation_player", request.evaluation_player)?;

            let state = PyDict::new(py);
            state.set_item("feature_version", request.feature_version)?;
            state.set_item("observer_idx", request.evaluation_player)?;
            state.set_item("features", request.state_features)?;
            position.set_item("state", state)?;

            let actions = PyList::empty(py);
            for (index, (key, feature_indices)) in request
                .action_keys
                .into_iter()
                .zip(request.action_feature_indices)
                .enumerate()
            {
                let action = PyDict::new(py);
                action.set_item("index", index)?;
                action.set_item("key", key)?;
                action.set_item("feature_indices", feature_indices)?;
                actions.append(action)?;
            }
            let legal_actions = PyDict::new(py);
            legal_actions.set_item("feature_version", request.feature_version)?;
            legal_actions.set_item("actions", actions)?;
            position.set_item("legal_actions", legal_actions)?;
            positions.append(position)?;
        }

        let out = PyDict::new(py);
        out.set_item("feature_version", TRAINING_FEATURE_VERSION)?;
        out.set_item("positions", positions)?;
        out.set_item("completed_simulations", self.search.completed_simulations())?;
        out.set_item("is_complete", self.search.is_complete())?;
        Ok(out.into())
    }

    #[pyo3(signature = (
        request_ids,
        action_keys,
        policy_probabilities,
        shared_win_rates,
        victory_point_margins,
        model_id,
        actor_victory_points=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit_inference_batch(
        &mut self,
        request_ids: Vec<u64>,
        action_keys: Vec<Vec<String>>,
        policy_probabilities: Vec<Vec<f64>>,
        shared_win_rates: Vec<f64>,
        victory_point_margins: Vec<f64>,
        model_id: String,
        actor_victory_points: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let batch_size = request_ids.len();
        let actor_victory_points = actor_victory_points.unwrap_or_else(|| vec![0.0; batch_size]);
        if action_keys.len() != batch_size
            || policy_probabilities.len() != batch_size
            || shared_win_rates.len() != batch_size
            || victory_point_margins.len() != batch_size
            || actor_victory_points.len() != batch_size
        {
            return Err(PyValueError::new_err(
                "neural inference submission arrays must have equal lengths",
            ));
        }
        let evaluations = (0..batch_size)
            .map(|index| NeuralLeafEvaluation {
                request_id: request_ids[index],
                model_id: model_id.clone(),
                action_keys: action_keys[index].clone(),
                policy_probabilities: policy_probabilities[index].clone(),
                shared_win_rate: shared_win_rates[index],
                victory_point_margin: victory_point_margins[index],
                actor_victory_points: actor_victory_points[index],
            })
            .collect();
        self.search
            .submit_inference_batch(evaluations)
            .map_err(PyValueError::new_err)
    }

    fn finish(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = self.search.finish_report().map_err(PyValueError::new_err)?;
        search_report_to_py(py, report, &self.legal_actions, self.forced_advances)
    }
}

#[pymethods]
impl BrassRLGame {
    #[new]
    #[pyo3(signature = (num_players=4, seed=None))]
    fn new(num_players: usize, seed: Option<u64>) -> PyResult<Self> {
        if !(2..=4).contains(&num_players) {
            return Err(PyValueError::new_err(
                "num_players must be in [2, 4] for Brass Birmingham",
            ));
        }
        let mut game = Self {
            runner: GameRunner::new(num_players, seed),
            current_shortfall: None,
            legal_action_cache: Vec::new(),
        };
        let _ = game.advance_to_next_decision()?;
        Ok(game)
    }

    #[pyo3(signature = (seed=None, num_players=None))]
    fn reset(&mut self, seed: Option<u64>, num_players: Option<usize>) -> PyResult<()> {
        let next_players = num_players.unwrap_or(self.runner.framework.board.state.players.len());
        if !(2..=4).contains(&next_players) {
            return Err(PyValueError::new_err(
                "num_players must be in [2, 4] for Brass Birmingham",
            ));
        }
        self.runner = GameRunner::new(next_players, seed);
        self.current_shortfall = None;
        self.legal_action_cache.clear();
        let _ = self.advance_to_next_decision()?;
        Ok(())
    }

    fn current_decision_player(&self) -> usize {
        self.current_shortfall
            .as_ref()
            .map(|s| s.player_idx)
            .unwrap_or(self.runner.framework.current_player)
    }

    fn current_decision_mode(&self) -> String {
        if self.current_shortfall.is_some() {
            "shortfall".to_string()
        } else {
            "turn".to_string()
        }
    }

    fn num_players(&self) -> usize {
        self.runner.framework.board.state.players.len()
    }

    fn is_done(&self) -> bool {
        self.runner.is_game_finished()
    }

    fn get_observation(&mut self, py: Python<'_>, observer_idx: usize) -> PyResult<PyObject> {
        self.validate_player_index(observer_idx)?;
        let out = PyDict::new(py);
        self.write_observation(py, observer_idx, &out)?;
        Ok(out.into())
    }

    fn get_action_masks(&mut self, py: Python<'_>, observer_idx: usize) -> PyResult<PyObject> {
        self.validate_player_index(observer_idx)?;
        let out = PyDict::new(py);
        self.write_action_masks(py, observer_idx, &out)?;
        Ok(out.into())
    }

    fn get_spaces(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("root_action_dim", ROOT_ACTION_COUNT)?;
        out.set_item("card_dim", MAX_HAND_MASK_DIM)?;
        out.set_item("industry_dim", 6)?;
        out.set_item("build_location_dim", N_BL)?;
        out.set_item("road_dim", N_ROAD_LOCATIONS)?;
        out.set_item("network_mode_dim", NETWORK_MODE_DIM)?;
        out.set_item("coal_source_dim", COAL_SOURCE_DIM)?;
        out.set_item("iron_source_dim", IRON_SOURCE_DIM)?;
        out.set_item("beer_source_dim", BEER_SOURCE_DIM)?;
        out.set_item("action_beer_source_dim", ACTION_BEER_SOURCE_DIM)?;
        out.set_item("sell_target_dim", SELL_TARGET_DIM)?;
        out.set_item("shortfall_tile_dim", SHORTFALL_TILE_DIM)?;
        Ok(out.into())
    }

    fn get_training_feature_schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        feature_schema_to_py(py)
    }

    fn get_training_state(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "training turn features are unavailable during shortfall resolution",
            ));
        }
        let observer_idx = self.current_decision_player();
        let features =
            encode_training_state(&self.runner, observer_idx).map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("feature_version", TRAINING_FEATURE_VERSION)?;
        out.set_item("observer_idx", observer_idx)?;
        out.set_item("decision_player", observer_idx)?;
        out.set_item("forced_advances", forced_advances)?;
        out.set_item("features", features)?;
        Ok(out.into())
    }

    fn available_root_actions(&mut self) -> PyResult<Vec<usize>> {
        let _ = self.advance_to_next_decision()?;
        Ok(self
            .filtered_root_actions()
            .into_iter()
            .filter_map(root_action_to_index)
            .collect())
    }

    fn get_legal_actions(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before enumerating turn actions",
            ));
        }
        self.refresh_legal_action_cache()?;

        let actions = PyList::empty(py);
        for (index, action) in self.legal_action_cache.iter().enumerate() {
            actions.append(legal_action_to_py(
                py,
                index,
                action,
                &self.runner,
                self.current_decision_player(),
            )?)?;
        }

        let out = PyDict::new(py);
        out.set_item("feature_version", TRAINING_FEATURE_VERSION)?;
        out.set_item("action_feature_dim", ACTION_FEATURE_DIM)?;
        out.set_item("decision_player", self.current_decision_player())?;
        out.set_item("forced_advances", forced_advances)?;
        out.set_item("actions", actions)?;
        Ok(out.into())
    }

    #[pyo3(signature = (determinizations_per_action=4, seed=None))]
    fn get_legal_action_successor_batch(
        &mut self,
        py: Python<'_>,
        determinizations_per_action: usize,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before encoding successors",
            ));
        }
        if self.runner.is_game_finished() {
            return Err(PyValueError::new_err(
                "cannot encode successors for a finished game",
            ));
        }
        self.refresh_legal_action_cache()?;
        let batch_seed = seed.unwrap_or_else(|| {
            self.runner.framework.board.state.seed
                ^ (self.runner.turn_count as u64).rotate_left(23)
                ^ (determinizations_per_action as u64).rotate_left(41)
        });
        let batch = encode_root_action_successor_batch(
            &self.runner,
            determinizations_per_action,
            batch_seed,
        )
        .map_err(PyValueError::new_err)?;
        let cached_keys = self
            .legal_action_cache
            .iter()
            .map(LegalAction::key)
            .collect::<Vec<_>>();
        if batch.action_keys != cached_keys {
            return Err(PyValueError::new_err(
                "successor batch changed stable legal-action order",
            ));
        }

        let states = PyList::empty(py);
        for (index, state) in batch.states.iter().enumerate() {
            let row = PyDict::new(py);
            row.set_item("index", index)?;
            row.set_item("feature_version", TRAINING_FEATURE_VERSION)?;
            row.set_item("observer_idx", state.observer_idx)?;
            row.set_item("features", state.features.clone())?;
            states.append(row)?;
        }
        let samples = PyList::empty(py);
        for sample in &batch.samples {
            let row = PyDict::new(py);
            row.set_item("action_index", sample.action_index)?;
            row.set_item("action_key", &sample.action_key)?;
            row.set_item("sample_index", sample.sample_index)?;
            row.set_item("evaluation_player", sample.evaluation_player)?;
            row.set_item("state_index", sample.state_index)?;
            row.set_item(
                "terminal_root_shared_win_rate",
                sample.terminal_root_shared_win_rate,
            )?;
            row.set_item(
                "terminal_root_victory_point_margin",
                sample.terminal_root_victory_point_margin,
            )?;
            samples.append(row)?;
        }
        let out = PyDict::new(py);
        out.set_item("feature_version", TRAINING_FEATURE_VERSION)?;
        out.set_item("num_players", batch.num_players)?;
        out.set_item("root_player", batch.root_player)?;
        out.set_item("action_keys", batch.action_keys)?;
        out.set_item(
            "determinizations_per_action",
            batch.determinizations_per_action,
        )?;
        out.set_item("seed", batch_seed)?;
        out.set_item("forced_advances", forced_advances)?;
        out.set_item("states", states)?;
        out.set_item("samples", samples)?;
        Ok(out.into())
    }

    #[pyo3(signature = (simulations=2000, seed=None))]
    fn search_legal_actions(
        &mut self,
        py: Python<'_>,
        simulations: u64,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        if simulations == 0 || simulations > 1_000_000 {
            return Err(PyValueError::new_err(
                "simulations must be between 1 and 1000000",
            ));
        }
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before searching turn actions",
            ));
        }
        if self.runner.is_game_finished() {
            return Err(PyValueError::new_err("cannot search a finished game"));
        }
        self.refresh_legal_action_cache()?;
        let action_count = self.legal_action_cache.len();
        let search_seed = seed.unwrap_or_else(|| {
            self.runner.framework.board.state.seed
                ^ (self.runner.turn_count as u64).rotate_left(19)
                ^ simulations.rotate_left(37)
        });
        let config = RootSearchConfig {
            simulations,
            seed: search_seed,
            recommendation_count: action_count.max(1),
            sample_continuation_length: 0,
            ..RootSearchConfig::default()
        };
        let report = search_top_actions(&self.runner, &config).map_err(PyValueError::new_err)?;
        search_report_to_py(py, report, &self.legal_action_cache, forced_advances)
    }

    #[pyo3(signature = (
        policy_probabilities,
        model_id,
        shared_win_rate,
        victory_point_margin,
        simulations=2000,
        seed=None,
        exploration_constant=1.5
    ))]
    fn search_legal_actions_with_policy(
        &mut self,
        py: Python<'_>,
        policy_probabilities: Vec<f64>,
        model_id: String,
        shared_win_rate: f64,
        victory_point_margin: f64,
        simulations: u64,
        seed: Option<u64>,
        exploration_constant: f64,
    ) -> PyResult<PyObject> {
        if simulations == 0 || simulations > 1_000_000 {
            return Err(PyValueError::new_err(
                "simulations must be between 1 and 1000000",
            ));
        }
        if !exploration_constant.is_finite() || exploration_constant < 0.0 {
            return Err(PyValueError::new_err(
                "exploration_constant must be finite and non-negative",
            ));
        }
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before searching turn actions",
            ));
        }
        if self.runner.is_game_finished() {
            return Err(PyValueError::new_err("cannot search a finished game"));
        }
        self.refresh_legal_action_cache()?;
        let action_count = self.legal_action_cache.len();
        let search_seed = seed.unwrap_or_else(|| {
            self.runner.framework.board.state.seed
                ^ (self.runner.turn_count as u64).rotate_left(19)
                ^ simulations.rotate_left(37)
        });
        let config = RootSearchConfig {
            simulations,
            exploration_constant,
            seed: search_seed,
            recommendation_count: action_count.max(1),
            sample_continuation_length: 0,
            ..RootSearchConfig::default()
        };
        let policy = RootPolicyEvaluation {
            model_id,
            action_keys: self
                .legal_action_cache
                .iter()
                .map(LegalAction::key)
                .collect(),
            policy_probabilities,
            shared_win_rate,
            victory_point_margin,
            actor_victory_points: 0.0,
        };
        let report = search_top_actions_with_policy(&self.runner, &config, &policy)
            .map_err(PyValueError::new_err)?;
        search_report_to_py(py, report, &self.legal_action_cache, forced_advances)
    }

    #[pyo3(signature = (
        policy_probabilities,
        action_keys,
        action_shared_win_rates,
        action_victory_point_margins,
        action_shared_win_standard_errors,
        action_value_sample_counts,
        model_id,
        shared_win_rate,
        victory_point_margin,
        simulations=2000,
        seed=None,
        exploration_constant=1.5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search_legal_actions_with_policy_and_action_values(
        &mut self,
        py: Python<'_>,
        policy_probabilities: Vec<f64>,
        action_keys: Vec<String>,
        action_shared_win_rates: Vec<f64>,
        action_victory_point_margins: Vec<f64>,
        action_shared_win_standard_errors: Vec<Option<f64>>,
        action_value_sample_counts: Vec<u64>,
        model_id: String,
        shared_win_rate: f64,
        victory_point_margin: f64,
        simulations: u64,
        seed: Option<u64>,
        exploration_constant: f64,
    ) -> PyResult<PyObject> {
        if simulations == 0 || simulations > 1_000_000 {
            return Err(PyValueError::new_err(
                "simulations must be between 1 and 1000000",
            ));
        }
        if !exploration_constant.is_finite() || exploration_constant < 0.0 {
            return Err(PyValueError::new_err(
                "exploration_constant must be finite and non-negative",
            ));
        }
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before searching turn actions",
            ));
        }
        if self.runner.is_game_finished() {
            return Err(PyValueError::new_err("cannot search a finished game"));
        }
        self.refresh_legal_action_cache()?;
        let action_count = self.legal_action_cache.len();
        let search_seed = seed.unwrap_or_else(|| {
            self.runner.framework.board.state.seed
                ^ (self.runner.turn_count as u64).rotate_left(19)
                ^ simulations.rotate_left(37)
        });
        let config = RootSearchConfig {
            simulations,
            exploration_constant,
            seed: search_seed,
            recommendation_count: action_count.max(1),
            sample_continuation_length: 0,
            ..RootSearchConfig::default()
        };
        let policy = RootPolicyEvaluation {
            model_id: model_id.clone(),
            action_keys: action_keys.clone(),
            policy_probabilities,
            shared_win_rate,
            victory_point_margin,
            actor_victory_points: 0.0,
        };
        let action_values = RootActionValueEvaluation {
            model_id,
            action_keys,
            shared_win_rates: action_shared_win_rates,
            victory_point_margins: action_victory_point_margins,
            shared_win_standard_errors: action_shared_win_standard_errors,
            sample_counts: action_value_sample_counts,
        };
        let report = search_top_actions_with_policy_and_action_values(
            &self.runner,
            &config,
            &policy,
            &action_values,
        )
        .map_err(PyValueError::new_err)?;
        search_report_to_py(py, report, &self.legal_action_cache, forced_advances)
    }

    #[pyo3(signature = (
        policy_probabilities,
        model_id,
        shared_win_rate,
        victory_point_margin,
        simulations=2000,
        seed=None,
        exploration_constant=1.5,
        determinizations=4,
        score_utility_weight=0.0,
        actor_victory_points=0.0,
        group_card_choices=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn start_batched_neural_search(
        &mut self,
        policy_probabilities: Vec<f64>,
        model_id: String,
        shared_win_rate: f64,
        victory_point_margin: f64,
        simulations: u64,
        seed: Option<u64>,
        exploration_constant: f64,
        determinizations: usize,
        score_utility_weight: f64,
        actor_victory_points: f64,
        group_card_choices: bool,
    ) -> PyResult<BrassNeuralSearch> {
        if simulations == 0 || simulations > 1_000_000 {
            return Err(PyValueError::new_err(
                "simulations must be between 1 and 1000000",
            ));
        }
        if !exploration_constant.is_finite() || exploration_constant < 0.0 {
            return Err(PyValueError::new_err(
                "exploration_constant must be finite and non-negative",
            ));
        }
        if !score_utility_weight.is_finite() || !(0.0..=1.0).contains(&score_utility_weight) {
            return Err(PyValueError::new_err(
                "score_utility_weight must be finite and in [0, 1]",
            ));
        }
        if !actor_victory_points.is_finite() || actor_victory_points < 0.0 {
            return Err(PyValueError::new_err(
                "actor_victory_points must be finite and non-negative",
            ));
        }
        let forced_advances = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before searching turn actions",
            ));
        }
        if self.runner.is_game_finished() {
            return Err(PyValueError::new_err("cannot search a finished game"));
        }
        self.refresh_legal_action_cache()?;
        let action_count = self.legal_action_cache.len();
        let search_seed = seed.unwrap_or_else(|| {
            self.runner.framework.board.state.seed
                ^ (self.runner.turn_count as u64).rotate_left(19)
                ^ simulations.rotate_left(37)
        });
        let policy = RootPolicyEvaluation {
            model_id,
            action_keys: self
                .legal_action_cache
                .iter()
                .map(LegalAction::key)
                .collect(),
            policy_probabilities,
            shared_win_rate,
            victory_point_margin,
            actor_victory_points,
        };
        let search = BatchedNeuralPuctSearch::new(
            &self.runner,
            BatchedNeuralPuctConfig {
                search: RootSearchConfig {
                    simulations,
                    exploration_constant,
                    seed: search_seed,
                    recommendation_count: action_count.max(1),
                    sample_continuation_length: 0,
                    ..RootSearchConfig::default()
                },
                determinizations,
                score_utility_weight,
                group_card_choices,
            },
            policy,
        )
        .map_err(PyValueError::new_err)?;
        Ok(BrassNeuralSearch {
            search,
            legal_actions: self.legal_action_cache.clone(),
            forced_advances,
        })
    }

    fn get_outcome(&self, py: Python<'_>) -> PyResult<PyObject> {
        if !self.runner.is_game_finished() {
            return Err(PyValueError::new_err(
                "official outcome is only available after game end",
            ));
        }
        outcome_to_py(py, &self.runner)
    }

    fn step_legal_action(&mut self, py: Python<'_>, action_index: usize) -> PyResult<PyObject> {
        let mut forced_passes = self.advance_to_next_decision()?;
        if self.current_shortfall.is_some() {
            return Err(PyValueError::new_err(
                "resolve the current income shortfall before applying a turn action",
            ));
        }

        let decision_mode_before = self.current_decision_mode();
        let acting_player = self.current_decision_player();
        let phase_before = self.runner.game_phase;
        let vps_before = current_vps(&self.runner);
        let potential_vps_before = current_potential_vps(&self.runner);
        let shortfall_before = (usize::MAX, 0);

        if self.runner.is_game_finished() {
            return self.build_step_delta(
                py,
                acting_player,
                &decision_mode_before,
                "game_end",
                -1,
                forced_passes,
                0,
                0,
                phase_before,
                vps_before,
                potential_vps_before,
                shortfall_before,
            );
        }

        if self.legal_action_cache.is_empty() {
            self.refresh_legal_action_cache()?;
        }
        let action = self
            .legal_action_cache
            .get(action_index)
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "legal action index {action_index} is out of range 0..{}",
                    self.legal_action_cache.len()
                ))
            })?;
        let action_key = legal_action_key(&action);
        let action_type_name = root_action_name(action.intent.action_type).to_string();
        let action_type_id = root_action_to_index(action.root)
            .map(|idx| idx as i32)
            .unwrap_or(-1);
        let sold_buildings_count = action.intent.sell_choices.len();

        action.apply(&mut self.runner).map_err(|error| {
            PyValueError::new_err(format!("legal action became stale: {error}"))
        })?;
        self.legal_action_cache.clear();
        forced_passes += self.advance_to_next_decision()?;

        let out = self.build_step_delta(
            py,
            acting_player,
            &decision_mode_before,
            &action_type_name,
            action_type_id,
            forced_passes,
            sold_buildings_count,
            0,
            phase_before,
            vps_before,
            potential_vps_before,
            shortfall_before,
        )?;
        let out_dict = out
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("step delta was not a dictionary"))?;
        out_dict.set_item("legal_action_index", action_index)?;
        out_dict.set_item("legal_action_key", action_key)?;
        Ok(out)
    }

    fn save_state(&self) -> BrassSavedState {
        BrassSavedState {
            runner: self.runner.clone(),
            current_shortfall: self.current_shortfall.clone(),
        }
    }

    fn restore_state(&mut self, state: &BrassSavedState) -> PyResult<()> {
        self.runner = state.runner.clone();
        self.current_shortfall = state.current_shortfall.clone();
        self.legal_action_cache.clear();
        let _ = self.advance_to_next_decision()?;
        Ok(())
    }

    fn randomize_hidden_information(&mut self, observer_idx: usize) -> PyResult<()> {
        self.validate_player_index(observer_idx)?;
        if self.runner.is_game_finished() {
            return Ok(());
        }
        let mut rng = self.runner.framework.board.state.rng.clone();
        determinize_hidden_information(&mut self.runner, observer_idx, &mut rng)
            .map_err(PyValueError::new_err)?;
        self.runner.framework.board.state.rng = rng;
        self.legal_action_cache.clear();
        Ok(())
    }

    fn step_composite_action(
        &mut self,
        py: Python<'_>,
        action: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        self.legal_action_cache.clear();
        let mut forced_passes = self.advance_to_next_decision()?;
        let decision_mode_before = self.current_decision_mode();
        let acting_player = self.current_decision_player();
        let phase_before = self.runner.game_phase;
        let vps_before = current_vps(&self.runner);
        let potential_vps_before = current_potential_vps(&self.runner);
        let shortfall_before = self
            .current_shortfall
            .as_ref()
            .map(|s| (s.player_idx, s.shortfall))
            .unwrap_or((usize::MAX, 0));
        let mut sold_buildings_count: usize = 0;
        let mut liquidation_tile_count: usize = 0;
        let mut action_type_name = "shortfall".to_string();
        let mut action_type_id: i32 = -1;

        if self.runner.is_game_finished() {
            let out = self.build_step_delta(
                py,
                acting_player,
                &decision_mode_before,
                &action_type_name,
                action_type_id,
                forced_passes,
                sold_buildings_count,
                liquidation_tile_count,
                phase_before,
                vps_before,
                potential_vps_before,
                shortfall_before,
            )?;
            return Ok(out);
        }

        if let Some(session) = self.current_shortfall.take() {
            let payload = CompositeActionPayload::from_py(action)?;
            let chosen_tiles = payload.shortfall_tile_order;
            liquidation_tile_count = chosen_tiles.len();
            self.runner
                .resolve_shortfall_with_tiles(session, chosen_tiles);
        } else {
            let mut payload = CompositeActionPayload::from_py(action)?;
            let valid_roots = self.filtered_root_actions();
            if valid_roots.is_empty() {
                return Err(PyValueError::new_err(
                    "No legal root action available for turn decision",
                ));
            }

            let desired_root_idx = payload
                .root_action
                .ok_or_else(|| PyValueError::new_err("missing required root_action"))?;
            let desired_root = index_to_root_action(desired_root_idx).ok_or_else(|| {
                PyValueError::new_err(format!("unknown root_action index {desired_root_idx}"))
            })?;
            if !valid_roots.contains(&desired_root) {
                return Err(PyValueError::new_err(format!(
                    "root action {} is not legal; expected one of {:?}",
                    root_action_name(desired_root),
                    valid_roots
                        .iter()
                        .map(|root| root_action_name(*root))
                        .collect::<Vec<_>>()
                )));
            }

            let _ = self.runner.start_action(desired_root);
            let mut choice_loop_guard = 0usize;
            loop {
                choice_loop_guard += 1;
                if choice_loop_guard > 512 {
                    self.runner.framework.cancel_action_session();
                    return Err(PyValueError::new_err(
                        "Action session choice loop exceeded safety limit",
                    ));
                }
                let Some(choice_set) = self.runner.framework.get_next_choice_set() else {
                    self.runner.framework.cancel_action_session();
                    return Err(PyValueError::new_err(
                        "Action session ended before confirmation",
                    ));
                };
                if matches!(choice_set, ChoiceSet::ConfirmOnly) {
                    break;
                }
                if matches!(choice_set, ChoiceSet::SellTarget(_))
                    && self.runner.framework.can_confirm()
                    && payload.peek_sell_target() == Some(STOP_INDEX_SELL_TARGET)
                {
                    let _ = payload.next_sell_target();
                    break;
                }
                let choice =
                    select_choice_from_payload(&choice_set, &mut payload).map_err(|err| {
                        self.runner.framework.cancel_action_session();
                        err
                    })?;
                self.runner
                    .framework
                    .apply_action_choice(choice)
                    .map_err(|err| {
                        self.runner.framework.cancel_action_session();
                        PyValueError::new_err(format!("apply_action_choice failed: {err}"))
                    })?;
            }

            if let Some(session) = self.runner.framework.current_session() {
                if session.action_type == ActionType::Sell {
                    sold_buildings_count = session.intent.sell_choices.len();
                }
            }

            self.runner.confirm_action().map_err(|err| {
                self.runner.framework.cancel_action_session();
                PyValueError::new_err(format!("confirm_action failed: {err}"))
            })?;

            action_type_name = root_action_name(desired_root).to_string();
            action_type_id = root_action_to_index(desired_root)
                .map(|idx| idx as i32)
                .unwrap_or(-1);

            if self.runner.actions_remaining_in_turn == 0 {
                self.runner.end_turn();
            }
        }

        forced_passes += self.advance_to_next_decision()?;
        let out = self.build_step_delta(
            py,
            acting_player,
            &decision_mode_before,
            &action_type_name,
            action_type_id,
            forced_passes,
            sold_buildings_count,
            liquidation_tile_count,
            phase_before,
            vps_before,
            potential_vps_before,
            shortfall_before,
        )?;
        Ok(out)
    }
}

impl BrassRLGame {
    fn refresh_legal_action_cache(&mut self) -> PyResult<()> {
        self.legal_action_cache = enumerate_legal_actions(&self.runner).map_err(|error| {
            PyValueError::new_err(format!("legal action enumeration failed: {error}"))
        })?;
        Ok(())
    }

    fn validate_player_index(&self, player_idx: usize) -> PyResult<()> {
        if player_idx >= self.runner.framework.board.state.players.len() {
            return Err(PyValueError::new_err(format!(
                "player index {} out of bounds for {} players",
                player_idx,
                self.runner.framework.board.state.players.len()
            )));
        }
        Ok(())
    }

    fn pull_shortfall_if_needed(&mut self) {
        if self.current_shortfall.is_none() && !self.runner.pending_shortfall_sessions.is_empty() {
            self.current_shortfall = Some(self.runner.pending_shortfall_sessions.remove(0));
        }
    }

    fn filtered_root_actions(&self) -> Vec<ActionType> {
        self.runner
            .framework
            .get_valid_root_actions()
            .into_iter()
            .filter(|a| *a != ActionType::BuildDoubleRailroad)
            .collect()
    }

    fn advance_to_next_decision(&mut self) -> PyResult<usize> {
        let mut forced_passes = 0usize;
        let mut guard = 0usize;

        while guard < 256 {
            guard += 1;
            if self.runner.is_game_finished() {
                break;
            }

            self.pull_shortfall_if_needed();
            if self.current_shortfall.is_some() {
                break;
            }

            if self.runner.actions_remaining_in_turn == 0 {
                let _ = self.runner.start_turn();
            }

            let roots = self.filtered_root_actions();
            if !roots.is_empty() {
                break;
            }

            // No card means no legal action, including Pass. Advance the empty slot;
            // a legal Pass is always exposed to the policy as a real decision.
            self.runner.end_action_slot();
            forced_passes += 1;

            if self.runner.actions_remaining_in_turn == 0 {
                self.runner.end_turn();
            }
        }

        if guard >= 256 {
            return Err(PyValueError::new_err(
                "advance_to_next_decision exceeded safety limit",
            ));
        }
        Ok(forced_passes)
    }

    fn write_observation(
        &self,
        _py: Python<'_>,
        observer_idx: usize,
        out: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let state = &self.runner.framework.board.state;
        let num_players = state.players.len();
        let decision_player = self
            .current_shortfall
            .as_ref()
            .map(|s| s.player_idx)
            .unwrap_or(self.runner.framework.current_player);
        let decision_mode = if self.current_shortfall.is_some() {
            "shortfall"
        } else {
            "turn"
        };

        let mut phase_one_hot = vec![0.0f32; 3];
        phase_one_hot[match self.runner.game_phase {
            GamePhase::Canal => 0,
            GamePhase::Railroad => 1,
            GamePhase::GameEnd => 2,
        }] = 1.0;
        let mut era_one_hot = vec![0.0f32; 2];
        era_one_hot[match state.era {
            Era::Canal => 0,
            Era::Railroad => 1,
        }] = 1.0;

        let global_features = vec![
            phase_one_hot[0],
            phase_one_hot[1],
            phase_one_hot[2],
            era_one_hot[0],
            era_one_hot[1],
            self.runner.turn_count as f32 / 200.0,
            self.runner.round_in_phase as f32 / 16.0,
            self.runner.actions_remaining_in_turn as f32 / 2.0,
            state.wild_location_cards_available as f32 / 4.0,
            state.wild_industry_cards_available as f32 / 4.0,
            state.remaining_market_coal as f32 / MAX_MARKET_COAL as f32,
            state.remaining_market_iron as f32 / MAX_MARKET_IRON as f32,
            if self.current_shortfall.is_some() {
                1.0
            } else {
                0.0
            },
            if self.runner.has_pending_shortfall() {
                1.0
            } else {
                0.0
            },
            observer_idx as f32 / (num_players.max(1) as f32),
            decision_player as f32 / (num_players.max(1) as f32),
        ];

        out.set_item("decision_mode", decision_mode)?;
        out.set_item("decision_player", decision_player)?;
        out.set_item("observer_idx", observer_idx)?;
        out.set_item("global_features", global_features)?;
        out.set_item(
            "turn_order",
            state
                .turn_order
                .iter()
                .map(|idx| *idx as i64)
                .collect::<Vec<i64>>(),
        )?;
        out.set_item("actions_remaining", self.runner.actions_remaining_in_turn)?;
        out.set_item("turn_count", self.runner.turn_count)?;
        out.set_item("round_in_phase", self.runner.round_in_phase)?;

        let mut discard_counts = vec![0.0f32; CARD_TYPE_DIM];
        for card in &state.discard_pile {
            let idx = card_type_index(card);
            discard_counts[idx] += 1.0;
        }
        out.set_item("discard_counts", discard_counts)?;

        let mut trade_post_slots = Vec::<i64>::new();
        let mut trade_post_beer = Vec::<f32>::new();
        for slot_idx in 0..(NUM_TRADE_POSTS * 2) {
            let slot_val = state
                .trade_post_slots
                .get(slot_idx)
                .and_then(|t| t.as_ref())
                .map(|tile| merchant_tile_type_to_index(&tile.tile_type))
                .unwrap_or(-1);
            trade_post_slots.push(slot_val);
            trade_post_beer.push(if state.trade_post_beer.contains(slot_idx) {
                1.0
            } else {
                0.0
            });
        }
        out.set_item("trade_post_slots", trade_post_slots)?;
        out.set_item("trade_post_beer", trade_post_beer)?;

        let mut buildings = Vec::<Vec<f32>>::with_capacity(N_BL);
        for bl_idx in 0..N_BL {
            let mut row = vec![0.0f32; 1 + N_PLAYERS + 6 + 1 + 1 + 1 + 6];
            if let Some(building) = state.bl_to_building.get(&bl_idx) {
                row[0] = 1.0;
                let owner = building.owner.as_usize().min(N_PLAYERS - 1);
                row[1 + owner] = 1.0;
                row[1 + N_PLAYERS + building.industry as usize] = 1.0;
                row[1 + N_PLAYERS + 6] = (building.level.as_usize() as f32 + 1.0) / 8.0;
                row[1 + N_PLAYERS + 6 + 1] = building.resource_amt as f32 / 5.0;
                row[1 + N_PLAYERS + 6 + 1 + 1] = if building.flipped { 1.0 } else { 0.0 };
            }
            for ind_idx in 0..6 {
                row[1 + N_PLAYERS + 6 + 1 + 1 + 1 + ind_idx] =
                    if BUILD_LOCATION_MASK[bl_idx].contains(ind_idx) {
                        1.0
                    } else {
                        0.0
                    };
            }
            buildings.push(row);
        }
        out.set_item("buildings", buildings)?;

        let mut roads = Vec::<Vec<f32>>::with_capacity(N_ROAD_LOCATIONS);
        for road_idx in 0..N_ROAD_LOCATIONS {
            let mut row = vec![0.0f32; 1 + N_PLAYERS + 2 + 3];
            row[1 + N_PLAYERS] = if LINK_LOCATIONS[road_idx].can_build_canal {
                1.0
            } else {
                0.0
            };
            row[1 + N_PLAYERS + 1] = if LINK_LOCATIONS[road_idx].can_build_rail {
                1.0
            } else {
                0.0
            };
            if state.built_roads.contains(road_idx) {
                row[0] = 1.0;
                for p_idx in 0..num_players {
                    if state.player_road_mask[p_idx].contains(road_idx) {
                        row[1 + p_idx.min(N_PLAYERS - 1)] = 1.0;
                        break;
                    }
                }
            }
            let mut loc_values = [-1.0f32; 3];
            for (k, loc_idx) in LINK_LOCATIONS[road_idx]
                .locations
                .ones()
                .enumerate()
                .take(3)
            {
                loc_values[k] = loc_idx as f32 / (N_LOCATIONS as f32);
            }
            row[1 + N_PLAYERS + 2] = loc_values[0];
            row[1 + N_PLAYERS + 2 + 1] = loc_values[1];
            row[1 + N_PLAYERS + 2 + 2] = loc_values[2];
            roads.push(row);
        }
        out.set_item("roads", roads)?;

        let mut players_public = Vec::<Vec<f32>>::with_capacity(num_players);
        let mut industry_mats = Vec::<Vec<f32>>::with_capacity(num_players);
        let mut hand_sizes = Vec::<f32>::with_capacity(num_players);
        let mut player_build_masks = Vec::<Vec<f32>>::with_capacity(num_players);
        let mut player_road_masks = Vec::<Vec<f32>>::with_capacity(num_players);

        for p_idx in 0..num_players {
            let player = &state.players[p_idx];
            players_public.push(vec![
                if p_idx == observer_idx { 1.0 } else { 0.0 },
                if p_idx == self.runner.framework.current_player {
                    1.0
                } else {
                    0.0
                },
                if p_idx == decision_player { 1.0 } else { 0.0 },
                player.money as f32 / 100.0,
                player.income_level as f32 / 100.0,
                player.get_income_amount(player.income_level) as f32 / 30.0,
                player.victory_points as f32 / 100.0,
                state.visible_vps[p_idx] as f32 / 100.0,
                player.hand.cards.len() as f32 / MAX_HAND_MASK_DIM as f32,
                player.spent_this_turn as f32 / 100.0,
            ]);

            hand_sizes.push(player.hand.cards.len() as f32 / MAX_HAND_MASK_DIM as f32);

            let mut mat_row = Vec::<f32>::with_capacity(6 * 3);
            for ind_idx in 0..6 {
                let industry = IndustryType::from_usize(ind_idx);
                let level = player.industry_mat.get_lowest_level(industry);
                let remaining = player.industry_mat.get_remaining_tiles_at_level(industry);
                mat_row.push((level.as_usize() as f32 + 1.0) / 8.0);
                mat_row.push(remaining as f32 / 3.0);
                mat_row.push(if player.industry_mat.has_tiles_left(industry) {
                    1.0
                } else {
                    0.0
                });
            }
            industry_mats.push(mat_row);

            let mut building_mask = vec![0.0f32; N_BL];
            for bl_idx in state.player_building_mask[p_idx].ones() {
                if bl_idx < N_BL {
                    building_mask[bl_idx] = 1.0;
                }
            }
            player_build_masks.push(building_mask);

            let mut road_mask = vec![0.0f32; N_ROAD_LOCATIONS];
            for road_idx in state.player_road_mask[p_idx].ones() {
                if road_idx < N_ROAD_LOCATIONS {
                    road_mask[road_idx] = 1.0;
                }
            }
            player_road_masks.push(road_mask);
        }

        out.set_item("players_public", players_public)?;
        out.set_item("industry_mats", industry_mats)?;
        out.set_item("hand_sizes", hand_sizes)?;
        out.set_item("player_building_masks", player_build_masks)?;
        out.set_item("player_road_masks", player_road_masks)?;

        let mut self_hand_counts = vec![0.0f32; CARD_TYPE_DIM];
        for card in &state.players[observer_idx].hand.cards {
            let idx = card_type_index(card);
            self_hand_counts[idx] += 1.0;
        }
        out.set_item("self_hand_counts", self_hand_counts)?;

        if let Some(shortfall) = &self.current_shortfall {
            let sf = PyDict::new(out.py());
            sf.set_item("player_idx", shortfall.player_idx)?;
            sf.set_item("shortfall", shortfall.shortfall)?;
            let tiles: Vec<(usize, u16)> = shortfall
                .removable_tiles
                .iter()
                .map(|t| (t.build_location_idx, t.liquidation_value))
                .collect();
            sf.set_item("removable_tiles", tiles)?;
            out.set_item("shortfall_state", sf)?;
        } else {
            out.set_item("shortfall_state", out.py().None())?;
        }
        Ok(())
    }

    fn write_action_masks(
        &mut self,
        _py: Python<'_>,
        observer_idx: usize,
        out: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        self.validate_player_index(observer_idx)?;

        let mut root_mask = vec![0.0f32; ROOT_ACTION_COUNT];
        let mut card_mask = vec![0.0f32; MAX_HAND_MASK_DIM];
        let mut industry_mask = vec![0.0f32; 6];
        let mut second_industry_mask = vec![0.0f32; 6];
        let mut build_location_mask = vec![0.0f32; N_BL];
        let mut network_mode_mask = vec![0.0f32; NETWORK_MODE_DIM];
        let mut road_mask = vec![0.0f32; N_ROAD_LOCATIONS];
        let mut second_road_mask = vec![0.0f32; N_ROAD_LOCATIONS];
        let mut coal_source_mask = vec![0.0f32; COAL_SOURCE_DIM];
        let mut iron_source_mask = vec![0.0f32; IRON_SOURCE_DIM];
        let mut beer_source_mask = vec![0.0f32; BEER_SOURCE_DIM];
        let mut action_beer_source_mask = vec![0.0f32; ACTION_BEER_SOURCE_DIM];
        let mut sell_target_mask = vec![0.0f32; SELL_TARGET_DIM];
        let mut shortfall_tile_mask = vec![0.0f32; SHORTFALL_TILE_DIM];

        // Resource and sell STOP tokens are phase-dependent, so they are not
        // legal in this root-level union mask. Atomic actions carry exact legality.
        shortfall_tile_mask[STOP_INDEX_SHORTFALL] = 1.0;

        if let Some(shortfall) = &self.current_shortfall {
            for tile in &shortfall.removable_tiles {
                if tile.build_location_idx < N_BL {
                    shortfall_tile_mask[tile.build_location_idx] = 1.0;
                }
            }
            out.set_item("decision_mode", "shortfall")?;
        } else {
            out.set_item("decision_mode", "turn")?;
            for action in self.filtered_root_actions() {
                if let Some(idx) = root_action_to_index(action) {
                    root_mask[idx] = 1.0;
                }
            }

            let state = &self.runner.framework.board.state;
            let player_idx = self.runner.framework.current_player;
            let hand_len = state.players[player_idx]
                .hand
                .cards
                .len()
                .min(MAX_HAND_MASK_DIM);
            for idx in 0..hand_len {
                card_mask[idx] = 1.0;
            }

            let validation = self.runner.framework.compute_valid_options();
            if let Some(build_opts) = validation.build_options.as_ref() {
                write_build_masks(
                    build_opts,
                    &mut industry_mask,
                    &mut build_location_mask,
                    &mut coal_source_mask,
                    &mut iron_source_mask,
                );
            }

            if let Some(dev_opts) = validation.dev_options.as_ref() {
                for ind_idx in dev_opts.ones() {
                    if ind_idx < 6 {
                        industry_mask[ind_idx] = 1.0;
                        second_industry_mask[ind_idx] = 1.0;
                    }
                }
            }

            let free_dev = self
                .runner
                .framework
                .board
                .get_valid_free_development_options(player_idx);
            for ind_idx in free_dev.ones() {
                if ind_idx < 6 {
                    second_industry_mask[ind_idx] = 1.0;
                }
            }

            let iron_sources_for_one = self
                .runner
                .framework
                .board
                .get_iron_sources_for_develop(player_idx, 1);
            for src in iron_sources_for_one {
                mark_resource_source(&mut iron_source_mask, src, STOP_INDEX_IRON);
            }
            let iron_sources_for_two = self
                .runner
                .framework
                .board
                .get_iron_sources_for_develop(player_idx, 2);
            for src in iron_sources_for_two {
                mark_resource_source(&mut iron_source_mask, src, STOP_INDEX_IRON);
            }

            if let Some(canal_opts) = validation.canal_options.as_ref() {
                if !canal_opts.is_empty() {
                    network_mode_mask[0] = 1.0;
                }
                for road_idx in canal_opts {
                    if *road_idx < N_ROAD_LOCATIONS {
                        road_mask[*road_idx] = 1.0;
                    }
                }
            }

            if let Some(single_opts) = validation.single_rail_options.as_ref() {
                if !single_opts.is_empty() {
                    network_mode_mask[0] = 1.0;
                }
                write_single_rail_masks(single_opts, &mut road_mask, &mut coal_source_mask);
            }

            if let Some(double_opts) = validation.double_rail_first_link_options.as_ref() {
                if !double_opts.is_empty() {
                    network_mode_mask[1] = 1.0;
                }
                write_single_rail_masks(double_opts, &mut road_mask, &mut coal_source_mask);
                write_double_rail_followup_masks(
                    &self.runner,
                    player_idx,
                    double_opts,
                    &mut second_road_mask,
                    &mut coal_source_mask,
                    &mut action_beer_source_mask,
                );
            }

            if let Some(sell_opts) = validation.sell_options.as_ref() {
                write_sell_masks(sell_opts, &mut sell_target_mask, &mut beer_source_mask);
            }
        }

        out.set_item("root_action_mask", root_mask)?;
        out.set_item("card_mask", card_mask)?;
        out.set_item("industry_mask", industry_mask)?;
        out.set_item("second_industry_mask", second_industry_mask)?;
        out.set_item("build_location_mask", build_location_mask)?;
        out.set_item("network_mode_mask", network_mode_mask)?;
        out.set_item("road_mask", road_mask)?;
        out.set_item("second_road_mask", second_road_mask)?;
        out.set_item("coal_source_mask", coal_source_mask)?;
        out.set_item("iron_source_mask", iron_source_mask)?;
        out.set_item("beer_source_mask", beer_source_mask)?;
        out.set_item("action_beer_source_mask", action_beer_source_mask)?;
        out.set_item("sell_target_mask", sell_target_mask)?;
        out.set_item("shortfall_tile_mask", shortfall_tile_mask)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_step_delta(
        &mut self,
        py: Python<'_>,
        acting_player: usize,
        decision_mode_before: &str,
        action_type_name: &str,
        action_type_id: i32,
        forced_passes: usize,
        sold_buildings_count: usize,
        liquidation_tile_count: usize,
        phase_before: GamePhase,
        vps_before: Vec<u16>,
        potential_vps_before: Vec<u16>,
        shortfall_before: (usize, u16),
    ) -> PyResult<PyObject> {
        let done = self.runner.is_game_finished();
        let vps_after = current_vps(&self.runner);
        let vp_delta: Vec<i32> = vps_after
            .iter()
            .zip(vps_before.iter())
            .map(|(after, before)| *after as i32 - *before as i32)
            .collect();
        let potential_vps_after = current_potential_vps(&self.runner);
        let potential_vp_delta: Vec<i32> = potential_vps_after
            .iter()
            .zip(potential_vps_before.iter())
            .map(|(after, before)| *after as i32 - *before as i32)
            .collect();
        let phase_transition = phase_before != self.runner.game_phase;

        let mut entered_shortfall = false;
        if let Some(sf) = &self.current_shortfall {
            if sf.player_idx == acting_player && shortfall_before.0 != acting_player {
                entered_shortfall = true;
            }
        } else if self
            .runner
            .pending_shortfall_sessions
            .iter()
            .any(|s| s.player_idx == acting_player)
            && shortfall_before.0 != acting_player
        {
            entered_shortfall = true;
        }

        let official_winner_indices = if done {
            official_winners(&self.runner).map_err(PyValueError::new_err)?
        } else {
            Vec::new()
        };
        let winner = official_winner_indices.first().map(|idx| *idx as i64);
        let shared_win_values = (0..self.runner.framework.board.state.players.len())
            .map(|player_idx| {
                if official_winner_indices.contains(&player_idx) {
                    1.0 / official_winner_indices.len() as f64
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();

        let mut next_root_mask = vec![0.0f32; ROOT_ACTION_COUNT];
        for action in self.filtered_root_actions() {
            if let Some(idx) = root_action_to_index(action) {
                next_root_mask[idx] = 1.0;
            }
        }

        let out = PyDict::new(py);
        out.set_item("acting_player", acting_player)?;
        out.set_item("decision_mode_before", decision_mode_before)?;
        out.set_item("action_type", action_type_name)?;
        out.set_item("action_type_id", action_type_id)?;
        out.set_item("forced_passes", forced_passes)?;
        out.set_item("sold_buildings", sold_buildings_count)?;
        out.set_item("liquidation_tiles", liquidation_tile_count)?;
        out.set_item("entered_shortfall", entered_shortfall)?;
        out.set_item("phase_transition", phase_transition)?;
        out.set_item("vp_delta", vp_delta)?;
        out.set_item("potential_vp_delta", potential_vp_delta)?;
        out.set_item("vps", vps_after)?;
        out.set_item("potential_vps", potential_vps_after)?;
        out.set_item("done", done)?;
        out.set_item("winner", winner)?;
        out.set_item("official_winners", official_winner_indices)?;
        out.set_item("shared_win_values", shared_win_values)?;
        out.set_item("next_decision_player", self.current_decision_player())?;
        out.set_item("next_decision_mode", self.current_decision_mode())?;
        out.set_item("next_root_action_mask", next_root_mask)?;
        out.set_item("round_in_phase", self.runner.round_in_phase)?;
        out.set_item("turn_count", self.runner.turn_count)?;
        Ok(out.into())
    }
}

fn search_report_to_py(
    py: Python<'_>,
    report: RootSearchReport,
    legal_actions: &[LegalAction],
    forced_advances: usize,
) -> PyResult<PyObject> {
    let estimates = report
        .recommendations
        .iter()
        .map(|estimate| (estimate.action_key.as_str(), estimate))
        .collect::<HashMap<_, _>>();

    let actions = PyList::empty(py);
    for (index, action) in legal_actions.iter().enumerate() {
        let row = PyDict::new(py);
        let action_key = action.key();
        let model_shared_win_rate = report
            .root_action_model_shared_win_rates
            .as_ref()
            .and_then(|values| values.get(index).copied());
        let model_victory_point_margin = report
            .root_action_model_victory_point_margins
            .as_ref()
            .and_then(|values| values.get(index).copied());
        let model_standard_error = report
            .root_action_model_shared_win_standard_errors
            .as_ref()
            .and_then(|values| values.get(index).copied())
            .flatten();
        let model_sample_count = report
            .root_action_model_sample_counts
            .as_ref()
            .and_then(|values| values.get(index).copied());
        row.set_item("index", index)?;
        row.set_item("key", &action_key)?;
        row.set_item(
            "policy_probability",
            report
                .root_policy_probabilities
                .as_ref()
                .and_then(|probabilities| probabilities.get(index).copied()),
        )?;
        if let Some(estimate) = estimates.get(action_key.as_str()) {
            row.set_item("rank", estimate.rank)?;
            row.set_item("visits", estimate.visits)?;
            row.set_item("visit_share", estimate.visit_share)?;
            row.set_item("value_source", &estimate.value_source)?;
            row.set_item("value_sample_count", estimate.value_sample_count)?;
            row.set_item(
                "estimated_shared_win_rate",
                estimate.estimated_shared_win_rate,
            )?;
            row.set_item(
                "estimated_outright_win_rate",
                estimate.estimated_outright_win_rate,
            )?;
            row.set_item(
                "estimated_tied_first_rate",
                estimate.estimated_tied_first_rate,
            )?;
            row.set_item(
                "average_final_victory_points",
                estimate.average_final_victory_points,
            )?;
            row.set_item(
                "average_victory_point_margin",
                estimate.average_victory_point_margin,
            )?;
            row.set_item(
                "shared_win_rate_standard_error",
                estimate.shared_win_rate_standard_error,
            )?;
        } else {
            row.set_item("rank", py.None())?;
            row.set_item("visits", 0)?;
            row.set_item("visit_share", 0.0)?;
            row.set_item("value_source", &report.value_source)?;
            row.set_item("value_sample_count", model_sample_count)?;
            row.set_item("estimated_shared_win_rate", model_shared_win_rate)?;
            row.set_item("estimated_outright_win_rate", py.None())?;
            row.set_item("estimated_tied_first_rate", py.None())?;
            row.set_item("average_final_victory_points", py.None())?;
            row.set_item("average_victory_point_margin", model_victory_point_margin)?;
            row.set_item("shared_win_rate_standard_error", model_standard_error)?;
        }
        actions.append(row)?;
    }

    let out = PyDict::new(py);
    out.set_item("method", report.method)?;
    out.set_item("value_source", report.value_source)?;
    out.set_item("model_id", report.model_id)?;
    out.set_item(
        "root_model_shared_win_rate",
        report.root_model_shared_win_rate,
    )?;
    out.set_item(
        "root_model_victory_point_margin",
        report.root_model_victory_point_margin,
    )?;
    out.set_item("root_player", report.root_player)?;
    out.set_item("forced_advances", forced_advances)?;
    out.set_item("requested_simulations", report.requested_simulations)?;
    out.set_item("completed_simulations", report.completed_simulations)?;
    out.set_item("root_action_count", report.root_action_count)?;
    out.set_item("evaluated_action_count", report.evaluated_action_count)?;
    out.set_item("visited_action_count", report.visited_action_count)?;
    out.set_item(
        "all_root_actions_evaluated",
        report.all_root_actions_evaluated,
    )?;
    out.set_item("max_search_depth", report.max_search_depth)?;
    out.set_item("neural_leaf_evaluations", report.neural_leaf_evaluations)?;
    out.set_item("inference_batches", report.inference_batches)?;
    out.set_item("actions", actions)?;
    Ok(out.into())
}

fn feature_schema_to_py(py: Python<'_>) -> PyResult<PyObject> {
    let schema = training_feature_schema();
    let state_blocks = PyList::empty(py);
    for block in schema.state_blocks {
        let row = PyDict::new(py);
        row.set_item("name", block.name)?;
        row.set_item("offset", block.offset)?;
        row.set_item("size", block.size)?;
        state_blocks.append(row)?;
    }
    let action_blocks = PyList::empty(py);
    for block in schema.action_blocks {
        let row = PyDict::new(py);
        row.set_item("name", block.name)?;
        row.set_item("offset", block.offset)?;
        row.set_item("size", block.size)?;
        action_blocks.append(row)?;
    }

    let out = PyDict::new(py);
    out.set_item("version", schema.version)?;
    out.set_item("state_dim", schema.state_dim)?;
    out.set_item("action_dim", schema.action_dim)?;
    out.set_item("card_type_dim", schema.card_type_dim)?;
    out.set_item("max_players", schema.max_players)?;
    out.set_item("state_blocks", state_blocks)?;
    out.set_item("action_blocks", action_blocks)?;
    Ok(out.into())
}

fn outcome_to_py(py: Python<'_>, runner: &GameRunner) -> PyResult<PyObject> {
    let state = &runner.framework.board.state;
    let winners = official_winners(runner).map_err(PyValueError::new_err)?;
    let winner_credit = 1.0 / winners.len() as f64;
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
    let victory_point_margins = state
        .players
        .iter()
        .enumerate()
        .map(|(player_idx, player)| {
            let best_opponent = state
                .players
                .iter()
                .enumerate()
                .filter_map(|(other_idx, other)| {
                    (other_idx != player_idx).then_some(other.victory_points)
                })
                .max()
                .unwrap_or(0);
            player.victory_points as i32 - best_opponent as i32
        })
        .collect::<Vec<_>>();

    let out = PyDict::new(py);
    out.set_item("official_winners", winners)?;
    out.set_item("shared_win_values", shared_win_values)?;
    out.set_item("placements", placements)?;
    out.set_item("finish_order", finish_order)?;
    out.set_item(
        "victory_points",
        state
            .players
            .iter()
            .map(|player| player.victory_points)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("victory_point_margins", victory_point_margins)?;
    out.set_item(
        "income_levels",
        state
            .players
            .iter()
            .map(|player| player.income_level)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "money",
        state
            .players
            .iter()
            .map(|player| player.money)
            .collect::<Vec<_>>(),
    )?;
    Ok(out.into())
}

fn next_from_list(values: &[usize], idx: &mut usize) -> Option<usize> {
    if *idx < values.len() {
        let value = values[*idx];
        *idx += 1;
        Some(value)
    } else {
        None
    }
}

fn extract_optional_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    let maybe = dict.get_item(key)?;
    let Some(value) = maybe else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<usize>()?))
}

fn extract_usize_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<usize>> {
    let maybe = dict.get_item(key)?;
    let Some(value) = maybe else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }

    if let Ok(list) = value.downcast::<PyList>() {
        return list
            .iter()
            .map(|item| item.extract::<usize>())
            .collect::<PyResult<Vec<usize>>>();
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| item.extract::<usize>())
            .collect::<PyResult<Vec<usize>>>();
    }
    if let Ok(single) = value.extract::<usize>() {
        return Ok(vec![single]);
    }
    Err(PyValueError::new_err(format!(
        "Value for '{}' must be an int or list of ints",
        key
    )))
}

fn legal_action_to_py(
    py: Python<'_>,
    index: usize,
    action: &LegalAction,
    runner: &GameRunner,
    observer_idx: usize,
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("index", index)?;
    out.set_item("key", legal_action_key(action))?;
    out.set_item("root_action", root_action_to_index(action.root))?;
    out.set_item("root_action_name", root_action_name(action.root))?;
    out.set_item("action_type", root_action_name(action.intent.action_type))?;

    let choices = PyList::empty(py);
    for choice in &action.choices {
        let encoded = PyDict::new(py);
        let (kind, value) = encode_action_choice(choice);
        encoded.set_item("kind", kind)?;
        encoded.set_item("value", value)?;
        choices.append(encoded)?;
    }
    out.set_item("choices", choices)?;
    out.set_item(
        "feature_indices",
        encode_training_action(runner, observer_idx, action).map_err(PyValueError::new_err)?,
    )?;
    let hand = &runner.framework.board.state.players[observer_idx]
        .hand
        .cards;
    let discard_card_types = action
        .choices
        .iter()
        .filter_map(|choice| match choice {
            ActionChoice::Card(card_idx) => hand.get(*card_idx).map(card_type_index),
            _ => None,
        })
        .collect::<Vec<_>>();
    out.set_item("discard_card_types", discard_card_types)?;

    out.set_item(
        "selected_industry",
        action
            .intent
            .selected_industry
            .map(|value| value.as_usize()),
    )?;
    out.set_item(
        "selected_second_industry",
        action
            .intent
            .selected_second_industry
            .map(|value| value.as_usize()),
    )?;
    out.set_item("selected_card", action.intent.selected_card_idx)?;
    out.set_item(
        "scout_additional_cards",
        &action.intent.scout_additional_discard_indices,
    )?;
    out.set_item("build_location", action.intent.selected_build_location)?;
    out.set_item("road", action.intent.selected_road_idx)?;
    out.set_item("second_road", action.intent.selected_second_road_idx)?;
    out.set_item(
        "sell_targets",
        action
            .intent
            .sell_choices
            .iter()
            .map(|choice| choice.location)
            .collect::<Vec<_>>(),
    )?;
    Ok(out.into())
}

fn legal_action_key(action: &LegalAction) -> String {
    action.key()
}

fn encode_action_choice(choice: &ActionChoice) -> (&'static str, Option<usize>) {
    match choice {
        ActionChoice::Industry(value) => ("industry", Some(value.as_usize())),
        ActionChoice::Card(value) => ("card", Some(*value)),
        ActionChoice::BuildLocation(value) => ("build_location", Some(*value)),
        ActionChoice::Road(value) => ("road", Some(*value)),
        ActionChoice::SellTarget(value) => ("sell_target", Some(*value)),
        ActionChoice::CoalSource(value) => ("coal_source", Some(encode_resource_source(*value))),
        ActionChoice::IronSource(value) => ("iron_source", Some(encode_resource_source(*value))),
        ActionChoice::BeerSource(value) => ("beer_source", Some(encode_beer_sell_source(*value))),
        ActionChoice::ActionBeerSource(value) => (
            "action_beer_source",
            Some(encode_action_beer_source(*value)),
        ),
        ActionChoice::FreeDevelopment(value) => ("free_development", Some(value.as_usize())),
        ActionChoice::NetworkMode(value) => (
            "network_mode",
            Some(match value {
                NetworkMode::Single => 0,
                NetworkMode::Double => 1,
            }),
        ),
        ActionChoice::Confirm => ("confirm", None),
        ActionChoice::Cancel => ("cancel", None),
    }
}

fn root_action_to_index(action: ActionType) -> Option<usize> {
    match action {
        ActionType::BuildBuilding => Some(ROOT_BUILD_BUILDING),
        ActionType::BuildRailroad => Some(ROOT_BUILD_RAILROAD),
        ActionType::Develop => Some(ROOT_DEVELOP),
        ActionType::DevelopDouble => Some(ROOT_DEVELOP_DOUBLE),
        ActionType::Sell => Some(ROOT_SELL),
        ActionType::Loan => Some(ROOT_LOAN),
        ActionType::Scout => Some(ROOT_SCOUT),
        ActionType::Pass => Some(ROOT_PASS),
        _ => None,
    }
}

fn index_to_root_action(index: usize) -> Option<ActionType> {
    match index {
        ROOT_BUILD_BUILDING => Some(ActionType::BuildBuilding),
        ROOT_BUILD_RAILROAD => Some(ActionType::BuildRailroad),
        ROOT_DEVELOP => Some(ActionType::Develop),
        ROOT_DEVELOP_DOUBLE => Some(ActionType::DevelopDouble),
        ROOT_SELL => Some(ActionType::Sell),
        ROOT_LOAN => Some(ActionType::Loan),
        ROOT_SCOUT => Some(ActionType::Scout),
        ROOT_PASS => Some(ActionType::Pass),
        _ => None,
    }
}

fn root_action_name(action: ActionType) -> &'static str {
    match action {
        ActionType::BuildBuilding => "build_building",
        ActionType::BuildRailroad => "build_railroad",
        ActionType::Develop => "develop",
        ActionType::DevelopDouble => "develop_double",
        ActionType::Sell => "sell",
        ActionType::Loan => "loan",
        ActionType::Scout => "scout",
        ActionType::Pass => "pass",
        ActionType::BuildDoubleRailroad => "build_double_railroad",
    }
}

fn current_vps(runner: &GameRunner) -> Vec<u16> {
    runner
        .framework
        .board
        .state
        .players
        .iter()
        .map(|p| p.victory_points)
        .collect()
}

fn current_potential_vps(runner: &GameRunner) -> Vec<u16> {
    let state = &runner.framework.board.state;
    let num_players = state.players.len();
    let mut potential_vps = vec![0u16; num_players];

    for road_idx in state.built_roads.ones() {
        let mut road_owner = None;
        for p_idx in 0..num_players {
            if state.player_road_mask[p_idx].contains(road_idx) {
                road_owner = Some(p_idx);
                break;
            }
        }

        if let Some(owner_idx) = road_owner {
            let mut road_vps: u16 = 0;
            for loc_idx in LINK_LOCATIONS[road_idx].locations.ones() {
                let bl_set = LocationName::from_usize(loc_idx).to_bl_set();
                for bl_idx in bl_set.ones() {
                    if let Some(building) = state
                        .bl_to_building
                        .get(&bl_idx)
                        .filter(|building| building.flipped)
                    {
                        let data =
                            &INDUSTRY_MAT[building.industry as usize][building.level.as_usize()];
                        road_vps = road_vps.saturating_add(data.road_vp as u16);
                    }
                }
            }
            potential_vps[owner_idx] = potential_vps[owner_idx].saturating_add(road_vps);
        }
    }

    for building in state.bl_to_building.values() {
        if building.flipped {
            let data = &INDUSTRY_MAT[building.industry as usize][building.level.as_usize()];
            let owner_idx = building.owner.as_usize();
            if owner_idx < potential_vps.len() {
                potential_vps[owner_idx] =
                    potential_vps[owner_idx].saturating_add(data.vp_on_flip as u16);
            }
        }
    }

    potential_vps
}

fn merchant_tile_type_to_index(tile: &MerchantTileType) -> i64 {
    match tile {
        MerchantTileType::All => 0,
        MerchantTileType::Cotton => 1,
        MerchantTileType::Goods => 2,
        MerchantTileType::Pottery => 3,
        MerchantTileType::Blank => 4,
    }
}

fn mark_resource_source(mask: &mut [f32], source: ResourceSource, stop_idx: usize) {
    let idx = match source {
        ResourceSource::Building(loc) => loc.min(stop_idx.saturating_sub(1)),
        ResourceSource::Market => N_BL,
    };
    if idx < stop_idx {
        mask[idx] = 1.0;
    }
}

fn write_build_masks(
    build_opts: &[BuildOption],
    industry_mask: &mut [f32],
    build_location_mask: &mut [f32],
    coal_source_mask: &mut [f32],
    iron_source_mask: &mut [f32],
) {
    for opt in build_opts {
        let ind_idx = opt.industry_type.as_usize();
        if ind_idx < industry_mask.len() {
            industry_mask[ind_idx] = 1.0;
        }
        if opt.build_location_idx < build_location_mask.len() {
            build_location_mask[opt.build_location_idx] = 1.0;
        }
        for src in &opt.coal_sources {
            mark_resource_source(coal_source_mask, *src, STOP_INDEX_COAL);
        }
        for src in &opt.iron_sources {
            mark_resource_source(iron_source_mask, *src, STOP_INDEX_IRON);
        }
    }
}

fn write_single_rail_masks(
    rail_opts: &[SingleRailroadOption],
    road_mask: &mut [f32],
    coal_source_mask: &mut [f32],
) {
    for option in rail_opts {
        if option.road_idx < road_mask.len() {
            road_mask[option.road_idx] = 1.0;
        }
        for src_idx in option.potential_coal_sources.ones() {
            let src = if src_idx >= N_BL {
                ResourceSource::Market
            } else {
                ResourceSource::Building(src_idx)
            };
            mark_resource_source(coal_source_mask, src, STOP_INDEX_COAL);
        }
    }
}

fn write_double_rail_followup_masks(
    runner: &GameRunner,
    player_idx: usize,
    first_link_opts: &[SingleRailroadOption],
    second_road_mask: &mut [f32],
    coal_source_mask: &mut [f32],
    action_beer_source_mask: &mut [f32],
) {
    let mut seen_second_roads = HashSet::<usize>::new();
    let mut seen_beer = HashSet::<usize>::new();
    let hypothetical_money_remaining = runner.framework.board.state.players[player_idx]
        .money
        .saturating_sub(TWO_RAILROAD_PRICE);

    for first_link in first_link_opts {
        for source_idx in first_link.potential_coal_sources.ones() {
            let coal_source = if source_idx >= N_BL {
                ResourceSource::Market
            } else {
                ResourceSource::Building(source_idx)
            };
            let followups = runner.framework.board.get_options_for_second_rail_link(
                player_idx,
                first_link.road_idx,
                &coal_source,
                hypothetical_money_remaining,
            );
            for followup in followups {
                if seen_second_roads.insert(followup.second_road_idx)
                    && followup.second_road_idx < second_road_mask.len()
                {
                    second_road_mask[followup.second_road_idx] = 1.0;
                }
                for src in followup.potential_coal_sources_for_second_link {
                    mark_resource_source(coal_source_mask, src, STOP_INDEX_COAL);
                }
                for src in followup.potential_beer_sources_for_action {
                    let idx = encode_action_beer_source(src);
                    if seen_beer.insert(idx) && idx < STOP_INDEX_ACTION_BEER {
                        action_beer_source_mask[idx] = 1.0;
                    }
                }
                for src in followup.own_brewery_sources {
                    let idx = encode_action_beer_source(src);
                    if seen_beer.insert(idx) && idx < STOP_INDEX_ACTION_BEER {
                        action_beer_source_mask[idx] = 1.0;
                    }
                }
            }
        }
    }
}

fn write_sell_masks(
    sell_opts: &[SellOption],
    sell_target_mask: &mut [f32],
    beer_source_mask: &mut [f32],
) {
    for sell_opt in sell_opts {
        if sell_opt.location < STOP_INDEX_SELL_TARGET {
            sell_target_mask[sell_opt.location] = 1.0;
        }
        for src_idx in sell_opt.beer_locations.ones() {
            if src_idx < STOP_INDEX_BEER {
                beer_source_mask[src_idx] = 1.0;
            }
        }
    }
}

fn select_choice_from_payload(
    choice_set: &ChoiceSet,
    payload: &mut CompositeActionPayload,
) -> PyResult<ActionChoice> {
    match choice_set {
        ChoiceSet::Industry(options) => Ok(ActionChoice::Industry(select_encoded_choice(
            payload.next_industry(),
            options,
            |value| value.as_usize(),
            "industry",
        )?)),
        ChoiceSet::SecondIndustry(options) => Ok(ActionChoice::Industry(select_encoded_choice(
            payload.next_second_industry(),
            options,
            |value| value.as_usize(),
            "second_industry",
        )?)),
        ChoiceSet::Card(options) => {
            let desired = payload.select_unique_card_from_options(options)?;
            Ok(ActionChoice::Card(desired))
        }
        ChoiceSet::BuildLocation(options) => {
            Ok(ActionChoice::BuildLocation(select_encoded_choice(
                payload.next_build_location(),
                options,
                |value| value,
                "build_location",
            )?))
        }
        ChoiceSet::Road(options) => Ok(ActionChoice::Road(select_encoded_choice(
            payload.next_road(),
            options,
            |value| value,
            "road",
        )?)),
        ChoiceSet::SecondRoad(options) => Ok(ActionChoice::Road(select_encoded_choice(
            payload.next_second_road(),
            options,
            |value| value,
            "second_road",
        )?)),
        ChoiceSet::CoalSource(options) => Ok(ActionChoice::CoalSource(select_encoded_choice(
            payload.next_coal_source(),
            options,
            encode_resource_source,
            "coal_source",
        )?)),
        ChoiceSet::IronSource(options) => Ok(ActionChoice::IronSource(select_encoded_choice(
            payload.next_iron_source(),
            options,
            encode_resource_source,
            "iron_source",
        )?)),
        ChoiceSet::BeerSource(options) => Ok(ActionChoice::BeerSource(select_encoded_choice(
            payload.next_beer_source(),
            options,
            encode_beer_sell_source,
            "beer_source",
        )?)),
        ChoiceSet::ActionBeerSource(options) => {
            Ok(ActionChoice::ActionBeerSource(select_encoded_choice(
                payload.next_action_beer_source(),
                options,
                encode_action_beer_source,
                "action_beer_source",
            )?))
        }
        ChoiceSet::SellTarget(options) => Ok(ActionChoice::SellTarget(select_encoded_choice(
            payload.next_sell_target(),
            options,
            |value| value,
            "sell_target",
        )?)),
        ChoiceSet::FreeDevelopment(options) => {
            let requested = payload
                .next_second_industry()
                .or_else(|| payload.next_industry());
            Ok(ActionChoice::FreeDevelopment(select_encoded_choice(
                requested,
                options,
                |value| value.as_usize(),
                "free_development",
            )?))
        }
        ChoiceSet::NetworkMode(options) => Ok(ActionChoice::NetworkMode(select_encoded_choice(
            payload.next_network_mode(),
            options,
            |mode| match mode {
                NetworkMode::Single => 0,
                NetworkMode::Double => 1,
            },
            "network_mode",
        )?)),
        ChoiceSet::ConfirmOnly => Ok(ActionChoice::Confirm),
    }
}

fn select_encoded_choice<T, F>(
    requested: Option<usize>,
    options: &[T],
    encode: F,
    label: &str,
) -> PyResult<T>
where
    T: Copy,
    F: Fn(T) -> usize,
{
    let requested = requested
        .ok_or_else(|| PyValueError::new_err(format!("missing required {label} choice")))?;
    options
        .iter()
        .copied()
        .find(|option| encode(*option) == requested)
        .ok_or_else(|| {
            let legal = options.iter().copied().map(&encode).collect::<Vec<_>>();
            PyValueError::new_err(format!(
                "{label} choice {requested} is not legal; expected one of {legal:?}"
            ))
        })
}

fn encode_resource_source(source: ResourceSource) -> usize {
    match source {
        ResourceSource::Building(loc) => loc,
        ResourceSource::Market => N_BL,
    }
}

fn encode_beer_sell_source(source: BeerSellSource) -> usize {
    match source {
        BeerSellSource::Building(loc) => loc,
        BeerSellSource::TradePost(slot) => N_BL + slot,
    }
}

fn encode_action_beer_source(source: BreweryBeerSource) -> usize {
    match source {
        BreweryBeerSource::OwnBrewery(loc) => loc,
        BreweryBeerSource::OpponentBrewery(loc) => N_BL + loc,
    }
}

#[pymodule]
pub fn fast_brass(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BrassRLGame>()?;
    m.add_class::<BrassNeuralSearch>()?;
    m.add_class::<BrassSavedState>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::building::BuiltBuilding;
    use crate::core::player::PlayerId;
    use crate::core::types::IndustryLevel;

    #[test]
    fn test_bridge_observation_and_masks_smoke() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(7)).expect("bridge init should work");
            let observer = game.current_decision_player();

            let obs_obj = game
                .get_observation(py, observer)
                .expect("observation call should work");
            let obs = obs_obj
                .bind(py)
                .downcast::<PyDict>()
                .expect("obs should be dict");
            assert!(obs.contains("global_features").unwrap());
            assert!(obs.contains("buildings").unwrap());
            assert!(obs.contains("self_hand_counts").unwrap());

            let masks_obj = game
                .get_action_masks(py, observer)
                .expect("masks call should work");
            let masks = masks_obj
                .bind(py)
                .downcast::<PyDict>()
                .expect("masks should be dict");
            assert!(masks.contains("root_action_mask").unwrap());
            assert!(masks.contains("shortfall_tile_mask").unwrap());
        });
    }

    #[test]
    fn test_training_features_and_search_distribution_smoke() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(70)).expect("bridge init should work");

            let schema_obj = game
                .get_training_feature_schema(py)
                .expect("feature schema should serialize");
            let schema = schema_obj.bind(py).downcast::<PyDict>().unwrap();
            let state_dim: usize = schema
                .get_item("state_dim")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(state_dim, crate::game::training::STATE_FEATURE_DIM);

            let state_obj = game
                .get_training_state(py)
                .expect("training state should encode");
            let state = state_obj.bind(py).downcast::<PyDict>().unwrap();
            let features = state
                .get_item("features")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert_eq!(features.len(), crate::game::training::STATE_FEATURE_DIM);

            let legal_obj = game
                .get_legal_actions(py)
                .expect("legal actions should encode");
            let legal = legal_obj.bind(py).downcast::<PyDict>().unwrap();
            let legal_actions = legal
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            let legal_count = legal_actions.len();
            let first_action = legal_actions
                .get_item(0)
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            let sparse_features = first_action
                .get_item("feature_indices")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert!(!sparse_features.is_empty());

            let search_obj = game
                .search_legal_actions(py, 32, Some(1234))
                .expect("training search should complete");
            let search = search_obj.bind(py).downcast::<PyDict>().unwrap();
            let searched_actions = search
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert_eq!(searched_actions.len(), legal_count);
            let visit_sum = searched_actions
                .iter()
                .map(|row| {
                    row.downcast_into::<PyDict>()
                        .unwrap()
                        .get_item("visits")
                        .unwrap()
                        .unwrap()
                        .extract::<u64>()
                        .unwrap()
                })
                .sum::<u64>();
            assert_eq!(visit_sum, 32);
        });
    }

    #[test]
    fn test_policy_guided_search_binding_preserves_priors_and_rejects_wrong_count() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(72)).expect("bridge init should work");
            let legal_obj = game
                .get_legal_actions(py)
                .expect("legal actions should encode");
            let legal = legal_obj.bind(py).downcast::<PyDict>().unwrap();
            let legal_actions = legal
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            let legal_count = legal_actions.len();
            assert!(legal_count > 1);

            let wrong_count_error = game
                .search_legal_actions_with_policy(
                    py,
                    vec![1.0],
                    "test-model".to_string(),
                    0.625,
                    3.5,
                    1,
                    Some(1234),
                    1.5,
                )
                .expect_err("a policy with the wrong action count must be rejected");
            assert!(wrong_count_error.to_string().contains("probabilities"));

            let selected_index = legal_count - 1;
            let mut probabilities = vec![0.0; legal_count];
            probabilities[selected_index] = 1.0;
            let search_obj = game
                .search_legal_actions_with_policy(
                    py,
                    probabilities.clone(),
                    "test-model".to_string(),
                    0.625,
                    3.5,
                    1,
                    Some(1234),
                    1.5,
                )
                .expect("policy-guided search should complete");
            let search = search_obj.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                search
                    .get_item("method")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "determinized_root_puct_policy_random_rollout"
            );
            assert_eq!(
                search
                    .get_item("model_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "test-model"
            );
            assert_eq!(
                search
                    .get_item("root_model_shared_win_rate")
                    .unwrap()
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                0.625
            );

            let searched_actions = search
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert_eq!(searched_actions.len(), legal_count);
            for (index, row) in searched_actions.iter().enumerate() {
                let row = row.downcast_into::<PyDict>().unwrap();
                assert_eq!(
                    row.get_item("policy_probability")
                        .unwrap()
                        .unwrap()
                        .extract::<f64>()
                        .unwrap(),
                    probabilities[index]
                );
                assert_eq!(
                    row.get_item("visits")
                        .unwrap()
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    u64::from(index == selected_index)
                );
            }
        });
    }

    #[test]
    fn test_successor_batch_and_action_value_search_binding() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(73)).expect("bridge init should work");
            let batch_obj = game
                .get_legal_action_successor_batch(py, 2, Some(4_321))
                .expect("successor batch should encode");
            let batch = batch_obj.bind(py).downcast::<PyDict>().unwrap();
            let action_keys = batch
                .get_item("action_keys")
                .unwrap()
                .unwrap()
                .extract::<Vec<String>>()
                .unwrap();
            let states = batch
                .get_item("states")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert_eq!(states.len(), action_keys.len() * 2);
            let selected_index = action_keys.len() - 1;
            let mut action_values = vec![0.25; action_keys.len()];
            action_values[selected_index] = 0.9;

            let search_obj = game
                .search_legal_actions_with_policy_and_action_values(
                    py,
                    vec![1.0; action_keys.len()],
                    action_keys.clone(),
                    action_values.clone(),
                    vec![0.0; action_keys.len()],
                    vec![Some(0.0); action_keys.len()],
                    vec![2; action_keys.len()],
                    "test-value-model".to_string(),
                    0.5,
                    0.0,
                    8,
                    Some(4_322),
                    0.0,
                )
                .expect("action-value search should complete");
            let search = search_obj.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                search
                    .get_item("method")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "determinized_root_puct_policy_batched_successor_value"
            );
            assert_eq!(
                search
                    .get_item("evaluated_action_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                action_keys.len()
            );
            assert_eq!(
                search
                    .get_item("visited_action_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            let searched_actions = search
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            for (index, row) in searched_actions.iter().enumerate() {
                let row = row.downcast_into::<PyDict>().unwrap();
                assert_eq!(
                    row.get_item("estimated_shared_win_rate")
                        .unwrap()
                        .unwrap()
                        .extract::<f64>()
                        .unwrap(),
                    action_values[index]
                );
                assert_eq!(
                    row.get_item("visits")
                        .unwrap()
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    if index == selected_index { 8 } else { 0 }
                );
            }
        });
    }

    #[test]
    fn test_batched_neural_search_binding_round_trips_leaf_batches() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(74)).expect("bridge init should work");
            let root_player = game.current_decision_player();
            let legal_obj = game
                .get_legal_actions(py)
                .expect("legal actions should encode");
            let legal = legal_obj.bind(py).downcast::<PyDict>().unwrap();
            let legal_count = legal
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap()
                .len();
            let mut search = game
                .start_batched_neural_search(
                    vec![1.0; legal_count],
                    "test-deep-model".to_string(),
                    0.5,
                    0.0,
                    2,
                    Some(4_401),
                    0.0,
                    1,
                    0.0,
                    70.0,
                    false,
                )
                .expect("deep search should start");

            for expected_depth in 1..=2 {
                let batch_obj = search
                    .next_inference_batch(py, 1)
                    .expect("leaf batch should encode");
                let batch = batch_obj.bind(py).downcast::<PyDict>().unwrap();
                let positions = batch
                    .get_item("positions")
                    .unwrap()
                    .unwrap()
                    .downcast_into::<PyList>()
                    .unwrap();
                assert_eq!(positions.len(), 1);
                let position = positions
                    .get_item(0)
                    .unwrap()
                    .downcast_into::<PyDict>()
                    .unwrap();
                assert_eq!(
                    position
                        .get_item("depth")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    expected_depth
                );
                let request_id = position
                    .get_item("request_id")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap();
                let evaluation_player = position
                    .get_item("evaluation_player")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let leaf_actions = position
                    .get_item("legal_actions")
                    .unwrap()
                    .unwrap()
                    .downcast_into::<PyDict>()
                    .unwrap()
                    .get_item("actions")
                    .unwrap()
                    .unwrap()
                    .downcast_into::<PyList>()
                    .unwrap();
                let action_keys = leaf_actions
                    .iter()
                    .map(|row| {
                        row.downcast_into::<PyDict>()
                            .unwrap()
                            .get_item("key")
                            .unwrap()
                            .unwrap()
                            .extract::<String>()
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let mut probabilities = vec![0.0; action_keys.len()];
                probabilities[0] = 1.0;
                let raw_value = if evaluation_player == root_player {
                    0.8
                } else {
                    0.2
                };
                search
                    .submit_inference_batch(
                        vec![request_id],
                        vec![action_keys],
                        vec![probabilities],
                        vec![raw_value],
                        vec![0.0],
                        "test-deep-model".to_string(),
                        Some(vec![70.0]),
                    )
                    .expect("leaf evaluation should submit");
            }

            assert!(search.is_complete());
            let report_obj = search.finish(py).expect("deep report should encode");
            let report = report_obj.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                report
                    .get_item("method")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "determinized_batched_neural_puct"
            );
            assert_eq!(
                report
                    .get_item("max_search_depth")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
        });
    }

    #[test]
    fn test_official_tied_outcome_uses_fractional_win_values() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(71)).expect("bridge init should work");
            let mut actions_played = 0usize;
            while !game.is_done() {
                let legal_obj = game
                    .get_legal_actions(py)
                    .expect("legal actions should enumerate");
                let legal = legal_obj.bind(py).downcast::<PyDict>().unwrap();
                let actions = legal
                    .get_item("actions")
                    .unwrap()
                    .unwrap()
                    .downcast_into::<PyList>()
                    .unwrap();
                let pass_index = actions
                    .iter()
                    .find_map(|row| {
                        let row = row.downcast_into::<PyDict>().ok()?;
                        let name = row
                            .get_item("root_action_name")
                            .ok()??
                            .extract::<String>()
                            .ok()?;
                        (name == "pass").then(|| {
                            row.get_item("index")
                                .unwrap()
                                .unwrap()
                                .extract::<usize>()
                                .unwrap()
                        })
                    })
                    .expect("Pass must remain available while cards remain");
                game.step_legal_action(py, pass_index)
                    .expect("Pass should execute");
                actions_played += 1;
                assert!(actions_played < 256, "pass-only game did not terminate");
            }

            let outcome_obj = game.get_outcome(py).expect("outcome should serialize");
            let outcome = outcome_obj.bind(py).downcast::<PyDict>().unwrap();
            let winners = outcome
                .get_item("official_winners")
                .unwrap()
                .unwrap()
                .extract::<Vec<usize>>()
                .unwrap();
            let shared = outcome
                .get_item("shared_win_values")
                .unwrap()
                .unwrap()
                .extract::<Vec<f64>>()
                .unwrap();
            assert_eq!(winners, vec![0, 1]);
            assert_eq!(shared, vec![0.5, 0.5]);
        });
    }

    #[test]
    fn test_atomic_legal_actions_are_stable_and_execute_exactly() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(8)).expect("bridge init should work");
            let first = game
                .get_legal_actions(py)
                .expect("atomic actions should enumerate");
            let first_dict = first.bind(py).downcast::<PyDict>().unwrap();
            let first_actions = first_dict
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert!(!first_actions.is_empty());

            let first_keys = first_actions
                .iter()
                .map(|item| {
                    item.downcast_into::<PyDict>()
                        .unwrap()
                        .get_item("key")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let second = game
                .get_legal_actions(py)
                .expect("repeated enumeration should succeed");
            let second_dict = second.bind(py).downcast::<PyDict>().unwrap();
            let second_actions = second_dict
                .get_item("actions")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            let second_keys = second_actions
                .iter()
                .map(|item| {
                    item.downcast_into::<PyDict>()
                        .unwrap()
                        .get_item("key")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(first_keys, second_keys);

            let (pass_index, pass_key) = second_actions
                .iter()
                .find_map(|item| {
                    let action = item.downcast_into::<PyDict>().ok()?;
                    let root = action
                        .get_item("root_action")
                        .ok()??
                        .extract::<usize>()
                        .ok()?;
                    if root != ROOT_PASS {
                        return None;
                    }
                    Some((
                        action.get_item("index").ok()??.extract::<usize>().ok()?,
                        action.get_item("key").ok()??.extract::<String>().ok()?,
                    ))
                })
                .expect("Pass should be present among atomic actions");

            let delta = game
                .step_legal_action(py, pass_index)
                .expect("selected atomic Pass should execute");
            let delta = delta.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                delta
                    .get_item("legal_action_key")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                pass_key
            );
            assert_eq!(
                delta
                    .get_item("action_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "pass"
            );
        });
    }

    #[test]
    fn test_atomic_action_index_out_of_range_does_not_mutate_state() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(9)).expect("bridge init should work");
            let player = game.runner.framework.current_player;
            let before_hand = game.runner.framework.board.state.players[player]
                .hand
                .cards
                .clone();
            let before_actions = game.runner.actions_remaining_in_turn;

            game.step_legal_action(py, usize::MAX)
                .expect_err("out-of-range action index must fail");

            assert_eq!(game.runner.framework.current_player, player);
            assert_eq!(game.runner.actions_remaining_in_turn, before_actions);
            assert_eq!(
                game.runner.framework.board.state.players[player].hand.cards,
                before_hand
            );
        });
    }

    #[test]
    fn test_save_restore_round_trip() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(11)).expect("bridge init should work");
            let saved = game.save_state();
            let before_player = game.current_decision_player();
            let before_mode = game.current_decision_mode();

            let roots = game
                .available_root_actions()
                .expect("roots should be available");
            let action = PyDict::new(py);
            assert!(roots.contains(&ROOT_PASS));
            action.set_item("root_action", ROOT_PASS).unwrap();
            action.set_item("card", vec![0usize]).unwrap();
            let _ = game
                .step_composite_action(py, &action)
                .expect("single step should work");

            game.restore_state(&saved).expect("restore should work");
            assert_eq!(before_player, game.current_decision_player());
            assert_eq!(before_mode, game.current_decision_mode());
        });
    }

    #[test]
    fn test_pass_is_exposed_as_a_strategic_action() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(13)).expect("bridge init should work");
            let player_idx = game.runner.framework.current_player;

            game.runner.framework.board.state.players[player_idx].money = 0;
            game.runner.framework.board.state.players[player_idx].income_level = 0;
            game.runner.framework.board.state.players[player_idx]
                .hand
                .cards
                .truncate(1);
            game.runner.framework.board.state.players[player_idx].industry_mat =
                crate::core::industry_mat::PlayerIndustryMat::new();
            game.runner
                .framework
                .board
                .state
                .wild_location_cards_available = 0;
            game.runner
                .framework
                .board
                .state
                .wild_industry_cards_available = 0;
            game.runner.framework.board.state.bl_to_building.clear();
            game.runner.framework.board.state.player_building_mask[player_idx].clear();
            game.runner
                .framework
                .board
                .state
                .build_locations_occupied
                .clear();
            game.runner.framework.board.state.built_roads.clear();
            game.runner.framework.board.state.player_road_mask[player_idx].clear();
            game.runner.actions_remaining_in_turn = 1;

            let forced = game
                .advance_to_next_decision()
                .expect("decision advance should not fail");
            assert_eq!(forced, 0, "a legal Pass must not be auto-executed");
            assert_eq!(game.runner.framework.current_player, player_idx);
            assert_eq!(
                game.available_root_actions()
                    .expect("roots should be available"),
                vec![ROOT_PASS]
            );

            let action = PyDict::new(py);
            action.set_item("root_action", ROOT_PASS).unwrap();
            action.set_item("card", vec![0usize]).unwrap();
            game.step_composite_action(py, &action)
                .expect("explicit pass should succeed");
        });
    }

    #[test]
    fn test_invalid_composite_action_is_rejected_without_fallback() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(14)).expect("bridge init should work");
            let before_player = game.runner.framework.current_player;
            let before_hand = game.runner.framework.board.state.players[before_player]
                .hand
                .cards
                .clone();
            let before_actions = game.runner.actions_remaining_in_turn;

            let action = PyDict::new(py);
            action.set_item("root_action", ROOT_PASS).unwrap();
            action.set_item("card", vec![usize::MAX]).unwrap();
            let error = game
                .step_composite_action(py, &action)
                .expect_err("invalid card must not fall back to another card or action");

            assert!(error.to_string().contains("card choice"));
            assert_eq!(game.runner.framework.current_player, before_player);
            assert_eq!(game.runner.actions_remaining_in_turn, before_actions);
            assert_eq!(
                game.runner.framework.board.state.players[before_player]
                    .hand
                    .cards,
                before_hand
            );
            assert!(game.runner.framework.action_context.is_none());
        });
    }

    #[test]
    fn test_shortfall_liquidation_event_fields() {
        Python::with_gil(|py| {
            let mut game = BrassRLGame::new(2, Some(21)).expect("bridge init should work");
            let player_idx = game.runner.framework.current_player;
            let loc = 27usize;

            let building = BuiltBuilding::build(
                IndustryType::Coal,
                IndustryLevel::I,
                loc as u8,
                PlayerId::from_usize(player_idx),
            );
            game.runner
                .framework
                .board
                .state
                .bl_to_building
                .insert(loc, building);
            game.runner
                .framework
                .board
                .state
                .build_locations_occupied
                .insert(loc);
            game.runner.framework.board.state.player_building_mask[player_idx].insert(loc);
            game.runner.framework.board.state.players[player_idx].money = 0;

            let session = game
                .runner
                .framework
                .start_shortfall_resolution_session(player_idx, 3);
            game.current_shortfall = Some(session);

            let action = PyDict::new(py);
            action.set_item("shortfall_tile_order", vec![loc]).unwrap();
            let delta_obj = game
                .step_composite_action(py, &action)
                .expect("shortfall step should work");
            let delta = delta_obj
                .bind(py)
                .downcast::<PyDict>()
                .expect("delta should be dict");
            let liquidation_tiles: usize = delta
                .get_item("liquidation_tiles")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(liquidation_tiles >= 1);
        });
    }

    #[test]
    fn test_hidden_information_randomization_preserves_public_constraints() {
        let mut game = BrassRLGame::new(4, Some(31)).expect("bridge init should work");
        let observer = 0usize;

        let observer_hand_before = game.runner.framework.board.state.players[observer]
            .hand
            .cards
            .clone();
        let opp_sizes_before: Vec<usize> = game
            .runner
            .framework
            .board
            .state
            .players
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != observer)
            .map(|(_, p)| p.hand.cards.len())
            .collect();

        let mut hidden_counts_before = vec![0usize; CARD_TYPE_DIM];
        for (idx, player) in game.runner.framework.board.state.players.iter().enumerate() {
            if idx != observer {
                for card in &player.hand.cards {
                    hidden_counts_before[card_type_index(card)] += 1;
                }
            }
        }
        for card in &game.runner.framework.board.state.deck.cards {
            hidden_counts_before[card_type_index(card)] += 1;
        }

        game.randomize_hidden_information(observer)
            .expect("hidden info randomization should succeed");

        assert_eq!(
            observer_hand_before,
            game.runner.framework.board.state.players[observer]
                .hand
                .cards
        );
        let opp_sizes_after: Vec<usize> = game
            .runner
            .framework
            .board
            .state
            .players
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != observer)
            .map(|(_, p)| p.hand.cards.len())
            .collect();
        assert_eq!(opp_sizes_before, opp_sizes_after);

        let mut hidden_counts_after = vec![0usize; CARD_TYPE_DIM];
        for (idx, player) in game.runner.framework.board.state.players.iter().enumerate() {
            if idx != observer {
                for card in &player.hand.cards {
                    hidden_counts_after[card_type_index(card)] += 1;
                }
            }
        }
        for card in &game.runner.framework.board.state.deck.cards {
            hidden_counts_after[card_type_index(card)] += 1;
        }
        assert_eq!(hidden_counts_before, hidden_counts_after);
    }
}
