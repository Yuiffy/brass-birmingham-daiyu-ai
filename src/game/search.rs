use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::core::locations::LocationName;
use crate::core::static_data::{INDUSTRY_MAT, LINK_LOCATIONS};
use crate::core::types::{ActionType, Era};
use crate::game::framework::{ActionChoice, ActionIntent, ChoiceSet};
use crate::game::hidden_information::determinize_hidden_information;
use crate::game::legal_actions::{enumerate_legal_actions, LegalAction};
use crate::game::runner::{GamePhase, GameRunner};
use crate::game::training::{
    advance_successor_to_decision, encode_training_action, encode_training_state,
    RootActionSuccessorBatch, TRAINING_FEATURE_VERSION,
};

pub const ROOT_UCB_METHOD: &str = "determinized_root_ucb_random_rollout";
pub const ROOT_PUCT_METHOD: &str = "determinized_root_puct_policy_random_rollout";
pub const ROOT_PUCT_ACTION_VALUE_METHOD: &str =
    "determinized_root_puct_policy_batched_successor_value";
pub const BATCHED_NEURAL_PUCT_METHOD: &str = "determinized_batched_neural_puct";
pub const RANDOM_ROLLOUT_VALUE_SOURCE: &str = "random_terminal_rollout";
pub const SUCCESSOR_MODEL_VALUE_SOURCE: &str = "batched_successor_model";
pub const NEURAL_TREE_VALUE_SOURCE: &str = "batched_neural_tree_search";
const SCORE_UTILITY_REFERENCE_VP: f64 = 140.0;

#[derive(Debug, Clone, PartialEq)]
pub struct RootSearchConfig {
    pub simulations: u64,
    pub exploration_constant: f64,
    pub seed: u64,
    pub recommendation_count: usize,
    pub max_rollout_actions: u32,
    pub sample_continuation_length: usize,
    pub sell_stop_probability: f64,
}

impl Default for RootSearchConfig {
    fn default() -> Self {
        Self {
            simulations: 2_000,
            exploration_constant: std::f64::consts::SQRT_2,
            seed: 0x4252_4153_535f_4149,
            recommendation_count: 3,
            max_rollout_actions: 256,
            sample_continuation_length: 8,
            sell_stop_probability: 0.35,
        }
    }
}

impl RootSearchConfig {
    fn validate(&self) -> Result<(), String> {
        if self.simulations == 0 {
            return Err("search requires at least one simulation".to_string());
        }
        if !self.exploration_constant.is_finite() || self.exploration_constant < 0.0 {
            return Err("exploration_constant must be finite and non-negative".to_string());
        }
        if self.recommendation_count == 0 {
            return Err("recommendation_count must be at least one".to_string());
        }
        if self.max_rollout_actions == 0 {
            return Err("max_rollout_actions must be at least one".to_string());
        }
        if !self.sell_stop_probability.is_finite()
            || !(0.0..=1.0).contains(&self.sell_stop_probability)
        {
            return Err("sell_stop_probability must be between zero and one".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootSearchReport {
    pub method: String,
    pub value_source: String,
    pub model_id: Option<String>,
    pub root_model_shared_win_rate: Option<f64>,
    pub root_model_victory_point_margin: Option<f64>,
    pub root_policy_probabilities: Option<Vec<f64>>,
    pub root_player: usize,
    pub requested_simulations: u64,
    pub completed_simulations: u64,
    pub root_action_count: usize,
    pub evaluated_action_count: usize,
    pub visited_action_count: usize,
    pub all_root_actions_evaluated: bool,
    pub max_search_depth: Option<usize>,
    pub neural_leaf_evaluations: Option<u64>,
    pub inference_batches: Option<u64>,
    pub root_action_model_shared_win_rates: Option<Vec<f64>>,
    pub root_action_model_victory_point_margins: Option<Vec<f64>>,
    pub root_action_model_shared_win_standard_errors: Option<Vec<Option<f64>>>,
    pub root_action_model_sample_counts: Option<Vec<u64>>,
    pub recommendations: Vec<RootActionEstimate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootPolicyEvaluation {
    pub model_id: String,
    pub action_keys: Vec<String>,
    pub policy_probabilities: Vec<f64>,
    pub shared_win_rate: f64,
    pub victory_point_margin: f64,
    pub actor_victory_points: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchedNeuralPuctConfig {
    pub search: RootSearchConfig,
    pub determinizations: usize,
    pub score_utility_weight: f64,
    pub group_card_choices: bool,
}

impl Default for BatchedNeuralPuctConfig {
    fn default() -> Self {
        Self {
            search: RootSearchConfig::default(),
            determinizations: 4,
            score_utility_weight: 0.0,
            group_card_choices: false,
        }
    }
}

impl BatchedNeuralPuctConfig {
    fn validate(&self) -> Result<(), String> {
        self.search.validate()?;
        if !(1..=64).contains(&self.determinizations) {
            return Err("neural search determinizations must be between 1 and 64".to_string());
        }
        if !self.score_utility_weight.is_finite()
            || !(0.0..=1.0).contains(&self.score_utility_weight)
        {
            return Err("neural score utility weight must be finite and in [0, 1]".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralLeafRequest {
    pub request_id: u64,
    pub depth: usize,
    pub evaluation_player: usize,
    pub feature_version: u32,
    pub state_features: Vec<f32>,
    pub action_keys: Vec<String>,
    pub action_feature_indices: Vec<Vec<u16>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralLeafEvaluation {
    pub request_id: u64,
    pub model_id: String,
    pub action_keys: Vec<String>,
    pub policy_probabilities: Vec<f64>,
    pub shared_win_rate: f64,
    pub victory_point_margin: f64,
    pub actor_victory_points: f64,
}

impl RootPolicyEvaluation {
    fn normalized_probabilities(&self, actions: &[LegalAction]) -> Result<Vec<f64>, String> {
        if self.model_id.trim().is_empty() {
            return Err("policy evaluation model_id must not be empty".to_string());
        }
        if self.action_keys.len() != actions.len()
            || self.policy_probabilities.len() != actions.len()
        {
            return Err(format!(
                "policy evaluation has {} keys and {} probabilities for {} legal actions",
                self.action_keys.len(),
                self.policy_probabilities.len(),
                actions.len()
            ));
        }
        for (index, (expected, actual)) in actions
            .iter()
            .map(LegalAction::key)
            .zip(&self.action_keys)
            .enumerate()
        {
            if &expected != actual {
                return Err(format!(
                    "policy action key mismatch at index {index}: expected {expected}, got {actual}"
                ));
            }
        }
        if !self.shared_win_rate.is_finite() || !(0.0..=1.0).contains(&self.shared_win_rate) {
            return Err("policy shared_win_rate must be finite and in [0, 1]".to_string());
        }
        if !self.victory_point_margin.is_finite() {
            return Err("policy victory_point_margin must be finite".to_string());
        }
        if !self.actor_victory_points.is_finite() || self.actor_victory_points < 0.0 {
            return Err("policy actor_victory_points must be finite and non-negative".to_string());
        }
        if self
            .policy_probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        {
            return Err("policy probabilities must be finite and non-negative".to_string());
        }
        let sum = self.policy_probabilities.iter().sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("policy probabilities must have positive mass".to_string());
        }
        Ok(self
            .policy_probabilities
            .iter()
            .map(|probability| probability / sum)
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootActionValueEvaluation {
    pub model_id: String,
    pub action_keys: Vec<String>,
    pub shared_win_rates: Vec<f64>,
    pub victory_point_margins: Vec<f64>,
    pub shared_win_standard_errors: Vec<Option<f64>>,
    pub sample_counts: Vec<u64>,
}

impl RootActionValueEvaluation {
    fn validate(&self, actions: &[LegalAction], policy_model_id: &str) -> Result<(), String> {
        if self.model_id.trim().is_empty() {
            return Err("action-value evaluation model_id must not be empty".to_string());
        }
        if self.model_id != policy_model_id {
            return Err(format!(
                "policy model {policy_model_id} and action-value model {} disagree",
                self.model_id
            ));
        }
        let expected = actions.len();
        if self.action_keys.len() != expected
            || self.shared_win_rates.len() != expected
            || self.victory_point_margins.len() != expected
            || self.shared_win_standard_errors.len() != expected
            || self.sample_counts.len() != expected
        {
            return Err(format!(
                "action-value evaluation lengths do not match {expected} legal actions"
            ));
        }
        for (index, (expected_key, actual_key)) in actions
            .iter()
            .map(LegalAction::key)
            .zip(&self.action_keys)
            .enumerate()
        {
            if &expected_key != actual_key {
                return Err(format!(
                    "action-value key mismatch at index {index}: expected {expected_key}, got {actual_key}"
                ));
            }
            if !self.shared_win_rates[index].is_finite()
                || !(0.0..=1.0).contains(&self.shared_win_rates[index])
            {
                return Err(format!(
                    "action-value shared-win estimate at index {index} is invalid"
                ));
            }
            if !self.victory_point_margins[index].is_finite() {
                return Err(format!(
                    "action-value VP-margin estimate at index {index} is invalid"
                ));
            }
            if self.sample_counts[index] == 0 {
                return Err(format!(
                    "action-value sample count at index {index} is zero"
                ));
            }
            if self.shared_win_standard_errors[index]
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            {
                return Err(format!(
                    "action-value standard error at index {index} is invalid"
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootActionEstimate {
    pub rank: usize,
    pub action_key: String,
    pub action: LegalAction,
    pub visits: u64,
    pub visit_share: f64,
    pub value_source: String,
    pub value_sample_count: u64,
    pub estimated_shared_win_rate: f64,
    pub estimated_outright_win_rate: Option<f64>,
    pub estimated_tied_first_rate: Option<f64>,
    pub average_final_victory_points: Option<f64>,
    pub average_victory_point_margin: f64,
    pub shared_win_rate_standard_error: Option<f64>,
    pub policy_probability: Option<f64>,
    pub calibrated_win_rate: Option<f64>,
    pub immediate_effect: ImmediateEffect,
    pub sample_random_continuation: SampleRandomContinuation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImmediateEffect {
    pub player_money_delta: Vec<i32>,
    pub player_income_level_delta: Vec<i16>,
    pub player_victory_points_delta: Vec<i32>,
    pub player_visible_victory_points_delta: Vec<i32>,
    pub player_potential_era_victory_points_delta: Vec<i32>,
    pub player_hand_size_delta: Vec<i16>,
    pub player_round_spend_delta: Vec<i32>,
    pub roads_on_board_delta: i32,
    pub buildings_on_board_delta: i32,
    pub flipped_buildings_delta: i32,
    pub draw_deck_size_delta: i32,
    pub market_coal_delta: i16,
    pub market_iron_delta: i16,
    pub wild_location_pool_delta: i16,
    pub wild_industry_pool_delta: i16,
    pub current_player_before: usize,
    pub current_player_after: usize,
    pub actions_remaining_before: u8,
    pub actions_remaining_after: u8,
    pub turn_count_delta: i64,
    pub phase_before: GamePhase,
    pub phase_after: GamePhase,
    pub era_before: Era,
    pub era_after: Era,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleRandomContinuation {
    pub steps: Vec<SampleContinuationStep>,
    pub final_victory_points: Vec<u16>,
    pub official_winners: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleContinuationStep {
    pub action_number: u32,
    pub player_idx: usize,
    pub action_key: String,
    pub action_type: ActionType,
    pub intent: ActionIntent,
}

#[derive(Debug, Clone, Default)]
struct CandidateStats {
    visits: u64,
    shared_win_sum: f64,
    shared_win_square_sum: f64,
    outright_wins: u64,
    tied_firsts: u64,
    final_victory_points_sum: f64,
    victory_point_margin_sum: f64,
    sample_random_continuation: Option<SampleRandomContinuation>,
}

pub struct BatchedNeuralPuctSearch {
    root_runner: GameRunner,
    root_player: usize,
    root_actions: Vec<LegalAction>,
    root_policy: RootPolicyEvaluation,
    normalized_root_policy: Vec<f64>,
    config: BatchedNeuralPuctConfig,
    trees: Vec<NeuralTree>,
    pending: HashMap<u64, PendingNeuralSimulation>,
    next_request_id: u64,
    next_tree_index: usize,
    completed_simulations: u64,
    neural_leaf_evaluations: u64,
    inference_batches: u64,
    max_search_depth: usize,
}

struct NeuralTree {
    nodes: Vec<NeuralTreeNode>,
}

struct NeuralTreeNode {
    runner: GameRunner,
    actor: usize,
    actions: Vec<LegalAction>,
    action_groups: Vec<Vec<usize>>,
    priors: Vec<f64>,
    model_root_shared_win_rate: f64,
    model_root_victory_point_margin: f64,
    model_root_score: f64,
    model_opponent_score: f64,
    edges: Vec<NeuralTreeEdge>,
    pending_evaluation: bool,
}

#[derive(Clone)]
struct NeuralTreeEdge {
    child: NeuralTreeChild,
    visits: u64,
    virtual_visits: u64,
    root_shared_win_sum: f64,
    root_shared_win_square_sum: f64,
    root_victory_point_margin_sum: f64,
    root_score_sum: f64,
    opponent_score_sum: f64,
}

impl Default for NeuralTreeEdge {
    fn default() -> Self {
        Self {
            child: NeuralTreeChild::Unexpanded,
            visits: 0,
            virtual_visits: 0,
            root_shared_win_sum: 0.0,
            root_shared_win_square_sum: 0.0,
            root_victory_point_margin_sum: 0.0,
            root_score_sum: 0.0,
            opponent_score_sum: 0.0,
        }
    }
}

#[derive(Clone)]
enum NeuralTreeChild {
    Unexpanded,
    Node(usize),
    Terminal {
        root_shared_win_rate: f64,
        root_victory_point_margin: f64,
        root_score: f64,
        opponent_score: f64,
    },
}

struct PendingNeuralSimulation {
    tree_index: usize,
    node_index: usize,
    path: Vec<(usize, usize)>,
    depth: usize,
}

struct ValidatedNeuralSubmission {
    request_id: u64,
    normalized_policy: Vec<f64>,
    root_shared_win_rate: f64,
    root_victory_point_margin: f64,
    root_score: f64,
    opponent_score: f64,
}

enum NeuralReservation {
    Pending(NeuralLeafRequest, PendingNeuralSimulation),
    Terminal,
    Unavailable,
}

#[derive(Debug)]
struct RolloutOutcome {
    shared_win_credit: f64,
    outright_win: bool,
    tied_first: bool,
    final_victory_points: Vec<u16>,
    victory_point_margin: i32,
    official_winners: Vec<usize>,
}

impl BatchedNeuralPuctSearch {
    pub fn new(
        runner: &GameRunner,
        config: BatchedNeuralPuctConfig,
        root_policy: RootPolicyEvaluation,
    ) -> Result<Self, String> {
        config.validate()?;
        if runner.is_game_finished() {
            return Err("cannot search a finished game".to_string());
        }
        let num_players = runner.framework.board.state.players.len();
        if num_players != 2 {
            return Err(format!(
                "batched neural PUCT currently requires exactly two players, got {num_players}"
            ));
        }

        let root_player = runner.framework.current_player;
        let root_actions = enumerate_legal_actions(runner)?;
        if root_actions.is_empty() {
            return Err("no legal root actions available for neural search".to_string());
        }
        let normalized_root_policy = root_policy.normalized_probabilities(&root_actions)?;
        let root_action_keys = root_actions
            .iter()
            .map(LegalAction::key)
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(config.search.seed ^ 0x6465_6570_5f70_7563);
        let mut trees = Vec::with_capacity(config.determinizations);
        for determinization_index in 0..config.determinizations {
            let mut determinized = runner.clone();
            determinize_hidden_information(&mut determinized, root_player, &mut rng)?;
            let actions = enumerate_legal_actions(&determinized)?;
            let action_keys = actions.iter().map(LegalAction::key).collect::<Vec<_>>();
            if action_keys != root_action_keys {
                return Err(format!(
                    "root legal actions changed in determinization {determinization_index}"
                ));
            }
            let (root_score, opponent_score) = two_player_predicted_scores(
                &determinized,
                root_player,
                root_player,
                root_policy.actor_victory_points,
                root_policy.victory_point_margin,
            )?;
            trees.push(NeuralTree {
                nodes: vec![NeuralTreeNode::expanded(
                    determinized,
                    root_player,
                    actions,
                    normalized_root_policy.clone(),
                    root_policy.shared_win_rate,
                    root_policy.victory_point_margin,
                    root_score,
                    opponent_score,
                    config.group_card_choices,
                )],
            });
        }

        Ok(Self {
            root_runner: runner.clone(),
            root_player,
            root_actions,
            root_policy,
            normalized_root_policy,
            config,
            trees,
            pending: HashMap::new(),
            next_request_id: 0,
            next_tree_index: 0,
            completed_simulations: 0,
            neural_leaf_evaluations: 0,
            inference_batches: 0,
            max_search_depth: 0,
        })
    }

    pub fn is_complete(&self) -> bool {
        self.completed_simulations == self.config.search.simulations && self.pending.is_empty()
    }

    pub fn completed_simulations(&self) -> u64 {
        self.completed_simulations
    }

    pub fn pending_evaluations(&self) -> usize {
        self.pending.len()
    }

    pub fn max_search_depth(&self) -> usize {
        self.max_search_depth
    }

    pub fn root_actions(&self) -> &[LegalAction] {
        &self.root_actions
    }

    pub fn next_inference_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<Vec<NeuralLeafRequest>, String> {
        self.next_inference_batch_until(max_batch_size, self.config.search.simulations)
    }

    pub fn next_inference_batch_until(
        &mut self,
        max_batch_size: usize,
        target_simulations: u64,
    ) -> Result<Vec<NeuralLeafRequest>, String> {
        if max_batch_size == 0 || max_batch_size > 256 {
            return Err("neural inference batch size must be between 1 and 256".to_string());
        }
        if target_simulations == 0 || target_simulations > self.config.search.simulations {
            return Err(format!(
                "target simulations must be between 1 and {}",
                self.config.search.simulations
            ));
        }
        if !self.pending.is_empty() {
            return Err(
                "submit the pending neural inference batch before requesting another".to_string(),
            );
        }
        if self.completed_simulations >= target_simulations {
            return Ok(Vec::new());
        }

        let mut requests = Vec::with_capacity(max_batch_size);
        let mut consecutive_unavailable = 0usize;
        while requests.len() < max_batch_size
            && self.completed_simulations + (requests.len() as u64) < target_simulations
        {
            let tree_index = self.next_tree_index;
            self.next_tree_index = (self.next_tree_index + 1) % self.trees.len();
            match self.reserve_simulation(tree_index)? {
                NeuralReservation::Pending(request, pending) => {
                    let request_id = request.request_id;
                    if self.pending.insert(request_id, pending).is_some() {
                        return Err(format!("duplicate pending neural request id {request_id}"));
                    }
                    requests.push(request);
                    consecutive_unavailable = 0;
                }
                NeuralReservation::Terminal => {
                    consecutive_unavailable = 0;
                }
                NeuralReservation::Unavailable => {
                    consecutive_unavailable += 1;
                    if consecutive_unavailable >= self.trees.len() {
                        break;
                    }
                }
            }
        }

        if requests.is_empty() && self.completed_simulations < target_simulations {
            return Err("neural search could not reserve an evaluable leaf".to_string());
        }
        Ok(requests)
    }

    pub fn submit_inference_batch(
        &mut self,
        evaluations: Vec<NeuralLeafEvaluation>,
    ) -> Result<(), String> {
        if self.pending.is_empty() {
            return Err("neural search has no pending inference batch".to_string());
        }
        if evaluations.len() != self.pending.len() {
            return Err(format!(
                "received {} neural evaluations for {} pending leaves",
                evaluations.len(),
                self.pending.len()
            ));
        }

        let mut seen = HashSet::with_capacity(evaluations.len());
        let mut validated = Vec::with_capacity(evaluations.len());
        for evaluation in evaluations {
            if !seen.insert(evaluation.request_id) {
                return Err(format!(
                    "duplicate neural evaluation request id {}",
                    evaluation.request_id
                ));
            }
            let pending = self.pending.get(&evaluation.request_id).ok_or_else(|| {
                format!(
                    "neural evaluation request id {} is not pending",
                    evaluation.request_id
                )
            })?;
            if evaluation.model_id != self.root_policy.model_id {
                return Err(format!(
                    "root model {} and neural leaf model {} disagree",
                    self.root_policy.model_id, evaluation.model_id
                ));
            }
            let node = self
                .trees
                .get(pending.tree_index)
                .and_then(|tree| tree.nodes.get(pending.node_index))
                .ok_or_else(|| "pending neural leaf points outside the search tree".to_string())?;
            if !node.pending_evaluation {
                return Err("pending neural leaf is already expanded".to_string());
            }
            let policy = RootPolicyEvaluation {
                model_id: evaluation.model_id,
                action_keys: evaluation.action_keys,
                policy_probabilities: evaluation.policy_probabilities,
                shared_win_rate: evaluation.shared_win_rate,
                victory_point_margin: evaluation.victory_point_margin,
                actor_victory_points: evaluation.actor_victory_points,
            };
            let normalized_policy = policy.normalized_probabilities(&node.actions)?;
            let (root_shared_win_rate, root_victory_point_margin) = to_root_perspective(
                self.root_player,
                node.actor,
                policy.shared_win_rate,
                policy.victory_point_margin,
            )?;
            let (root_score, opponent_score) = two_player_predicted_scores(
                &node.runner,
                self.root_player,
                node.actor,
                policy.actor_victory_points,
                policy.victory_point_margin,
            )?;
            validated.push(ValidatedNeuralSubmission {
                request_id: evaluation.request_id,
                normalized_policy,
                root_shared_win_rate,
                root_victory_point_margin,
                root_score,
                opponent_score,
            });
        }
        validated.sort_by_key(|submission| submission.request_id);

        for submission in validated {
            let pending = self
                .pending
                .remove(&submission.request_id)
                .expect("validated neural request remains pending");
            let tree = &mut self.trees[pending.tree_index];
            let node = &mut tree.nodes[pending.node_index];
            node.priors = submission.normalized_policy;
            node.edges = vec![NeuralTreeEdge::default(); node.actions.len()];
            node.model_root_shared_win_rate = submission.root_shared_win_rate;
            node.model_root_victory_point_margin = submission.root_victory_point_margin;
            node.model_root_score = submission.root_score;
            node.model_opponent_score = submission.opponent_score;
            node.pending_evaluation = false;
            backup_neural_path(
                tree,
                &pending.path,
                submission.root_shared_win_rate,
                submission.root_victory_point_margin,
                submission.root_score,
                submission.opponent_score,
                true,
            )?;
            self.completed_simulations += 1;
            self.neural_leaf_evaluations += 1;
            self.max_search_depth = self.max_search_depth.max(pending.depth);
        }
        self.inference_batches += 1;
        Ok(())
    }

    pub fn finish_report(&self) -> Result<RootSearchReport, String> {
        let report = self.report_at_current_progress()?;
        if report.completed_simulations != self.config.search.simulations {
            return Err(format!(
                "neural search completed {} of {} requested simulations",
                report.completed_simulations, self.config.search.simulations
            ));
        }
        Ok(report)
    }

    pub fn report_at_current_progress(&self) -> Result<RootSearchReport, String> {
        if !self.pending.is_empty() {
            return Err("cannot finish neural search with pending inference leaves".to_string());
        }

        let mut stats = vec![CandidateStats::default(); self.root_actions.len()];
        for tree in &self.trees {
            let root = tree
                .nodes
                .first()
                .ok_or_else(|| "neural search tree has no root".to_string())?;
            if root.edges.len() != stats.len() {
                return Err("neural search root action count changed".to_string());
            }
            for (index, edge) in root.edges.iter().enumerate() {
                stats[index].visits += edge.visits;
                stats[index].shared_win_sum += edge.root_shared_win_sum;
                stats[index].shared_win_square_sum += edge.root_shared_win_square_sum;
                stats[index].victory_point_margin_sum += edge.root_victory_point_margin_sum;
            }
        }
        let root_visit_sum = stats.iter().map(|candidate| candidate.visits).sum::<u64>();
        if root_visit_sum != self.completed_simulations {
            return Err(format!(
                "neural root visits {root_visit_sum} disagree with {} completed simulations",
                self.completed_simulations
            ));
        }

        let immediate_effects = self
            .root_actions
            .iter()
            .map(|action| immediate_effect_for_action(&self.root_runner, action))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ranked_indices = stats
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| (candidate.visits > 0).then_some(index))
            .collect::<Vec<_>>();
        ranked_indices.sort_by(|left, right| {
            stats[*right]
                .visits
                .cmp(&stats[*left].visits)
                .then_with(|| {
                    mean_shared_win(&stats[*right]).total_cmp(&mean_shared_win(&stats[*left]))
                })
                .then_with(|| {
                    self.root_actions[*left]
                        .key()
                        .cmp(&self.root_actions[*right].key())
                })
        });

        let mut rng = StdRng::seed_from_u64(self.config.search.seed ^ 0x7265_706f_7274_5f72);
        let visit_denominator = self.completed_simulations.max(1) as f64;
        let mut recommendations = Vec::new();
        for (rank_index, action_index) in ranked_indices
            .into_iter()
            .take(self.config.search.recommendation_count)
            .enumerate()
        {
            let candidate = &stats[action_index];
            let sample_random_continuation = if self.config.search.sample_continuation_length == 0 {
                SampleRandomContinuation {
                    steps: Vec::new(),
                    final_victory_points: Vec::new(),
                    official_winners: Vec::new(),
                }
            } else {
                sample_random_continuation_for_action(
                    &self.root_runner,
                    &self.root_actions[action_index],
                    self.root_player,
                    &self.config.search,
                    &mut rng,
                )?
            };
            recommendations.push(RootActionEstimate {
                rank: rank_index + 1,
                action_key: self.root_actions[action_index].key(),
                action: self.root_actions[action_index].clone(),
                visits: candidate.visits,
                visit_share: candidate.visits as f64 / visit_denominator,
                value_source: NEURAL_TREE_VALUE_SOURCE.to_string(),
                value_sample_count: candidate.visits,
                estimated_shared_win_rate: mean_shared_win(candidate),
                estimated_outright_win_rate: None,
                estimated_tied_first_rate: None,
                average_final_victory_points: None,
                average_victory_point_margin: candidate.victory_point_margin_sum
                    / candidate.visits as f64,
                shared_win_rate_standard_error: sample_standard_error(
                    candidate.shared_win_sum,
                    candidate.shared_win_square_sum,
                    candidate.visits,
                ),
                policy_probability: Some(self.normalized_root_policy[action_index]),
                calibrated_win_rate: None,
                immediate_effect: immediate_effects[action_index].clone(),
                sample_random_continuation,
            });
        }

        let visited_action_count = stats
            .iter()
            .filter(|candidate| candidate.visits > 0)
            .count();
        Ok(RootSearchReport {
            method: BATCHED_NEURAL_PUCT_METHOD.to_string(),
            value_source: NEURAL_TREE_VALUE_SOURCE.to_string(),
            model_id: Some(self.root_policy.model_id.clone()),
            root_model_shared_win_rate: Some(self.root_policy.shared_win_rate),
            root_model_victory_point_margin: Some(self.root_policy.victory_point_margin),
            root_policy_probabilities: Some(self.normalized_root_policy.clone()),
            root_player: self.root_player,
            requested_simulations: self.config.search.simulations,
            completed_simulations: self.completed_simulations,
            root_action_count: self.root_actions.len(),
            evaluated_action_count: visited_action_count,
            visited_action_count,
            all_root_actions_evaluated: visited_action_count == self.root_actions.len(),
            max_search_depth: Some(self.max_search_depth),
            neural_leaf_evaluations: Some(self.neural_leaf_evaluations),
            inference_batches: Some(self.inference_batches),
            root_action_model_shared_win_rates: None,
            root_action_model_victory_point_margins: None,
            root_action_model_shared_win_standard_errors: None,
            root_action_model_sample_counts: None,
            recommendations,
        })
    }

    fn reserve_simulation(&mut self, tree_index: usize) -> Result<NeuralReservation, String> {
        let mut node_index = 0usize;
        let mut path = Vec::new();
        loop {
            let edge_index = {
                let tree = &self.trees[tree_index];
                select_neural_edge(
                    tree,
                    node_index,
                    self.root_player,
                    self.config.search.exploration_constant,
                    self.config.score_utility_weight,
                )
            };
            let Some(edge_index) = edge_index else {
                return Ok(NeuralReservation::Unavailable);
            };
            path.push((node_index, edge_index));
            let child = self.trees[tree_index].nodes[node_index].edges[edge_index]
                .child
                .clone();
            match child {
                NeuralTreeChild::Node(child_index) => {
                    node_index = child_index;
                }
                NeuralTreeChild::Terminal {
                    root_shared_win_rate,
                    root_victory_point_margin,
                    root_score,
                    opponent_score,
                } => {
                    backup_neural_path(
                        &mut self.trees[tree_index],
                        &path,
                        root_shared_win_rate,
                        root_victory_point_margin,
                        root_score,
                        opponent_score,
                        false,
                    )?;
                    self.completed_simulations += 1;
                    self.max_search_depth = self.max_search_depth.max(path.len());
                    return Ok(NeuralReservation::Terminal);
                }
                NeuralTreeChild::Unexpanded => {
                    let (mut successor, action) = {
                        let node = &self.trees[tree_index].nodes[node_index];
                        (node.runner.clone(), node.actions[edge_index].clone())
                    };
                    action.apply(&mut successor).map_err(|error| {
                        format!(
                            "neural tree action {} failed at depth {}: {error}",
                            action.key(),
                            path.len()
                        )
                    })?;
                    advance_successor_to_decision(&mut successor)?;
                    if successor.is_game_finished() {
                        let outcome = evaluate_finished_game(&successor, self.root_player)?;
                        let root_shared_win_rate = outcome.shared_win_credit;
                        let root_victory_point_margin = outcome.victory_point_margin as f64;
                        let root_score = outcome.final_victory_points[self.root_player] as f64;
                        let opponent_score =
                            outcome.final_victory_points[1 - self.root_player] as f64;
                        self.trees[tree_index].nodes[node_index].edges[edge_index].child =
                            NeuralTreeChild::Terminal {
                                root_shared_win_rate,
                                root_victory_point_margin,
                                root_score,
                                opponent_score,
                            };
                        backup_neural_path(
                            &mut self.trees[tree_index],
                            &path,
                            root_shared_win_rate,
                            root_victory_point_margin,
                            root_score,
                            opponent_score,
                            false,
                        )?;
                        self.completed_simulations += 1;
                        self.max_search_depth = self.max_search_depth.max(path.len());
                        return Ok(NeuralReservation::Terminal);
                    }

                    let actor = successor.framework.current_player;
                    let actions = enumerate_legal_actions(&successor)?;
                    if actions.is_empty() {
                        return Err(format!(
                            "neural leaf at depth {} has no legal actions",
                            path.len()
                        ));
                    }
                    let state_features = encode_training_state(&successor, actor)?;
                    let action_keys = actions.iter().map(LegalAction::key).collect::<Vec<_>>();
                    let action_feature_indices = actions
                        .iter()
                        .map(|candidate| encode_training_action(&successor, actor, candidate))
                        .collect::<Result<Vec<_>, _>>()?;
                    let request_id = self.next_request_id;
                    self.next_request_id += 1;
                    let child_index = self.trees[tree_index].nodes.len();
                    self.trees[tree_index].nodes.push(NeuralTreeNode::pending(
                        successor,
                        actor,
                        actions,
                        self.config.group_card_choices,
                    ));
                    self.trees[tree_index].nodes[node_index].edges[edge_index].child =
                        NeuralTreeChild::Node(child_index);
                    add_virtual_visits(&mut self.trees[tree_index], &path)?;
                    let request = NeuralLeafRequest {
                        request_id,
                        depth: path.len(),
                        evaluation_player: actor,
                        feature_version: TRAINING_FEATURE_VERSION,
                        state_features,
                        action_keys,
                        action_feature_indices,
                    };
                    let pending = PendingNeuralSimulation {
                        tree_index,
                        node_index: child_index,
                        depth: path.len(),
                        path,
                    };
                    return Ok(NeuralReservation::Pending(request, pending));
                }
            }
        }
    }
}

impl NeuralTreeNode {
    fn expanded(
        runner: GameRunner,
        actor: usize,
        actions: Vec<LegalAction>,
        priors: Vec<f64>,
        model_root_shared_win_rate: f64,
        model_root_victory_point_margin: f64,
        model_root_score: f64,
        model_opponent_score: f64,
        group_card_choices: bool,
    ) -> Self {
        debug_assert_eq!(actions.len(), priors.len());
        let edge_count = actions.len();
        let action_groups = build_neural_action_groups(&actions, group_card_choices);
        Self {
            runner,
            actor,
            actions,
            action_groups,
            priors,
            model_root_shared_win_rate,
            model_root_victory_point_margin,
            model_root_score,
            model_opponent_score,
            edges: vec![NeuralTreeEdge::default(); edge_count],
            pending_evaluation: false,
        }
    }

    fn pending(
        runner: GameRunner,
        actor: usize,
        actions: Vec<LegalAction>,
        group_card_choices: bool,
    ) -> Self {
        let action_groups = build_neural_action_groups(&actions, group_card_choices);
        Self {
            runner,
            actor,
            actions,
            action_groups,
            priors: Vec::new(),
            model_root_shared_win_rate: 0.0,
            model_root_victory_point_margin: 0.0,
            model_root_score: 0.0,
            model_opponent_score: 0.0,
            edges: Vec::new(),
            pending_evaluation: true,
        }
    }
}

fn select_neural_edge(
    tree: &NeuralTree,
    node_index: usize,
    root_player: usize,
    exploration_constant: f64,
    score_utility_weight: f64,
) -> Option<usize> {
    let node = tree.nodes.get(node_index)?;
    if node.pending_evaluation || node.actions.len() != node.edges.len() {
        return None;
    }
    let total_visits = node
        .edges
        .iter()
        .map(|edge| edge.visits + edge.virtual_visits)
        .sum::<u64>();
    let sqrt_total = (total_visits + 1) as f64;
    let sqrt_total = sqrt_total.sqrt();
    let mut best_group: Option<(usize, f64)> = None;
    for (group_index, members) in node.action_groups.iter().enumerate() {
        let available = members.iter().any(|&edge_index| {
            !matches!(
                node.edges[edge_index].child,
                NeuralTreeChild::Node(child) if tree.nodes[child].pending_evaluation
            )
        });
        if !available {
            continue;
        }
        let group_visits = members
            .iter()
            .map(|&edge_index| node.edges[edge_index].visits)
            .sum::<u64>();
        let group_virtual_visits = members
            .iter()
            .map(|&edge_index| node.edges[edge_index].virtual_visits)
            .sum::<u64>();
        let group_prior = members
            .iter()
            .map(|&edge_index| node.priors[edge_index])
            .sum::<f64>();
        let (root_win_value, root_score, opponent_score, root_margin) = if group_visits == 0 {
            (
                node.model_root_shared_win_rate,
                node.model_root_score,
                node.model_opponent_score,
                node.model_root_victory_point_margin,
            )
        } else {
            let root_win_sum = members
                .iter()
                .map(|&edge_index| node.edges[edge_index].root_shared_win_sum)
                .sum::<f64>();
            let root_score_sum = members
                .iter()
                .map(|&edge_index| node.edges[edge_index].root_score_sum)
                .sum::<f64>();
            let opponent_score_sum = members
                .iter()
                .map(|&edge_index| node.edges[edge_index].opponent_score_sum)
                .sum::<f64>();
            let root_margin_sum = members
                .iter()
                .map(|&edge_index| node.edges[edge_index].root_victory_point_margin_sum)
                .sum::<f64>();
            (
                root_win_sum / group_visits as f64,
                root_score_sum / group_visits as f64,
                opponent_score_sum / group_visits as f64,
                root_margin_sum / group_visits as f64,
            )
        };
        let actor_value = actor_utility_from_root_values(
            node,
            root_player,
            root_win_value,
            root_score,
            opponent_score,
            root_margin,
            score_utility_weight,
        );
        let exploration = exploration_constant * group_prior * sqrt_total
            / (1.0 + group_visits as f64 + group_virtual_visits as f64);
        let score = actor_value + exploration;
        if best_group.is_none_or(|(_, best_score)| score > best_score) {
            best_group = Some((group_index, score));
        }
    }

    let (group_index, _) = best_group?;
    select_neural_edge_within_group(
        tree,
        node,
        &node.action_groups[group_index],
        root_player,
        exploration_constant,
        score_utility_weight,
    )
}

fn select_neural_edge_within_group(
    tree: &NeuralTree,
    node: &NeuralTreeNode,
    members: &[usize],
    root_player: usize,
    exploration_constant: f64,
    score_utility_weight: f64,
) -> Option<usize> {
    let group_visits = members
        .iter()
        .map(|&edge_index| {
            let edge = &node.edges[edge_index];
            edge.visits + edge.virtual_visits
        })
        .sum::<u64>();
    let sqrt_group_visits = ((group_visits + 1) as f64).sqrt();
    let group_prior = members
        .iter()
        .map(|&edge_index| node.priors[edge_index])
        .sum::<f64>();
    let fallback_prior = 1.0 / members.len().max(1) as f64;
    let mut best: Option<(usize, f64)> = None;
    for &edge_index in members {
        let edge = &node.edges[edge_index];
        if matches!(edge.child, NeuralTreeChild::Node(child) if tree.nodes[child].pending_evaluation)
        {
            continue;
        }
        let (root_win_value, root_score, opponent_score, root_margin) = if edge.visits == 0 {
            (
                node.model_root_shared_win_rate,
                node.model_root_score,
                node.model_opponent_score,
                node.model_root_victory_point_margin,
            )
        } else {
            (
                edge.root_shared_win_sum / edge.visits as f64,
                edge.root_score_sum / edge.visits as f64,
                edge.opponent_score_sum / edge.visits as f64,
                edge.root_victory_point_margin_sum / edge.visits as f64,
            )
        };
        let actor_value = actor_utility_from_root_values(
            node,
            root_player,
            root_win_value,
            root_score,
            opponent_score,
            root_margin,
            score_utility_weight,
        );
        let conditional_prior = if group_prior > 0.0 {
            node.priors[edge_index] / group_prior
        } else {
            fallback_prior
        };
        let exploration = exploration_constant * conditional_prior * sqrt_group_visits
            / (1.0 + edge.visits as f64 + edge.virtual_visits as f64);
        let score = actor_value + exploration;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((edge_index, score));
        }
    }
    best.map(|(edge_index, _)| edge_index)
}

fn actor_utility_from_root_values(
    node: &NeuralTreeNode,
    root_player: usize,
    root_win_value: f64,
    root_score: f64,
    opponent_score: f64,
    root_victory_point_margin: f64,
    score_utility_weight: f64,
) -> f64 {
    if node.actor == root_player {
        combined_actor_utility(
            root_win_value,
            root_score,
            root_victory_point_margin,
            score_utility_weight,
        )
    } else {
        combined_actor_utility(
            1.0 - root_win_value,
            opponent_score,
            -root_victory_point_margin,
            score_utility_weight,
        )
    }
}

fn build_neural_action_groups(
    actions: &[LegalAction],
    group_card_choices: bool,
) -> Vec<Vec<usize>> {
    if !group_card_choices {
        return (0..actions.len()).map(|index| vec![index]).collect();
    }

    let mut group_indices = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (action_index, action) in actions.iter().enumerate() {
        let key = action.card_invariant_key();
        let group_index = match group_indices.get(&key) {
            Some(&group_index) => group_index,
            None => {
                let group_index = groups.len();
                group_indices.insert(key, group_index);
                groups.push(Vec::new());
                group_index
            }
        };
        groups[group_index].push(action_index);
    }
    groups
}

fn combined_actor_utility(
    shared_win_value: f64,
    actor_score: f64,
    victory_point_margin: f64,
    score_utility_weight: f64,
) -> f64 {
    let absolute_score_quality = (actor_score / SCORE_UTILITY_REFERENCE_VP).clamp(0.0, 1.0);
    let margin_quality =
        (0.5 + victory_point_margin / (2.0 * SCORE_UTILITY_REFERENCE_VP)).clamp(0.0, 1.0);
    let score_quality = 0.5 * (absolute_score_quality + margin_quality);
    (1.0 - score_utility_weight) * shared_win_value + score_utility_weight * score_quality
}

fn add_virtual_visits(tree: &mut NeuralTree, path: &[(usize, usize)]) -> Result<(), String> {
    for &(node_index, edge_index) in path {
        let edge = tree
            .nodes
            .get_mut(node_index)
            .and_then(|node| node.edges.get_mut(edge_index))
            .ok_or_else(|| "neural virtual-visit path is invalid".to_string())?;
        edge.virtual_visits += 1;
    }
    Ok(())
}

fn backup_neural_path(
    tree: &mut NeuralTree,
    path: &[(usize, usize)],
    root_shared_win_rate: f64,
    root_victory_point_margin: f64,
    root_score: f64,
    opponent_score: f64,
    remove_virtual_visit: bool,
) -> Result<(), String> {
    for &(node_index, edge_index) in path {
        let edge = tree
            .nodes
            .get_mut(node_index)
            .and_then(|node| node.edges.get_mut(edge_index))
            .ok_or_else(|| "neural backup path is invalid".to_string())?;
        if remove_virtual_visit {
            if edge.virtual_visits == 0 {
                return Err("neural backup has no matching virtual visit".to_string());
            }
            edge.virtual_visits -= 1;
        }
        edge.visits += 1;
        edge.root_shared_win_sum += root_shared_win_rate;
        edge.root_shared_win_square_sum += root_shared_win_rate * root_shared_win_rate;
        edge.root_victory_point_margin_sum += root_victory_point_margin;
        edge.root_score_sum += root_score;
        edge.opponent_score_sum += opponent_score;
    }
    Ok(())
}

fn to_root_perspective(
    root_player: usize,
    evaluation_player: usize,
    shared_win_rate: f64,
    victory_point_margin: f64,
) -> Result<(f64, f64), String> {
    if !shared_win_rate.is_finite() || !(0.0..=1.0).contains(&shared_win_rate) {
        return Err("neural leaf shared-win estimate must be finite and in [0, 1]".to_string());
    }
    if !victory_point_margin.is_finite() {
        return Err("neural leaf VP-margin estimate must be finite".to_string());
    }
    if root_player > 1 || evaluation_player > 1 {
        return Err("two-player neural perspective index is out of bounds".to_string());
    }
    Ok(if evaluation_player == root_player {
        (shared_win_rate, victory_point_margin)
    } else {
        (1.0 - shared_win_rate, -victory_point_margin)
    })
}

pub fn aggregate_two_player_action_values(
    batch: &RootActionSuccessorBatch,
    model_id: String,
    shared_win_rates: &[f64],
    victory_point_margins: &[f64],
) -> Result<RootActionValueEvaluation, String> {
    if batch.num_players != 2 {
        return Err(format!(
            "scalar perspective conversion requires exactly two players, got {}",
            batch.num_players
        ));
    }
    if batch.root_player >= batch.num_players {
        return Err("successor batch root player is out of bounds".to_string());
    }
    if shared_win_rates.len() != batch.states.len()
        || victory_point_margins.len() != batch.states.len()
    {
        return Err(format!(
            "received {} shared-win and {} VP-margin predictions for {} successor states",
            shared_win_rates.len(),
            victory_point_margins.len(),
            batch.states.len()
        ));
    }
    for (state_index, (shared_win_rate, victory_point_margin)) in shared_win_rates
        .iter()
        .zip(victory_point_margins)
        .enumerate()
    {
        if !shared_win_rate.is_finite() || !(0.0..=1.0).contains(shared_win_rate) {
            return Err(format!(
                "successor shared-win prediction at state {state_index} is invalid"
            ));
        }
        if !victory_point_margin.is_finite() {
            return Err(format!(
                "successor VP-margin prediction at state {state_index} is invalid"
            ));
        }
    }

    let action_count = batch.action_keys.len();
    let mut win_sums = vec![0.0; action_count];
    let mut win_square_sums = vec![0.0; action_count];
    let mut margin_sums = vec![0.0; action_count];
    let mut sample_counts = vec![0u64; action_count];
    for sample in &batch.samples {
        if sample.action_index >= action_count
            || sample.action_key != batch.action_keys[sample.action_index]
        {
            return Err("successor sample action identity is inconsistent".to_string());
        }
        if sample.sample_index >= batch.determinizations_per_action {
            return Err("successor sample index is out of bounds".to_string());
        }
        let (root_shared_win_rate, root_victory_point_margin) = if let Some(state_index) =
            sample.state_index
        {
            let state = batch
                .states
                .get(state_index)
                .ok_or_else(|| "successor sample state index is out of bounds".to_string())?;
            if state.observer_idx != sample.evaluation_player
                || sample.evaluation_player >= batch.num_players
            {
                return Err("successor sample evaluation player is inconsistent".to_string());
            }
            if sample.terminal_root_shared_win_rate.is_some()
                || sample.terminal_root_victory_point_margin.is_some()
            {
                return Err("non-terminal successor sample contains terminal values".to_string());
            }
            let value = shared_win_rates[state_index];
            let margin = victory_point_margins[state_index];
            if sample.evaluation_player == batch.root_player {
                (value, margin)
            } else {
                (1.0 - value, -margin)
            }
        } else {
            match (
                sample.terminal_root_shared_win_rate,
                sample.terminal_root_victory_point_margin,
            ) {
                (Some(value), Some(margin)) => (value, margin),
                _ => {
                    return Err("terminal successor sample is missing exact root values".to_string())
                }
            }
        };
        let index = sample.action_index;
        win_sums[index] += root_shared_win_rate;
        win_square_sums[index] += root_shared_win_rate * root_shared_win_rate;
        margin_sums[index] += root_victory_point_margin;
        sample_counts[index] += 1;
    }

    let expected_samples = batch.determinizations_per_action as u64;
    if sample_counts.iter().any(|count| *count != expected_samples) {
        return Err(format!(
            "successor batch must contain {expected_samples} samples per action"
        ));
    }
    let shared_win_rates = win_sums
        .iter()
        .zip(&sample_counts)
        .map(|(sum, count)| sum / *count as f64)
        .collect::<Vec<_>>();
    let victory_point_margins = margin_sums
        .iter()
        .zip(&sample_counts)
        .map(|(sum, count)| sum / *count as f64)
        .collect::<Vec<_>>();
    let shared_win_standard_errors = win_sums
        .iter()
        .zip(&win_square_sums)
        .zip(&sample_counts)
        .map(|((sum, square_sum), count)| sample_standard_error(*sum, *square_sum, *count))
        .collect();

    Ok(RootActionValueEvaluation {
        model_id,
        action_keys: batch.action_keys.clone(),
        shared_win_rates,
        victory_point_margins,
        shared_win_standard_errors,
        sample_counts,
    })
}

/// Evaluates complete root actions with hidden-hand determinization and root UCB.
///
/// The returned win estimates come from random continuations. They are useful as
/// a deterministic baseline, but are neither calibrated probabilities nor a
/// learned policy.
pub fn search_top_actions(
    runner: &GameRunner,
    config: &RootSearchConfig,
) -> Result<RootSearchReport, String> {
    search_top_actions_internal(runner, config, None, None)
}

/// Uses learned root action priors with PUCT while retaining hidden-hand
/// determinization and complete random continuations for outcome estimates.
pub fn search_top_actions_with_policy(
    runner: &GameRunner,
    config: &RootSearchConfig,
    policy: &RootPolicyEvaluation,
) -> Result<RootSearchReport, String> {
    search_top_actions_internal(runner, config, Some(policy), None)
}

/// Uses policy priors and batched one-step successor values for two-player PUCT.
/// The scalar next-actor values must already be aggregated into root perspective.
pub fn search_top_actions_with_policy_and_action_values(
    runner: &GameRunner,
    config: &RootSearchConfig,
    policy: &RootPolicyEvaluation,
    action_values: &RootActionValueEvaluation,
) -> Result<RootSearchReport, String> {
    search_top_actions_internal(runner, config, Some(policy), Some(action_values))
}

fn search_top_actions_internal(
    runner: &GameRunner,
    config: &RootSearchConfig,
    policy: Option<&RootPolicyEvaluation>,
    action_values: Option<&RootActionValueEvaluation>,
) -> Result<RootSearchReport, String> {
    config.validate()?;
    if runner.is_game_finished() {
        return Err("cannot search a finished game".to_string());
    }

    let root_player = runner.framework.current_player;
    let actions = enumerate_legal_actions(runner)?;
    if actions.is_empty() {
        return Err("no legal root actions available for search".to_string());
    }
    let policy_probabilities = policy
        .map(|evaluation| evaluation.normalized_probabilities(&actions))
        .transpose()?;
    if let Some(values) = action_values {
        if runner.framework.board.state.players.len() != 2 {
            return Err(
                "scalar successor action values are only valid for two-player search".to_string(),
            );
        }
        let policy = policy.ok_or_else(|| {
            "successor action-value search requires a root policy evaluation".to_string()
        })?;
        values.validate(&actions, &policy.model_id)?;
    }

    let immediate_effects = actions
        .iter()
        .map(|action| immediate_effect_for_action(runner, action))
        .collect::<Result<Vec<_>, _>>()?;
    let mut stats = vec![CandidateStats::default(); actions.len()];
    let mut rng = StdRng::seed_from_u64(config.seed);
    if let Some(values) = action_values {
        let probabilities = policy_probabilities
            .as_ref()
            .expect("action-value search validated a policy above");
        for simulation_number in 0..config.simulations {
            let candidate_idx = select_puct_candidate_with_action_values(
                &stats,
                simulation_number,
                config.exploration_constant,
                probabilities,
                &values.shared_win_rates,
            );
            stats[candidate_idx].visits += 1;
        }
    } else {
        let mut unvisited_order = (0..actions.len()).collect::<Vec<_>>();
        if let Some(probabilities) = &policy_probabilities {
            unvisited_order.sort_by(|left, right| {
                probabilities[*right]
                    .total_cmp(&probabilities[*left])
                    .then_with(|| actions[*left].key().cmp(&actions[*right].key()))
            });
        } else {
            unvisited_order.shuffle(&mut rng);
        }
        let mut next_unvisited = 0usize;

        for simulation_number in 0..config.simulations {
            let candidate_idx = if next_unvisited < unvisited_order.len() {
                let index = unvisited_order[next_unvisited];
                next_unvisited += 1;
                index
            } else {
                match &policy_probabilities {
                    Some(probabilities) => select_puct_candidate(
                        &stats,
                        simulation_number,
                        config.exploration_constant,
                        probabilities,
                    ),
                    None => {
                        select_ucb_candidate(&stats, simulation_number, config.exploration_constant)
                    }
                }
            };

            let mut simulation = runner.clone();
            determinize_hidden_information(&mut simulation, root_player, &mut rng)?;
            actions[candidate_idx]
                .apply(&mut simulation)
                .map_err(|error| {
                    format!(
                        "root action {} failed after determinization: {error}",
                        actions[candidate_idx].key()
                    )
                })?;

            let continuation_steps = random_rollout_to_end(&mut simulation, config, &mut rng)?;
            let outcome = evaluate_finished_game(&simulation, root_player)?;
            let candidate_stats = &mut stats[candidate_idx];
            candidate_stats.visits += 1;
            candidate_stats.shared_win_sum += outcome.shared_win_credit;
            candidate_stats.shared_win_square_sum +=
                outcome.shared_win_credit * outcome.shared_win_credit;
            candidate_stats.outright_wins += u64::from(outcome.outright_win);
            candidate_stats.tied_firsts += u64::from(outcome.tied_first);
            candidate_stats.final_victory_points_sum +=
                outcome.final_victory_points[root_player] as f64;
            candidate_stats.victory_point_margin_sum += outcome.victory_point_margin as f64;
            if candidate_stats.sample_random_continuation.is_none() {
                candidate_stats.sample_random_continuation = Some(SampleRandomContinuation {
                    steps: continuation_steps,
                    final_victory_points: outcome.final_victory_points,
                    official_winners: outcome.official_winners,
                });
            }
        }
    }

    let visited_action_count = stats
        .iter()
        .filter(|candidate| candidate.visits > 0)
        .count();
    let evaluated_action_count = if action_values.is_some() {
        actions.len()
    } else {
        visited_action_count
    };
    let mut ranked_indices = stats
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| (candidate.visits > 0).then_some(index))
        .collect::<Vec<_>>();
    ranked_indices.sort_by(|left, right| {
        stats[*right]
            .visits
            .cmp(&stats[*left].visits)
            .then_with(|| match action_values {
                Some(values) => {
                    values.shared_win_rates[*right].total_cmp(&values.shared_win_rates[*left])
                }
                None => mean_shared_win(&stats[*right]).total_cmp(&mean_shared_win(&stats[*left])),
            })
            .then_with(|| actions[*left].key().cmp(&actions[*right].key()))
    });

    let mut recommendations = Vec::new();
    for (rank_index, candidate_idx) in ranked_indices
        .into_iter()
        .take(config.recommendation_count)
        .enumerate()
    {
        let estimate = match action_values {
            Some(values) => {
                let sample = if config.sample_continuation_length == 0 {
                    SampleRandomContinuation {
                        steps: Vec::new(),
                        final_victory_points: Vec::new(),
                        official_winners: Vec::new(),
                    }
                } else {
                    sample_random_continuation_for_action(
                        runner,
                        &actions[candidate_idx],
                        root_player,
                        config,
                        &mut rng,
                    )?
                };
                estimate_from_action_value(
                    rank_index + 1,
                    &actions[candidate_idx],
                    &immediate_effects[candidate_idx],
                    &stats[candidate_idx],
                    config.simulations,
                    policy_probabilities
                        .as_ref()
                        .map(|probabilities| probabilities[candidate_idx]),
                    values.shared_win_rates[candidate_idx],
                    values.victory_point_margins[candidate_idx],
                    values.shared_win_standard_errors[candidate_idx],
                    values.sample_counts[candidate_idx],
                    sample,
                )
            }
            None => estimate_from_stats(
                rank_index + 1,
                &actions[candidate_idx],
                &immediate_effects[candidate_idx],
                &stats[candidate_idx],
                config.simulations,
                policy_probabilities
                    .as_ref()
                    .map(|probabilities| probabilities[candidate_idx]),
            ),
        };
        recommendations.push(estimate);
    }

    Ok(RootSearchReport {
        method: if action_values.is_some() {
            ROOT_PUCT_ACTION_VALUE_METHOD.to_string()
        } else if policy.is_some() {
            ROOT_PUCT_METHOD.to_string()
        } else {
            ROOT_UCB_METHOD.to_string()
        },
        value_source: if action_values.is_some() {
            SUCCESSOR_MODEL_VALUE_SOURCE.to_string()
        } else {
            RANDOM_ROLLOUT_VALUE_SOURCE.to_string()
        },
        model_id: policy.map(|evaluation| evaluation.model_id.clone()),
        root_model_shared_win_rate: policy.map(|evaluation| evaluation.shared_win_rate),
        root_model_victory_point_margin: policy.map(|evaluation| evaluation.victory_point_margin),
        root_policy_probabilities: policy_probabilities,
        root_player,
        requested_simulations: config.simulations,
        completed_simulations: stats.iter().map(|candidate| candidate.visits).sum(),
        root_action_count: actions.len(),
        evaluated_action_count,
        visited_action_count,
        all_root_actions_evaluated: evaluated_action_count == actions.len(),
        max_search_depth: None,
        neural_leaf_evaluations: None,
        inference_batches: None,
        root_action_model_shared_win_rates: action_values
            .map(|values| values.shared_win_rates.clone()),
        root_action_model_victory_point_margins: action_values
            .map(|values| values.victory_point_margins.clone()),
        root_action_model_shared_win_standard_errors: action_values
            .map(|values| values.shared_win_standard_errors.clone()),
        root_action_model_sample_counts: action_values.map(|values| values.sample_counts.clone()),
        recommendations,
    })
}

/// Returns every player tied for first after the official VP, income-level, and
/// money tie breakers.
pub fn official_winners(runner: &GameRunner) -> Result<Vec<usize>, String> {
    if !runner.is_game_finished() {
        return Err("official winners are only available after game end".to_string());
    }
    let players = &runner.framework.board.state.players;
    let best_key = players
        .iter()
        .map(|player| (player.victory_points, player.income_level, player.money))
        .max()
        .ok_or_else(|| "cannot rank a game with no players".to_string())?;
    Ok(players
        .iter()
        .enumerate()
        .filter_map(|(player_idx, player)| {
            ((player.victory_points, player.income_level, player.money) == best_key)
                .then_some(player_idx)
        })
        .collect())
}

fn select_ucb_candidate(
    stats: &[CandidateStats],
    total_visits: u64,
    exploration_constant: f64,
) -> usize {
    debug_assert!(stats.iter().all(|candidate| candidate.visits > 0));
    let log_total = (total_visits as f64).ln();
    let mut best_index = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (candidate_idx, candidate) in stats.iter().enumerate() {
        let exploitation = mean_shared_win(candidate);
        let exploration = exploration_constant * (log_total / candidate.visits as f64).sqrt();
        let score = exploitation + exploration;
        if score > best_score {
            best_score = score;
            best_index = candidate_idx;
        }
    }
    best_index
}

fn select_puct_candidate(
    stats: &[CandidateStats],
    total_visits: u64,
    exploration_constant: f64,
    policy_probabilities: &[f64],
) -> usize {
    debug_assert_eq!(stats.len(), policy_probabilities.len());
    debug_assert!(stats.iter().all(|candidate| candidate.visits > 0));
    let sqrt_total = (total_visits.max(1) as f64).sqrt();
    let mut best_index = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (candidate_idx, candidate) in stats.iter().enumerate() {
        let exploitation = mean_shared_win(candidate);
        let exploration = exploration_constant * policy_probabilities[candidate_idx] * sqrt_total
            / (1.0 + candidate.visits as f64);
        let score = exploitation + exploration;
        if score > best_score {
            best_score = score;
            best_index = candidate_idx;
        }
    }
    best_index
}

fn select_puct_candidate_with_action_values(
    stats: &[CandidateStats],
    total_visits: u64,
    exploration_constant: f64,
    policy_probabilities: &[f64],
    action_shared_win_rates: &[f64],
) -> usize {
    debug_assert_eq!(stats.len(), policy_probabilities.len());
    debug_assert_eq!(stats.len(), action_shared_win_rates.len());
    let sqrt_total = (total_visits + 1) as f64;
    let sqrt_total = sqrt_total.sqrt();
    let mut best_index = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (candidate_idx, candidate) in stats.iter().enumerate() {
        let exploration = exploration_constant * policy_probabilities[candidate_idx] * sqrt_total
            / (1.0 + candidate.visits as f64);
        let score = action_shared_win_rates[candidate_idx] + exploration;
        if score > best_score {
            best_score = score;
            best_index = candidate_idx;
        }
    }
    best_index
}

fn mean_shared_win(stats: &CandidateStats) -> f64 {
    if stats.visits == 0 {
        0.0
    } else {
        stats.shared_win_sum / stats.visits as f64
    }
}

fn estimate_from_stats(
    rank: usize,
    action: &LegalAction,
    immediate_effect: &ImmediateEffect,
    stats: &CandidateStats,
    total_simulations: u64,
    policy_probability: Option<f64>,
) -> RootActionEstimate {
    let visits = stats.visits as f64;
    RootActionEstimate {
        rank,
        action_key: action.key(),
        action: action.clone(),
        visits: stats.visits,
        visit_share: visits / total_simulations as f64,
        value_source: RANDOM_ROLLOUT_VALUE_SOURCE.to_string(),
        value_sample_count: stats.visits,
        estimated_shared_win_rate: stats.shared_win_sum / visits,
        estimated_outright_win_rate: Some(stats.outright_wins as f64 / visits),
        estimated_tied_first_rate: Some(stats.tied_firsts as f64 / visits),
        average_final_victory_points: Some(stats.final_victory_points_sum / visits),
        average_victory_point_margin: stats.victory_point_margin_sum / visits,
        shared_win_rate_standard_error: sample_standard_error(
            stats.shared_win_sum,
            stats.shared_win_square_sum,
            stats.visits,
        ),
        policy_probability,
        calibrated_win_rate: None,
        immediate_effect: immediate_effect.clone(),
        sample_random_continuation: stats
            .sample_random_continuation
            .clone()
            .expect("visited candidates always store one sample continuation"),
    }
}

#[allow(clippy::too_many_arguments)]
fn estimate_from_action_value(
    rank: usize,
    action: &LegalAction,
    immediate_effect: &ImmediateEffect,
    stats: &CandidateStats,
    total_simulations: u64,
    policy_probability: Option<f64>,
    shared_win_rate: f64,
    victory_point_margin: f64,
    shared_win_standard_error: Option<f64>,
    value_sample_count: u64,
    sample_random_continuation: SampleRandomContinuation,
) -> RootActionEstimate {
    RootActionEstimate {
        rank,
        action_key: action.key(),
        action: action.clone(),
        visits: stats.visits,
        visit_share: stats.visits as f64 / total_simulations as f64,
        value_source: SUCCESSOR_MODEL_VALUE_SOURCE.to_string(),
        value_sample_count,
        estimated_shared_win_rate: shared_win_rate,
        estimated_outright_win_rate: None,
        estimated_tied_first_rate: None,
        average_final_victory_points: None,
        average_victory_point_margin: victory_point_margin,
        shared_win_rate_standard_error: shared_win_standard_error,
        policy_probability,
        calibrated_win_rate: None,
        immediate_effect: immediate_effect.clone(),
        sample_random_continuation,
    }
}

fn sample_standard_error(sum: f64, square_sum: f64, count: u64) -> Option<f64> {
    if count < 2 {
        return None;
    }
    let count_f64 = count as f64;
    let corrected_sum_of_squares = (square_sum - sum * sum / count_f64).max(0.0);
    let sample_variance = corrected_sum_of_squares / (count_f64 - 1.0);
    Some((sample_variance / count_f64).sqrt())
}

fn sample_random_continuation_for_action(
    runner: &GameRunner,
    action: &LegalAction,
    root_player: usize,
    config: &RootSearchConfig,
    rng: &mut StdRng,
) -> Result<SampleRandomContinuation, String> {
    let mut simulation = runner.clone();
    determinize_hidden_information(&mut simulation, root_player, rng)?;
    action.apply(&mut simulation).map_err(|error| {
        format!(
            "recommended action {} failed in sample continuation: {error}",
            action.key()
        )
    })?;
    let steps = random_rollout_to_end(&mut simulation, config, rng)?;
    let outcome = evaluate_finished_game(&simulation, root_player)?;
    Ok(SampleRandomContinuation {
        steps,
        final_victory_points: outcome.final_victory_points,
        official_winners: outcome.official_winners,
    })
}

fn random_rollout_to_end(
    runner: &mut GameRunner,
    config: &RootSearchConfig,
    rng: &mut StdRng,
) -> Result<Vec<SampleContinuationStep>, String> {
    let mut action_count = 0u32;
    let mut sample_steps = Vec::new();
    while !runner.is_game_finished() {
        resolve_random_shortfalls(runner, rng);
        if runner.is_game_finished() {
            break;
        }
        if action_count >= config.max_rollout_actions {
            return Err(format!(
                "random rollout exceeded {} actions at turn {}",
                config.max_rollout_actions, runner.turn_count
            ));
        }

        let player_idx = runner.framework.current_player;
        let action = play_random_staged_action(runner, config.sell_stop_probability, rng)?;
        action_count += 1;
        if sample_steps.len() < config.sample_continuation_length {
            sample_steps.push(SampleContinuationStep {
                action_number: action_count,
                player_idx,
                action_key: action.key(),
                action_type: action.intent.action_type,
                intent: action.intent.clone(),
            });
        }

        if runner.actions_remaining_in_turn == 0 {
            runner.end_turn();
        }
    }
    resolve_random_shortfalls(runner, rng);
    Ok(sample_steps)
}

fn resolve_random_shortfalls(runner: &mut GameRunner, rng: &mut StdRng) {
    for session in runner.take_shortfall_sessions() {
        let mut locations = session
            .removable_tiles
            .iter()
            .map(|tile| tile.build_location_idx)
            .collect::<Vec<_>>();
        locations.shuffle(rng);
        runner.resolve_shortfall_with_tiles(session, locations);
    }
}

fn play_random_staged_action(
    runner: &mut GameRunner,
    sell_stop_probability: f64,
    rng: &mut StdRng,
) -> Result<LegalAction, String> {
    let roots = runner.start_turn();
    let root = *roots
        .choose(rng)
        .ok_or_else(|| random_action_error(runner, "empty root action set"))?;
    runner.framework.start_action_session(root);

    let mut choices = Vec::new();
    let mut used_card_indices = Vec::new();
    for choice_number in 0..64usize {
        let session = runner
            .framework
            .current_session()
            .ok_or_else(|| random_action_error(runner, "action session disappeared"))?;
        let choice_set = runner
            .framework
            .get_next_choice_set()
            .ok_or_else(|| random_action_error(runner, "missing staged choice set"))?;

        let stop_sell = root == ActionType::Sell
            && session.can_confirm
            && !matches!(choice_set, ChoiceSet::ConfirmOnly)
            && rng.gen_bool(sell_stop_probability);
        if stop_sell || matches!(choice_set, ChoiceSet::ConfirmOnly) {
            if !runner.framework.can_confirm() {
                return Err(random_action_error(
                    runner,
                    "confirmation was exposed before the action was complete",
                ));
            }
            let intent = session.intent;
            runner.confirm_action()?;
            choices.push(ActionChoice::Confirm);
            return Ok(LegalAction {
                root,
                choices,
                intent,
            });
        }

        let choice = random_choice(choice_set, &used_card_indices, rng)?;
        if let ActionChoice::Card(card_idx) = choice {
            used_card_indices.push(card_idx);
        }
        let before = session;
        let after = runner
            .framework
            .apply_action_choice(choice.clone())?
            .ok_or_else(|| random_action_error(runner, "staged choice ended session early"))?;
        if before.intent == after.intent
            && before.next_choices == after.next_choices
            && before.can_confirm == after.can_confirm
        {
            return Err(random_action_error(
                runner,
                &format!("staged choice {choice_number} did not advance the action"),
            ));
        }
        choices.push(choice);
    }

    Err(random_action_error(
        runner,
        "action exceeded 64 staged choices",
    ))
}

fn random_choice(
    set: ChoiceSet,
    used_card_indices: &[usize],
    rng: &mut StdRng,
) -> Result<ActionChoice, String> {
    let choice = match set {
        ChoiceSet::Industry(values) => ActionChoice::Industry(pick(&values, rng, "industry")?),
        ChoiceSet::Card(values) => {
            let available = values
                .into_iter()
                .filter(|card_idx| !used_card_indices.contains(card_idx))
                .collect::<Vec<_>>();
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
        ChoiceSet::ConfirmOnly => {
            return Err("ConfirmOnly must be handled before random choice selection".to_string())
        }
    };
    Ok(choice)
}

fn pick<T: Clone>(values: &[T], rng: &mut StdRng, label: &str) -> Result<T, String> {
    values
        .choose(rng)
        .cloned()
        .ok_or_else(|| format!("empty random choice set: {label}"))
}

fn random_action_error(runner: &GameRunner, message: &str) -> String {
    format!(
        "{message}; phase {:?}, turn {}, player {}, actions remaining {}",
        runner.game_phase,
        runner.turn_count,
        runner.framework.current_player,
        runner.actions_remaining_in_turn
    )
}

fn evaluate_finished_game(
    runner: &GameRunner,
    root_player: usize,
) -> Result<RolloutOutcome, String> {
    let winners = official_winners(runner)?;
    let final_victory_points = runner
        .framework
        .board
        .state
        .players
        .iter()
        .map(|player| player.victory_points)
        .collect::<Vec<_>>();
    if root_player >= final_victory_points.len() {
        return Err(format!(
            "root player {root_player} out of bounds for {} final scores",
            final_victory_points.len()
        ));
    }
    let root_is_winner = winners.contains(&root_player);
    let best_opponent_vp = final_victory_points
        .iter()
        .enumerate()
        .filter_map(|(player_idx, score)| (player_idx != root_player).then_some(*score))
        .max()
        .unwrap_or(final_victory_points[root_player]);

    Ok(RolloutOutcome {
        shared_win_credit: if root_is_winner {
            1.0 / winners.len() as f64
        } else {
            0.0
        },
        outright_win: root_is_winner && winners.len() == 1,
        tied_first: root_is_winner && winners.len() > 1,
        victory_point_margin: final_victory_points[root_player] as i32 - best_opponent_vp as i32,
        final_victory_points,
        official_winners: winners,
    })
}

fn immediate_effect_for_action(
    runner: &GameRunner,
    action: &LegalAction,
) -> Result<ImmediateEffect, String> {
    let before = runner;
    let mut after = runner.clone();
    action.apply(&mut after)?;
    let before_state = &before.framework.board.state;
    let after_state = &after.framework.board.state;
    let before_potential_vps = potential_era_victory_points(before);
    let after_potential_vps = potential_era_victory_points(&after);

    Ok(ImmediateEffect {
        player_money_delta: zip_player_delta(before, &after, |player| player.money as i32),
        player_income_level_delta: before_state
            .players
            .iter()
            .zip(&after_state.players)
            .map(|(left, right)| right.income_level as i16 - left.income_level as i16)
            .collect(),
        player_victory_points_delta: zip_player_delta(before, &after, |player| {
            player.victory_points as i32
        }),
        player_visible_victory_points_delta: (0..before_state.players.len())
            .map(|player_idx| {
                after_state.visible_vps[player_idx] as i32
                    - before_state.visible_vps[player_idx] as i32
            })
            .collect(),
        player_potential_era_victory_points_delta: before_potential_vps
            .iter()
            .zip(after_potential_vps)
            .map(|(left, right)| right as i32 - *left as i32)
            .collect(),
        player_hand_size_delta: before_state
            .players
            .iter()
            .zip(&after_state.players)
            .map(|(left, right)| right.hand.cards.len() as i16 - left.hand.cards.len() as i16)
            .collect(),
        player_round_spend_delta: zip_player_delta(before, &after, |player| {
            player.spent_this_turn as i32
        }),
        roads_on_board_delta: after_state.built_roads.ones().count() as i32
            - before_state.built_roads.ones().count() as i32,
        buildings_on_board_delta: after_state.bl_to_building.len() as i32
            - before_state.bl_to_building.len() as i32,
        flipped_buildings_delta: flipped_building_count(&after) as i32
            - flipped_building_count(before) as i32,
        draw_deck_size_delta: after_state.deck.cards.len() as i32
            - before_state.deck.cards.len() as i32,
        market_coal_delta: after_state.remaining_market_coal as i16
            - before_state.remaining_market_coal as i16,
        market_iron_delta: after_state.remaining_market_iron as i16
            - before_state.remaining_market_iron as i16,
        wild_location_pool_delta: after_state.wild_location_cards_available as i16
            - before_state.wild_location_cards_available as i16,
        wild_industry_pool_delta: after_state.wild_industry_cards_available as i16
            - before_state.wild_industry_cards_available as i16,
        current_player_before: before.framework.current_player,
        current_player_after: after.framework.current_player,
        actions_remaining_before: before.actions_remaining_in_turn,
        actions_remaining_after: after.actions_remaining_in_turn,
        turn_count_delta: after.turn_count as i64 - before.turn_count as i64,
        phase_before: before.game_phase,
        phase_after: after.game_phase,
        era_before: before_state.era,
        era_after: after_state.era,
    })
}

fn zip_player_delta<F>(before: &GameRunner, after: &GameRunner, value: F) -> Vec<i32>
where
    F: Fn(&crate::core::player::Player) -> i32,
{
    before
        .framework
        .board
        .state
        .players
        .iter()
        .zip(&after.framework.board.state.players)
        .map(|(left, right)| value(right) - value(left))
        .collect()
}

fn flipped_building_count(runner: &GameRunner) -> usize {
    runner
        .framework
        .board
        .state
        .bl_to_building
        .values()
        .filter(|building| building.flipped)
        .count()
}

fn potential_era_victory_points(runner: &GameRunner) -> Vec<u16> {
    let state = &runner.framework.board.state;
    let mut potential_vps = vec![0u16; state.players.len()];
    for road_idx in state.built_roads.ones() {
        let owner = (0..state.players.len())
            .find(|player_idx| state.player_road_mask[*player_idx].contains(road_idx));
        let Some(owner_idx) = owner else {
            continue;
        };
        for location_idx in LINK_LOCATIONS[road_idx].locations.ones() {
            for building_idx in LocationName::from_usize(location_idx).to_bl_set().ones() {
                if let Some(building) = state
                    .bl_to_building
                    .get(&building_idx)
                    .filter(|building| building.flipped)
                {
                    potential_vps[owner_idx] = potential_vps[owner_idx].saturating_add(
                        INDUSTRY_MAT[building.industry as usize][building.level.as_usize()].road_vp
                            as u16,
                    );
                }
            }
        }
    }
    for building in state
        .bl_to_building
        .values()
        .filter(|building| building.flipped)
    {
        let owner_idx = building.owner.as_usize();
        potential_vps[owner_idx] = potential_vps[owner_idx].saturating_add(
            INDUSTRY_MAT[building.industry as usize][building.level.as_usize()].vp_on_flip as u16,
        );
    }
    potential_vps
}

fn two_player_score_progress(
    runner: &GameRunner,
    root_player: usize,
) -> Result<(f64, f64), String> {
    let state = &runner.framework.board.state;
    if state.players.len() != 2 || root_player >= 2 {
        return Err("score progress requires a valid two-player root".to_string());
    }
    let potential = potential_era_victory_points(runner);
    let score = |player_idx: usize| {
        f64::from(state.players[player_idx].victory_points) + f64::from(potential[player_idx])
    };
    Ok((score(root_player), score(1 - root_player)))
}

fn two_player_predicted_scores(
    runner: &GameRunner,
    root_player: usize,
    evaluation_player: usize,
    actor_victory_points: f64,
    actor_victory_point_margin: f64,
) -> Result<(f64, f64), String> {
    if !actor_victory_points.is_finite() || actor_victory_points < 0.0 {
        return Err("predicted actor victory points must be finite and non-negative".to_string());
    }
    if !actor_victory_point_margin.is_finite() {
        return Err("predicted victory-point margin must be finite".to_string());
    }
    if evaluation_player >= 2 {
        return Err("predicted score requires a valid evaluation player".to_string());
    }

    let actor_score = actor_victory_points;
    let other_score = (actor_victory_points - actor_victory_point_margin).max(0.0);
    let (predicted_root, predicted_opponent) = if evaluation_player == root_player {
        (actor_score, other_score)
    } else {
        (other_score, actor_score)
    };
    let (secured_root, secured_opponent) = two_player_score_progress(runner, root_player)?;
    Ok((
        predicted_root.max(secured_root),
        predicted_opponent.max(secured_opponent),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_error_is_zero_for_constant_fractional_ties() {
        assert_eq!(sample_standard_error(1.5, 0.75, 3), Some(0.0));
    }

    #[test]
    fn score_utility_distinguishes_low_and_high_scoring_ties() {
        assert_eq!(combined_actor_utility(0.5, 0.0, 0.0, 0.0), 0.5);
        assert!(
            combined_actor_utility(0.5, 140.0, 0.0, 0.2)
                > combined_actor_utility(0.5, 0.0, 0.0, 0.2)
        );
        assert!(
            combined_actor_utility(0.5, 50.0, 40.0, 0.2)
                > combined_actor_utility(0.5, 50.0, -40.0, 0.2)
        );
        assert!(
            combined_actor_utility(1.0, 10.0, 10.0, 0.2)
                > combined_actor_utility(0.0, 140.0, -10.0, 0.2)
        );
    }

    #[test]
    fn official_winners_apply_all_rulebook_tie_breakers() {
        let mut runner = GameRunner::new(3, Some(77));
        runner.game_phase = GamePhase::GameEnd;
        runner.framework.board.state.players[0].victory_points = 100;
        runner.framework.board.state.players[1].victory_points = 100;
        runner.framework.board.state.players[2].victory_points = 99;
        runner.framework.board.state.players[0].income_level = 20;
        runner.framework.board.state.players[1].income_level = 21;
        assert_eq!(official_winners(&runner).unwrap(), vec![1]);

        runner.framework.board.state.players[0].income_level = 21;
        runner.framework.board.state.players[0].money = 30;
        runner.framework.board.state.players[1].money = 29;
        assert_eq!(official_winners(&runner).unwrap(), vec![0]);

        runner.framework.board.state.players[1].money = 30;
        assert_eq!(official_winners(&runner).unwrap(), vec![0, 1]);
    }
}
