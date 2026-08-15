use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::game::legal_actions::{enumerate_legal_actions, LegalAction};
use crate::game::runner::GameRunner;
use crate::game::search::{NeuralLeafEvaluation, NeuralLeafRequest, RootPolicyEvaluation};
use crate::game::training::{
    encode_training_action, encode_training_state, training_feature_schema, TrainingFeatureSchema,
    TRAINING_FEATURE_VERSION,
};

const INFERENCE_URL_ENV: &str = "FAST_BRASS_INFERENCE_URL";

#[derive(Clone)]
pub struct ModelInferenceClient {
    endpoint: String,
    batch_endpoint: String,
    client: Client,
}

impl ModelInferenceClient {
    pub fn from_env() -> Result<Option<Self>, String> {
        let Ok(base_url) = std::env::var(INFERENCE_URL_ENV) else {
            return Ok(None);
        };
        let base_url = base_url.trim();
        if base_url.is_empty() {
            return Err(format!("{INFERENCE_URL_ENV} must not be empty"));
        }
        let endpoint = format!("{}/evaluate", base_url.trim_end_matches('/'));
        let batch_endpoint = format!("{}/evaluate-batch", base_url.trim_end_matches('/'));
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(90))
            .build()
            .map_err(|error| format!("failed to build inference HTTP client: {error}"))?;
        Ok(Some(Self {
            endpoint,
            batch_endpoint,
            client,
        }))
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub async fn evaluate(&self, runner: &GameRunner) -> Result<RootPolicyEvaluation, String> {
        let request = build_inference_request(runner)?;
        let response = self
            .client
            .post(&self.endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|error| format!("model inference request failed: {error}"))?;
        let status = response.status();
        let payload = response
            .json::<InferenceResponse>()
            .await
            .map_err(|error| format!("model inference returned invalid JSON: {error}"))?;
        if !status.is_success() || !payload.ok {
            return Err(format!(
                "model inference failed with {status}: {}",
                payload.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }
        Ok(RootPolicyEvaluation {
            model_id: payload
                .model_id
                .ok_or_else(|| "model inference response omitted model_id".to_string())?,
            action_keys: payload
                .action_keys
                .ok_or_else(|| "model inference response omitted action_keys".to_string())?,
            policy_probabilities: payload.policy_probabilities.ok_or_else(|| {
                "model inference response omitted policy_probabilities".to_string()
            })?,
            shared_win_rate: payload
                .shared_win_rate
                .ok_or_else(|| "model inference response omitted shared_win_rate".to_string())?,
            victory_point_margin: payload.victory_point_margin.ok_or_else(|| {
                "model inference response omitted victory_point_margin".to_string()
            })?,
        })
    }

    pub async fn evaluate_leaf_batch(
        &self,
        leaves: &[NeuralLeafRequest],
        expected_model_id: &str,
    ) -> Result<Vec<NeuralLeafEvaluation>, String> {
        let request = build_batch_inference_request(leaves)?;
        let response = self
            .client
            .post(&self.batch_endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|error| format!("batched policy/value inference request failed: {error}"))?;
        let status = response.status();
        let payload = response
            .json::<InferenceBatchResponse>()
            .await
            .map_err(|error| {
                format!("batched policy/value inference returned invalid JSON: {error}")
            })?;
        if !status.is_success() || !payload.ok {
            return Err(format!(
                "batched policy/value inference failed with {status}: {}",
                payload.error.unwrap_or_else(|| "unknown error".to_string())
            ));
        }
        let model_id = payload
            .model_id
            .ok_or_else(|| "batched policy/value response omitted model_id".to_string())?;
        if model_id != expected_model_id {
            return Err(format!(
                "root policy model {expected_model_id} and batched leaf model {model_id} disagree"
            ));
        }
        let evaluations = payload
            .evaluations
            .ok_or_else(|| "batched policy/value response omitted evaluations".to_string())?;
        if evaluations.len() != leaves.len() {
            return Err(format!(
                "received {} batched leaf evaluations for {} requests",
                evaluations.len(),
                leaves.len()
            ));
        }
        Ok(evaluations
            .into_iter()
            .map(|evaluation| NeuralLeafEvaluation {
                request_id: evaluation.request_id,
                model_id: model_id.clone(),
                action_keys: evaluation.action_keys,
                policy_probabilities: evaluation.policy_probabilities,
                shared_win_rate: evaluation.shared_win_rate,
                victory_point_margin: evaluation.victory_point_margin,
            })
            .collect())
    }
}

#[derive(Debug, Serialize)]
struct InferenceRequest {
    feature_schema: TrainingFeatureSchema,
    state: InferenceState,
    legal_actions: InferenceLegalActions,
}

#[derive(Debug, Serialize)]
struct InferenceBatchRequest {
    feature_schema: TrainingFeatureSchema,
    positions: Vec<InferenceBatchPosition>,
}

#[derive(Debug, Serialize)]
struct InferenceBatchPosition {
    request_id: u64,
    state: InferenceState,
    legal_actions: InferenceLegalActions,
}

#[derive(Debug, Serialize)]
struct InferenceState {
    feature_version: u32,
    features: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct InferenceLegalActions {
    feature_version: u32,
    actions: Vec<InferenceAction>,
}

#[derive(Debug, Serialize)]
struct InferenceAction {
    index: usize,
    key: String,
    feature_indices: Vec<u16>,
}

#[derive(Debug, Deserialize)]
struct InferenceResponse {
    ok: bool,
    error: Option<String>,
    model_id: Option<String>,
    action_keys: Option<Vec<String>>,
    policy_probabilities: Option<Vec<f64>>,
    shared_win_rate: Option<f64>,
    victory_point_margin: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct InferenceBatchResponse {
    ok: bool,
    error: Option<String>,
    model_id: Option<String>,
    evaluations: Option<Vec<InferenceBatchEvaluationResponse>>,
}

#[derive(Debug, Deserialize)]
struct InferenceBatchEvaluationResponse {
    request_id: u64,
    action_keys: Vec<String>,
    policy_probabilities: Vec<f64>,
    shared_win_rate: f64,
    victory_point_margin: f64,
}

fn build_inference_request(runner: &GameRunner) -> Result<InferenceRequest, String> {
    let observer_idx = runner.framework.current_player;
    let actions = enumerate_legal_actions(runner)?;
    if actions.is_empty() {
        return Err("cannot infer a position with no legal actions".to_string());
    }
    Ok(InferenceRequest {
        feature_schema: training_feature_schema(),
        state: InferenceState {
            feature_version: TRAINING_FEATURE_VERSION,
            features: encode_training_state(runner, observer_idx)?,
        },
        legal_actions: InferenceLegalActions {
            feature_version: TRAINING_FEATURE_VERSION,
            actions: encode_actions(runner, observer_idx, &actions)?,
        },
    })
}

fn build_batch_inference_request(
    leaves: &[NeuralLeafRequest],
) -> Result<InferenceBatchRequest, String> {
    if leaves.is_empty() || leaves.len() > 256 {
        return Err("batched leaf inference requires between 1 and 256 positions".to_string());
    }
    let positions = leaves
        .iter()
        .map(|leaf| {
            if leaf.feature_version != TRAINING_FEATURE_VERSION {
                return Err(format!(
                    "neural leaf {} uses feature version {}, expected {}",
                    leaf.request_id, leaf.feature_version, TRAINING_FEATURE_VERSION
                ));
            }
            if leaf.action_keys.is_empty()
                || leaf.action_keys.len() != leaf.action_feature_indices.len()
            {
                return Err(format!(
                    "neural leaf {} has inconsistent legal actions",
                    leaf.request_id
                ));
            }
            Ok(InferenceBatchPosition {
                request_id: leaf.request_id,
                state: InferenceState {
                    feature_version: leaf.feature_version,
                    features: leaf.state_features.clone(),
                },
                legal_actions: InferenceLegalActions {
                    feature_version: leaf.feature_version,
                    actions: leaf
                        .action_keys
                        .iter()
                        .zip(&leaf.action_feature_indices)
                        .enumerate()
                        .map(|(index, (key, feature_indices))| InferenceAction {
                            index,
                            key: key.clone(),
                            feature_indices: feature_indices.clone(),
                        })
                        .collect(),
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InferenceBatchRequest {
        feature_schema: training_feature_schema(),
        positions,
    })
}

fn encode_actions(
    runner: &GameRunner,
    observer_idx: usize,
    actions: &[LegalAction],
) -> Result<Vec<InferenceAction>, String> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            Ok(InferenceAction {
                index,
                key: action.key(),
                feature_indices: encode_training_action(runner, observer_idx, action)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::training::{ACTION_FEATURE_DIM, STATE_FEATURE_DIM};

    #[test]
    fn inference_request_uses_complete_stable_training_features() {
        let runner = GameRunner::new(2, Some(4_404));
        let expected_actions = enumerate_legal_actions(&runner).unwrap();
        let request = build_inference_request(&runner).unwrap();

        assert_eq!(request.state.features.len(), STATE_FEATURE_DIM);
        assert_eq!(request.feature_schema.state_dim, STATE_FEATURE_DIM);
        assert_eq!(request.feature_schema.action_dim, ACTION_FEATURE_DIM);
        assert_eq!(request.legal_actions.actions.len(), expected_actions.len());
        for (index, (encoded, expected)) in request
            .legal_actions
            .actions
            .iter()
            .zip(&expected_actions)
            .enumerate()
        {
            assert_eq!(encoded.index, index);
            assert_eq!(encoded.key, expected.key());
            assert!(!encoded.feature_indices.is_empty());
            assert!(encoded
                .feature_indices
                .iter()
                .all(|feature| (*feature as usize) < ACTION_FEATURE_DIM));
        }
    }

    #[test]
    fn batched_leaf_request_preserves_stable_position_and_action_order() {
        let leaves = vec![
            NeuralLeafRequest {
                request_id: 19,
                depth: 1,
                evaluation_player: 1,
                feature_version: TRAINING_FEATURE_VERSION,
                state_features: vec![0.25; STATE_FEATURE_DIM],
                action_keys: vec!["build:first".to_string(), "pass:second".to_string()],
                action_feature_indices: vec![vec![2, 7], vec![11]],
            },
            NeuralLeafRequest {
                request_id: 23,
                depth: 2,
                evaluation_player: 0,
                feature_version: TRAINING_FEATURE_VERSION,
                state_features: vec![-0.5; STATE_FEATURE_DIM],
                action_keys: vec!["loan:only".to_string()],
                action_feature_indices: vec![vec![3, 5, 13]],
            },
        ];

        let first = build_batch_inference_request(&leaves).unwrap();
        let second = build_batch_inference_request(&leaves).unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        assert_eq!(first.feature_schema.state_dim, STATE_FEATURE_DIM);
        assert_eq!(first.feature_schema.action_dim, ACTION_FEATURE_DIM);
        assert_eq!(first.positions.len(), 2);
        assert_eq!(first.positions[0].request_id, 19);
        assert_eq!(first.positions[1].request_id, 23);
        assert_eq!(first.positions[0].legal_actions.actions[0].index, 0);
        assert_eq!(
            first.positions[0].legal_actions.actions[0].key,
            "build:first"
        );
        assert_eq!(
            first.positions[0].legal_actions.actions[0].feature_indices,
            vec![2, 7]
        );
        assert_eq!(first.positions[0].legal_actions.actions[1].index, 1);
        assert_eq!(
            first.positions[0].legal_actions.actions[1].key,
            "pass:second"
        );
    }
}
