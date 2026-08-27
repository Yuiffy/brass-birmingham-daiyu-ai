use fast_brass::core::types::{ActionType, Era};
use fast_brass::game::legal_actions::enumerate_legal_actions;
use fast_brass::game::runner::{GamePhase, GameRunner};
use fast_brass::game::search::{
    aggregate_two_player_action_values, search_top_actions, search_top_actions_with_policy,
    search_top_actions_with_policy_and_action_values, BatchedNeuralPuctConfig,
    BatchedNeuralPuctSearch, NeuralLeafEvaluation, NeuralLeafRequest, RootPolicyEvaluation,
    RootSearchConfig, BATCHED_NEURAL_PUCT_METHOD, NEURAL_TREE_VALUE_SOURCE,
    ROOT_PUCT_ACTION_VALUE_METHOD, ROOT_PUCT_METHOD, ROOT_UCB_METHOD,
};
use fast_brass::game::training::encode_root_action_successor_batch;

fn compact_rail_endgame(cards_per_player: usize, income_level: u8) -> GameRunner {
    let mut runner = GameRunner::new(2, Some(8_101));
    runner.game_phase = GamePhase::Railroad;
    runner.framework.board.state.era = Era::Railroad;
    runner.framework.board.state.deck.cards.clear();
    runner.framework.board.state.discard_pile.clear();
    runner.actions_remaining_in_turn = 0;
    runner.personal_turns_taken.fill(0);
    for player in &mut runner.framework.board.state.players {
        player.hand.cards.truncate(cards_per_player);
        player.money = 0;
        player.income_level = income_level;
        player.victory_points = 0;
        player.spent_this_turn = 0;
    }
    runner
}

fn root_policy(runner: &GameRunner, model_id: &str) -> RootPolicyEvaluation {
    let actions = enumerate_legal_actions(runner).unwrap();
    RootPolicyEvaluation {
        model_id: model_id.to_string(),
        action_keys: actions.iter().map(|action| action.key()).collect(),
        policy_probabilities: vec![1.0; actions.len()],
        shared_win_rate: 0.5,
        victory_point_margin: 0.0,
        actor_victory_points: 40.0,
    }
}

fn leaf_evaluation(
    request: &NeuralLeafRequest,
    model_id: &str,
    root_player: usize,
    root_shared_win_rate: f64,
) -> NeuralLeafEvaluation {
    let (shared_win_rate, victory_point_margin) = if request.evaluation_player == root_player {
        (root_shared_win_rate, 10.0)
    } else {
        (1.0 - root_shared_win_rate, -10.0)
    };
    let mut policy_probabilities = vec![0.0; request.action_keys.len()];
    policy_probabilities[0] = 1.0;
    NeuralLeafEvaluation {
        request_id: request.request_id,
        model_id: model_id.to_string(),
        action_keys: request.action_keys.clone(),
        policy_probabilities,
        shared_win_rate,
        victory_point_margin,
        actor_victory_points: 40.0,
    }
}

#[test]
fn root_search_is_deterministic_conserves_visits_and_returns_applicable_actions() {
    let runner = compact_rail_endgame(2, 10);
    let root_action_count = enumerate_legal_actions(&runner).unwrap().len();
    assert!(root_action_count >= 2);
    let config = RootSearchConfig {
        simulations: 64,
        recommendation_count: usize::MAX,
        max_rollout_actions: 8,
        sample_continuation_length: 4,
        seed: 9_001,
        ..RootSearchConfig::default()
    };

    let first = search_top_actions(&runner, &config).expect("first search should complete");
    let second = search_top_actions(&runner, &config).expect("second search should complete");

    assert_eq!(first, second);
    assert_eq!(first.method, ROOT_UCB_METHOD);
    assert_eq!(first.completed_simulations, config.simulations);
    assert_eq!(first.root_action_count, root_action_count);
    assert_eq!(first.evaluated_action_count, root_action_count);
    assert!(first.all_root_actions_evaluated);
    assert_eq!(
        first
            .recommendations
            .iter()
            .map(|candidate| candidate.visits)
            .sum::<u64>(),
        config.simulations
    );
    assert!(
        (first
            .recommendations
            .iter()
            .map(|candidate| candidate.visit_share)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );

    for candidate in &first.recommendations {
        assert!((0.0..=1.0).contains(&candidate.visit_share));
        assert!((0.0..=1.0).contains(&candidate.estimated_shared_win_rate));
        assert!(candidate
            .estimated_outright_win_rate
            .is_some_and(|value| (0.0..=1.0).contains(&value)));
        assert!(candidate
            .estimated_tied_first_rate
            .is_some_and(|value| (0.0..=1.0).contains(&value)));
        assert_eq!(candidate.policy_probability, None);
        assert_eq!(candidate.calibrated_win_rate, None);
        assert_eq!(candidate.immediate_effect.player_money_delta.len(), 2);
        assert!(candidate.sample_random_continuation.steps.len() <= 4);
        assert_eq!(
            candidate
                .sample_random_continuation
                .final_victory_points
                .len(),
            2
        );

        let mut branch = runner.clone();
        candidate
            .action
            .apply(&mut branch)
            .unwrap_or_else(|error| panic!("{} was not applicable: {error}", candidate.action_key));
    }
}

#[test]
fn tied_first_place_receives_fractional_shared_win_credit() {
    let runner = compact_rail_endgame(1, 0);
    let actions = enumerate_legal_actions(&runner).unwrap();
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].root, ActionType::Pass);
    let config = RootSearchConfig {
        simulations: 8,
        recommendation_count: 1,
        max_rollout_actions: 4,
        sample_continuation_length: 2,
        seed: 9_002,
        ..RootSearchConfig::default()
    };

    let report = search_top_actions(&runner, &config).expect("tie search should complete");
    let estimate = &report.recommendations[0];

    assert_eq!(estimate.visits, 8);
    assert_eq!(estimate.estimated_shared_win_rate, 0.5);
    assert_eq!(estimate.estimated_outright_win_rate, Some(0.0));
    assert_eq!(estimate.estimated_tied_first_rate, Some(1.0));
    assert_eq!(estimate.shared_win_rate_standard_error, Some(0.0));
    assert_eq!(
        estimate.sample_random_continuation.official_winners,
        vec![0, 1]
    );
}

#[test]
fn root_search_rolls_real_initial_games_to_completion() {
    let runner = GameRunner::new(2, Some(8_102));
    let config = RootSearchConfig {
        simulations: 3,
        recommendation_count: 3,
        max_rollout_actions: 160,
        sample_continuation_length: 3,
        seed: 9_003,
        ..RootSearchConfig::default()
    };

    let report = search_top_actions(&runner, &config).expect("initial search should complete");

    assert_eq!(report.completed_simulations, 3);
    assert_eq!(report.evaluated_action_count, 3);
    assert!(!report.recommendations.is_empty());
    for candidate in &report.recommendations {
        assert_eq!(
            candidate
                .sample_random_continuation
                .final_victory_points
                .len(),
            2
        );
        assert!(!candidate
            .sample_random_continuation
            .official_winners
            .is_empty());
    }
}

#[test]
fn root_puct_uses_the_highest_model_prior_before_other_actions() {
    let runner = compact_rail_endgame(2, 10);
    let actions = enumerate_legal_actions(&runner).unwrap();
    assert!(actions.len() >= 2);
    let selected_index = actions.len() - 1;
    let mut probabilities = vec![0.0; actions.len()];
    probabilities[selected_index] = 1.0;
    let policy = RootPolicyEvaluation {
        model_id: "test-policy-v1".to_string(),
        action_keys: actions.iter().map(|action| action.key()).collect(),
        policy_probabilities: probabilities,
        shared_win_rate: 0.625,
        victory_point_margin: 4.5,
        actor_victory_points: 40.0,
    };
    let config = RootSearchConfig {
        simulations: 1,
        recommendation_count: actions.len(),
        max_rollout_actions: 8,
        sample_continuation_length: 0,
        seed: 9_004,
        ..RootSearchConfig::default()
    };

    let report = search_top_actions_with_policy(&runner, &config, &policy)
        .expect("policy-guided search should complete");

    assert_eq!(report.method, ROOT_PUCT_METHOD);
    assert_eq!(report.model_id.as_deref(), Some("test-policy-v1"));
    assert_eq!(report.root_model_shared_win_rate, Some(0.625));
    assert_eq!(report.root_model_victory_point_margin, Some(4.5));
    assert_eq!(
        report.root_policy_probabilities.as_ref().unwrap()[selected_index],
        1.0
    );
    assert_eq!(report.evaluated_action_count, 1);
    assert_eq!(report.recommendations.len(), 1);
    assert_eq!(
        report.recommendations[0].action_key,
        actions[selected_index].key()
    );
    assert_eq!(report.recommendations[0].policy_probability, Some(1.0));
}

#[test]
fn root_puct_rejects_policy_probabilities_for_reordered_actions() {
    let runner = compact_rail_endgame(2, 10);
    let actions = enumerate_legal_actions(&runner).unwrap();
    assert!(actions.len() >= 2);
    let mut action_keys = actions
        .iter()
        .map(|action| action.key())
        .collect::<Vec<_>>();
    action_keys.swap(0, 1);
    let policy = RootPolicyEvaluation {
        model_id: "test-policy-v1".to_string(),
        action_keys,
        policy_probabilities: vec![1.0; actions.len()],
        shared_win_rate: 0.5,
        victory_point_margin: 0.0,
        actor_victory_points: 40.0,
    };

    let error = search_top_actions_with_policy(
        &runner,
        &RootSearchConfig {
            simulations: 1,
            max_rollout_actions: 8,
            ..RootSearchConfig::default()
        },
        &policy,
    )
    .expect_err("reordered actions must be rejected");

    assert!(error.contains("policy action key mismatch at index 0"));
}

#[test]
fn batched_successor_values_are_converted_to_root_perspective_and_drive_puct() {
    let runner = compact_rail_endgame(2, 10);
    let actions = enumerate_legal_actions(&runner).unwrap();
    assert!(actions.len() >= 2);
    let selected_index = actions.len() - 1;
    let successors = encode_root_action_successor_batch(&runner, 2, 9_005).unwrap();
    let mut raw_win_rates = vec![0.0; successors.states.len()];
    let mut raw_margins = vec![0.0; successors.states.len()];
    for sample in &successors.samples {
        let state_index = sample.state_index.unwrap();
        let desired_root_win = if sample.action_index == selected_index {
            0.9
        } else {
            0.1
        };
        let desired_root_margin = if sample.action_index == selected_index {
            12.0
        } else {
            -8.0
        };
        if sample.evaluation_player == successors.root_player {
            raw_win_rates[state_index] = desired_root_win;
            raw_margins[state_index] = desired_root_margin;
        } else {
            raw_win_rates[state_index] = 1.0 - desired_root_win;
            raw_margins[state_index] = -desired_root_margin;
        }
    }
    let values = aggregate_two_player_action_values(
        &successors,
        "test-policy-v2".to_string(),
        &raw_win_rates,
        &raw_margins,
    )
    .unwrap();
    assert_eq!(values.shared_win_rates[selected_index], 0.9);
    assert_eq!(values.victory_point_margins[selected_index], 12.0);
    assert_eq!(values.shared_win_standard_errors[selected_index], Some(0.0));
    assert_eq!(values.sample_counts[selected_index], 2);

    let policy = RootPolicyEvaluation {
        model_id: "test-policy-v2".to_string(),
        action_keys: actions.iter().map(|action| action.key()).collect(),
        policy_probabilities: vec![1.0; actions.len()],
        shared_win_rate: 0.5,
        victory_point_margin: 0.0,
        actor_victory_points: 40.0,
    };
    let report = search_top_actions_with_policy_and_action_values(
        &runner,
        &RootSearchConfig {
            simulations: 32,
            exploration_constant: 0.0,
            recommendation_count: 3,
            max_rollout_actions: 8,
            sample_continuation_length: 0,
            seed: 9_006,
            ..RootSearchConfig::default()
        },
        &policy,
        &values,
    )
    .unwrap();

    assert_eq!(report.method, ROOT_PUCT_ACTION_VALUE_METHOD);
    assert_eq!(report.evaluated_action_count, actions.len());
    assert_eq!(report.visited_action_count, 1);
    assert!(report.all_root_actions_evaluated);
    assert_eq!(
        report.recommendations[0].action_key,
        actions[selected_index].key()
    );
    assert_eq!(report.recommendations[0].visits, 32);
    assert_eq!(report.recommendations[0].estimated_shared_win_rate, 0.9);
    assert_eq!(report.recommendations[0].estimated_outright_win_rate, None);
    assert_eq!(report.recommendations[0].average_final_victory_points, None);
    assert_eq!(report.recommendations[0].value_sample_count, 2);
}

#[test]
fn batched_neural_puct_expands_beyond_one_ply_and_backs_up_root_values() {
    let runner = compact_rail_endgame(2, 10);
    let root_player = runner.framework.current_player;
    let model_id = "deep-test-model";
    let mut search = BatchedNeuralPuctSearch::new(
        &runner,
        BatchedNeuralPuctConfig {
            search: RootSearchConfig {
                simulations: 2,
                exploration_constant: 0.0,
                recommendation_count: 3,
                max_rollout_actions: 8,
                sample_continuation_length: 0,
                seed: 9_101,
                ..RootSearchConfig::default()
            },
            determinizations: 1,
            score_utility_weight: 0.0,
            group_card_choices: false,
        },
        root_policy(&runner, model_id),
    )
    .unwrap();

    let first_batch = search.next_inference_batch(1).unwrap();
    assert_eq!(first_batch.len(), 1);
    assert_eq!(first_batch[0].depth, 1);
    search
        .submit_inference_batch(vec![leaf_evaluation(
            &first_batch[0],
            model_id,
            root_player,
            0.9,
        )])
        .unwrap();

    let second_batch = search.next_inference_batch(1).unwrap();
    assert_eq!(second_batch.len(), 1);
    assert_eq!(second_batch[0].depth, 2);
    search
        .submit_inference_batch(vec![leaf_evaluation(
            &second_batch[0],
            model_id,
            root_player,
            0.8,
        )])
        .unwrap();

    assert!(search.is_complete());
    assert_eq!(search.max_search_depth(), 2);
    let report = search.finish_report().unwrap();
    assert_eq!(report.method, BATCHED_NEURAL_PUCT_METHOD);
    assert_eq!(report.value_source, NEURAL_TREE_VALUE_SOURCE);
    assert_eq!(report.completed_simulations, 2);
    assert_eq!(report.max_search_depth, Some(2));
    assert_eq!(report.neural_leaf_evaluations, Some(2));
    assert_eq!(report.inference_batches, Some(2));
    assert_eq!(report.recommendations[0].visits, 2);
    assert!((report.recommendations[0].estimated_shared_win_rate - 0.85).abs() < 1e-12);
    assert_eq!(report.recommendations[0].average_victory_point_margin, 10.0);
}

#[test]
fn batched_neural_puct_reserves_distinct_leaves_within_a_batch() {
    let runner = GameRunner::new(2, Some(8_104));
    let root_player = runner.framework.current_player;
    let model_id = "batch-test-model";
    let mut search = BatchedNeuralPuctSearch::new(
        &runner,
        BatchedNeuralPuctConfig {
            search: RootSearchConfig {
                simulations: 4,
                recommendation_count: 8,
                sample_continuation_length: 0,
                seed: 9_102,
                ..RootSearchConfig::default()
            },
            determinizations: 1,
            score_utility_weight: 0.0,
            group_card_choices: false,
        },
        root_policy(&runner, model_id),
    )
    .unwrap();

    let batch = search.next_inference_batch(4).unwrap();
    assert_eq!(batch.len(), 4);
    assert!(batch.iter().all(|request| request.depth == 1));
    let request_ids = batch
        .iter()
        .map(|request| request.request_id)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(request_ids.len(), batch.len());
    search
        .submit_inference_batch(
            batch
                .iter()
                .map(|request| leaf_evaluation(request, model_id, root_player, 0.5))
                .collect(),
        )
        .unwrap();

    let report = search.finish_report().unwrap();
    assert_eq!(report.completed_simulations, 4);
    assert_eq!(report.visited_action_count, 4);
    assert_eq!(
        report
            .recommendations
            .iter()
            .map(|candidate| candidate.visits)
            .sum::<u64>(),
        4
    );
}

#[test]
fn batched_neural_puct_groups_card_variants_before_reserving_root_leaves() {
    let runner = GameRunner::new(2, Some(8_106));
    let root_player = runner.framework.current_player;
    let model_id = "grouped-card-test-model";
    let actions = enumerate_legal_actions(&runner).unwrap();
    let mut group_indices = std::collections::HashMap::new();
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
    let multi_card_group = groups
        .iter()
        .position(|members| members.len() > 1)
        .expect("initial actions should include discard variants");
    let mut selected_groups = vec![multi_card_group];
    selected_groups.extend(
        (0..groups.len())
            .filter(|group_index| *group_index != multi_card_group)
            .take(3),
    );
    assert_eq!(selected_groups.len(), 4);

    let mut probabilities = vec![0.0; actions.len()];
    for group_index in &selected_groups {
        let members = &groups[*group_index];
        for action_index in members {
            probabilities[*action_index] = 0.25 / members.len() as f64;
        }
    }
    let policy = RootPolicyEvaluation {
        model_id: model_id.to_string(),
        action_keys: actions.iter().map(|action| action.key()).collect(),
        policy_probabilities: probabilities,
        shared_win_rate: 0.5,
        victory_point_margin: 0.0,
        actor_victory_points: 40.0,
    };
    let mut search = BatchedNeuralPuctSearch::new(
        &runner,
        BatchedNeuralPuctConfig {
            search: RootSearchConfig {
                simulations: 4,
                recommendation_count: 8,
                sample_continuation_length: 0,
                seed: 9_104,
                ..RootSearchConfig::default()
            },
            determinizations: 1,
            score_utility_weight: 0.0,
            group_card_choices: true,
        },
        policy,
    )
    .unwrap();

    let batch = search.next_inference_batch(4).unwrap();
    assert_eq!(batch.len(), 4);
    search
        .submit_inference_batch(
            batch
                .iter()
                .map(|request| leaf_evaluation(request, model_id, root_player, 0.5))
                .collect(),
        )
        .unwrap();

    let report = search.finish_report().unwrap();
    let visited_groups = report
        .recommendations
        .iter()
        .map(|recommendation| recommendation.action.card_invariant_key())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(report.visited_action_count, 4);
    assert_eq!(visited_groups.len(), 4);
}

#[test]
fn batched_neural_puct_rejects_invalid_leaf_evaluation_atomically() {
    let runner = GameRunner::new(2, Some(8_105));
    let root_player = runner.framework.current_player;
    let model_id = "validation-test-model";
    let mut search = BatchedNeuralPuctSearch::new(
        &runner,
        BatchedNeuralPuctConfig {
            search: RootSearchConfig {
                simulations: 1,
                sample_continuation_length: 0,
                seed: 9_103,
                ..RootSearchConfig::default()
            },
            determinizations: 1,
            score_utility_weight: 0.0,
            group_card_choices: false,
        },
        root_policy(&runner, model_id),
    )
    .unwrap();
    let batch = search.next_inference_batch(1).unwrap();
    let mut invalid = leaf_evaluation(&batch[0], "wrong-model", root_player, 0.5);
    invalid.action_keys.swap(0, 1);
    let error = search
        .submit_inference_batch(vec![invalid])
        .expect_err("wrong model and action order must be rejected");
    assert!(error.contains("model"));
    assert_eq!(search.pending_evaluations(), 1);
    assert_eq!(search.completed_simulations(), 0);

    search
        .submit_inference_batch(vec![leaf_evaluation(&batch[0], model_id, root_player, 0.5)])
        .unwrap();
    assert!(search.is_complete());
}
