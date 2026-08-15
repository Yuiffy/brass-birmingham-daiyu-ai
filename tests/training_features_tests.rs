use fast_brass::game::hidden_information::determinize_hidden_information;
use fast_brass::game::legal_actions::enumerate_legal_actions;
use fast_brass::game::runner::GameRunner;
use fast_brass::game::training::{
    encode_root_action_successor_batch, encode_training_action, encode_training_state,
    training_feature_schema, ACTION_FEATURE_DIM, STATE_FEATURE_DIM, TRAINING_FEATURE_VERSION,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn training_schema_is_contiguous_and_versioned() {
    let schema = training_feature_schema();
    assert_eq!(schema.version, TRAINING_FEATURE_VERSION);
    assert_eq!(schema.state_dim, STATE_FEATURE_DIM);
    assert_eq!(schema.action_dim, ACTION_FEATURE_DIM);

    let mut next_offset = 0usize;
    for block in &schema.state_blocks {
        assert_eq!(
            block.offset, next_offset,
            "state block {} has a gap",
            block.name
        );
        next_offset += block.size;
    }
    assert_eq!(next_offset, STATE_FEATURE_DIM);
    assert_eq!(
        schema
            .action_blocks
            .last()
            .map(|block| block.offset + block.size),
        Some(ACTION_FEATURE_DIM)
    );
}

#[test]
fn state_features_are_fixed_finite_and_deterministic_for_all_player_counts() {
    for num_players in 2..=4 {
        let runner = GameRunner::new(num_players, Some(10_000 + num_players as u64));
        for observer_idx in 0..num_players {
            let first = encode_training_state(&runner, observer_idx).unwrap();
            let second = encode_training_state(&runner, observer_idx).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), STATE_FEATURE_DIM);
            assert!(first.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn hidden_information_determinization_does_not_change_observer_features() {
    let runner = GameRunner::new(4, Some(22_222));
    let observer_idx = runner.framework.current_player;
    let before = encode_training_state(&runner, observer_idx).unwrap();

    let mut determinized = runner.clone();
    let mut rng = StdRng::seed_from_u64(98_765);
    determinize_hidden_information(&mut determinized, observer_idx, &mut rng).unwrap();
    let after = encode_training_state(&determinized, observer_idx).unwrap();

    assert_eq!(before, after);
}

#[test]
fn every_initial_atomic_action_has_bounded_sparse_features() {
    for num_players in 2..=4 {
        let runner = GameRunner::new(num_players, Some(30_000 + num_players as u64));
        let observer_idx = runner.framework.current_player;
        let actions = enumerate_legal_actions(&runner).unwrap();
        assert!(!actions.is_empty());

        for action in &actions {
            let features = encode_training_action(&runner, observer_idx, action).unwrap();
            assert!(!features.is_empty(), "{} has no features", action.key());
            assert_eq!(
                features[0],
                0,
                "{} is missing the bias feature",
                action.key()
            );
            assert!(
                features.windows(2).all(|pair| pair[0] < pair[1]),
                "{} contains duplicate or unsorted features",
                action.key()
            );
            assert!(
                features
                    .iter()
                    .all(|feature_idx| (*feature_idx as usize) < ACTION_FEATURE_DIM),
                "{} contains an out-of-range feature",
                action.key()
            );
        }
    }
}

#[test]
fn action_features_require_the_current_decision_players_view() {
    let runner = GameRunner::new(2, Some(44_444));
    let action = enumerate_legal_actions(&runner).unwrap().remove(0);
    let current = runner.framework.current_player;
    let other = (current + 1) % 2;

    let error = encode_training_action(&runner, other, &action).unwrap_err();
    assert!(error.contains("decision player as observer"));
}

#[test]
fn successor_value_batch_is_deterministic_complete_and_stably_ordered() {
    let runner = GameRunner::new(2, Some(7_303));
    let actions = enumerate_legal_actions(&runner).unwrap();
    let first = encode_root_action_successor_batch(&runner, 3, 8_404).unwrap();
    let second = encode_root_action_successor_batch(&runner, 3, 8_404).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.root_player, runner.framework.current_player);
    assert_eq!(
        first.action_keys,
        actions
            .iter()
            .map(|action| action.key())
            .collect::<Vec<_>>()
    );
    assert_eq!(first.samples.len(), actions.len() * 3);
    assert_eq!(first.states.len(), first.samples.len());
    for (flat_index, sample) in first.samples.iter().enumerate() {
        let action_index = flat_index / 3;
        let sample_index = flat_index % 3;
        assert_eq!(sample.action_index, action_index);
        assert_eq!(sample.sample_index, sample_index);
        assert_eq!(sample.action_key, first.action_keys[action_index]);
        assert_eq!(sample.state_index, Some(flat_index));
        assert_eq!(sample.terminal_root_shared_win_rate, None);
        assert_eq!(sample.terminal_root_victory_point_margin, None);
        assert_eq!(first.states[flat_index].features.len(), STATE_FEATURE_DIM);
        assert!(first.states[flat_index]
            .features
            .iter()
            .all(|value| value.is_finite()));
    }
}

#[test]
fn successor_value_batch_rejects_invalid_sample_counts() {
    let runner = GameRunner::new(2, Some(7_304));
    assert!(encode_root_action_successor_batch(&runner, 0, 1)
        .unwrap_err()
        .contains("between 1 and 64"));
    assert!(encode_root_action_successor_batch(&runner, 65, 1)
        .unwrap_err()
        .contains("between 1 and 64"));
}
