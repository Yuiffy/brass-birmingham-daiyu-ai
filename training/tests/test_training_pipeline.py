from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import (
    LEGACY_ACTOR_VP_HEAD_MARKER,
    load_model_checkpoint,
    save_checkpoint,
)
from training.data import SelfPlayDataset, collate_positions
from training.inference import (
    BatchValuePrediction,
    CheckpointEvaluator,
    aggregate_two_player_successor_values,
    build_value_state_batch,
)
from training.model import (
    BrassPolicyValueNet,
    ModelConfig,
    compute_losses,
    expand_with_policy_adapter,
    fuse_policy_and_value_models,
)


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shard = self.root / "synthetic.jsonl"
        _write_synthetic_shard(self.shard)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_dataset_indexes_validates_and_collates_variable_actions(self) -> None:
        dataset = SelfPlayDataset([self.shard])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.schema.state_dim, 4)
        self.assertEqual(dataset.schema.action_dim, 10)
        batch = collate_positions([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch.states.shape), (2, 4))
        self.assertEqual(batch.action_counts.tolist(), [2, 3])
        self.assertEqual(batch.action_feature_offsets.tolist(), [0, 2, 5, 7, 9, 12])
        self.assertEqual(tuple(batch.policy_targets.shape), (5,))
        self.assertEqual(batch.actor_victory_points_targets.tolist(), [40.0, 80.0])
        self.assertTrue(
            torch.allclose(batch.policy_targets[:2], torch.tensor([0.75, 0.25]))
        )
        dataset.close()

    def test_dataset_keeps_strategy_behavior_target_separate_from_search_visits(self) -> None:
        records = [
            json.loads(line)
            for line in self.shard.read_text(encoding="utf-8").splitlines()
        ]
        position = next(record for record in records if record.get("record_type") == "position")
        actions = position["legal_actions"]
        actions[0]["policy_target"] = 0.9
        actions[1]["policy_target"] = 0.1
        for action in actions:
            action["search_policy_target"] = action["visits"] / 4.0
        guided_shard = self.root / "guided-targets.jsonl"
        guided_shard.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        dataset = SelfPlayDataset([guided_shard])
        self.assertTrue(
            torch.allclose(dataset[0].policy_target, torch.tensor([0.9, 0.1]))
        )
        dataset.close()

    def test_model_forward_loss_backward_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(123)
        dataset = SelfPlayDataset([self.shard])
        batch = collate_positions([dataset[0], dataset[1]])
        config = ModelConfig(
            state_dim=4,
            action_dim=10,
            state_hidden_dim=16,
            action_embedding_dim=8,
            trunk_dim=12,
            vp_margin_scale=20.0,
            actor_vp_scale=100.0,
        )
        model = BrassPolicyValueNet(config)
        output = model(batch)
        self.assertEqual(tuple(output.policy_logits.shape), (5,))
        self.assertEqual(tuple(output.shared_win_logits.shape), (2,))
        self.assertEqual(tuple(output.actor_victory_points_normalized.shape), (2,))
        losses = compute_losses(
            output,
            batch,
            vp_margin_scale=config.vp_margin_scale,
            actor_vp_scale=config.actor_vp_scale,
        )
        self.assertTrue(torch.isfinite(losses.total).item())
        losses.total.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
        )

        checkpoint = self.root / "model.pt"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        save_checkpoint(
            checkpoint,
            model,
            dataset.schema,
            optimizer=optimizer,
            training_metadata={"global_step": 7},
        )
        restored, payload = load_model_checkpoint(
            checkpoint, expected_schema=dataset.schema
        )
        restored.eval()
        model.eval()
        with torch.no_grad():
            expected = model(batch)
            actual = restored(batch)
        self.assertTrue(torch.equal(expected.policy_logits, actual.policy_logits))
        self.assertTrue(
            torch.equal(expected.shared_win_logits, actual.shared_win_logits)
        )
        self.assertTrue(
            torch.equal(
                expected.actor_victory_points_normalized,
                actual.actor_victory_points_normalized,
            )
        )
        self.assertEqual(payload["metadata"]["training"]["global_step"], 7)
        dataset.close()

    def test_zero_initialized_policy_adapter_preserves_base_values(self) -> None:
        torch.manual_seed(124)
        dataset = SelfPlayDataset([self.shard])
        batch = collate_positions([dataset[0], dataset[1]])
        base = BrassPolicyValueNet(
            ModelConfig(
                state_dim=4,
                action_dim=10,
                state_hidden_dim=16,
                action_embedding_dim=8,
                trunk_dim=12,
            )
        )
        expanded = expand_with_policy_adapter(base, policy_adapter_dim=16)
        base.eval()
        expanded.eval()
        with torch.no_grad():
            baseline = base(batch)
            initial = expanded(batch)
        self.assertTrue(torch.equal(baseline.policy_logits, initial.policy_logits))
        self.assertTrue(
            torch.equal(baseline.shared_win_logits, initial.shared_win_logits)
        )
        self.assertTrue(
            torch.equal(
                baseline.victory_point_margin_normalized,
                initial.victory_point_margin_normalized,
            )
        )
        self.assertTrue(
            torch.equal(
                baseline.actor_victory_points_normalized,
                initial.actor_victory_points_normalized,
            )
        )

        for name, parameter in expanded.named_parameters():
            parameter.requires_grad_(name.startswith("policy_adapter_"))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in expanded.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        expanded.train()
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(
            expanded(batch),
            batch,
            vp_margin_scale=expanded.config.vp_margin_scale,
            actor_vp_scale=expanded.config.actor_vp_scale,
        )
        losses.total.backward()
        optimizer.step()
        expanded.eval()
        with torch.no_grad():
            trained = expanded(batch)
        self.assertFalse(torch.equal(initial.policy_logits, trained.policy_logits))
        self.assertTrue(
            torch.equal(initial.shared_win_logits, trained.shared_win_logits)
        )
        self.assertTrue(
            torch.equal(
                initial.victory_point_margin_normalized,
                trained.victory_point_margin_normalized,
            )
        )
        self.assertTrue(
            torch.equal(
                initial.actor_victory_points_normalized,
                trained.actor_victory_points_normalized,
            )
        )
        dataset.close()

    def test_fused_model_preserves_policy_and_value_experts(self) -> None:
        dataset = SelfPlayDataset([self.shard])
        batch = collate_positions([dataset[0], dataset[1]])
        config = ModelConfig(
            state_dim=4,
            action_dim=10,
            state_hidden_dim=16,
            action_embedding_dim=8,
            trunk_dim=12,
        )
        torch.manual_seed(125)
        policy_model = BrassPolicyValueNet(config).eval()
        torch.manual_seed(126)
        value_model = BrassPolicyValueNet(config).eval()

        fused = fuse_policy_and_value_models(policy_model, value_model).eval()
        with torch.no_grad():
            policy_output = policy_model(batch)
            value_output = value_model(batch)
            fused_output = fused(batch)
        self.assertTrue(fused.config.separate_value_encoder)
        self.assertTrue(
            torch.equal(policy_output.policy_logits, fused_output.policy_logits)
        )
        self.assertTrue(
            torch.equal(
                value_output.shared_win_logits, fused_output.shared_win_logits
            )
        )
        self.assertTrue(
            torch.equal(
                value_output.victory_point_margin_normalized,
                fused_output.victory_point_margin_normalized,
            )
        )
        self.assertTrue(
            torch.equal(
                value_output.actor_victory_points_normalized,
                fused_output.actor_victory_points_normalized,
            )
        )

        checkpoint = self.root / "fused.pt"
        save_checkpoint(checkpoint, fused, dataset.schema)
        restored, _ = load_model_checkpoint(
            checkpoint, expected_schema=dataset.schema
        )
        restored.eval()
        with torch.no_grad():
            restored_output = restored(batch)
        self.assertTrue(
            torch.equal(fused_output.policy_logits, restored_output.policy_logits)
        )
        self.assertTrue(
            torch.equal(
                fused_output.shared_win_logits, restored_output.shared_win_logits
            )
        )
        dataset.close()

    def test_checkpoint_rejects_feature_schema_mismatch(self) -> None:
        dataset = SelfPlayDataset([self.shard])
        model = BrassPolicyValueNet(ModelConfig(state_dim=4, action_dim=10))
        checkpoint = self.root / "model.pt"
        save_checkpoint(checkpoint, model, dataset.schema)
        altered_shard = self.root / "altered.jsonl"
        _write_synthetic_shard(altered_shard, feature_version=2)
        altered_dataset = SelfPlayDataset([altered_shard])
        with self.assertRaisesRegex(ValueError, "checkpoint feature schema mismatch"):
            load_model_checkpoint(checkpoint, expected_schema=altered_dataset.schema)
        dataset.close()
        altered_dataset.close()

    def test_legacy_checkpoint_initializes_actor_vp_head_deterministically(self) -> None:
        dataset = SelfPlayDataset([self.shard])
        model = BrassPolicyValueNet(ModelConfig(state_dim=4, action_dim=10))
        current = self.root / "current.pt"
        save_checkpoint(current, model, dataset.schema)
        payload = torch.load(current, map_location="cpu", weights_only=True)
        payload["model_state_dict"] = {
            key: value
            for key, value in payload["model_state_dict"].items()
            if not key.startswith("actor_victory_points_head.")
        }
        payload["metadata"]["model_config"].pop("actor_vp_scale")
        legacy = self.root / "legacy.pt"
        torch.save(payload, legacy)

        first, first_payload = load_model_checkpoint(legacy)
        second, second_payload = load_model_checkpoint(legacy)
        first_head = first.actor_victory_points_head.state_dict()
        second_head = second.actor_victory_points_head.state_dict()
        self.assertTrue(first_payload[LEGACY_ACTOR_VP_HEAD_MARKER])
        self.assertTrue(second_payload[LEGACY_ACTOR_VP_HEAD_MARKER])
        self.assertTrue(
            all(torch.equal(first_head[key], second_head[key]) for key in first_head)
        )
        self.assertGreater(first_head["0.weight"].abs().sum().item(), 0.0)
        self.assertGreater(first_head["2.weight"].abs().sum().item(), 0.0)
        dataset.close()

    def test_checkpoint_evaluator_returns_stable_keyed_policy_and_values(self) -> None:
        torch.manual_seed(321)
        dataset = SelfPlayDataset([self.shard])
        assert dataset.schema is not None
        model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=dataset.schema.state_dim,
                action_dim=dataset.schema.action_dim,
                state_hidden_dim=16,
                action_embedding_dim=8,
                trunk_dim=12,
                vp_margin_scale=20.0,
            )
        )
        checkpoint = self.root / "inference-model.pt"
        save_checkpoint(
            checkpoint,
            model,
            dataset.schema,
            training_metadata={"global_step": 11},
        )
        evaluator = CheckpointEvaluator(checkpoint)
        state = {
            "feature_version": dataset.schema.version,
            "features": [0.0, 1.0, 0.5, -0.5],
        }
        legal = {
            "feature_version": dataset.schema.version,
            "actions": [
                {"index": 0, "key": "action-a", "feature_indices": [0, 2]},
                {"index": 1, "key": "action-b", "feature_indices": [0, 4, 7]},
            ],
        }

        prediction = evaluator.predict(state, legal, dataset.schema)

        self.assertTrue(prediction.model_id.startswith("sha256:"))
        self.assertEqual(prediction.checkpoint_step, 11)
        self.assertEqual(prediction.action_keys, ("action-a", "action-b"))
        self.assertAlmostEqual(sum(prediction.policy_probabilities), 1.0, places=6)
        self.assertTrue(
            all(0.0 <= value <= 1.0 for value in prediction.policy_probabilities)
        )
        self.assertTrue(0.0 <= prediction.shared_win_rate <= 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(prediction.victory_point_margin)))
        self.assertGreaterEqual(prediction.actor_victory_points, 0.0)
        self.assertIn(evaluator.select_action(state, legal), (0, 1))

        reordered = {**legal, "actions": list(reversed(legal["actions"]))}
        with self.assertRaisesRegex(ValueError, "stable index order"):
            evaluator.predict(state, reordered, dataset.schema)
        dataset.close()

    def test_checkpoint_evaluator_batches_value_only_states(self) -> None:
        torch.manual_seed(654)
        dataset = SelfPlayDataset([self.shard])
        assert dataset.schema is not None
        model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=dataset.schema.state_dim,
                action_dim=dataset.schema.action_dim,
                state_hidden_dim=16,
                action_embedding_dim=8,
                trunk_dim=12,
                vp_margin_scale=20.0,
            )
        )
        checkpoint = self.root / "batched-value-model.pt"
        save_checkpoint(checkpoint, model, dataset.schema)
        evaluator = CheckpointEvaluator(checkpoint)
        states = [
            {"feature_version": dataset.schema.version, "features": [0.0, 1.0, 0.5, -0.5]},
            {"feature_version": dataset.schema.version, "features": [1.0, 0.0, -0.5, 0.5]},
        ]

        prediction = evaluator.predict_values(states, dataset.schema)
        dense = build_value_state_batch(states, dataset.schema)
        with torch.inference_mode():
            expected_win_logits, expected_margins, expected_actor_vps = (
                model.forward_values(dense)
            )

        self.assertEqual(len(prediction.shared_win_rates), 2)
        self.assertEqual(len(prediction.victory_point_margins), 2)
        self.assertEqual(len(prediction.actor_victory_points), 2)
        self.assertTrue(
            torch.allclose(
                torch.tensor(prediction.shared_win_rates),
                torch.sigmoid(expected_win_logits),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(prediction.victory_point_margins),
                expected_margins * model.config.vp_margin_scale,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(prediction.actor_victory_points),
                (expected_actor_vps * model.config.actor_vp_scale).clamp_min(0.0),
            )
        )
        with self.assertRaisesRegex(ValueError, "non-empty states"):
            evaluator.predict_values([], dataset.schema)
        with self.assertRaisesRegex(ValueError, "value state 0 feature version mismatch"):
            evaluator.predict_values([{**states[0], "feature_version": 99}], dataset.schema)
        dataset.close()

    def test_checkpoint_evaluator_batches_variable_action_policies(self) -> None:
        torch.manual_seed(777)
        dataset = SelfPlayDataset([self.shard])
        assert dataset.schema is not None
        model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=dataset.schema.state_dim,
                action_dim=dataset.schema.action_dim,
                state_hidden_dim=16,
                action_embedding_dim=8,
                trunk_dim=12,
                vp_margin_scale=20.0,
            )
        )
        checkpoint = self.root / "batched-policy-model.pt"
        save_checkpoint(checkpoint, model, dataset.schema)
        evaluator = CheckpointEvaluator(checkpoint)
        states = [
            {"feature_version": 1, "features": [0.0, 1.0, 0.5, -0.5]},
            {"feature_version": 1, "features": [1.0, 0.0, -0.25, 0.75]},
        ]
        legal = [
            {
                "feature_version": 1,
                "actions": [
                    {"index": 0, "key": "a-0", "feature_indices": [0, 2]},
                    {"index": 1, "key": "a-1", "feature_indices": [0, 4, 7]},
                ],
            },
            {
                "feature_version": 1,
                "actions": [
                    {"index": 0, "key": "b-0", "feature_indices": [0, 1]},
                    {"index": 1, "key": "b-1", "feature_indices": [0, 5]},
                    {"index": 2, "key": "b-2", "feature_indices": [0, 6, 9]},
                ],
            },
        ]

        predictions = evaluator.predict_batch(states, legal, dataset.schema)
        individual = tuple(
            evaluator.predict(state, actions, dataset.schema)
            for state, actions in zip(states, legal, strict=True)
        )

        self.assertEqual(len(predictions), 2)
        self.assertEqual([len(row.policy_probabilities) for row in predictions], [2, 3])
        for batched, single in zip(predictions, individual, strict=True):
            self.assertEqual(batched.action_keys, single.action_keys)
            self.assertAlmostEqual(sum(batched.policy_probabilities), 1.0, places=6)
            self.assertTrue(
                torch.allclose(
                    torch.tensor(batched.policy_probabilities),
                    torch.tensor(single.policy_probabilities),
                    atol=1e-7,
                )
            )
            self.assertAlmostEqual(batched.shared_win_rate, single.shared_win_rate)
            self.assertAlmostEqual(
                batched.victory_point_margin, single.victory_point_margin
            )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            evaluator.predict_batch(states, legal[:1], dataset.schema)
        dataset.close()

    def test_successor_values_convert_next_actor_to_root_perspective(self) -> None:
        batch = {
            "num_players": 2,
            "root_player": 0,
            "action_keys": ["same-player", "opponent", "terminal"],
            "determinizations_per_action": 1,
            "states": [
                {"index": 0, "observer_idx": 0},
                {"index": 1, "observer_idx": 1},
            ],
            "samples": [
                {
                    "action_index": 0,
                    "action_key": "same-player",
                    "sample_index": 0,
                    "evaluation_player": 0,
                    "state_index": 0,
                    "terminal_root_shared_win_rate": None,
                    "terminal_root_victory_point_margin": None,
                },
                {
                    "action_index": 1,
                    "action_key": "opponent",
                    "sample_index": 0,
                    "evaluation_player": 1,
                    "state_index": 1,
                    "terminal_root_shared_win_rate": None,
                    "terminal_root_victory_point_margin": None,
                },
                {
                    "action_index": 2,
                    "action_key": "terminal",
                    "sample_index": 0,
                    "evaluation_player": 0,
                    "state_index": None,
                    "terminal_root_shared_win_rate": 1.0,
                    "terminal_root_victory_point_margin": 9.0,
                },
            ],
        }
        prediction = BatchValuePrediction(
            model_id="sha256:test",
            checkpoint_step=12,
            shared_win_rates=(0.2, 0.7),
            victory_point_margins=(2.0, 4.0),
            actor_victory_points=(40.0, 60.0),
        )

        values = aggregate_two_player_successor_values(batch, prediction)

        self.assertEqual(values.action_keys, ("same-player", "opponent", "terminal"))
        self.assertAlmostEqual(values.shared_win_rates[0], 0.2)
        self.assertAlmostEqual(values.shared_win_rates[1], 0.3)
        self.assertAlmostEqual(values.shared_win_rates[2], 1.0)
        self.assertEqual(values.victory_point_margins, (2.0, -4.0, 9.0))
        self.assertEqual(values.shared_win_standard_errors, (None, None, None))
        self.assertEqual(values.sample_counts, (1, 1, 1))


def _write_synthetic_shard(path: Path, feature_version: int = 1) -> None:
    schema = {
        "version": feature_version,
        "state_dim": 4,
        "action_dim": 10,
        "card_type_dim": 2,
        "max_players": 2,
        "state_blocks": [{"name": "all", "offset": 0, "size": 4}],
        "action_blocks": [{"name": "all", "offset": 0, "size": 10}],
    }
    records = [
        {
            "record_type": "metadata",
            "format": "fast_brass_self_play_jsonl",
            "format_version": 1,
            "engine_revision": "synthetic-test-engine",
            "feature_schema": schema,
        },
        {
            "record_type": "game",
            "format_version": 1,
            "game_index": 0,
            "victory_points": [40, 80],
        },
        _position(
            feature_version,
            0,
            [0.0, 1.0, 0.5, -0.5],
            [([0, 2], 3), ([0, 4, 7], 1)],
            0.5,
            4,
        ),
        _position(
            feature_version,
            1,
            [1.0, 0.0, -0.25, 0.75],
            [([0, 1], 1), ([0, 5], 1), ([0, 6, 9], 2)],
            1.0,
            12,
        ),
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")


def _position(
    feature_version: int,
    position_index: int,
    state: list[float],
    actions: list[tuple[list[int], int]],
    shared_win: float,
    vp_margin: int,
) -> dict:
    visit_sum = sum(visits for _, visits in actions)
    return {
        "record_type": "position",
        "format_version": 1,
        "feature_version": feature_version,
        "game_index": 0,
        "position_index": position_index,
        "actor": position_index,
        "state_features": state,
        "legal_actions": [
            {
                "index": index,
                "key": f"action-{position_index}-{index}",
                "feature_indices": features,
                "visits": visits,
                "policy_target": visits / visit_sum,
            }
            for index, (features, visits) in enumerate(actions)
        ],
        "selected_action_index": max(
            range(len(actions)), key=lambda index: actions[index][1]
        ),
        "selected_action_key": f"action-{position_index}-{max(range(len(actions)), key=lambda index: actions[index][1])}",
        "value_target": {
            "shared_win": shared_win,
            "victory_point_margin": vp_margin,
        },
    }


if __name__ == "__main__":
    unittest.main()
