from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import load_model_checkpoint, save_checkpoint
from training.data import SelfPlayDataset, collate_positions
from training.inference import (
    BatchValuePrediction,
    CheckpointEvaluator,
    aggregate_two_player_successor_values,
    build_value_state_batch,
)
from training.model import BrassPolicyValueNet, ModelConfig, compute_losses


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
        self.assertTrue(
            torch.allclose(batch.policy_targets[:2], torch.tensor([0.75, 0.25]))
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
        )
        model = BrassPolicyValueNet(config)
        output = model(batch)
        self.assertEqual(tuple(output.policy_logits.shape), (5,))
        self.assertEqual(tuple(output.shared_win_logits.shape), (2,))
        losses = compute_losses(output, batch, vp_margin_scale=config.vp_margin_scale)
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
        self.assertEqual(payload["metadata"]["training"]["global_step"], 7)
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
            expected_win_logits, expected_margins = model.forward_values(dense)

        self.assertEqual(len(prediction.shared_win_rates), 2)
        self.assertEqual(len(prediction.victory_point_margins), 2)
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
