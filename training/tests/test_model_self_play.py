from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.data import FeatureSchema, SelfPlayDataset
from training.inference import PolicyValuePrediction
from training.model_self_play import ModelSelfPlayConfig, export_model_self_play

SCHEMA = {
    "version": 1,
    "state_dim": 4,
    "action_dim": 8,
    "card_type_dim": 2,
    "max_players": 2,
    "state_blocks": [{"name": "all", "offset": 0, "size": 4}],
    "action_blocks": [{"name": "all", "offset": 0, "size": 8}],
}


class _FakeEvaluator:
    def __init__(self) -> None:
        self.schema = FeatureSchema.from_schema_dict(SCHEMA)
        self.model_id = "sha256:test-model"
        self.checkpoint_step = 17

    def predict(
        self,
        state_record: dict,
        legal_record: dict,
        request_schema: FeatureSchema,
    ) -> PolicyValuePrediction:
        self.schema.assert_compatible(request_schema, "fake request")
        return PolicyValuePrediction(
            model_id=self.model_id,
            checkpoint_step=self.checkpoint_step,
            action_keys=tuple(action["key"] for action in legal_record["actions"]),
            policy_probabilities=(0.75, 0.25),
            shared_win_rate=0.6,
            victory_point_margin=4.0,
        )

    def predict_batch(
        self,
        state_records: list[dict],
        legal_records: list[dict],
        request_schema: FeatureSchema,
    ) -> tuple[PolicyValuePrediction, ...]:
        self.schema.assert_compatible(request_schema, "fake batch request")
        return tuple(
            PolicyValuePrediction(
                model_id=self.model_id,
                checkpoint_step=self.checkpoint_step,
                action_keys=tuple(action["key"] for action in legal["actions"]),
                policy_probabilities=tuple(
                    1.0 if index == 0 else 0.0
                    for index in range(len(legal["actions"]))
                ),
                shared_win_rate=0.4,
                victory_point_margin=-2.0,
            )
            for legal in legal_records
        )


class _FakeNeuralSearch:
    def __init__(self, simulations: int, model_id: str, probabilities: list[float]):
        self.simulations = simulations
        self.model_id = model_id
        self.probabilities = probabilities
        self.submitted = False

    def is_complete(self) -> bool:
        return self.submitted

    def next_inference_batch(self, max_batch_size: int) -> dict:
        if max_batch_size <= 0 or self.submitted:
            raise AssertionError("unexpected fake neural batch request")
        return {
            "positions": [
                {
                    "request_id": 0,
                    "depth": 2,
                    "evaluation_player": 1,
                    "state": {"feature_version": 1, "features": [0.0, 1.0, 0.5, -0.5]},
                    "legal_actions": {
                        "feature_version": 1,
                        "actions": [
                            {"index": 0, "key": "leaf-a", "feature_indices": [0, 2]},
                            {"index": 1, "key": "leaf-b", "feature_indices": [0, 5]},
                        ],
                    },
                }
            ]
        }

    def submit_inference_batch(
        self,
        request_ids: list[int],
        action_keys: list[list[str]],
        policy_probabilities: list[list[float]],
        shared_win_rates: list[float],
        victory_point_margins: list[float],
        model_id: str,
    ) -> None:
        if request_ids != [0] or model_id != self.model_id:
            raise AssertionError("fake neural submission identity changed")
        if action_keys != [["leaf-a", "leaf-b"]]:
            raise AssertionError("fake neural action order changed")
        self.submitted = True

    def finish(self) -> dict:
        if not self.submitted:
            raise AssertionError("fake neural search finished before inference")
        return {
            "method": "determinized_batched_neural_puct",
            "model_id": self.model_id,
            "root_player": 0,
            "forced_advances": 0,
            "completed_simulations": self.simulations,
            "max_search_depth": 2,
            "neural_leaf_evaluations": 1,
            "inference_batches": 1,
            "actions": [
                {
                    "index": 0,
                    "key": "action-a",
                    "visits": self.simulations,
                    "policy_probability": self.probabilities[0],
                    "value_source": "batched_neural_tree_search",
                    "value_sample_count": self.simulations,
                    "estimated_shared_win_rate": 0.8,
                    "average_victory_point_margin": 6.0,
                },
                {
                    "index": 1,
                    "key": "action-b",
                    "visits": 0,
                    "policy_probability": self.probabilities[1],
                    "value_source": "batched_neural_tree_search",
                    "value_sample_count": None,
                    "estimated_shared_win_rate": None,
                    "average_victory_point_margin": None,
                },
            ],
        }


class _FakeGame:
    def __init__(self, num_players: int, seed: int) -> None:
        if num_players != 2:
            raise AssertionError("fake engine only supports two players")
        self.seed = seed
        self.done = False

    def get_training_feature_schema(self) -> dict:
        return SCHEMA

    def is_done(self) -> bool:
        return self.done

    def current_decision_mode(self) -> str:
        return "turn"

    def current_decision_player(self) -> int:
        return 0

    def get_training_state(self) -> dict:
        return {
            "feature_version": 1,
            "decision_player": 0,
            "forced_advances": 0,
            "features": [1.0, 0.0, 0.5, -0.5],
        }

    def get_legal_actions(self) -> dict:
        return {
            "feature_version": 1,
            "decision_player": 0,
            "forced_advances": 0,
            "actions": [
                {"index": 0, "key": "action-a", "feature_indices": [0, 2]},
                {"index": 1, "key": "action-b", "feature_indices": [0, 5]},
            ],
        }

    def get_legal_action_successor_batch(
        self, determinizations_per_action: int, seed: int
    ) -> dict:
        return {
            "num_players": 2,
            "root_player": 0,
            "action_keys": ["action-a", "action-b"],
            "determinizations_per_action": determinizations_per_action,
            "seed": seed,
            "states": [],
            "samples": [],
        }

    def start_batched_neural_search(
        self,
        probabilities: list[float],
        model_id: str,
        shared_win_rate: float,
        victory_point_margin: float,
        simulations: int,
        seed: int,
        exploration_constant: float,
        determinizations: int,
    ) -> _FakeNeuralSearch:
        self.last_search = {
            "probabilities": probabilities,
            "model_id": model_id,
            "simulations": simulations,
            "seed": seed,
            "determinizations": determinizations,
        }
        return _FakeNeuralSearch(simulations, model_id, probabilities)

    def search_legal_actions_with_policy_and_action_values(
        self,
        probabilities: list[float],
        action_keys: list[str],
        action_shared_win_rates: list[float],
        action_victory_point_margins: list[float],
        action_shared_win_standard_errors: list[float | None],
        action_value_sample_counts: list[int],
        model_id: str,
        shared_win_rate: float,
        victory_point_margin: float,
        simulations: int,
        seed: int,
        exploration_constant: float,
    ) -> dict:
        self.last_search = {
            "probabilities": probabilities,
            "action_keys": action_keys,
            "action_shared_win_rates": action_shared_win_rates,
            "action_victory_point_margins": action_victory_point_margins,
            "model_id": model_id,
            "simulations": simulations,
            "seed": seed,
        }
        return {
            "method": "determinized_root_puct_policy_batched_successor_value",
            "model_id": model_id,
            "root_player": 0,
            "forced_advances": 0,
            "completed_simulations": simulations,
            "actions": [
                {
                    "index": 0,
                    "key": "action-a",
                    "visits": simulations,
                    "policy_probability": probabilities[0],
                    "value_source": "batched_successor_model",
                    "value_sample_count": action_value_sample_counts[0],
                    "estimated_shared_win_rate": action_shared_win_rates[0],
                    "average_victory_point_margin": action_victory_point_margins[0],
                },
                {
                    "index": 1,
                    "key": "action-b",
                    "visits": 0,
                    "policy_probability": probabilities[1],
                    "value_source": "batched_successor_model",
                    "value_sample_count": action_value_sample_counts[1],
                    "estimated_shared_win_rate": action_shared_win_rates[1],
                    "average_victory_point_margin": action_victory_point_margins[1],
                },
            ],
        }

    def search_legal_actions_with_policy(
        self,
        probabilities: list[float],
        model_id: str,
        shared_win_rate: float,
        victory_point_margin: float,
        simulations: int,
        seed: int,
        exploration_constant: float,
    ) -> dict:
        self.last_search = {
            "probabilities": probabilities,
            "model_id": model_id,
            "shared_win_rate": shared_win_rate,
            "victory_point_margin": victory_point_margin,
            "simulations": simulations,
            "seed": seed,
            "exploration_constant": exploration_constant,
        }
        return {
            "method": "determinized_root_puct_policy_random_rollout",
            "model_id": model_id,
            "root_player": 0,
            "forced_advances": 0,
            "completed_simulations": simulations,
            "actions": [
                {
                    "index": 0,
                    "key": "action-a",
                    "visits": simulations,
                    "policy_probability": probabilities[0],
                    "estimated_shared_win_rate": 0.75,
                    "average_victory_point_margin": 8.0,
                },
                {
                    "index": 1,
                    "key": "action-b",
                    "visits": 0,
                    "policy_probability": probabilities[1],
                    "estimated_shared_win_rate": None,
                    "average_victory_point_margin": None,
                },
            ],
        }

    def get_observation(self, actor: int) -> dict:
        if actor != 0:
            raise AssertionError("unexpected actor")
        return {
            "global_features": [1.0, 0.0, 0.0],
            "turn_count": 0,
            "round_in_phase": 0,
            "actions_remaining": 1,
        }

    def step_legal_action(self, action_index: int) -> None:
        if action_index != 0:
            raise AssertionError("zero-visit action was selected")
        self.done = True

    def get_outcome(self) -> dict:
        return {
            "official_winners": [0],
            "shared_win_values": [1.0, 0.0],
            "placements": [1, 2],
            "finish_order": [0, 1],
            "victory_points": [12, 7],
            "victory_point_margins": [5, -5],
            "income_levels": [4, 2],
            "money": [9, 3],
        }


class _FakeEngine:
    BrassRLGame = _FakeGame


class ModelSelfPlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_config_requires_exactly_one_model_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ModelSelfPlayConfig(output=self.root / "missing.jsonl").validate()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ModelSelfPlayConfig(
                output=self.root / "both.jsonl",
                checkpoint=self.root / "model.pt",
                inference_url="http://inference.test",
            ).validate()
        ModelSelfPlayConfig(
            output=self.root / "remote.jsonl",
            inference_url="http://inference.test",
        ).validate()

    def test_export_is_dataset_compatible_and_keeps_model_search_evidence(self) -> None:
        output = self.root / "model-self-play.jsonl"
        config = ModelSelfPlayConfig(
            output=output,
            checkpoint=self.root / "model.pt",
            games=1,
            simulations_per_decision=4,
            engine_revision="test-engine",
        )
        evaluator = _FakeEvaluator()

        summary = export_model_self_play(
            config,
            engine_module=_FakeEngine,
            evaluator=evaluator,
        )

        self.assertEqual(summary.games_written, 1)
        self.assertEqual(summary.positions_written, 1)
        self.assertEqual(summary.model_id, evaluator.model_id)
        self.assertTrue(output.is_file())
        self.assertFalse(Path(f"{output}.partial").exists())
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([record["record_type"] for record in records], ["metadata", "game", "position"])
        self.assertEqual(
            records[0]["search_method"],
            "determinized_batched_neural_puct",
        )
        position = records[2]
        self.assertEqual(position["selected_action_index"], 0)
        self.assertEqual(position["value_target"], {"shared_win": 1.0, "victory_point_margin": 5})
        self.assertEqual(
            [action["model_policy_probability"] for action in position["legal_actions"]],
            [0.75, 0.25],
        )
        self.assertEqual(
            [action["policy_target"] for action in position["legal_actions"]],
            [1.0, 0.0],
        )
        self.assertEqual(
            [action["estimated_shared_win_rate"] for action in position["legal_actions"]],
            [0.8, None],
        )
        self.assertEqual(
            [action["value_sample_count"] for action in position["legal_actions"]],
            [4, None],
        )
        self.assertEqual(position["max_search_depth"], 2)
        self.assertEqual(position["inference_batches"], 1)

        dataset = SelfPlayDataset([output])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].policy_target.tolist(), [1.0, 0.0])
        dataset.close()

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            export_model_self_play(
                config,
                engine_module=_FakeEngine,
                evaluator=evaluator,
            )


if __name__ == "__main__":
    unittest.main()
