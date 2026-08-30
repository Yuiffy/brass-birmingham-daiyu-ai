from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from training.data import FeatureSchema, SelfPlayDataset
from training.demonstration_prior import load_demonstration_prior
from training.inference import PolicyValuePrediction
from training.model_self_play import (
    ModelSelfPlayConfig,
    _lifecycle_selection_weights,
    _sample_action_index,
    export_model_self_play,
)

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
            actor_victory_points=72.0,
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
                actor_victory_points=55.0,
            )
            for legal in legal_records
        )


class _FakeNeuralSearch:
    def __init__(self, simulations: int, model_id: str, probabilities: list[float]):
        self.simulations = simulations
        self.model_id = model_id
        self.probabilities = probabilities
        self.submitted = False
        self.requested_batch_sizes: list[int] = []

    def is_complete(self) -> bool:
        return self.submitted

    def next_inference_batch(self, max_batch_size: int) -> dict:
        if max_batch_size <= 0 or self.submitted:
            raise AssertionError("unexpected fake neural batch request")
        self.requested_batch_sizes.append(max_batch_size)
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
        actor_victory_points: list[float],
    ) -> None:
        if request_ids != [0] or model_id != self.model_id:
            raise AssertionError("fake neural submission identity changed")
        if action_keys != [["leaf-a", "leaf-b"]]:
            raise AssertionError("fake neural action order changed")
        if actor_victory_points != [55.0]:
            raise AssertionError("fake neural actor-VP values changed")
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
    last_instance = None

    def __init__(self, num_players: int, seed: int) -> None:
        if num_players not in (2, 3, 4):
            raise AssertionError("fake engine only supports two to four players")
        self.num_players = num_players
        self.seed = seed
        self.done = False
        self.last_search = None
        self.last_search_object = None
        type(self).last_instance = self

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
            "num_players": self.num_players,
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
        score_utility_weight: float,
        actor_victory_points: float,
        group_card_choices: bool,
        final_vp_utility_weight: float,
    ) -> _FakeNeuralSearch:
        self.last_search = {
            "probabilities": probabilities,
            "model_id": model_id,
            "simulations": simulations,
            "seed": seed,
            "determinizations": determinizations,
            "score_utility_weight": score_utility_weight,
            "actor_victory_points": actor_victory_points,
            "group_card_choices": group_card_choices,
            "final_vp_utility_weight": final_vp_utility_weight,
        }
        self.last_search_object = _FakeNeuralSearch(simulations, model_id, probabilities)
        return self.last_search_object

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
            "shared_win_values": [1.0] + [0.0] * (self.num_players - 1),
            "placements": [1] + [2] * (self.num_players - 1),
            "finish_order": list(range(self.num_players)),
            "victory_points": [12] + [7] * (self.num_players - 1),
            "victory_point_margins": [5] + [-5] * (self.num_players - 1),
            "income_levels": [4] + [2] * (self.num_players - 1),
            "money": [9] + [3] * (self.num_players - 1),
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
        with self.assertRaisesRegex(ValueError, "game_index_offset"):
            ModelSelfPlayConfig(
                output=self.root / "negative-offset.jsonl",
                checkpoint=self.root / "model.pt",
                game_index_offset=-1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "selection_temperature"):
            ModelSelfPlayConfig(
                output=self.root / "bad-temperature.jsonl",
                checkpoint=self.root / "model.pt",
                selection_temperature=-0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "score_utility_weight"):
            ModelSelfPlayConfig(
                output=self.root / "bad-score-weight.jsonl",
                checkpoint=self.root / "model.pt",
                score_utility_weight=1.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "final_vp_utility_weight"):
            ModelSelfPlayConfig(
                output=self.root / "bad-final-vp-weight.jsonl",
                checkpoint=self.root / "model.pt",
                final_vp_utility_weight=1.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "demonstration_prior_strength"):
            ModelSelfPlayConfig(
                output=self.root / "missing-demonstration.jsonl",
                checkpoint=self.root / "model.pt",
                demonstration_prior_strength=0.5,
            ).validate()
        with self.assertRaisesRegex(ValueError, "valid seat indices"):
            ModelSelfPlayConfig(
                output=self.root / "bad-demonstration-seat.jsonl",
                checkpoint=self.root / "model.pt",
                num_players=2,
                demonstration_prior_players=(2,),
            ).validate()
        with self.assertRaisesRegex(ValueError, "must be unique"):
            ModelSelfPlayConfig(
                output=self.root / "duplicate-demonstration-seats.jsonl",
                checkpoint=self.root / "model.pt",
                demonstration_prior_players=(0, 0),
            ).validate()

    def test_zero_temperature_selects_stable_highest_visit_action(self) -> None:
        targets = [
            {"index": 0, "visits": 3},
            {"index": 1, "visits": 9},
            {"index": 2, "visits": 9},
        ]
        self.assertEqual(_sample_action_index(targets, 123, 0.0), 1)

    def test_lifecycle_selection_aggregates_visited_card_variants(self) -> None:
        targets = [
            {
                "key": "loan|c0,confirm",
                "visits": 3,
                "strategy_guarded": False,
                "strategy_prior_score": 1.0,
                "strategy_prior_probability": 0.45,
            },
            {
                "key": "loan|c1,confirm",
                "visits": 1,
                "strategy_guarded": False,
                "strategy_prior_score": 1.0,
                "strategy_prior_probability": 0.05,
            },
            {
                "key": "loan|c2,confirm",
                "visits": 0,
                "strategy_guarded": False,
                "strategy_prior_score": 1.0,
                "strategy_prior_probability": 0.05,
            },
            {
                "key": "build|i5,c0,b36,confirm",
                "visits": 2,
                "strategy_guarded": False,
                "strategy_prior_score": 0.0,
                "strategy_prior_probability": 0.45,
            },
        ]

        weights = _lifecycle_selection_weights(targets, 1.0)

        self.assertAlmostEqual(weights[0] + weights[1], 4.0 * math.exp(2.0))
        self.assertAlmostEqual(weights[0] / weights[1], 9.0)
        self.assertEqual(weights[2], 0.0)
        self.assertAlmostEqual(weights[3], 2.0)

    def test_game_index_offset_is_applied_to_every_game_record(self) -> None:
        output = self.root / "offset-self-play.jsonl"
        config = ModelSelfPlayConfig(
            output=output,
            checkpoint=self.root / "model.pt",
            games=1,
            game_index_offset=9,
            simulations_per_decision=4,
            engine_revision="test-engine",
            group_card_choices=True,
            selection_temperature=0.0,
        )

        export_model_self_play(
            config,
            engine_module=_FakeEngine,
            evaluator=_FakeEvaluator(),
        )

        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(records[0]["game_index_offset"], 9)
        self.assertEqual(records[1]["game_index"], 9)
        self.assertEqual(records[2]["game_index"], 9)
        self.assertEqual(records[1]["game_seed"], records[2]["game_seed"])

    def test_three_player_export_uses_batched_neural_puct(self) -> None:
        output = self.root / "three-player-self-play.jsonl"
        config = ModelSelfPlayConfig(
            output=output,
            checkpoint=self.root / "model.pt",
            games=1,
            num_players=3,
            simulations_per_decision=5,
            search_determinizations=3,
            inference_batch_size=7,
            engine_revision="test-engine",
            selection_temperature=0.0,
        )

        export_model_self_play(
            config,
            engine_module=_FakeEngine,
            evaluator=_FakeEvaluator(),
        )

        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(records[0]["num_players"], 3)
        self.assertEqual(
            records[0]["search_method"],
            "determinized_batched_neural_puct",
        )
        self.assertEqual(records[1]["shared_win_values"], [1.0, 0.0, 0.0])
        self.assertIsNotNone(_FakeGame.last_instance)
        self.assertEqual(_FakeGame.last_instance.last_search["simulations"], 5)
        self.assertEqual(_FakeGame.last_instance.last_search["determinizations"], 3)
        self.assertEqual(
            _FakeGame.last_instance.last_search_object.requested_batch_sizes,
            [7],
        )

    def test_export_is_dataset_compatible_and_keeps_model_search_evidence(self) -> None:
        output = self.root / "model-self-play.jsonl"
        config = ModelSelfPlayConfig(
            output=output,
            checkpoint=self.root / "model.pt",
            games=1,
            simulations_per_decision=4,
            engine_revision="test-engine",
            group_card_choices=True,
            selection_temperature=0.0,
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
        self.assertEqual(
            records[0]["card_choice_grouping_version"],
            "card-invariant-intent-v1-max",
        )
        self.assertEqual(records[0]["selection_temperature"], 0.0)
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
        self.assertEqual(position["root_model_actor_victory_points"], 72.0)

        dataset = SelfPlayDataset([output])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].policy_target.tolist(), [1.0, 0.0])
        self.assertEqual(dataset[0].actor_victory_points_target.item(), 12.0)
        dataset.close()

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            export_model_self_play(
                config,
                engine_module=_FakeEngine,
                evaluator=evaluator,
            )

    def test_demonstration_prior_reaches_root_search_and_export_metadata(self) -> None:
        demonstration_path = self.root / "human.jsonl"
        demonstration_path.write_text(
            "\n".join(
                json.dumps(record, separators=(",", ":"))
                for record in (
                    {"record_type": "metadata"},
                    {"record_type": "game"},
                    {
                        "record_type": "position",
                        "phase": "canal",
                        "round_in_phase": 0,
                        "actions_remaining_in_turn": 1,
                        "selected_action_key": "action-b",
                    },
                    {
                        "record_type": "position",
                        "phase": "canal",
                        "round_in_phase": 0,
                        "actions_remaining_in_turn": 1,
                        "selected_action_key": "action-b",
                    },
                    {
                        "record_type": "position",
                        "phase": "canal",
                        "round_in_phase": 0,
                        "actions_remaining_in_turn": 1,
                        "selected_action_key": "action-a",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        prior = load_demonstration_prior((demonstration_path,))
        output = self.root / "demonstration-self-play.jsonl"
        config = ModelSelfPlayConfig(
            output=output,
            checkpoint=self.root / "model.pt",
            games=1,
            simulations_per_decision=4,
            engine_revision="test-engine",
            selection_temperature=0.0,
            demonstration_prior=prior,
            demonstration_prior_strength=1.0,
        )

        export_model_self_play(
            config,
            engine_module=_FakeEngine,
            evaluator=_FakeEvaluator(),
        )

        self.assertGreater(
            _FakeGame.last_instance.last_search["probabilities"][1],
            _FakeGame.last_instance.last_search["probabilities"][0],
        )
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(records[0]["demonstration_prior_positions"], 3)
        self.assertEqual(records[0]["demonstration_prior_strength"], 1.0)
        self.assertEqual(records[2]["demonstration_prior_positions"], 3)
        self.assertGreater(
            records[2]["legal_actions"][1]["demonstration_prior_probability"],
            records[2]["legal_actions"][0]["demonstration_prior_probability"],
        )

        # A seat-restricted prior leaves an opponent's root policy untouched.
        restricted_output = self.root / "restricted-demonstration-self-play.jsonl"
        restricted_config = ModelSelfPlayConfig(
            output=restricted_output,
            checkpoint=self.root / "model.pt",
            games=1,
            simulations_per_decision=4,
            engine_revision="test-engine",
            selection_temperature=0.0,
            demonstration_prior=prior,
            demonstration_prior_strength=1.0,
            demonstration_prior_players=(1,),
        )
        export_model_self_play(
            restricted_config,
            engine_module=_FakeEngine,
            evaluator=_FakeEvaluator(),
        )
        self.assertEqual(
            _FakeGame.last_instance.last_search["probabilities"],
            [0.75, 0.25],
        )
        restricted_records = [
            json.loads(line) for line in restricted_output.read_text().splitlines()
        ]
        self.assertFalse(restricted_records[2]["demonstration_prior_applied"])
        self.assertIsNone(
            restricted_records[2]["legal_actions"][0][
                "demonstration_prior_probability"
            ]
        )

if __name__ == "__main__":
    unittest.main()
