from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from training.evaluate import (
    EvaluationPolicy,
    MatchResult,
    _select_search_action,
    _student_t_critical_95,
    build_parser,
    multiplayer_score_delta,
    summarize_gate,
)
from training.schema import FeatureSchema


class EvaluationGateTests(unittest.TestCase):
    def test_parser_accepts_independent_remote_model_sources(self) -> None:
        args = build_parser().parse_args(
            [
                "--candidate-inference-url",
                "http://host.test:8766",
                "--champion-inference-url",
                "http://host.test:8765",
            ]
        )
        self.assertIsNone(args.candidate)
        self.assertIsNone(args.champion)
        self.assertEqual(args.candidate_inference_url, "http://host.test:8766")
        self.assertEqual(args.champion_inference_url, "http://host.test:8765")
        self.assertEqual(args.candidate_score_utility_weight, 0.0)
        self.assertEqual(args.champion_score_utility_weight, 0.0)

    def test_evaluation_policy_selects_greedy_action_via_common_protocol(self) -> None:
        schema = FeatureSchema.from_schema_dict(
            {"version": 1, "state_dim": 2, "action_dim": 4}
        )
        evaluator = _FakeEvaluator(schema)
        policy = EvaluationPolicy(
            evaluator, schema, "http://candidate.test", "remote_inference"
        )

        selected = policy.select_action(
            {},
            {"feature_version": 1, "features": [0.0, 1.0]},
            {
                "feature_version": 1,
                "actions": [
                    {"index": 0, "key": "a", "feature_indices": [0]},
                    {"index": 1, "key": "b", "feature_indices": [1]},
                ],
            },
        )

        self.assertEqual(selected, 1)

    def test_evaluation_policy_passes_score_utility_weight_to_neural_search(self) -> None:
        schema = FeatureSchema.from_schema_dict(
            {"version": 1, "state_dim": 2, "action_dim": 4}
        )
        evaluator = _FakeEvaluator(schema)
        policy = EvaluationPolicy(
            evaluator,
            schema,
            "utility-candidate",
            "checkpoint",
            score_utility_weight=0.35,
        )
        report = {
            "method": "determinized_batched_neural_puct",
            "model_id": evaluator.model_id,
            "completed_simulations": 4,
            "actions": [
                {
                    "index": 0,
                    "key": "a",
                    "visits": 3,
                    "estimated_shared_win_rate": 0.5,
                    "policy_probability": 0.5,
                },
                {
                    "index": 1,
                    "key": "b",
                    "visits": 1,
                    "estimated_shared_win_rate": 0.5,
                    "policy_probability": 0.5,
                },
            ],
        }
        with patch(
            "training.evaluate.run_batched_neural_puct", return_value=report
        ) as search:
            selected = policy.select_action_with_search(
                None,
                {},
                {"feature_version": 1, "features": [0.0, 1.0]},
                {
                    "feature_version": 1,
                    "actions": [
                        {"index": 0, "key": "a", "feature_indices": [0]},
                        {"index": 1, "key": "b", "feature_indices": [1]},
                    ],
                },
                2,
                4,
                1.5,
                4,
                32,
                99,
            )

        self.assertEqual(selected, 0)
        self.assertEqual(search.call_args.kwargs["score_utility_weight"], 0.35)

    def test_evaluation_policy_uses_batched_neural_search_for_three_players(self) -> None:
        schema = FeatureSchema.from_schema_dict(
            {"version": 1, "state_dim": 2, "action_dim": 4}
        )
        evaluator = _FakeEvaluator(schema)
        policy = EvaluationPolicy(
            evaluator,
            schema,
            "three-player-candidate",
            "checkpoint",
        )
        report = {
            "method": "determinized_batched_neural_puct",
            "model_id": evaluator.model_id,
            "completed_simulations": 6,
            "actions": [
                {
                    "index": 0,
                    "key": "a",
                    "visits": 1,
                    "estimated_shared_win_rate": 0.4,
                    "policy_probability": 0.25,
                },
                {
                    "index": 1,
                    "key": "b",
                    "visits": 5,
                    "estimated_shared_win_rate": 0.6,
                    "policy_probability": 0.75,
                },
            ],
        }
        game = object()
        with patch(
            "training.evaluate.run_batched_neural_puct", return_value=report
        ) as search:
            selected = policy.select_action_with_search(
                game,
                {},
                {"feature_version": 1, "features": [0.0, 1.0]},
                {
                    "feature_version": 1,
                    "actions": [
                        {"index": 0, "key": "a", "feature_indices": [0]},
                        {"index": 1, "key": "b", "feature_indices": [1]},
                    ],
                },
                3,
                6,
                1.5,
                5,
                16,
                123,
            )

        self.assertEqual(selected, 1)
        self.assertIs(search.call_args.args[0], game)
        self.assertEqual(search.call_args.kwargs["determinizations"], 5)
        self.assertEqual(search.call_args.kwargs["inference_batch_size"], 16)

    def test_evaluation_policy_can_apply_strategy_prior_at_the_root(self) -> None:
        schema = FeatureSchema.from_schema_dict(
            {"version": 1, "state_dim": 2, "action_dim": 4}
        )
        evaluator = _FakeEvaluator(schema, probabilities=(0.99, 0.01))
        policy = EvaluationPolicy(
            evaluator,
            schema,
            "guided-candidate",
            "checkpoint",
            strategy_prior_strength=1.0,
            strategy_prior_version="human-strategy-v5-map-aware-lifecycle",
        )
        observation = {
            "decision_player": 0,
            "global_features": [1.0, 0.0, 0.0],
            "players_public": [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
            "industry_mats": [[0.125, 1.0, 1.0] * 6],
            "buildings": [],
            "roads": [],
            "self_hand_counts": [],
        }
        legal_record = {
            "feature_version": 1,
            "actions": [
                {"index": 0, "key": "pass|c0,confirm", "feature_indices": [0]},
                {"index": 1, "key": "loan|c0,confirm", "feature_indices": [1]},
            ],
        }

        selected = policy.select_action(
            observation,
            {"feature_version": 1, "features": [0.0, 1.0]},
            legal_record,
        )

        self.assertEqual(selected, 1)

    def test_evaluation_module_import_does_not_require_torch(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('training.evaluate imported torch eagerly')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import training.evaluate
print('torch-free import ok')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("torch-free import ok", completed.stdout)

    def test_multiplayer_score_delta_compares_candidate_to_each_champion_copy(
        self,
    ) -> None:
        self.assertEqual(multiplayer_score_delta(1.0, 4), 1.0)
        self.assertAlmostEqual(multiplayer_score_delta(0.0, 4), -1.0 / 3.0)
        self.assertAlmostEqual(multiplayer_score_delta(0.25, 4), 0.0)
        self.assertAlmostEqual(multiplayer_score_delta(0.5, 2), 0.0)

    def test_gate_requires_strict_positive_lower_confidence_bound(self) -> None:
        winning = [_result(index, 1.0) for index in range(8)]
        winning_summary = summarize_gate(winning, promotion_margin=0.0, minimum_games=8)
        self.assertTrue(winning_summary.promote)
        self.assertEqual(winning_summary.score_delta_lower_95, 1.0)

        tied = [_result(index, 0.0) for index in range(8)]
        tied_summary = summarize_gate(tied, promotion_margin=0.0, minimum_games=8)
        self.assertFalse(tied_summary.promote)
        self.assertEqual(tied_summary.score_delta_lower_95, 0.0)

    def test_gate_respects_minimum_game_count(self) -> None:
        summary = summarize_gate(
            [_result(index, 1.0) for index in range(4)],
            promotion_margin=0.0,
            minimum_games=8,
        )
        self.assertFalse(summary.promote)

    def test_confidence_interval_uses_seat_rotated_seed_groups(self) -> None:
        results = [
            _result(0, 1.0),
            _result(1, -1.0),
            _result(2, 1.0),
            _result(3, -1.0),
        ]

        summary = summarize_gate(
            results, promotion_margin=0.0, minimum_games=4
        )

        self.assertEqual(summary.games, 4)
        self.assertEqual(summary.independent_seed_groups, 2)
        self.assertEqual(summary.candidate_mean_score_delta, 0.0)
        self.assertEqual(summary.score_delta_standard_error, 0.0)
        self.assertEqual(summary.score_delta_lower_95, 0.0)
        self.assertFalse(summary.promote)

    def test_default_forty_game_gate_uses_student_t_not_normal_limit(self) -> None:
        self.assertAlmostEqual(_student_t_critical_95(39), 2.02269092, places=7)
        self.assertGreater(_student_t_critical_95(39), 1.96)

    def test_student_t_critical_rejects_nonpositive_degrees_of_freedom(self) -> None:
        with self.assertRaises(ValueError):
            _student_t_critical_95(0)

    def test_search_selector_uses_visits_and_validates_stable_order(self) -> None:
        report = {
            "method": "value-puct",
            "model_id": "model-a",
            "completed_simulations": 8,
            "actions": [
                {
                    "index": 0,
                    "key": "a",
                    "visits": 2,
                    "estimated_shared_win_rate": 0.9,
                    "policy_probability": 0.8,
                },
                {
                    "index": 1,
                    "key": "b",
                    "visits": 6,
                    "estimated_shared_win_rate": 0.4,
                    "policy_probability": 0.2,
                },
            ],
        }
        self.assertEqual(
            _select_search_action(report, ("a", "b"), "model-a", "value-puct", 8),
            1,
        )
        report["actions"][1]["key"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "order changed"):
            _select_search_action(report, ("a", "b"), "model-a", "value-puct", 8)


def _result(index: int, score_delta: float) -> MatchResult:
    return MatchResult(
        round_index=index // 2,
        game_seed=123 + index // 2,
        candidate_seat=index % 2,
        actions=10,
        candidate_shared_win=1.0 if score_delta > 0.0 else 0.5,
        candidate_score_delta=score_delta,
        candidate_victory_point_margin=1 if score_delta > 0.0 else 0,
        candidate_placement=1,
        candidate_is_official_winner=True,
        official_winners=(index % 2,),
        victory_points=(10, 9),
    )


class _FakeEvaluator:
    def __init__(
        self,
        schema: FeatureSchema,
        probabilities: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.schema = schema
        self.model_id = "sha256:fake"
        self.checkpoint_step = 9
        self.probabilities = probabilities

    def predict(self, state_record, legal_record, request_schema=None):
        self.schema.assert_compatible(request_schema, "fake request")
        return SimpleNamespace(
            model_id=self.model_id,
            checkpoint_step=self.checkpoint_step,
            action_keys=tuple(action["key"] for action in legal_record["actions"]),
            policy_probabilities=self.probabilities,
            shared_win_rate=0.5,
            victory_point_margin=0.0,
            actor_victory_points=20.0,
        )

    def predict_batch(self, state_records, legal_records, request_schema=None):
        return tuple(
            self.predict(state, legal, request_schema)
            for state, legal in zip(state_records, legal_records, strict=True)
        )


if __name__ == "__main__":
    unittest.main()
