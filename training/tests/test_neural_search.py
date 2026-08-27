from __future__ import annotations

import unittest
from types import SimpleNamespace

from training.neural_search import run_batched_neural_puct
from training.schema import FeatureSchema


class _CompletingSearch:
    def __init__(self, simulations: int) -> None:
        self.simulations = simulations
        self.complete = False

    def is_complete(self) -> bool:
        return self.complete

    def next_inference_batch(self, _max_batch_size: int) -> dict:
        self.complete = True
        return {
            "positions": [],
            "completed_simulations": self.simulations,
            "is_complete": True,
        }

    def finish(self) -> dict:
        return {
            "method": "determinized_batched_neural_puct",
            "model_id": "sha256:test-model",
            "completed_simulations": self.simulations,
            "max_search_depth": 1,
        }


class _CompletingGame:
    def start_batched_neural_search(self, *args) -> _CompletingSearch:
        return _CompletingSearch(int(args[4]))


class _UnusedEvaluator:
    def predict_batch(self, *_args, **_kwargs):
        raise AssertionError("a completed terminal batch must not call inference")


class NeuralSearchTests(unittest.TestCase):
    def test_search_can_complete_while_reserving_a_terminal_only_batch(self) -> None:
        schema = FeatureSchema.from_schema_dict(
            {"version": 1, "state_dim": 4, "action_dim": 8}
        )
        prediction = SimpleNamespace(
            model_id="sha256:test-model",
            policy_probabilities=(0.75, 0.25),
            shared_win_rate=0.6,
            victory_point_margin=2.0,
            actor_victory_points=70.0,
        )
        report = run_batched_neural_puct(
            _CompletingGame(),
            _UnusedEvaluator(),
            schema,
            prediction,
            simulations=3,
            search_seed=42,
            exploration_constant=1.5,
            determinizations=2,
            inference_batch_size=8,
        )
        self.assertEqual(report["completed_simulations"], 3)


if __name__ == "__main__":
    unittest.main()
