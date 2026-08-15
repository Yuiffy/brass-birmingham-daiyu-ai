from __future__ import annotations

import unittest

from training.evaluate import (
    MatchResult,
    _select_search_action,
    _student_t_critical_95,
    multiplayer_score_delta,
    summarize_gate,
)


class EvaluationGateTests(unittest.TestCase):
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
        round_index=0,
        game_seed=123,
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


if __name__ == "__main__":
    unittest.main()
