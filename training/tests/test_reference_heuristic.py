from __future__ import annotations

import unittest

from training.reference_heuristic import ReferenceHeuristicPolicy


class ReferenceHeuristicTests(unittest.TestCase):
    def test_pass_is_selected_when_it_is_the_only_action(self) -> None:
        observation = {
            "global_features": [1.0, 0.0, 0.0],
            "players_public": [[1.0] * 10, [0.0] * 10],
            "hand_sizes": [0.125, 0.125],
            "discard_counts": [0.0] * 29,
            "industry_mats": [[0.125, 1.0, 1.0] * 6, [0.125, 1.0, 1.0] * 6],
            "buildings": [],
            "roads": [],
            "trade_post_slots": [],
            "trade_post_beer": [],
            "self_hand_counts": [0.0] * 29,
            "decision_player": 0,
        }
        legal = {
            "actions": [
                {
                    "key": "pass|c0,confirm",
                    "root_action_name": "pass",
                    "selected_card": 0,
                    "discard_card_types": [0],
                    "choices": [
                        {"kind": "card", "value": 0},
                        {"kind": "confirm", "value": None},
                    ],
                }
            ]
        }
        policy = ReferenceHeuristicPolicy()
        self.assertAlmostEqual(policy.score_actions(observation, legal)[0], -0.5 + 0.0001)

    def test_scoring_is_deterministic(self) -> None:
        observation = {
            "global_features": [1.0, 0.0, 0.0],
            "players_public": [[1.0] * 10, [0.0] * 10],
            "hand_sizes": [0.125, 0.125],
            "discard_counts": [0.0] * 29,
            "industry_mats": [[0.125, 1.0, 1.0] * 6, [0.125, 1.0, 1.0] * 6],
            "buildings": [],
            "roads": [],
            "trade_post_slots": [],
            "trade_post_beer": [],
            "self_hand_counts": [0.0] * 29,
            "decision_player": 0,
        }
        legal = {
            "actions": [
                {
                    "key": "pass|c0,confirm",
                    "root_action_name": "pass",
                    "selected_card": 0,
                    "discard_card_types": [0],
                    "choices": [{"kind": "card", "value": 0}],
                },
                {
                    "key": "loan|c0,confirm",
                    "root_action_name": "loan",
                    "selected_card": 0,
                    "discard_card_types": [0],
                    "choices": [{"kind": "card", "value": 0}],
                },
            ]
        }
        policy = ReferenceHeuristicPolicy()
        first = policy.score_actions(observation, legal)
        second = policy.score_actions(observation, legal)
        self.assertEqual(first, second)
if __name__ == "__main__":
    unittest.main()
