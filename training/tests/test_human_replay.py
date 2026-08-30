from __future__ import annotations

import unittest

from training.human_replay import (
    ConfirmedAction,
    audit_replay_events,
    _choices_to_payload,
    _normalized_event_choices,
    _normalized_legal_choices,
    _semantic_choice_signature,
    _semantic_legal_signature,
    action_type_key,
)


class HumanReplayTests(unittest.TestCase):
    def test_clean_replay_is_value_eligible(self):
        events = [
            {"kind": "start_turn"},
            {"kind": "start_action", "action_type": "Loan"},
            {"kind": "apply_choice", "choice_kind": "card", "value": 0},
            {"kind": "confirm_action"},
            {
                "kind": "replay_move",
                "position_key": "Canal:0:0:0:1:true",
                "player_idx": 0,
                "before_state": {
                    "phase": "Canal",
                    "era": "Canal",
                    "round_in_phase": 0,
                    "turn_count": 0,
                    "current_player": 0,
                    "actions_remaining": 1,
                    "choice_set": {"kind": "card"},
                },
                "after_state": {},
            },
            {"kind": "end_turn"},
        ]
        audit = audit_replay_events(events, num_players=2)
        self.assertEqual(audit.replay_integrity, "clean")
        self.assertTrue(audit.value_target_usable)
        self.assertEqual(audit.anomaly_counts, {})

    def test_duplicate_end_turn_contaminates_value_targets(self):
        events = [
            {"kind": "start_turn"},
            {"kind": "start_action", "action_type": "Pass"},
            {"kind": "confirm_action"},
            {
                "kind": "replay_move",
                "position_key": "Canal:0:0:0:0:true",
                "player_idx": 0,
                "before_state": {
                    "phase": "Canal",
                    "era": "Canal",
                    "round_in_phase": 0,
                    "turn_count": 0,
                    "current_player": 0,
                    "actions_remaining": 0,
                    "choice_set": {"kind": "confirm"},
                },
            },
            {"kind": "end_turn"},
            {"kind": "end_turn"},
        ]
        audit = audit_replay_events(events, num_players=2)
        self.assertEqual(audit.replay_integrity, "contaminated")
        self.assertFalse(audit.value_target_usable)
        self.assertEqual(audit.anomaly_counts["consecutive_end_turn"], 1)
        self.assertEqual(audit.anomaly_counts["end_turn_without_active_turn"], 1)
        self.assertEqual(audit.anomaly_counts["end_turn_without_confirmed_action"], 1)

    def test_composite_payload_encodes_duplicate_develop_and_double_rail_choices(self):
        payload = _choices_to_payload(
            "DevelopDouble",
            [
                ("card", 7),
                ("industry", "Beer"),
                ("industry", "Cotton"),
                ("iron_source", "Market"),
            ],
        )
        self.assertEqual(payload["root_action"], 3)
        self.assertEqual(payload["industry"], [2])
        self.assertEqual(payload["second_industry"], [5])
        self.assertEqual(payload["iron_sources"], [49])

        rail_payload = _choices_to_payload(
            "BuildRailroad",
            [
                ("card", 1),
                ("network_mode", "Double"),
                ("road", 18),
                ("coal_source", "Market"),
                ("road", 5),
                ("coal_source", {"Building": 2}),
                ("action_beer_source", {"OpponentBrewery": 15}),
            ],
        )
        self.assertEqual(rail_payload["root_action"], 1)
        self.assertEqual(rail_payload["road"], [18])
        self.assertEqual(rail_payload["second_road"], [5])
        self.assertEqual(rail_payload["coal_sources"], [49, 2])
        self.assertEqual(rail_payload["action_beer_source"], [64])

    def test_semantic_card_matching_handles_engine_card_deduplication(self):
        record = ConfirmedAction(
            action_type="Loan",
            choices=[("card", 6)],
            state_record={"features": []},
            legal_record={"actions": []},
            before_save=None,
            position_key="Canal:0:1:0:1:true",
            replay_move={
                "player_idx": 0,
                "before_state": {
                    "players": [
                        {
                            "hand": [
                                {"card_type": "Location(BurtonUponTrent)"}
                            ]
                        }
                    ]
                },
            },
        )
        expected = _normalized_event_choices("Loan", [("card", 6)])
        legal = {
            "choices": [
                {"kind": "card", "value": 2},
                {"kind": "confirm", "value": None},
            ],
            "discard_card_types": [1],
        }
        # The fixture's hand needs the selected slot to exist; append duplicate
        # cards as they would appear in the browser snapshot.
        record.replay_move["before_state"]["players"][0]["hand"] = [
            {"card_type": "Location(Stafford)"},
            {"card_type": "Location(Stafford)"},
            {"card_type": "Location(BurtonUponTrent)"},
            {"card_type": "Location(Stafford)"},
            {"card_type": "Location(Stafford)"},
            {"card_type": "Location(Stafford)"},
            {"card_type": "Location(BurtonUponTrent)"},
        ]
        self.assertEqual(_semantic_choice_signature(record, expected), [
            ("card_type", 1)
        ])
        self.assertEqual(_semantic_legal_signature(legal), [("card_type", 1)])

    def test_action_type_key_accepts_engine_alias(self):
        self.assertEqual(action_type_key("BuildDoubleRailroad"), "builddoublerailroad")


if __name__ == "__main__":
    unittest.main()
