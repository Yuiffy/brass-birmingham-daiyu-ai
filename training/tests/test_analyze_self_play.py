from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.analyze_self_play import analyze_self_play_shards


class AnalyzeSelfPlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reports_score_action_and_lifecycle_quality(self) -> None:
        shard = self.root / "sample.jsonl"
        records = [
            {
                "record_type": "metadata",
                "model_id": "sha256:test",
                "checkpoint_step": 7,
                "simulations_per_decision": 64,
                "strategy_prior_version": "test-prior",
                "strategy_prior_strength": 0.0,
                "card_choice_grouping_version": "test-grouping",
                "selection_temperature": 0.0,
            },
            {
                "record_type": "game",
                "game_index": 0,
                "positions": 6,
                "victory_points": [0, 120],
                "income_levels": [-10, 12],
                "money": [0, 9],
            },
            _position(0, "canal", "loan|c0,confirm"),
            _position(1, "canal", "build|i2,c0,b14,confirm"),
            _position(0, "canal", "loan|c1,confirm"),
            _position(0, "canal", "network|c2,r4,confirm"),
            _position(1, "railroad", "network|c1,r8,confirm"),
            _position(1, "railroad", "build|i5,c2,b20,confirm"),
        ]
        shard.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        report = analyze_self_play_shards((shard,))

        self.assertEqual(report["games"], 1)
        self.assertEqual(report["scores"]["all_players"]["mean"], 60.0)
        self.assertEqual(report["scores"]["all_players"]["at_least_100"], 1)
        self.assertEqual(report["lifecycle"]["loans"], 2)
        self.assertEqual(report["lifecycle"]["repeat_loans"], 1)
        self.assertEqual(report["lifecycle"]["network_before_first_build"], 1)
        self.assertEqual(report["actions"]["build_industries"], {"2": 1, "5": 1})
        self.assertEqual(report["outcomes"]["nonpositive_final_income_levels"], 1)

    def test_declared_position_count_is_verified(self) -> None:
        shard = self.root / "invalid.jsonl"
        records = [
            {"record_type": "metadata"},
            {
                "record_type": "game",
                "game_index": 0,
                "positions": 1,
                "victory_points": [1, 2],
                "income_levels": [0, 0],
                "money": [0, 0],
            },
        ]
        shard.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "declares 1 positions"):
            analyze_self_play_shards((shard,))


def _position(actor: int, phase: str, key: str) -> dict:
    return {
        "record_type": "position",
        "actor": actor,
        "phase": phase,
        "selected_action_key": key,
    }


if __name__ == "__main__":
    unittest.main()
