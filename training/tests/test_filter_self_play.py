from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.data import SelfPlayDataset
from training.filter_self_play import filter_self_play_shards


class FilterSelfPlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_filters_positions_by_actor_score_and_remains_dataset_compatible(
        self,
    ) -> None:
        source = self.root / "source.jsonl"
        _write_source(source)

        summaries = filter_self_play_shards(
            (source,), self.root / "quality", minimum_actor_vp=50
        )

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.source_positions, 2)
        self.assertEqual(summary.positions_written, 1)
        self.assertEqual(summary.eligible_player_trajectories, 1)
        output = Path(summary.output)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(records[0]["derived_dataset"], "actor_final_vp_quality_filter")
        self.assertEqual(records[0]["minimum_actor_vp"], 50)
        self.assertEqual(records[1]["source_positions"], 2)
        self.assertEqual(records[1]["positions"], 1)
        self.assertEqual(records[1]["quality_eligible_players"], [0])
        self.assertEqual(records[2]["actor"], 0)

        dataset = SelfPlayDataset([output])
        self.assertEqual(len(dataset), 1)
        dataset.close()

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            filter_self_play_shards(
                (source,), self.root / "quality", minimum_actor_vp=50
            )

    def test_rejects_game_without_victory_points(self) -> None:
        source = self.root / "missing-vp.jsonl"
        records = _records()
        records[1].pop("victory_points")
        _write_records(source, records)

        with self.assertRaisesRegex(ValueError, "victory_points is required"):
            filter_self_play_shards(
                (source,), self.root / "invalid", minimum_actor_vp=50
            )
        self.assertFalse((self.root / "invalid-000.jsonl.partial").exists())


def _write_source(path: Path) -> None:
    _write_records(path, _records())


def _records() -> list[dict]:
    schema = {
        "version": 1,
        "state_dim": 2,
        "action_dim": 2,
        "state_blocks": [{"name": "all", "offset": 0, "size": 2}],
        "action_blocks": [{"name": "all", "offset": 0, "size": 2}],
    }
    return [
        {
            "record_type": "metadata",
            "format": "fast_brass_self_play_jsonl",
            "format_version": 1,
            "engine_revision": "test-engine",
            "feature_schema": schema,
        },
        {
            "record_type": "game",
            "format_version": 1,
            "game_index": 0,
            "game_seed": 101,
            "positions": 2,
            "victory_points": [60, 20],
        },
        _position(0, 0),
        _position(1, 1),
    ]


def _position(actor: int, position_index: int) -> dict:
    return {
        "record_type": "position",
        "format_version": 1,
        "feature_version": 1,
        "game_index": 0,
        "position_index": position_index,
        "actor": actor,
        "state_features": [float(actor), float(1 - actor)],
        "legal_actions": [
            {
                "index": 0,
                "key": "pass|confirm",
                "feature_indices": [0],
                "visits": 1,
                "policy_target": 1.0,
            }
        ],
        "selected_action_index": 0,
        "selected_action_key": "pass|confirm",
        "value_target": {"shared_win": float(actor == 0), "victory_point_margin": 40},
    }


def _write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    unittest.main()
