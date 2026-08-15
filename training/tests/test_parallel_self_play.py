from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.parallel_self_play import (
    ParallelSelfPlayConfig,
    build_shard_plans,
    export_parallel_self_play,
)


class ParallelSelfPlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def config(self, **overrides: object) -> ParallelSelfPlayConfig:
        values = {
            "output_prefix": self.root / "train",
            "checkpoint": self.root / "model.pt",
            "games": 10,
            "workers": 3,
            "game_index_offset": 7,
            "engine_revision": "test-engine",
        }
        values.update(overrides)
        return ParallelSelfPlayConfig(**values)

    def test_plans_cover_contiguous_global_game_indices(self) -> None:
        plans = build_shard_plans(self.config())

        self.assertEqual([plan.shard_index for plan in plans], [0, 1, 2])
        self.assertEqual([plan.games for plan in plans], [4, 3, 3])
        self.assertEqual(
            [plan.game_index_offset for plan in plans],
            [7, 11, 14],
        )
        self.assertEqual(
            [plan.output.name for plan in plans],
            ["train-000.jsonl", "train-001.jsonl", "train-002.jsonl"],
        )

    def test_config_rejects_unsafe_worker_and_prefix_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            self.config(games=2, workers=3).validate()
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            self.config(workers=65).validate()
        with self.assertRaisesRegex(ValueError, "must not end in .jsonl"):
            self.config(output_prefix=self.root / "train.jsonl").validate()

    def test_existing_output_is_rejected_before_workers_start(self) -> None:
        config = self.config(games=2, workers=2, game_index_offset=0)
        first_output = build_shard_plans(config)[0].output
        first_output.write_text("already complete\n", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            export_parallel_self_play(config)

    def test_existing_partial_is_rejected_before_workers_start(self) -> None:
        config = self.config(games=2, workers=2, game_index_offset=0)
        first_output = build_shard_plans(config)[0].output
        Path(f"{first_output}.partial").write_text("incomplete\n", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "partial"):
            export_parallel_self_play(config)


if __name__ == "__main__":
    unittest.main()
