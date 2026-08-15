from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.checkpoint import load_model_checkpoint
from training.data import SelfPlayDataset, collate_positions
from training.model import BrassPolicyValueNet, ModelConfig
from training.schema import FeatureSchema
from training.train import _evaluate, _load_datasets, _validate_args, build_parser, train


SCHEMA = {
    "version": 1,
    "state_dim": 2,
    "action_dim": 4,
    "state_blocks": [{"name": "all", "offset": 0, "size": 2}],
    "action_blocks": [{"name": "all", "offset": 0, "size": 4}],
}


class TrainingPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_independent_validation_shards_are_loaded_separately(self) -> None:
        training = self.root / "training.jsonl"
        validation = self.root / "validation.jsonl"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        _write_shard(validation, engine_revision="engine-a", game_seed=202)

        training_dataset, validation_dataset = _load_datasets(
            [str(training)], [str(validation)]
        )

        self.assertEqual(len(training_dataset), 1)
        self.assertIsNotNone(validation_dataset)
        assert validation_dataset is not None
        self.assertEqual(len(validation_dataset), 1)
        self.assertEqual(training_dataset.game_seeds, {101})
        self.assertEqual(validation_dataset.game_seeds, {202})
        training_dataset.close()
        validation_dataset.close()

    def test_training_and_validation_paths_must_not_overlap(self) -> None:
        shard = self.root / "same.jsonl"
        _write_shard(shard, engine_revision="engine-a", game_seed=101)
        with self.assertRaisesRegex(ValueError, "shards overlap"):
            _load_datasets([str(shard)], [str(shard)])

    def test_validation_engine_revision_must_match_training(self) -> None:
        training = self.root / "training.jsonl"
        validation = self.root / "validation.jsonl"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        _write_shard(validation, engine_revision="engine-b", game_seed=202)
        with self.assertRaisesRegex(ValueError, "engine revisions differ"):
            _load_datasets([str(training)], [str(validation)])

    def test_validation_game_seeds_must_not_overlap_training(self) -> None:
        training = self.root / "training.jsonl"
        validation = self.root / "validation.jsonl"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        _write_shard(validation, engine_revision="engine-a", game_seed=101)
        with self.assertRaisesRegex(ValueError, "reuse game seeds"):
            _load_datasets([str(training)], [str(validation)])

    def test_random_position_validation_split_is_rejected(self) -> None:
        args = build_parser().parse_args(
            [
                "--shards",
                "training.jsonl",
                "--output",
                "model.pt",
                "--validation-fraction",
                "0.1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "leaks positions"):
            _validate_args(args)

    def test_training_checkpoint_selects_and_records_best_validation_epoch(self) -> None:
        training = self.root / "training.jsonl"
        validation = self.root / "validation.jsonl"
        checkpoint = self.root / "model.pt"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        _write_shard(validation, engine_revision="engine-a", game_seed=202)
        args = build_parser().parse_args(
            [
                "--shards",
                str(training),
                "--validation-shards",
                str(validation),
                "--output",
                str(checkpoint),
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--batch-size",
                "1",
                "--state-hidden-dim",
                "8",
                "--action-embedding-dim",
                "4",
                "--trunk-dim",
                "8",
                "--log-every",
                "1",
            ]
        )

        with redirect_stdout(io.StringIO()):
            train(args)

        schema = FeatureSchema.from_schema_dict(SCHEMA)
        _model, payload = load_model_checkpoint(
            checkpoint, expected_schema=schema
        )
        metadata = payload["metadata"]["training"]
        self.assertIn("loss", metadata["initial_validation_metrics"])
        self.assertIn("loss", metadata["selected_validation_metrics"])
        self.assertIn("loss", metadata["last_validation_metrics"])
        self.assertLessEqual(
            metadata["selected_validation_metrics"]["loss"],
            metadata["initial_validation_metrics"]["loss"],
        )
        self.assertLessEqual(metadata["global_step"], metadata["last_global_step"])

    def test_validation_metrics_are_independent_of_batch_boundaries(self) -> None:
        validation = self.root / "validation.jsonl"
        _write_shard(
            validation,
            engine_revision="engine-a",
            game_seed=202,
            position_count=3,
        )
        dataset = SelfPlayDataset([validation])
        torch.manual_seed(73)
        model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=2,
                action_dim=4,
                state_hidden_dim=8,
                action_embedding_dim=4,
                trunk_dim=8,
                vp_margin_scale=100.0,
            )
        )
        loaders = [
            DataLoader(dataset, batch_size=size, collate_fn=collate_positions)
            for size in (2, 3)
        ]

        metrics = [
            _evaluate(model, loader, torch.device("cpu"), 1.0, 0.25)
            for loader in loaders
        ]

        self.assertEqual(metrics[0].keys(), metrics[1].keys())
        for name in metrics[0]:
            self.assertAlmostEqual(metrics[0][name], metrics[1][name], places=6)
        dataset.close()


def _write_shard(
    path: Path,
    *,
    engine_revision: str,
    game_seed: int,
    position_count: int = 1,
) -> None:
    records = [
        {
            "record_type": "metadata",
            "format": "fast_brass_self_play_jsonl",
            "format_version": 1,
            "engine_revision": engine_revision,
            "feature_schema": SCHEMA,
        },
        {
            "record_type": "game",
            "format_version": 1,
            "game_index": 0,
            "game_seed": game_seed,
        },
    ]
    for position_index in range(position_count):
        state_fraction = position_index / max(1, position_count - 1)
        shared_win = (0.0, 1.0, 0.5)[position_index % 3]
        records.append(
            {
                "record_type": "position",
                "format_version": 1,
                "feature_version": 1,
                "game_index": 0,
                "position_index": position_index,
                "state_features": [state_fraction, 1.0 - state_fraction],
                "legal_actions": [
                    {
                        "index": 0,
                        "key": "action-a",
                        "feature_indices": [0],
                        "visits": 1,
                        "policy_target": 1.0,
                    }
                ],
                "selected_action_index": 0,
                "selected_action_key": "action-a",
                "value_target": {
                    "shared_win": shared_win,
                    "victory_point_margin": position_index * 3 - 3,
                },
            },
        )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    unittest.main()
