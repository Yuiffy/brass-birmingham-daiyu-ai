from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from training.checkpoint import load_model_checkpoint, save_checkpoint
from training.data import SelfPlayDataset, collate_positions
from training.model import (
    BrassPolicyValueNet,
    ModelConfig,
    fuse_policy_and_value_models,
)
from training.schema import FeatureSchema
from training.train import (
    _evaluate,
    _load_datasets,
    _make_loader,
    _restore_optimizer_state,
    _should_restore_optimizer_state,
    _validate_args,
    build_parser,
    train,
)


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

    def test_expert_iteration_source_fractions_use_replacement_sampling(self) -> None:
        teacher = self.root / "teacher.jsonl"
        replay = self.root / "replay.jsonl"
        human = self.root / "human.jsonl"
        _write_shard(teacher, engine_revision="engine-a", game_seed=101, position_count=2)
        _write_shard(replay, engine_revision="engine-a", game_seed=202, position_count=3)
        _write_shard(human, engine_revision="engine-a", game_seed=303, position_count=1)
        human_records = [json.loads(line) for line in human.read_text().splitlines()]
        human_records[0]["source_type"] = "human_replay"
        human_records[0]["value_target_usable"] = False
        human.write_text(
            "\n".join(json.dumps(record, separators=(",", ":")) for record in human_records)
            + "\n",
            encoding="utf-8",
        )

        training_dataset, validation_dataset = _load_datasets(
            [str(teacher)],
            None,
            replay_shard_paths=[str(replay)],
            human_shard_paths=[str(human)],
            replay_fraction=0.2,
            human_fraction=0.1,
        )
        self.assertIsNone(validation_dataset)
        self.assertIsNotNone(training_dataset.sampling_weights)
        assert training_dataset.sampling_weights is not None
        counts = training_dataset.shard_position_counts
        by_path = {
            path: float(training_dataset.sampling_weights[
                [locator.path for locator in training_dataset._positions].index(path)
            ]) * count
            for path, count in counts.items()
        }
        self.assertAlmostEqual(by_path[teacher.resolve()], 0.7)
        self.assertAlmostEqual(by_path[replay.resolve()], 0.2)
        self.assertAlmostEqual(by_path[human.resolve()], 0.1)
        groups = training_dataset.sampling_metadata["groups"]
        self.assertEqual([group["name"] for group in groups], ["teacher", "replay", "human"])
        self.assertEqual(training_dataset[0].value_loss_weight, 1.0)
        human_index = next(
            index
            for index, locator in enumerate(training_dataset._positions)
            if locator.path == human.resolve()
        )
        self.assertEqual(training_dataset[human_index].value_loss_weight, 0.0)

        args = build_parser().parse_args(
            [
                "--shards",
                str(teacher),
                "--replay-shards",
                str(replay),
                "--human-shards",
                str(human),
                "--output",
                str(self.root / "model.pt"),
                "--batch-size",
                "2",
            ]
        )
        loader = _make_loader(
            training_dataset,
            args,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
        )
        self.assertIsInstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
        training_dataset.close()

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

    def test_early_stopping_requires_independent_validation_shards(self) -> None:
        args = build_parser().parse_args(
            [
                "--shards",
                "training.jsonl",
                "--output",
                "model.pt",
                "--early-stopping-patience",
                "2",
            ]
        )

        with self.assertRaisesRegex(ValueError, "independent validation shards"):
            _validate_args(args)

    def test_early_stopping_rejects_invalid_thresholds(self) -> None:
        cases = (
            (["--early-stopping-patience", "-1"], "patience must be non-negative"),
            (
                ["--early-stopping-min-delta", "-0.1"],
                "min-delta must be non-negative",
            ),
            (
                ["--early-stopping-min-delta", "0.1"],
                "requires positive early-stopping-patience",
            ),
        )

        for extra_args, expected_error in cases:
            with self.subTest(extra_args=extra_args):
                args = build_parser().parse_args(
                    [
                        "--shards",
                        "training.jsonl",
                        "--output",
                        "model.pt",
                        *extra_args,
                    ]
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    _validate_args(args)

    def test_source_mixture_fractions_are_bounded(self) -> None:
        cases = (
            ("--replay-fraction", "1.0", "between 0 and 1"),
            ("--human-fraction", "-0.1", "between 0 and 1"),
        )
        for option, value, expected in cases:
            with self.subTest(option=option):
                args = build_parser().parse_args(
                    [
                        "--shards",
                        "training.jsonl",
                        "--output",
                        "model.pt",
                        option,
                        value,
                    ]
                )
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_args(args)

        args = build_parser().parse_args(
            [
                "--shards",
                "training.jsonl",
                "--replay-shards",
                "replay.jsonl",
                "--human-shards",
                "human.jsonl",
                "--output",
                "model.pt",
                "--replay-fraction",
                "0.7",
                "--human-fraction",
                "0.3",
            ]
        )
        with self.assertRaisesRegex(ValueError, "positive teacher fraction"):
            _validate_args(args)

    def test_policy_target_exponent_must_be_finite_and_positive(self) -> None:
        for exponent in ("0", "-1", "nan", "inf"):
            args = build_parser().parse_args(
                [
                    "--shards",
                    "training.jsonl",
                    "--output",
                    "model.pt",
                    "--policy-target-exponent",
                    exponent,
                ]
            )
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                _validate_args(args)

    def test_final_vp_quality_weight_validates_target_mode_and_range(self) -> None:
        args = build_parser().parse_args(
            [
                "--shards",
                "training.jsonl",
                "--output",
                "model.pt",
                "--final-vp-quality-weight",
                "1.0",
                "--actor-vp-target-mode",
                "remaining_vp",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires .*absolute_final_vp"):
            _validate_args(args)

        args = build_parser().parse_args(
            [
                "--shards",
                "training.jsonl",
                "--output",
                "model.pt",
                "--final-vp-quality-weight",
                "2.1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            _validate_args(args)

    def test_early_stopping_preserves_exact_best_validation_epoch(self) -> None:
        training = self.root / "training.jsonl"
        validation = self.root / "validation.jsonl"
        checkpoint = self.root / "early-stopped.pt"
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
                "10",
                "--early-stopping-patience",
                "2",
                "--early-stopping-min-delta",
                "0.05",
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
        validation_results = [
            {"loss": 1.0},
            {"loss": 0.9},
            {"loss": 0.89},
            {"loss": 0.88},
        ]
        output = io.StringIO()

        with patch("training.train._evaluate", side_effect=validation_results):
            with redirect_stdout(output):
                train(args)

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        early_stop_events = [
            event for event in events if event["event"] == "training_early_stopped"
        ]
        self.assertEqual(len(early_stop_events), 1)
        self.assertEqual(early_stop_events[0]["epoch"], 2)
        self.assertEqual(early_stop_events[0]["selected_epoch"], 2)

        schema = FeatureSchema.from_schema_dict(SCHEMA)
        _model, payload = load_model_checkpoint(
            checkpoint, expected_schema=schema
        )
        metadata = payload["metadata"]["training"]
        self.assertEqual(metadata["stop_reason"], "early_stopping")
        self.assertTrue(metadata["early_stopping_triggered"])
        self.assertEqual(metadata["epochs_ran"], 3)
        self.assertEqual(metadata["steps_this_run"], 3)
        self.assertEqual(metadata["selected_epoch"], 2)
        self.assertEqual(metadata["last_completed_epoch"], 2)
        self.assertEqual(metadata["selected_validation_metrics"]["loss"], 0.88)
        self.assertEqual(metadata["early_stopping_reference_loss"], 0.9)
        self.assertEqual(metadata["epochs_without_significant_improvement"], 2)

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

    def test_resume_keeps_optimizer_moments_but_uses_explicit_hyperparameters(
        self,
    ) -> None:
        model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=2,
                action_dim=4,
                state_hidden_dim=8,
                action_embedding_dim=4,
                trunk_dim=8,
            )
        )
        previous = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        resumed_payload = {"optimizer_state_dict": previous.state_dict()}
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=5e-5,
        )
        args = build_parser().parse_args(
            [
                "--shards",
                "training.jsonl",
                "--output",
                "model.pt",
                "--learning-rate",
                "0.0001",
                "--weight-decay",
                "0.00005",
            ]
        )

        _restore_optimizer_state(optimizer, resumed_payload, args)

        self.assertTrue(
            all(group["lr"] == 1e-4 for group in optimizer.param_groups)
        )
        self.assertTrue(
            all(group["weight_decay"] == 5e-5 for group in optimizer.param_groups)
        )

    def test_full_finetune_restarts_optimizer_after_head_only_training(self) -> None:
        head_only_payload = {
            "metadata": {"training": {"actor_vp_head_only": True}}
        }
        full_model_payload = {
            "metadata": {"training": {"actor_vp_head_only": False}}
        }

        self.assertFalse(
            _should_restore_optimizer_state(
                head_only_payload, actor_vp_head_only=False
            )
        )
        self.assertTrue(
            _should_restore_optimizer_state(
                full_model_payload, actor_vp_head_only=False
            )
        )
        self.assertFalse(
            _should_restore_optimizer_state(
                full_model_payload, actor_vp_head_only=True
            )
        )
        self.assertFalse(
            _should_restore_optimizer_state(
                full_model_payload,
                actor_vp_head_only=False,
                policy_adapter_only=True,
            )
        )
        self.assertFalse(
            _should_restore_optimizer_state(
                full_model_payload,
                actor_vp_head_only=False,
                value_tower_only=True,
            )
        )
        self.assertFalse(
            _should_restore_optimizer_state(
                full_model_payload,
                actor_vp_head_only=False,
                architecture_expanded=True,
            )
        )
        self.assertFalse(
            _should_restore_optimizer_state(
                full_model_payload,
                actor_vp_head_only=False,
                policy_target_configuration_changed=True,
            )
        )
        self.assertFalse(
            _should_restore_optimizer_state(
                full_model_payload,
                actor_vp_head_only=False,
                final_vp_quality_weight_changed=True,
            )
        )

    def test_policy_adapter_only_training_preserves_base_parameters(self) -> None:
        training = self.root / "training.jsonl"
        base_checkpoint = self.root / "base.pt"
        adapted_checkpoint = self.root / "adapted.pt"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        dataset = SelfPlayDataset([training])
        assert dataset.schema is not None
        base_model = BrassPolicyValueNet(
            ModelConfig(
                state_dim=2,
                action_dim=4,
                state_hidden_dim=8,
                action_embedding_dim=4,
                trunk_dim=8,
            )
        )
        save_checkpoint(base_checkpoint, base_model, dataset.schema)
        dataset.close()
        args = build_parser().parse_args(
            [
                "--shards",
                str(training),
                "--output",
                str(adapted_checkpoint),
                "--resume",
                str(base_checkpoint),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--policy-adapter-dim",
                "8",
                "--policy-adapter-only",
            ]
        )

        with redirect_stdout(io.StringIO()):
            train(args)

        base, _base_payload = load_model_checkpoint(base_checkpoint)
        adapted, payload = load_model_checkpoint(adapted_checkpoint)
        self.assertEqual(adapted.config.policy_adapter_dim, 8)
        adapted_state = adapted.state_dict()
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, adapted_state[name]), name)
        metadata = payload["metadata"]["training"]
        self.assertTrue(metadata["policy_adapter_only"])
        self.assertTrue(metadata["policy_adapter_expanded"])
        self.assertFalse(metadata["optimizer_state_restored"])

    def test_value_tower_only_training_preserves_policy_parameters(self) -> None:
        training = self.root / "training.jsonl"
        base_checkpoint = self.root / "fused-base.pt"
        trained_checkpoint = self.root / "value-trained.pt"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        dataset = SelfPlayDataset([training])
        assert dataset.schema is not None
        config = ModelConfig(
            state_dim=2,
            action_dim=4,
            state_hidden_dim=8,
            action_embedding_dim=4,
            trunk_dim=8,
        )
        torch.manual_seed(81)
        policy_model = BrassPolicyValueNet(config)
        torch.manual_seed(82)
        value_model = BrassPolicyValueNet(config)
        fused_model = fuse_policy_and_value_models(policy_model, value_model)
        save_checkpoint(base_checkpoint, fused_model, dataset.schema)
        dataset.close()
        args = build_parser().parse_args(
            [
                "--shards",
                str(training),
                "--output",
                str(trained_checkpoint),
                "--resume",
                str(base_checkpoint),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--learning-rate",
                "0.01",
                "--value-tower-only",
            ]
        )

        with redirect_stdout(io.StringIO()):
            train(args)

        base, _ = load_model_checkpoint(base_checkpoint)
        trained, payload = load_model_checkpoint(trained_checkpoint)
        value_prefixes = (
            "value_state_encoder.",
            "shared_win_head.",
            "victory_point_margin_head.",
            "actor_victory_points_head.",
        )
        base_state = base.state_dict()
        trained_state = trained.state_dict()
        for name, value in base_state.items():
            if not name.startswith(value_prefixes):
                self.assertTrue(torch.equal(value, trained_state[name]), name)
        self.assertTrue(
            any(
                not torch.equal(value, trained_state[name])
                for name, value in base_state.items()
                if name.startswith(value_prefixes)
            )
        )
        metadata = payload["metadata"]["training"]
        self.assertTrue(metadata["value_tower_only"])
        self.assertFalse(metadata["optimizer_state_restored"])

    def test_max_steps_counts_only_steps_from_the_current_resume(self) -> None:
        training = self.root / "training.jsonl"
        initial_checkpoint = self.root / "initial.pt"
        resumed_checkpoint = self.root / "resumed.pt"
        _write_shard(training, engine_revision="engine-a", game_seed=101)
        model_args = [
            "--state-hidden-dim",
            "8",
            "--action-embedding-dim",
            "4",
            "--trunk-dim",
            "8",
        ]
        initial_args = build_parser().parse_args(
            [
                "--shards",
                str(training),
                "--output",
                str(initial_checkpoint),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                *model_args,
            ]
        )
        with redirect_stdout(io.StringIO()):
            train(initial_args)

        resumed_args = build_parser().parse_args(
            [
                "--shards",
                str(training),
                "--output",
                str(resumed_checkpoint),
                "--resume",
                str(initial_checkpoint),
                "--device",
                "cpu",
                "--epochs",
                "5",
                "--batch-size",
                "1",
                "--max-steps",
                "2",
                *model_args,
            ]
        )
        with redirect_stdout(io.StringIO()):
            train(resumed_args)

        schema = FeatureSchema.from_schema_dict(SCHEMA)
        _model, initial_payload = load_model_checkpoint(
            initial_checkpoint,
            expected_schema=schema,
        )
        _model, resumed_payload = load_model_checkpoint(
            resumed_checkpoint,
            expected_schema=schema,
        )
        initial_step = initial_payload["metadata"]["training"]["global_step"]
        metadata = resumed_payload["metadata"]["training"]
        self.assertEqual(metadata["starting_global_step"], initial_step)
        self.assertEqual(metadata["steps_this_run"], 2)
        self.assertEqual(metadata["global_step"], initial_step + 2)


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
            "victory_points": [30, 60],
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
                "actor": position_index % 2,
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
