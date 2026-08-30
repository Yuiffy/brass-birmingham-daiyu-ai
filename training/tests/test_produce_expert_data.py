from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.model_self_play import (
    ModelSelfPlaySummary,
)
from training.parallel_self_play import (
    ParallelSelfPlayConfig,
    ParallelSelfPlaySummary,
    SelfPlayShardResult,
)
from training.produce_expert_data import (
    ExpertDataProductionConfig,
    QualityGate,
    assess_game_quality,
    build_generation_recipes,
    filter_quality_shard,
    produce_expert_data,
)


class ProduceExpertDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_balanced_recipe_plan_is_deterministic(self) -> None:
        recipes = build_generation_recipes(
            games=10,
            strategy_prior_strengths=(0.6, 0.8),
            selection_temperatures=(0.3, 0.7),
            maximum_recipes=8,
        )
        self.assertEqual([recipe.games for recipe in recipes], [3, 3, 2, 2])
        self.assertEqual(
            [
                (recipe.strategy_prior_strength, recipe.selection_temperature)
                for recipe in recipes
            ],
            [(0.6, 0.3), (0.6, 0.7), (0.8, 0.3), (0.8, 0.7)],
        )
        with self.assertRaisesRegex(ValueError, "too small"):
            build_generation_recipes(
                games=1,
                strategy_prior_strengths=(0.6, 0.8),
                selection_temperatures=(0.3, 0.7),
            )

    def test_demonstration_strength_is_a_recipe_dimension(self) -> None:
        recipes = build_generation_recipes(
            games=4,
            strategy_prior_strengths=(0.6,),
            selection_temperatures=(0.3,),
            demonstration_prior_strengths=(0.0, 0.5),
            maximum_recipes=4,
        )
        self.assertEqual([recipe.games for recipe in recipes], [2, 2])
        self.assertEqual(
            [recipe.demonstration_prior_strength for recipe in recipes],
            [0.0, 0.5],
        )

    def test_positive_demonstration_strength_requires_shards(self) -> None:
        with self.assertRaisesRegex(ValueError, "demonstration_shards"):
            ExpertDataProductionConfig(
                output_dir=self.root / "invalid",
                checkpoint=self.root / "champion.pt",
                demonstration_prior_strengths=(0.4,),
            ).validate()

    def test_demonstration_prior_players_are_validated_against_player_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid seat indices"):
            ExpertDataProductionConfig(
                output_dir=self.root / "invalid-seat",
                checkpoint=self.root / "champion.pt",
                num_players=3,
                demonstration_prior_players=(3,),
            ).validate()
        config = ExpertDataProductionConfig(
            output_dir=self.root / "valid-seat",
            checkpoint=self.root / "champion.pt",
            num_players=3,
            demonstration_prior_players=(1,),
        )
        config.validate()

    def test_quality_filter_rejects_zero_score_and_lifecycle_fault(self) -> None:
        source = self.root / "source.jsonl"
        _write_shard(
            source,
            games=[
                _game(
                    game_index=0,
                    game_seed=101,
                    scores=[100, 0],
                    positions=[
                        _position(0, 0, "network|c0,r1,confirm"),
                        _position(1, 1, "build|i2,c0,b14,confirm"),
                    ],
                )
            ],
        )
        gate = QualityGate(minimum_actor_vp=80)
        summary = filter_quality_shard(source, self.root / "filtered.jsonl", gate)
        self.assertEqual(summary.games_seen, 1)
        self.assertEqual(summary.games_accepted, 0)
        self.assertEqual(summary.positions_written, 0)
        self.assertIn("zero_score", summary.rejection_reasons)
        self.assertIn("network_before_first_build", summary.rejection_reasons)

        quality = assess_game_quality(
            _game(
                game_index=0,
                game_seed=101,
                scores=[100, 0],
                positions=[
                    _position(0, 0, "network|c0,r1,confirm"),
                    _position(1, 1, "build|i2,c0,b14,confirm"),
                ],
            ),
            [
                _position(0, 0, "network|c0,r1,confirm"),
                _position(1, 1, "build|i2,c0,b14,confirm"),
            ],
            gate,
        )
        self.assertFalse(quality.accepted)

    def test_quality_filter_keeps_only_healthy_top_trajectory(self) -> None:
        source = self.root / "source.jsonl"
        output = self.root / "filtered.jsonl"
        positions = [
            _position(0, 0, "build|i2,c0,b14,confirm"),
            _position(1, 1, "build|i2,c1,b15,confirm"),
            _position(0, 2, "sell|i5,c2,b20,confirm"),
        ]
        _write_shard(
            source,
            games=[
                _game(
                    game_index=0,
                    game_seed=202,
                    scores=[110, 90],
                    positions=positions,
                )
            ],
        )
        summary = filter_quality_shard(
            source,
            output,
            QualityGate(minimum_actor_vp=100),
        )
        self.assertEqual(summary.games_accepted, 1)
        self.assertEqual(summary.eligible_player_trajectories, 1)
        self.assertEqual(summary.positions_written, 2)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(records[0]["source_type"], "expert_iteration")
        self.assertEqual(records[1]["quality_eligible_players"], [0])
        self.assertEqual(records[1]["positions"], 2)
        self.assertEqual([record["actor"] for record in records[2:]], [0, 0])

    def test_producer_writes_disjoint_train_validation_manifest(self) -> None:
        def fake_export(config: ParallelSelfPlayConfig) -> ParallelSelfPlaySummary:
            output = config.output_prefix.with_name(
                f"{config.output_prefix.name}-000.jsonl"
            )
            _write_shard(
                output,
                games=[
                    _game(
                        game_index=config.game_index_offset + index,
                        game_seed=config.base_seed + config.game_index_offset + index,
                        scores=[110, 90],
                        positions=[
                            _position(0, 0, "build|i2,c0,b14,confirm"),
                            _position(1, 1, "build|i2,c1,b15,confirm"),
                        ],
                    )
                    for index in range(config.games)
                ],
                metadata_extra={
                    "model_id": "sha256:test",
                    "checkpoint_step": 9,
                },
            )
            model_summary = ModelSelfPlaySummary(
                output=str(output),
                games_written=config.games,
                positions_written=config.games * 2,
                model_id="sha256:test",
                checkpoint_step=9,
            )
            return ParallelSelfPlaySummary(
                games_written=config.games,
                positions_written=config.games * 2,
                workers=config.workers,
                elapsed_seconds=0.01,
                positions_per_second=100.0,
                model_id="sha256:test",
                checkpoint_step=9,
                shards=(
                    SelfPlayShardResult(
                        shard_index=0,
                        game_index_offset=config.game_index_offset,
                        games=config.games,
                        summary=model_summary,
                    ),
                ),
            )

        config = ExpertDataProductionConfig(
            output_dir=self.root / "iteration",
            checkpoint=self.root / "champion.pt",
            train_games=4,
            validation_games=2,
            workers=2,
            strategy_prior_strengths=(0.8,),
            selection_temperatures=(0.5,),
            quality_gate=QualityGate(minimum_actor_vp=100),
        )
        summary = produce_expert_data(config, exporter=fake_export)
        self.assertEqual(summary.train_games_accepted, 4)
        self.assertEqual(summary.validation_games_accepted, 2)
        self.assertEqual(len(summary.train_shards), 1)
        manifest = json.loads(Path(summary.manifest).read_text())
        self.assertTrue(manifest["seed_partition"]["disjoint"])
        self.assertFalse(manifest["playwright_required"])
        self.assertEqual(
            set(manifest["seed_partition"]["train_game_seeds"])
            & set(manifest["seed_partition"]["validation_game_seeds"]),
            set(),
        )


def _metadata() -> dict:
    return {
        "record_type": "metadata",
        "format": "fast_brass_self_play_jsonl",
        "format_version": 1,
        "engine_revision": "test-engine",
        "feature_schema": {
            "version": 1,
            "state_dim": 2,
            "action_dim": 2,
            "state_blocks": [{"name": "all", "offset": 0, "size": 2}],
            "action_blocks": [{"name": "all", "offset": 0, "size": 2}],
        },
    }


def _game(
    *,
    game_index: int,
    game_seed: int,
    scores: list[int],
    positions: list[dict],
) -> dict:
    players = len(scores)
    winners = [index for index, score in enumerate(scores) if score == max(scores)]
    return {
        "record_type": "game",
        "format_version": 1,
        "game_index": game_index,
        "game_seed": game_seed,
        "positions": len(positions),
        "official_winners": winners,
        "shared_win_values": [1.0 if index in winners else 0.0 for index in range(players)],
        "placements": [0 if index in winners else 1 for index in range(players)],
        "finish_order": list(range(players)),
        "victory_points": scores,
        "victory_point_margins": [
            score - max((other for index, other in enumerate(scores) if index != actor), default=score)
            for actor, score in enumerate(scores)
        ],
        "income_levels": [5 for _ in range(players)],
        "money": [10 for _ in range(players)],
        "_positions": positions,
    }


def _position(actor: int, position_index: int, key: str) -> dict:
    return {
        "record_type": "position",
        "format_version": 1,
        "feature_version": 1,
        "game_index": 0,
        "position_index": position_index,
        "actor": actor,
        "state_features": [0.0, 1.0],
        "legal_actions": [
            {
                "index": 0,
                "key": key,
                "feature_indices": [0],
                "visits": 1,
                "policy_target": 1.0,
                "search_policy_target": 1.0,
            }
        ],
        "selected_action_index": 0,
        "selected_action_key": key,
        "value_target": {"shared_win": 1.0, "victory_point_margin": 10},
    }


def _write_shard(path: Path, *, games: list[dict], metadata_extra: dict | None = None) -> None:
    metadata = _metadata()
    metadata.update(metadata_extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        for game in games:
            positions = game.get("_positions", [])
            serialized_game = {key: value for key, value in game.items() if key != "_positions"}
            handle.write(json.dumps(serialized_game, separators=(",", ":")) + "\n")
            for position in positions or []:
                position = dict(position)
                position["game_index"] = game["game_index"]
                handle.write(json.dumps(position, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    unittest.main()
