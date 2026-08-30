from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from training.evaluate import MatchResult, _splitmix64
from training.parallel_evaluate import (
    EvaluationPartitionResult,
    EvaluationRoundPlan,
    build_round_plans,
    combine_evaluation_reports,
    write_evaluation_report,
)


class ParallelEvaluationTests(unittest.TestCase):
    def test_round_plans_are_contiguous_and_balanced(self) -> None:
        plans = build_round_plans(20, 4)

        self.assertEqual(
            [(plan.round_offset, plan.rounds) for plan in plans],
            [(0, 5), (5, 5), (10, 5), (15, 5)],
        )
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            build_round_plans(3, 4)

    def test_combined_report_recomputes_one_global_gate(self) -> None:
        args = argparse.Namespace(
            rounds=2,
            workers=2,
            seed=1234,
            players=2,
            promotion_margin=0.0,
            minimum_games=4,
        )
        results = tuple(
            EvaluationPartitionResult(
                plan=EvaluationRoundPlan(
                    worker_index=round_index,
                    round_offset=round_index,
                    rounds=1,
                ),
                report=self.partition_report(args, round_index),
            )
            for round_index in range(2)
        )

        combined = combine_evaluation_reports(args, results)

        self.assertEqual(combined["rounds"], 2)
        self.assertEqual(combined["base_seed"], 1234)
        self.assertEqual(combined["parallel_workers"], 2)
        self.assertEqual(len(combined["games"]), 4)
        self.assertEqual(combined["summary"]["independent_seed_groups"], 2)
        self.assertEqual(combined["summary"]["candidate_mean_score_delta"], 1.0)
        self.assertTrue(combined["summary"]["promote"])

    def test_report_write_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "gate.json"
            resolved = write_evaluation_report(destination, {"promote": True})

            self.assertEqual(resolved, destination.resolve())
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"promote": True},
            )
            self.assertEqual(list(destination.parent.glob(".gate.json.tmp-*")), [])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_evaluation_report(destination, {"promote": False})

    def partition_report(self, args: argparse.Namespace, round_index: int) -> dict:
        game_seed = _splitmix64(args.seed + round_index)
        games = [
            MatchResult(
                round_index=0,
                game_seed=game_seed,
                candidate_seat=seat,
                actions=79,
                candidate_shared_win=1.0,
                candidate_score_delta=1.0,
                candidate_victory_point_margin=4,
                candidate_placement=1,
                candidate_is_official_winner=True,
                official_winners=(seat,),
                victory_points=(12, 8),
            )
            for seat in range(2)
        ]
        return {
            "candidate": "candidate.pt",
            "champion": "champion.pt",
            "candidate_source_kind": "checkpoint",
            "champion_source_kind": "checkpoint",
            "candidate_model_id": "sha256:candidate",
            "champion_model_id": "sha256:champion",
            "candidate_checkpoint_step": 2,
            "champion_checkpoint_step": 1,
            "players": 2,
            "rounds": 1,
            "base_seed": args.seed + round_index,
            "feature_version": 1,
            "agent_mode": "puct_search",
            "search_simulations": 64,
            "exploration_constant": 1.5,
            "search_determinizations": 4,
            "inference_batch_size": 64,
            "candidate_strategy_prior_strength": 0.0,
            "candidate_strategy_prior_version": "default",
            "candidate_score_utility_weight": 0.0,
            "champion_strategy_prior_strength": 0.0,
            "champion_strategy_prior_version": "default",
            "champion_score_utility_weight": 0.0,
            "gate_confidence_unit": "seat_rotated_seed_group",
            "summary": {},
            "games": [asdict(game) for game in games],
        }


if __name__ == "__main__":
    unittest.main()
