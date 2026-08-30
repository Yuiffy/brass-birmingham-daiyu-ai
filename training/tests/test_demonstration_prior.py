from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.demonstration_prior import (
    DEMONSTRATION_PRIOR_VERSION,
    action_intent_signature,
    blend_policy_with_demonstration,
    load_demonstration_prior,
)


class DemonstrationPriorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_action_signature_removes_cards_and_keeps_industry(self) -> None:
        self.assertEqual(
            action_intent_signature("build|i5,c7,b36,coalb5,confirm"),
            "build:i5",
        )
        self.assertEqual(
            action_intent_signature("develop_double|c6,i5,i5,ironm,confirm"),
            "develop_double:i5,i5",
        )
        self.assertEqual(action_intent_signature("loan|c2,confirm"), "loan")

    def test_context_prior_prefers_matching_human_intent_without_masking_others(self) -> None:
        shard = self.root / "human.jsonl"
        _write_shard(
            shard,
            [
                _position("canal", 0, 1, "build|i5,c1,b36,confirm"),
                _position("canal", 0, 1, "build|i5,c2,b35,confirm"),
                _position("canal", 0, 1, "loan|c3,confirm"),
            ],
        )
        prior = load_demonstration_prior((shard,))
        self.assertEqual(prior.version, DEMONSTRATION_PRIOR_VERSION)
        self.assertEqual(prior.positions, 3)
        probabilities = prior.probabilities(
            {
                "global_features": [1.0, 0.0, 0.0],
                "round_in_phase": 0,
                "actions_remaining": 1,
            },
            {
                "actions": [
                    {"index": 0, "key": "build|i5,c9,b40,confirm"},
                    {"index": 1, "key": "loan|c8,confirm"},
                    {"index": 2, "key": "sell|c4,s35,confirm"},
                ]
            },
        )
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])
        self.assertTrue(all(value > 0.0 for value in probabilities))

    def test_loader_rejects_missing_action_label(self) -> None:
        shard = self.root / "bad.jsonl"
        _write_shard(shard, [{"record_type": "position"}])
        with self.assertRaisesRegex(ValueError, "selected_action_key"):
            load_demonstration_prior((shard,))

    def test_geometric_blend_preserves_model_at_zero_and_demo_at_one(self) -> None:
        model = (0.8, 0.2)
        demonstration = (0.1, 0.9)
        self.assertEqual(
            blend_policy_with_demonstration(model, demonstration, 0.0),
            model,
        )
        for actual, expected in zip(
            blend_policy_with_demonstration(model, demonstration, 1.0),
            demonstration,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        middle = blend_policy_with_demonstration(model, demonstration, 0.5)
        self.assertAlmostEqual(sum(middle), 1.0)
        self.assertGreater(middle[1], model[1])

    def test_blend_rejects_mismatched_or_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            blend_policy_with_demonstration((1.0,), (1.0, 0.0), 0.5)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            blend_policy_with_demonstration((1.0,), (1.0,), 1.5)
        with self.assertRaisesRegex(ValueError, "guarded action mask"):
            blend_policy_with_demonstration(
                (0.5, 0.5),
                (0.5, 0.5),
                0.5,
                guarded_actions=(True,),
            )

    def test_blend_keeps_active_guarded_actions_at_zero(self) -> None:
        blended = blend_policy_with_demonstration(
            (0.5, 0.5),
            (0.01, 0.99),
            1.0,
            guarded_actions=(True, False),
        )
        self.assertEqual(blended[0], 0.0)
        self.assertAlmostEqual(blended[1], 1.0)


def _position(phase: str, round_in_phase: int, actions: int, key: str) -> dict:
    return {
        "record_type": "position",
        "phase": phase,
        "round_in_phase": round_in_phase,
        "actions_remaining_in_turn": actions,
        "selected_action_key": key,
    }


def _write_shard(path: Path, positions: list[dict]) -> None:
    records = [{"record_type": "metadata"}, {"record_type": "game"}, *positions]
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
