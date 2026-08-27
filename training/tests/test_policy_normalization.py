from __future__ import annotations

import unittest

from training.policy_normalization import (
    card_invariant_action_intent,
    rebalance_card_choice_groups,
)


class PolicyNormalizationTests(unittest.TestCase):
    def test_card_tokens_are_removed_without_touching_resource_tokens(self) -> None:
        self.assertEqual(
            card_invariant_action_intent("build|i2,c7,b14,coalb3,confirm"),
            "build|i2,b14,coalb3,confirm",
        )
        self.assertEqual(
            card_invariant_action_intent("scout|c0,c3,c7,confirm"),
            "scout|confirm",
        )

    def test_duplicate_card_choices_do_not_inflate_intent_mass(self) -> None:
        keys = (
            "network|c0,r4,confirm",
            "network|c1,r4,confirm",
            "build|i2,c0,b14,confirm",
        )
        probabilities = rebalance_card_choice_groups(keys, (0.4, 0.4, 0.2))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], 1.0 / 3.0)
        self.assertAlmostEqual(probabilities[1], 1.0 / 3.0)
        self.assertAlmostEqual(probabilities[2], 1.0 / 3.0)

    def test_model_preference_is_preserved_inside_an_intent(self) -> None:
        keys = (
            "loan|c0,confirm",
            "loan|c1,confirm",
            "pass|c0,confirm",
        )
        probabilities = rebalance_card_choice_groups(keys, (0.45, 0.15, 0.40))
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertAlmostEqual(probabilities[0] / probabilities[1], 3.0)

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            rebalance_card_choice_groups(("pass|c0,confirm",), (0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "positive mass"):
            rebalance_card_choice_groups(("pass|c0,confirm",), (0.0,))


if __name__ == "__main__":
    unittest.main()
