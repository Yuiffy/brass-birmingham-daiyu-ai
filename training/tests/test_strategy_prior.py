from __future__ import annotations

import unittest

from training.strategy_prior import (
    ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
    CONSERVATIVE_ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
    GROUPED_STRATEGY_PRIOR_VERSION,
    LEGACY_STRATEGY_PRIOR_VERSION,
    LIFECYCLE_STRATEGY_PRIOR_VERSION,
    MAP_AWARE_STRATEGY_PRIOR_VERSION,
    RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
    STRATEGY_PRIOR_VERSION,
    blend_policy_with_strategy,
    build_strategy_prior,
)


def _building(
    industry: int,
    *,
    resource: float = 1.0,
    flipped: bool = False,
    level: int = 0,
    owner: int = 0,
) -> list[float]:
    row = [0.0] * 14
    row[0] = 1.0
    row[1 + owner] = 1.0
    row[5 + industry] = 1.0
    row[11] = (level + 1) / 8.0
    row[12] = resource
    row[13] = 1.0 if flipped else 0.0
    return row


def _action(
    key: str,
    root: str,
    *,
    industry: int | None = None,
    location: int | None = None,
    sell_targets: list[int] | None = None,
    choices: list[dict] | None = None,
    discard_card_types: list[int] | None = None,
) -> dict:
    return {
        "key": key,
        "root_action_name": root,
        "selected_industry": industry,
        "build_location": location,
        "sell_targets": sell_targets or [],
        "choices": choices or [],
        "discard_card_types": discard_card_types or [],
    }


def _road(*locations: int, built: bool = False, owner: int | None = None) -> list[float]:
    row = [0.0] * 10
    row[0] = 1.0 if built else 0.0
    if owner is not None:
        row[1 + owner] = 1.0
    row[5] = 1.0
    row[6] = 1.0
    for index, location in enumerate(locations[:3]):
        row[7 + index] = location / 54.0
    for index in range(len(locations), 3):
        row[7 + index] = -1.0
    return row


class StrategyPriorTests(unittest.TestCase):
    def observation(self) -> dict:
        return {
            "global_features": [1.0, 0.0, 0.0],
            "decision_player": 0,
            "players_public": [[1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            "industry_mats": [[
                0.125,
                1.0,
                1.0,
                0.125,
                1.0,
                1.0,
                0.125,
                1.0,
                1.0,
                0.125,
                1.0,
                1.0,
                0.125,
                1.0,
                1.0,
                0.125,
                1.0,
                1.0,
            ]],
            "buildings": [],
        }

    def test_canal_prior_prefers_brewery_and_sell_over_unproductive_actions(self) -> None:
        actions = [
            _action("build|beer", "build", industry=2, location=47),
            _action("sell|target", "sell", sell_targets=[0]),
            _action("network|road", "network"),
            _action("pass|confirm", "pass"),
            _action("scout|confirm", "scout"),
        ]
        prior = build_strategy_prior(self.observation(), {"actions": actions})

        self.assertEqual(prior.version, STRATEGY_PRIOR_VERSION)
        self.assertEqual(prior.phase, "canal")
        self.assertGreater(prior.probabilities[0], prior.probabilities[2])
        self.assertGreater(prior.probabilities[1], prior.probabilities[3])
        self.assertGreater(prior.probabilities[3], prior.probabilities[2])
        self.assertGreater(prior.probabilities[3], prior.probabilities[4])
        self.assertTrue(prior.guarded_actions[2])
        self.assertEqual(prior.probabilities[2], 0.0)
        self.assertAlmostEqual(sum(prior.probabilities), 1.0, places=7)

    def test_grouped_prior_does_not_reward_action_family_size(self) -> None:
        actions = [
            _action("build|beer", "build", industry=2, location=47),
            *[
                _action(f"network|road-{index}", "network")
                for index in range(100)
            ],
            _action("pass|confirm", "pass"),
        ]
        observation = self.observation()
        observation["buildings"] = [[
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]]

        prior = build_strategy_prior(
            observation,
            {"actions": actions},
            version=GROUPED_STRATEGY_PRIOR_VERSION,
        )
        network_mass = sum(prior.probabilities[1:101])

        self.assertFalse(any(prior.guarded_actions))
        self.assertGreater(prior.probabilities[0], network_mass)
        self.assertAlmostEqual(sum(prior.probabilities), 1.0, places=7)

    def test_default_prior_caps_canal_network_family_mass(self) -> None:
        actions = [
            _action("build|beer", "build", industry=2, location=47),
            *[
                _action(f"network|road-{index}", "network")
                for index in range(100)
            ],
            _action("pass|confirm", "pass"),
        ]
        observation = self.observation()
        observation["buildings"] = [[
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]]

        prior = build_strategy_prior(observation, {"actions": actions})

        self.assertAlmostEqual(sum(prior.probabilities[1:101]), 0.18, places=7)
        self.assertAlmostEqual(sum(prior.probabilities), 1.0, places=7)

    def test_default_prior_balances_build_probability_by_industry(self) -> None:
        actions = [
            *[
                _action(
                    f"build|coal-{index}",
                    "build",
                    industry=0,
                    location=index % 47,
                )
                for index in range(100)
            ],
            _action("build|cotton", "build", industry=5, location=47),
            _action("pass|confirm", "pass"),
        ]
        prior = build_strategy_prior(self.observation(), {"actions": actions})
        coal_mass = sum(prior.probabilities[:100])
        cotton_mass = prior.probabilities[100]

        self.assertGreater(cotton_mass, coal_mass)
        self.assertAlmostEqual(sum(prior.probabilities), 1.0, places=7)

    def test_low_income_penalizes_another_loan(self) -> None:
        observation = self.observation()
        observation["players_public"][0][3] = 0.0
        observation["players_public"][0][4] = -0.08
        actions = [
            _action("loan|confirm", "loan"),
            _action("pass|confirm", "pass"),
        ]

        prior = build_strategy_prior(observation, {"actions": actions})

        self.assertLess(prior.scores[0], prior.scores[1])

    def test_lifecycle_prior_avoids_surplus_beer_before_sellable_industry(self) -> None:
        observation = self.observation()
        observation["buildings"] = [_building(2)]
        actions = [
            _action("build|i2,c0,b47,confirm", "build", industry=2, location=47),
            _action("build|i5,c0,b36,confirm", "build", industry=5, location=36),
            _action("build|i1,c0,b26,confirm", "build", industry=1, location=26),
        ]

        prior = build_strategy_prior(
            observation,
            {"actions": actions},
            version=LIFECYCLE_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(prior.scores[1], prior.scores[0])
        self.assertGreater(prior.scores[2], prior.scores[0])
        self.assertGreater(prior.probabilities[1], prior.probabilities[0])

    def test_lifecycle_prior_connects_an_unflipped_sellable_industry(self) -> None:
        observation = self.observation()
        observation["buildings"] = [_building(5, resource=0.0)]
        actions = [
            _action("network|c0,r4,confirm", "network"),
            _action("build|i2,c0,b47,confirm", "build", industry=2, location=47),
            _action("develop|i5,c0,ironm,confirm", "develop", industry=5),
        ]

        prior = build_strategy_prior(
            observation,
            {"actions": actions},
            version=LIFECYCLE_STRATEGY_PRIOR_VERSION,
        )

        self.assertFalse(prior.guarded_actions[0])
        self.assertGreater(prior.scores[0], prior.scores[1])
        self.assertGreater(prior.probabilities[0], prior.probabilities[1])

    def test_lifecycle_prior_finishes_a_legal_sale_before_expanding(self) -> None:
        observation = self.observation()
        observation["buildings"] = [_building(5, resource=0.0), _building(2)]
        actions = [
            _action("sell|c0,s0,confirm", "sell", sell_targets=[0]),
            _action("network|c0,r4,confirm", "network"),
            _action("build|i5,c0,b36,confirm", "build", industry=5, location=36),
        ]

        prior = build_strategy_prior(
            observation,
            {"actions": actions},
            version=LIFECYCLE_STRATEGY_PRIOR_VERSION,
        )

        self.assertEqual(max(range(len(actions)), key=prior.scores.__getitem__), 0)
        self.assertEqual(
            max(range(len(actions)), key=prior.probabilities.__getitem__),
            0,
        )

    def test_lifecycle_prior_card_variants_do_not_inflate_intent_mass(self) -> None:
        observation = self.observation()
        base_actions = [
            _action("loan|c0,confirm", "loan"),
            _action("build|i5,c0,b36,confirm", "build", industry=5, location=36),
        ]
        duplicate_actions = [
            _action("loan|c0,confirm", "loan"),
            _action("loan|c1,confirm", "loan"),
            base_actions[1],
        ]

        base = build_strategy_prior(
            observation,
            {"actions": base_actions},
            version=LIFECYCLE_STRATEGY_PRIOR_VERSION,
        )
        duplicated = build_strategy_prior(
            observation,
            {"actions": duplicate_actions},
            version=LIFECYCLE_STRATEGY_PRIOR_VERSION,
        )

        self.assertAlmostEqual(
            base.probabilities[0],
            duplicated.probabilities[0] + duplicated.probabilities[1],
        )

    def test_map_aware_prior_prefers_a_road_that_completes_a_merchant_route(self) -> None:
        observation = self.observation()
        observation["buildings"] = [[0.0] * 20 for _ in range(49)]
        observation["buildings"][36] = _building(5)
        observation["roads"] = [_road() for _ in range(39)]
        observation["roads"][28] = _road(16, 4)
        observation["roads"][31] = _road(16, 23)
        observation["trade_post_slots"] = [-1, 1, -1, -1, -1, -1, -1, -1, -1]
        observation["trade_post_beer"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        actions = [
            _action("network|c0,r31,confirm", "network"),
            _action("network|c0,r28,confirm", "network"),
        ]

        lifecycle = build_strategy_prior(
            observation,
            {"actions": actions},
            version=LIFECYCLE_STRATEGY_PRIOR_VERSION,
        )
        map_aware = build_strategy_prior(
            observation,
            {"actions": actions},
            version=MAP_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertEqual(lifecycle.scores[0], lifecycle.scores[1])
        self.assertGreater(map_aware.scores[0], map_aware.scores[1] + 4.0)
        self.assertGreater(map_aware.probabilities[0], map_aware.probabilities[1])

    def test_map_aware_prior_follows_the_shortest_route_to_a_matching_merchant(self) -> None:
        observation = self.observation()
        observation["buildings"] = [[0.0] * 20 for _ in range(49)]
        observation["buildings"][0] = _building(5)
        observation["roads"] = [_road() for _ in range(39)]
        observation["roads"][0] = _road(25, 6)
        observation["roads"][8] = _road(6, 7)
        observation["roads"][10] = _road(7, 0)
        observation["roads"][12] = _road(0, 2)
        observation["trade_post_slots"] = [-1, -1, -1, -1, -1, 1, -1, -1, -1]
        observation["trade_post_beer"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        actions = [
            _action("network|c0,r10,confirm", "network"),
            _action("network|c0,r12,confirm", "network"),
        ]

        prior = build_strategy_prior(
            observation,
            {"actions": actions},
            version=MAP_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(prior.scores[0], prior.scores[1] + 2.0)
        self.assertGreater(prior.probabilities[0], prior.probabilities[1])

    def test_legacy_prior_remains_available_for_ab_comparison(self) -> None:
        actions = [
            _action("build|beer", "build", industry=2, location=47),
            _action("network|road", "network"),
        ]
        prior = build_strategy_prior(
            self.observation(),
            {"actions": actions},
            version=LEGACY_STRATEGY_PRIOR_VERSION,
        )

        self.assertEqual(prior.version, LEGACY_STRATEGY_PRIOR_VERSION)
        self.assertFalse(any(prior.guarded_actions))
        self.assertGreater(prior.probabilities[1], 0.0)

    def test_v6_uses_beer_units_instead_of_brewery_count(self) -> None:
        actions = [
            _action("build|beer", "build", industry=2, location=47),
            _action("build|cotton", "build", industry=5, location=36),
        ]
        high_supply = self.observation()
        high_supply["buildings"] = [_building(2, resource=1.0), _building(5)]
        low_supply = self.observation()
        low_supply["buildings"] = [_building(2, resource=0.2), _building(5)]

        high = build_strategy_prior(
            high_supply,
            {"actions": actions},
            version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        )
        low = build_strategy_prior(
            low_supply,
            {"actions": actions},
            version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertLess(high.scores[0], low.scores[0])
        self.assertGreater(low.scores[1], low.scores[0])

    def test_v6_prices_the_selected_market_source(self) -> None:
        observation = self.observation()
        observation["global_features"] = [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0 / 14.0,
            1.0,
        ]
        observation["buildings"] = [[0.0] * 14, _building(0, resource=1.0)]
        market_action = _action(
            "build|market",
            "build",
            industry=3,
            location=2,
            choices=[{"kind": "coal_source", "value": 49}],
        )
        board_action = _action(
            "build|board",
            "build",
            industry=3,
            location=2,
            choices=[{"kind": "coal_source", "value": 1}],
        )

        prior = build_strategy_prior(
            observation,
            {"actions": [market_action, board_action]},
            version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertLess(prior.scores[0], prior.scores[1])

    def test_v6_scout_keeps_wild_cards_for_future_value(self) -> None:
        actions = [
            _action(
                "scout|wild",
                "scout",
                discard_card_types=[27, 0, 1],
            ),
            _action(
                "scout|locations",
                "scout",
                discard_card_types=[0, 1, 2],
            ),
        ]

        prior = build_strategy_prior(
            self.observation(),
            {"actions": actions},
            version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(prior.scores[1], prior.scores[0])

    def test_v6_develop_prefers_a_sellable_industry_path(self) -> None:
        actions = [
            _action(
                "develop|cotton",
                "develop",
                industry=5,
                choices=[{"kind": "iron_source", "value": 49}],
            ),
            _action(
                "develop|coal",
                "develop",
                industry=0,
                choices=[{"kind": "iron_source", "value": 49}],
            ),
        ]

        prior = build_strategy_prior(
            self.observation(),
            {"actions": actions},
            version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(prior.scores[0], prior.scores[1])

    def test_v6_double_rail_rewards_cash_backed_route_value(self) -> None:
        observation = self.observation()
        observation["global_features"] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        observation["players_public"][0][3] = 0.8
        observation["buildings"] = [[0.0] * 20 for _ in range(49)]
        observation["buildings"][0] = _building(3, flipped=True)
        observation["roads"] = [_road() for _ in range(39)]
        observation["roads"][0] = _road(0, 1)
        observation["roads"][1] = _road(1, 2)
        observation["roads"][2] = _road(10, 11)
        double_action = _action(
            "double_network|r0,r1",
            "double_network",
            choices=[
                {"kind": "coal_source", "value": 49},
                {"kind": "coal_source", "value": 49},
                {"kind": "action_beer_source", "value": 0},
            ],
        )
        double_action["road"] = 0
        double_action["second_road"] = 1
        single_action = _action(
            "network|r2",
            "network",
            choices=[{"kind": "coal_source", "value": 49}],
        )
        single_action["road"] = 2

        prior = build_strategy_prior(
            observation,
            {"actions": [double_action, single_action]},
            version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(prior.scores[0], prior.scores[1])

    def test_v7_uses_public_beer_supply_when_pricing_a_brewery(self) -> None:
        actions = [
            _action("build|beer", "build", industry=2, location=47),
            _action("build|cotton", "build", industry=5, location=0),
        ]
        scarce = self.observation()
        scarce["buildings"] = [_building(5)]
        supplied = self.observation()
        supplied["buildings"] = [_building(5), _building(2, owner=1)]

        scarce_prior = build_strategy_prior(
            scarce,
            {"actions": actions},
            version=ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
        )
        supplied_prior = build_strategy_prior(
            supplied,
            {"actions": actions},
            version=ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(scarce_prior.scores[0], supplied_prior.scores[0])

    def test_v7_prices_a_high_value_sale_by_the_tile_not_just_the_action_family(self) -> None:
        observation = self.observation()
        observation["buildings"] = [
            _building(3, level=0),
            _building(4, level=4),
        ]
        actions = [
            _action("sell|goods", "sell", sell_targets=[0]),
            _action("sell|pottery", "sell", sell_targets=[1]),
        ]

        prior = build_strategy_prior(
            observation,
            {"actions": actions},
            version=ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
        )

        self.assertGreater(prior.scores[1], prior.scores[0])

    def test_v8_keeps_action_efficiency_as_a_small_v6_residual(self) -> None:
        actions = [
            _action("build|coal", "build", industry=0, location=0),
            _action("build|cotton", "build", industry=5, location=0),
            _action("sell|goods", "sell", sell_targets=[0]),
        ]
        legal = {"actions": actions}
        v6 = build_strategy_prior(
            self.observation(), legal, version=RESOURCE_AWARE_STRATEGY_PRIOR_VERSION
        )
        v7 = build_strategy_prior(
            self.observation(), legal, version=ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION
        )
        v8 = build_strategy_prior(
            self.observation(),
            legal,
            version=CONSERVATIVE_ACTION_EFFICIENCY_STRATEGY_PRIOR_VERSION,
        )

        for v6_score, v7_score, v8_score in zip(
            v6.scores, v7.scores, v8.scores, strict=True
        ):
            self.assertAlmostEqual(v8_score, v6_score + 0.25 * (v7_score - v6_score))

    def test_blend_is_normalized_and_strength_zero_preserves_model(self) -> None:
        model = (0.8, 0.15, 0.05)
        strategy = (0.05, 0.8, 0.15)

        self.assertEqual(blend_policy_with_strategy(model, strategy, 0.0), model)
        blended = blend_policy_with_strategy(model, strategy, 0.5)
        self.assertAlmostEqual(sum(blended), 1.0, places=7)
        self.assertGreater(blended[1], model[1])
        self.assertGreater(blended[2], model[2])

    def test_blend_rejects_mismatched_distributions(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            blend_policy_with_strategy((1.0,), (0.5, 0.5), 0.5)

    def test_blend_applies_hard_guard_only_when_prior_is_enabled(self) -> None:
        model = (0.8, 0.2)
        strategy = (0.5, 0.5)

        self.assertEqual(
            blend_policy_with_strategy(
                model,
                strategy,
                0.0,
                guarded_actions=(True, False),
            ),
            model,
        )
        blended = blend_policy_with_strategy(
            model,
            strategy,
            0.7,
            guarded_actions=(True, False),
        )
        self.assertEqual(blended, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
