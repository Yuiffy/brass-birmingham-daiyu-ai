# Economic Planning AI

The current CPU policy is `rule_economic_conversion_v2`, selected by
`RuleDecisionConfig::default()`. It evaluates eight distinct strategic candidates with a
second action in the same turn. It uses the existing legal-action engine and known hand;
it does not use neural checkpoints, hidden opponent cards or full-game rollouts.

## V2 Changes

- Combine all pending income on one nonlinear income track. Borrowing no longer makes each
  idle Coal Mine and Brewery independently appear more valuable.
- Discount resource investments without connected customers, and reserve a cash runway when
  remaining legal loans cannot cover negative-income settlements and productive actions.
- Plan both actions in a turn, including Loan -> Build, Build -> Sell and route -> Sell.
  Card-equivalent discards share a strategic expansion budget. Continuations stop at a turn
  or era boundary even if the next current player happens to be the same seat.
- Allow a planned shortest route to use the destination merchant's existing barrel.
  Visible Brewery and merchant Beer is still allocated at most once across products.
- Value future link icons beside currently unflipped industries, as well as secured link VP.
- Correct economic rules first: loan income bands and floor, turn-end refill, one setup discard
  per player, Scout restrictions, printed merchant link icons and income tiebreaks.
  See [the rulebook audit](official-rules-audit.md) for sources and replay compatibility.

`RuleDecisionConfig::economic_v1()` retains the preceding evaluator with one-ply selection.
`RuleDecisionConfig::legacy()` retains the August policy. Both use the corrected engine in
current comparisons. `deep()` expands sixteen strategic candidates as a diagnostic option.

## Independent Self-Play Scores

All seats use the named policy, with the same eight new seeds
`2026092900..2026092907` and corrected rules for each policy. These seeds were not used for
tuning. Every game and low score is included.

| Players | Games per Policy | V1 Mean VP | V2 Mean VP | V2 Minimum | V2 Below 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 120.69 | 151.44 | 126 | 0/16 |
| 3 | 8 | 121.75 | 133.13 | 113 | 0/24 |
| 4 | 8 | 103.88 | 116.69 | 81 | 2/32 |

The paired game-mean gain is +30.75 / +11.38 / +12.81 VP. Its Student-t 95% intervals
are [+4.91,+56.59] / [-4.40,+27.15] / [+9.66,+15.96]. Eight games are a small sample:
the three-player self-play gain is not statistically established by this set alone.
V1 has a zero score in both the two- and three-player control sets; neither is excluded.
V2 has no zero scores. Four-player V2 still has 81- and 96-point finishes.

## Direct Matches Against V1

One V2 seat plays against V1 opponents, rotating through every seat for each of 24 previously
unused seeds `2026092800..2026092823`. Both sides use exactly the same corrected rules.
The margin is V2 VP minus the best opponent's VP, with a 95% Student-t interval grouped
by seed. In multiplayer, beating the strongest of several opponents is a harder margin
than beating an average opponent.

| Players | Games | V2 Mean VP | Opponent Mean VP | V2 Wins | V2 Minimum | Below 100 | Margin (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 48 | 150.06 | 126.58 | 38 (79.17%) | 113 | 0 | +23.48 [+16.57,+30.39] |
| 3 | 72 | 136.69 | 113.87 | 49 (68.06%) | 101 | 0 | +10.14 [+5.82,+14.46] |
| 4 | 96 | 120.96 | 103.19 | 41 (42.71%) | 91 | 7 | -0.13 [-4.54,+4.29] |

All 216 games completed; V2 had no shortfall sessions and no score below 50. The four-player
margin interval includes zero. V2 is not consistently beating the best of three V1 opponents.

The income tiebreak correction was completed while the frozen matches were running. It does
not affect action selection or VP. All five tied-highest-score games involving V2 were replayed
under the corrected ranking: their score vectors reproduced exactly. One four-player win
(seed 2026092813, seat 0) becomes a loss because both contenders earn +13, but the opponent
has more cash. The table and JSON report include that correction.

## Reproduce and Inspect

```bash
cargo test --all-targets --features python-bindings
cargo run --release --example rule_ai_match -- \
  --rounds 24 --players 4 --seed 2026092800 --workers 4 \
  --candidate economy --opponent economy-v1 --output output/v2-match.json
cargo run --release --example rule_ai_bench -- 8 4 2026092900
```

`rule_ai_match` also accepts `--seat N` for a single-seat diagnostic, `--trace`,
`--depth`, `--branching`, and `--opponent legacy` or `--opponent economy`.
Reports include action families by era, Canal VP, cash, income, unflipped tiles, shortfalls,
win shares and timings. Policy inference is deterministic; timings depend on concurrent load.

Generated evidence remains in ignored `output/`:

- `economy-v2-heldout-{2,3,4}p.json`: complete direct matches.
- `economy-v2-heldout-self-{2,3,4}p.log`: eight complete V2 self-play games each.
- `economy-v1-paired-self-{2,3,4}p.json`: matching V1 controls.
- `economy-v2-tie-*.json`: exact tied-score replay records.
- `economy-v2-validation-summary.json`: compact scores and paired intervals.

The frozen evaluator SHA-256, including tests, is
`C6F0FFADDC78DBF0C35D2634AC6E0A9B10A56D3564CD702228D5028D9C5CAF3D`.
Development used seeds 2026092700..2026092707 and previously observed September failure seeds.
No policy parameters changed while either independent evaluation set was running.

## Validation and Limits

The full Rust suite with Python bindings passes (128 library tests plus all integration and
example targets). New regressions cover the corrected economic rules, shared pending income,
merchant route Beer, hidden-card invariance, and exact turn-boundary termination. The API
created 2/3/4-player games and analyzed/applied ten actions each. The browser created a new
game, executed an AI move, and displayed the v2 analysis label without console warnings/errors.

The preview is [http://127.0.0.1:5175/](http://127.0.0.1:5175/), with API port 3011.
Start a fresh game: historical saved action streams, training shards and checkpoints predate
these rule corrections. They have not been migrated or retrained.

These results show higher and more stable scores against the available controls, not a human
rating. Resource demand and future conversion remain approximate, the search does not model
opponents' responses, and the four-player tail remains weak. The income and money utilities
are estimates, not calibrated win probabilities. The rule audit covers the documented issues;
it is not an independent certification of the complete game engine.

## Historical V1 Results

The earlier v1-versus-August evaluation used seeds 2026092600..2026092623 under the old,
uncorrected engine. It produced mean candidate VP 134.44 / 118.39 / 100.88, with 47/48,
63/72 and 64/96 wins for 2/3/4 players. The four-player minimum was zero at seed 2026092610,
seat 2. That failure repeatedly built isolated resources, borrowed to the income floor and
passed through almost the entire Railroad era.

Those historical scores cannot isolate improvement against v2 because the engine rules
changed. The preceding source is retained in `src/game/rule_ai/economy_v1.rs`, and its raw
reports remain `output/economy-final-{2,3,4}p.json`. Its original module hash, including the
then-current tests, was `2A759829AFD6B72ACBEA1ED077DFF5BB7AA505E0A6073A3EB336B8D37AD38395`.
