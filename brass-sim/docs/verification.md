# MVP verification — 2026-09-14

Historical verification of the initial model. Its weights/reports are now archived under `world_model/experiments/v1/`. See [continued-training verification](continued-verification.md) for the current version.

- Source: `Yuiffy/brass-birmingham-sim`, commit `cdd9ac458106fa34e626168cf36c82d9f3a38cf7`.
- Generated 1,000 complete four-player games, 128,000 transitions. Re-generated the first 10 games with the final generator and compared the float32 file byte-for-byte with the corresponding prefix of the original data: identical.
- Split integrity: 128,000 unique row IDs across disjoint train/validation/test partitions, with each complete game confined to one split.
- Trained the 96-hidden-unit dynamics MLP and independent action-scoring MLP for 24 epochs on CPU. The dynamics model selected epoch 24 by validation MSE; the policy selected epoch 23 by its own validation MAE.
- Evaluated on 100 held-out games / 12,800 transitions. Reported field groups, change precision/recall/F1, action breakdowns, a persistence baseline, recursive depth 1–5 errors and example discrepancies.
- Compared 80 / 320 / 800 training-game subsets using the same 100-game validation set and hyperparameters. Results and training histories are in `world_model/reports/learning-curve.json`.
- Completed 100 additional tournament games with rotating seats, seed 800001, depth 2 and root candidate width 8. No games were truncated. Full records are in `world_model/reports/tournament.json`.
- `npm test`: 11 tests passed, 0 failed. Includes original AI regression, full state cloning, valid/failed transitions, 2/3/4-player games through both eras, cross-language model output agreement, learned planning without ground-truth transitions, and tournament accounting.
- All JavaScript files passed `node --check`; the Python modules passed `py_compile`; `git diff --check` passed.
- Chrome verification: model loading, AI-type selects, four-AI game, autonomous turns, canal scoring and continuation, world-model candidates, recursive prediction, real execution and duplicate-execution prevention. No JavaScript errors observed.
- With AI disabled, a human Loan changed Alice from £17 / income 10 to £47 / income 7 and advanced play to Bob, using the original action workflow. No AI toolbar appeared.
- Browser Web Worker completed a four-game tournament. Per-AI average VP and win rates matched the equivalent first four Node tournament games: world 81 / 50%; search 77.75 / 50%; heuristic 78.5 / 0%; neural 65.5 / 0%.

Captured UI views: `screenshots/world-model/imagination.png` and `screenshots/world-model/arena.png`.

The model is demonstrably approximate. Current-player accuracy is only 21.34%; its dense whole-state prediction errors exceed a state-persistence baseline. The UI and documentation disclose this alongside the successful pipeline and tournament results.
