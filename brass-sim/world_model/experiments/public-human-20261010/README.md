# Audited public human intent and critical controls

See `../../../docs/public-human-training-results.md` for the full results and limits.

This directory includes:

- `audit.json`: eight complete native replay audits; only six four-player human-vs-AI games enter the intent corpus. Exact source commit/tree and compiled module hashes are recorded. Human flags are source assertions; expert skill is unverified.
- `human-intents.jsonl.gz`: 372 derived public-feature/action-intent rows, whole-game split 248/62/62. No foreign-engine JS value targets or raw private logs.
- `human-intent.json`, `training.json`, `fixture.json`: selected intent MLP, training history and Python/JS probability parity fixture.
- `critical-world-model.json`, `critical-training.json`, `critical-fixture.json`: categorical controls trained on synthetic teacher transitions; all original board deltas/gates and value weights retained. Optional experiment, not a default promotion.
- `protocol.json`, `development-protocol.json`, `frozen-source.zip`: candidates selected before fresh final seeds, execution sources frozen as exact bytes.
- `reproducibility/`: new independent synthetic teacher prediction data and checksums. Every game in `fresh-holdout` is test-only; `collectionSplit` records the collector's original bookkeeping, not training usage in this round.
- `local-private-audit.json`: aggregate audit of the private local SQLite only; no accepted complete trajectory and no raw private actions exported.
- Compressed game reports, `prediction-holdout.json` and `summary.json`: frozen final playing-strength and prediction evidence.

Extract `human-intents.jsonl.gz` into a new directory beside a copy of `audit.json`
before invoking `train_human_intent.py`. Decompress `.f32.gz` files before using
`Dataset`. The critical-control trainer uses the prior experiment's
`reproducibility/augmented-resource-v2` data as both primary and validation input;
its immutable train/validation splits remain separate.

For exact tournament reproduction, extract `frozen-source.zip` into `brass-sim`
first. Git checkout may normalize line endings and change source checksums.
All test seeds and planning budgets are in `protocol.json`. Analysis resamples
whole seed groups; identical-profile rotations are correlated.

The optional prior is available to `BrassPlanner.plan` through `intentPrior` and
`intentWeight` (0–2), using calibrated guided strategy profiles. The benchmark CLI
accepts `--intent-prior` and `--intent-weight`. The browser still selects the
previous teacher model. Neither this native-intent transfer nor the public-hands
JS tournament certifies human expert or tabletop strength.
