# Synthetic teacher evidence

All data here comes from the calibrated JavaScript engine and `human-guide-v1` self-play. Human demonstration rows: **0**. Float32 data is compressed without changing its contents. `manifest.json` records compressed and original SHA-256 hashes; the source snapshots preserve the exact collector and final match runner.

Run commands from `brass-sim`, with Python + NumPy and Node 18+. On PowerShell, set `$env:BRASS_RULES='economy-v2'` before invoking Node runners. No PyTorch/GPU is needed.

```text
python world_model/unpack_teacher_evidence.py --out world_model/.local-experiments/human-teacher-repro
```

This verifies every packaged hash, checks archive paths, and restores binary datasets and source snapshots. Use a fresh output directory. The original machine's absolute paths in collection metadata are provenance, not portable command inputs.

## Collection and augmentation

Teacher collection: 160 games; seed 313100001 + game index × 9973; four guided seats; strategy `human-guide-v1`; depth 2; width 8; exploration .04; stride 2; 6 workers. Invoke the unpacked `frozen-teacher/world_model/cli.js collect-value` with the equivalent options shown by `cli.js`; incumbent models are `world_model/experiments/economy-20260920/4p/direct`. Whole games are split 128/16/16. The development test was later used to diagnose action encoding and is explicitly **not** final holdout evidence.

The snapshot CLI changes its working directory, so collection inputs and outputs must be absolute paths. Example in PowerShell:

```powershell
$reproRoot = (Resolve-Path 'world_model/.local-experiments/human-teacher-repro').Path
$incumbentRoot = (Resolve-Path 'world_model/experiments/economy-20260920/4p/direct').Path
node "$reproRoot/frozen-teacher/world_model/cli.js" collect-value --games 160 --players 4 --seed 313100001 --lineup guided,guided,guided,guided --strategy human-guide-v1 --exploration .04 --stride 2 --workers 6 --models $incumbentRoot --out "$reproRoot/recollected-values" --transitions "$reproRoot/recollected-transitions"
```

```text
node world_model/augment_double_rails.js --data world_model/.local-experiments/human-teacher-repro/teacher-transitions --out world_model/.local-experiments/human-teacher-repro/augmented-rebuilt --teacherRoot world_model/.local-experiments/human-teacher-repro/frozen-teacher --maxPerState 6
```

Every original state/action/successor is audited by replay. Up to six legal alternative doubles are appended at rail states in **training games only** (10,874 additional rows). Validation/test states and successors are unchanged; action vectors are reencoded to retain rail order and brewery identity. State schema and dimensions stay compatible; action encoding is explicitly `resource-network-v2`.

## Training

```text
python world_model/train_value.py --data world_model/.local-experiments/human-teacher-repro/teacher-transitions --validation-data world_model/.local-experiments/human-teacher-repro/teacher-transitions --out world_model/.local-experiments/human-teacher-repro/value-rebuilt --resume world_model/experiments/economy-20260920/4p/direct --epochs 48 --stride 4 --batch-size 512 --lr .0005 --opponent-weight 0 --feature-version brass-value-v2
python world_model/train_structured.py --data world_model/.local-experiments/human-teacher-repro/augmented-resource-v2 --validation-data world_model/.local-experiments/human-teacher-repro/augmented-resource-v2 --out world_model/.local-experiments/human-teacher-repro/dynamics-rebuilt --resume world_model/experiments/economy-20260920/4p/direct --epochs 40 --batch-size 512 --lr .0008 --policy world_model/experiments/economy-20260920/4p/direct/neural-policy.json
```

```text
python world_model/compose.py --dynamics world_model/.local-experiments/human-teacher-repro/dynamics-rebuilt --value world_model/.local-experiments/human-teacher-repro/value-rebuilt --policy world_model/experiments/economy-20260920/4p/direct/neural-policy.json --out world_model/.local-experiments/human-teacher-repro/composed-rebuilt --candidates learned-pool-v1 --shortlist score-diverse-v2 --continuation diverse-pool-12
```

The published artifact then adds metadata `planningDoubleRail=true`, `teacherStrategy="human-guide-v1"`, `syntheticTeacherGames=160`, `humanDemonstrations=0`, and `actionEncoding="resource-network-v2"`. Final `models/composition.json`, `training.json`, and `value-training.json` record component parameters, selections and history. The published policy is unchanged. Final model hash is frozen in `../protocol.json`. Floating point differences across BLAS implementations may prevent byte-identical retraining.

## Independent evaluation

```text
python world_model/evaluate_teacher.py --data world_model/.local-experiments/human-teacher-repro/fresh-reencoded-r2 --incumbent world_model/experiments/economy-20260920/4p/direct --candidate world_model/experiments/human-teacher-20261009/models --out world_model/.local-experiments/human-teacher-repro/prediction.json
```

All 40 fresh games (seed 995048391) are designated test; they were collected after candidate freeze, have disjoint seeds, receive no counterfactual augmentation and are never fitted or used to select checkpoints. Value errors here use all transition states, weighted by state count. Prediction metrics measure this teacher distribution, not human skill or out-of-distribution planning reliability.

Use unpacked `frozen-final/world_model/human_benchmark.js` for the three 120-game match runs and 40-game world comparison. Explicitly pass the incumbent model via `--models`, candidate via `--candidate`, lineups and seeds from `../protocol.json`, and fresh `--out` paths. Depth 2, width 8 and knowledge weight .8 are fixed. `analyze_teacher_league.js` pairs seeds/rotations/seat and bootstraps whole seed groups. Each rotation of a self-play game is correlated. Final results did not trigger further tuning.

The published full match reports are `../mixed.json.gz`, `baseline.json.gz`, `selfplay.json.gz`, and `world-comparison.json.gz`. `../match-report-manifest.json` records compressed and uncompressed hashes. The analyzer reads gzip automatically; standard gzip tools or Python's `gzip` module can restore the exact original JSON bytes.
