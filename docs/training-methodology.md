# Efficient Training Methodology for Board-Game Analysis AI

## Objective and claim standard

The product objective is not merely to beat the current checkpoint. It is to provide analysis that
helps strong players improve, and ultimately to reach top-human Brass: Birmingham strength. A model
is not top-human merely because it wins self-play, reduces validation loss, or occasionally scores
100 VP. That claim requires independent expert evidence.

The current project has **not** reached that standard. Its strongest absolute-score candidate and
its deployed head-to-head champion are different checkpoints, neither has been evaluated against a
qualified human panel, and no complete machine-readable corpus of expert games has been found.

## Evidence accumulated so far

| Finding | Evidence | Decision |
| --- | --- | --- |
| Local CUDA inference is the efficient generation path | One process reached 2.66 positions/s; four reached 8.95 positions/s, a 3.36x gain. Local inference was 2.92x faster than HTTP for a fixed game. | Use four local workers on the measured 16 GB RTX 5080. Do not add workers without a memory/throughput benchmark. |
| Waiting to merge HTTP requests is counterproductive | Shared dynamic batching reduced measured throughput from 608.54 to 521-548 positions/s. | Keep pipelining or remove HTTP/JSON; do not reintroduce wait-based batching. |
| Map-aware lifecycle knowledge has the largest measured behavioral effect | V5 strategy blending raised a same-checkpoint 64-search mean from single digits to 41.69 VP and a 256-search pilot to 54.56 VP while sharply reducing zero scores. | Keep V5 as an auditable search teacher until learned policy/value behavior surpasses it. |
| More policy accuracy is not enough | Iter12 reached held-out policy Top-1 26.90% but averaged 25.75 VP in its pilot. | Never promote from supervised metrics alone. |
| Better value loss is not enough | Iter14 improved all three held-out value losses and won 6/8 no-prior games, but its locked 256-search mean was 63.44 versus iter11's 66.81. | Keep absolute-quality and direct-opponent gates independent. |
| Larger or separated towers are not automatically stronger | Iter13 exact policy/value fusion and iter14 value-only training both failed the locked absolute-quality gate. | Do not scale architecture before fixing targets and external evaluation. |
| A point estimate is not promotion evidence | Several candidates had positive win rates or VP margins while the paired 95% lower bound stayed below zero. | Promotion requires the predeclared confidence bound, not a favorable headline win rate. |

## The compute funnel

Every hypothesis must move through the following stages. A failure stops that branch; later stages
must not be used to rescue a failed cheap gate.

### 0. Rules and information integrity

- Complete games must terminate legally for every supported player count.
- Legal actions, hidden-information determinizations, replay, seeds, and terminal scoring must be
  deterministic under a fixed configuration.
- Training observations must not expose opponent cards or other private state.
- Engine revision, feature schema, model identity, seed stream, and search settings must be recorded.

No training result is meaningful until this stage passes.

### 1. Search-only pilot

Test a strategy, utility, normalization, or search change with the same checkpoint and same seeds.
Use four games at 64 searches first. Inspect:

- mean, median, minimum, maximum, zero scores, and 50+/100+ counts;
- build/sell/network/develop/loan/pass counts by era;
- network before first industry, repeat loans, money, and income failures;
- per-game actions and exact settings, not just aggregate VP.

Reject immediately for repeated zero scores, lifecycle collapse, or a large regression. Only a
healthy variant advances to an eight-game 256-search confirmation.

### 2. Data and training pilot

- Generate training and validation by independent game seeds. Never split positions from one game
  across partitions.
- Start with the measured 40-game / 3,160-position training set and eight-game / 632-position
  validation set. Increase data only after evidence of underfitting or seed instability.
- Change one causal variable per candidate: target, loss weight, architecture, data mix, or teacher.
- Resume model weights deliberately; record whether optimizer state is restored.
- Use validation rollback and early stopping. The next initial setting is patience `3` and composite
  loss min-delta `0.001`; it is a starting hypothesis, not a universal constant.

Example:

```bash
python -m training.train \
  --shards output/train-*.jsonl \
  --validation-shards output/validation-*.jsonl \
  --resume output/baseline.pt \
  --output output/candidate.pt \
  --epochs 24 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.001 \
  --device cuda
```

The exact lowest validation-loss state remains the saved checkpoint even when a smaller improvement
does not reset patience.

For the fast expert-iteration path, pass the current teacher shards as `--shards`, old champion
trajectories as `--replay-shards`, and audited human positions as `--human-shards`. Set
`--replay-fraction` and `--human-fraction` to reserve replacement-sampling mass for those pools;
the remaining mass is assigned to the teacher. This keeps a small human shard visible without
duplicating it into a permanent corpus. The loader records the source counts and fractions in
checkpoint metadata, and human shards marked `value_target_usable=false` remain policy-only.

This is a local Python/Rust/CUDA operation. It has no browser, Svelte, or Playwright dependency;
browser smoke tests belong only to the separate UI release check.

### 2.5. Human-prior-guided expert data production

Small human replays are a prior, not a complete corpus. The efficient production step is to use the
human-adapted champion plus the versioned strategy prior as a teacher, sample several controlled
temperatures/strengths, and run many independent Rust games. This is **human-prior-guided
self-play**. Keeping only complete, healthy, high-scoring trajectories is **quality-filtered
trajectory generation**, or **Best-of-N/rejection sampling**. The learner then distills the search
policy in an **expert-iteration** loop.

Use the repository entry point rather than manually composing several shell loops:

```bash
python -m training.produce_expert_data \
  --checkpoint output/champion-iter3-mixed-v5-v6.pt \
  --output-dir output/iter19-expert \
  --train-games 128 --validation-games 16 --workers 4 \
  --strategy-prior-version human-strategy-v6-resource-aware \
  --strategy-prior-strengths 0.65 0.85 \
  --selection-temperatures 0.35 0.70 \
  --demonstration-shards output/human-replay-game6-audited.jsonl \
  --demonstration-prior-strengths 0.25 0.40 \
  --minimum-actor-vp 80 --minimum-vp-margin 0 --device cuda
```

The producer writes `raw/`, quality-filtered `train/` and `validation/` shards, and an atomic
`manifest.json`. The manifest records every recipe, model/engine identity, accepted/rejected counts,
rejection reasons, and generated game seeds. Training and validation use distinct seed namespaces;
the producer fails if any seed overlaps. By default it rejects any zero-score player, non-positive
ending income/cash, a network action before the actor's first build, and an immediately repeated
loan. Adjust those gates explicitly for a diagnostic run, never silently.

The resulting teacher shards can be mixed with the immutable champion replay buffer and audited
human policy shards using the source-fraction options shown below. A failed quality gate leaves the
raw artifacts and manifest for diagnosis but does not alter a checkpoint.

The demonstration flags are an optional **human-prior-guided self-play** layer. The loader extracts
only confirmed action-intent frequencies, smoothed by phase and round context; it does not treat the
human terminal score as a value label. The resulting distribution is blended into root PUCT with a
bounded strength, while the Rust legal-action list and lifecycle guards remain authoritative. Keep a
`0.0` control recipe in larger runs so the contribution can be measured instead of assumed.

### 3. Locked absolute-quality gate

Run eight blind, same-seed games at 256 searches, four determinizations, stable argmax, inference
batch 64, and V5 prior strength 0.7. The current bar is iter11:

- mean VP greater than `66.8125`;
- no zero-score player;
- no clear median or lifecycle regression;
- the action distribution remains plausible rather than gaming one score statistic.

Reuse fixed seeds for paired development comparisons, but keep an untouched confirmation seed set
to detect pilot overfitting.

### 4. Direct pilot

Run four independent seeds with the candidate rotated through both seats: eight games total. Turn
off strategy priors for both agents when measuring what the networks themselves learned. A candidate
must clearly improve on the previous pilot frontier without failing the absolute-quality gate.

The current direct frontier is iter14 at 6/8 wins and mean VP margin `+15.125`, but its confidence
lower bound was still `-1.091`; this is a screening result, not promotion proof.

### 5. Formal promotion gate

Only survivors receive the 20-seed / 40-game seat-rotated PUCT gate. Treat each seed group, not each
correlated seat game, as an independent unit. Promote only when the two-sided 95% Student-t lower
bound on candidate score delta is greater than the declared margin (currently zero).

The deployed champion remains unchanged on every failure. Never replace it manually because a
candidate looks promising. Use `python -m training.promote` with the full evaluation report for the
final handoff: it verifies both checkpoint hashes and the passing summary, keeps the old checkpoint
as a recoverable backup, and writes a promotion manifest.

### 6. Human-strength and coaching gate

Self-play gates establish relative project progress, not top-human strength. The final claim needs:

- a versioned corpus of complete expert action logs with player qualification and game context;
- held-out move agreement and search-regret analysis at decision, turn, and era level;
- blinded expert review of Top-N recommendations and explanations;
- seat-rotated human matches or an equivalent independently administered evaluation;
- value calibration on external games, including uncertainty and out-of-distribution detection;
- evidence that analysis improves human decisions in pre/post or controlled training exercises.

VP distributions are diagnostics. Compare them against matched expert games once such data exists;
do not use an isolated 100+ score as a substitute for this gate.

## Timeboxing and stopping rules

- Predeclare the hypothesis, baseline, seeds, metrics, threshold, and maximum compute before running.
- Allow at most two cheap variants and one blind confirmation per hypothesis. Three failures require
  a different hypothesis, not more epochs or a larger model.
- Do not generate a full new corpus for a change that can be tested at search time.
- Do not run 40 games for a candidate that fails either the absolute-quality or direct pilot.
- Stop an epoch run after validation patience is exhausted; retain the exact best state.
- Keep a result manifest and failed experiments. Negative evidence prevents expensive repetition.
- Rebenchmark throughput only after changing hardware, process count, model size, transport, or batch
  shape. Optimization without a changed bottleneck is not an experiment.

## Fast path: expert iteration with a league

The shortest reliable route to a strong player is an **expert-iteration** loop rather than a
sequence of isolated hyperparameter bets. Keep the current champion immutable and run this loop:

```text
human/teacher games + champion replay buffer
                |
                v
strong teacher = V6/V8 lifecycle prior + batched PUCT
                |
                v
search-policy targets -> learner -> candidate checkpoint
                |
                v
fixed benchmark + checkpoint league + human-position review
                |
        pass ----+---- fail
        |              |
  promote champion    keep candidate only as an artifact
```

Operationally, use three independent data pools:

- **Replay buffer:** retain a reservoir of old champion/search positions so a small new human
  shard cannot erase general play. Add clean human decisions as policy-only labels when the game
  has replay anomalies; enable value labels only for a complete, audited game.
- **Teacher pool:** generate fresh positions with the best known search configuration (currently
  the V6 resource-aware prior plus batched PUCT). Store the search distribution, uncertainty, seed,
  and engine/model identities, not just the selected move.
- **Evaluation pool:** never train on the fixed seed groups used for the absolute-quality and direct
  strength suites. Add a separate expert holdout for Top-k agreement and decision regret.

Use a two-speed schedule. Every candidate first gets a four-game/64-search smoke screen; stop on a
zero-score trajectory, lifecycle violation, or large regression. Only survivors receive an eight-game
256-search confirmation and then the seat-rotated league gate. Within a surviving training run,
validate after every epoch and retain the best validation state. Run multiple candidates from the
same data snapshot only when they differ in one declared teacher/target variable, so compute is spent
on breadth rather than repeated long runs.

For human improvement, add DAgger-style collection to the UI: record the human choice, the current
model's Top-N disagreement, and later outcome/regret. Ask for another human label only at high
uncertainty or high disagreement positions. A single spectacular score is not a label: replay
integrity and cross-game agreement come first. The current three-player scalar value backup is a
useful bridge, but the final multiplayer system should learn a per-player value vector before making
top-strength claims.

This loop is deliberately asymmetric: use the engineered teacher to obtain immediate playing
strength, then let the network distill it and gradually reduce the prior/search crutch. That gives a
usable opponent now while preserving a monotonic, auditable path toward a stronger champion.

## Highest-value next Brass work

1. **Acquire external evidence.** Extend replay export into an opt-in, privacy-reviewed expert-log
   format and recruit qualified players. Public search has not produced usable complete logs.
2. **Audit value targets before scaling.** Iter14 shows that terminal BCE/Huber improvements do not
   guarantee stronger score-seeking search. Compare phase-conditioned return targets, distributional
   final VP, and opponent-relative utility with fixed policy tensors.
3. **Create two untouched benchmark suites.** One measures absolute lifecycle/VP quality; the other
   measures direct strength. Rotate them only after a hypothesis family is closed.
4. **Measure coaching value.** Record whether users accept, reject, or later regret recommendations,
   while keeping private information and consent explicit.
5. **Scale model/data only after target validation.** A larger network is justified only if learning
   curves show underfitting and the candidate clears behavioral pilots.

## Porting the method to another board game

The reusable core requires a small game-specific contract:

1. deterministic state transition and terminal scoring;
2. complete legal-action enumeration with stable semantic keys;
3. player-relative information-set observations;
4. hidden-state sampling consistent with public evidence;
5. replayable seeds and engine revision identity;
6. game-specific lifecycle diagnostics in addition to win/loss;
7. seat-rotated paired evaluation and a domain-appropriate human benchmark.

Policy/value networks, variable-action batching, checkpoint metadata, independent game partitions,
early stopping, paired confidence gates, staged browser analysis, and the compute funnel are generic.
Rules, observation encoding, determinization, strategy priors, lifecycle checks, and expert standards
remain game-specific.

## Minimal experiment record

Every run should answer these fields before another run starts:

```text
hypothesis:
single changed variable:
baseline and candidate IDs:
engine/schema revision:
train/validation/pilot seeds:
search and inference settings:
predeclared pass/fail thresholds:
elapsed time and peak memory:
validation result and selected epoch:
absolute-quality result:
direct result and confidence interval:
decision: reject / confirm / formal gate / promote
next action:
```

This record is the main defense against an indefinitely expanding training schedule.
