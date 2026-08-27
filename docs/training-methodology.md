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
candidate looks promising.

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
