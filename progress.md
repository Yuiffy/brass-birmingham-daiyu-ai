Original prompt: 我想找一个工业革命伯明翰的顶级AI进行人机对练或者观察ai局，来提升我的伯明翰实力。你帮我找找有没有现成的，没有的话找找有没有开源游戏我们拿来自己训练，还没有的话我们看规则书开发一个游戏然后练ai。我们项目里有ai训练经验。

## 2026-10-09 teacher training and strategy experiments

- Executed docs/human-replay-next-experiment.md on codex/brass-human-strategy; preserve the unrelated Manila checkout.
- Engine-based financing/production chains implemented and tested but did not beat the incumbent in development; retained experimental profiles.
- Teacher: 160 synthetic human-guide-v1 self-play games, 128/16/16 whole-game split; no human training rows. Audited 10,874 extra legal training-only doubles.
- Explicit resource-network-v2 encoding captures rail order/brewery source. Value trained 48 epochs, dynamics 40; immutable candidate hash in protocol.json.
- Original 16-game prediction test reused for encoding design, relabeled development. New 40-game prediction holdout fully audited after model freeze: value RMSE 59.90 -> 11.91 VP; double link exact 0 -> 62.70%.
- Formal 360 games completed, 30 fresh seed groups per setting: old selfplay 123.675, candidate selfplay 130.967, candidate vs three incumbent 127.842. Paired fixed gain +4.17 CI [-.175,8.55], not confirmed; selfplay +7.29 CI [3.025,11.658]. No default promotion or post-holdout tuning.
- World 40-game mixed comparison completed: new 45.70, old 38.125; still poor. Public hands / partial JS rules do not establish tabletop or human strength.
- New opt-in teacher-trained-v2 UI; only four-player AI. Browser pause/step/inspect/rethink/execute/resume/double preview verified; screenshots inspected; no console errors. Standard skill client also passed.
- Full 76 Node + 11 Python tests pass. Published dataset/source snapshots verify and unpack all 20 manifest artifacts; additional runtime guard audit checks equivalent accepted inference.
- Results, reproducibility, evidence limitations and remaining causal diagnostics in brass-sim/docs/human-teacher-training-results.md.
- Remaining delivery: verify final staged evidence, commit/push same branch, update draft PR #1, verify remote SHA. Real expert replay acquisition, complete official/hidden-information rules and unsold/development causal attribution remain follow-up research, not claims of this release.

## 2026-10-09 active goal: human strategy for guided search

User goal: 我看网上都是说稳140 150分的。你去看他们的攻略或者教程，也可以看B站的视频教程或者比赛对局，学习他们的经验用到我们的学习增强搜索里，是不是就能达到他们的水平了？你试试

- Working branch: `codex/brass-human-strategy`, managed worktree `brass-human-strategy`.
- Research: read ChrisLuv, Erik Twice and Dave C / Skypray Huang guides; checked network rules.
  Bilibili public API verifies the 150+ video's description says three CPU opponents; subtitles are absent.
- Scope: calibrated JavaScript `guided` search, exact legal double rails, human action-efficiency guidance,
  frozen-opponent comparisons and full candidate self-play on independent seeds. No human-level claim from VP alone.
- Completed: atomic single/double network plans, canal own overbuild, same-type/higher-tier opponent overbuild;
  source-derived strategy blends 80% guidance / 20% frozen learned value, root 64 / continuation 12.
- Frozen holdout: 120 mixed + 120 baseline + 120 candidate self-play, 30 independent groups each.
  Candidate mixed mean 121.61; candidate self-play 124.38; baseline self-play 78.28.
  Paired gain +43.33 [39.97, 46.66] VP. Self-play >=140 19.17%, >=150 6.67%; stable targets NOT achieved.
- UI: explicit experimental option in calibrated JavaScript guided search; inspection follows actual agent.
  World dynamics / weights and Rust AI unchanged. Double preview uses real engine, not untrained dynamics.
- Validation: 59 Node tests, 10 Python tests; provided Playwright client and four-player control/double-rail
  smoke; screenshots inspected, controls verified. Arena final result table verified. Removed request for
  absent calibrated prediction report (existing 404); explicitly shows that this report is unavailable.
- Remaining research: financing / production-chain failures (heldout worst 71 VP), farm building, complete
  resource/card/merchant choices, hidden hands and official map/industry audit. No further tuning on holdout.

## Product direction

- Browser UI remains SvelteKit/TypeScript.
- Rust owns rules, search, and native self-play; browser delivery can use HTTP first and WASM Worker later.
- The analysis inspector must distinguish visit share, estimated shared win rate, policy probability, and calibrated win rate.
- Search follow-up explanations must be grounded in the selected legal action, immediate deltas, search samples, and current game state.

## Visual thesis

A restrained industrial drafting desk: the real board dominates, neutral graphite and white frame the tools, and color is reserved for players, recommendation rank, and evaluation state.

## Content plan

- Primary board workspace with compact turn and era status.
- Existing hand and action controls remain close to the board.
- Right analysis inspector shows Top 1/2/3, coverage, uncertainty, and one selected move's evidence.
- A contextual why-question thread stays in the inspector without obscuring the board.

## Interaction thesis

- Analysis runs with stable progress/status dimensions so the board never shifts.
- Selecting a candidate reveals its exact action and immediate deltas, with restrained highlight transitions.
- Asking why appends a contextual answer while preserving the selected candidate and board state.

## Completed

- Repaired major rules and turn-flow defects in the upstream AGPL engine.
- Added deterministic complete legal actions and strict Python atomic APIs.
- Added hidden-information determinization preserving observer hand and wild-card ownership.
- Added determinized root-UCB random-rollout search with Top-N metrics and sample continuations.
- Verified standard Rust tests, Python-feature tests, deterministic search, fractional ties, and 2/3/4-player complete rollouts.
- Measured release search throughput: about 972 simulations/s for 2 players and 450 simulations/s for 4 players on initial positions.

## Current

- The Axum analysis API and Svelte analysis inspector are integrated and live-tested.
- The current recommendation engine is deliberately labeled as a random-rollout baseline. Its
  visit share and sampled shared win rate are not policy probability or calibrated win rate.

## 2026-08-15 integration and browser QA

- Added `POST /api/analyze`, `POST /api/explain`, and
  `POST /api/apply_analyzed_action` with revision-checked analysis caching.
- Verified a real two-player HTTP flow: create, start turn, analyze 800 continuations, ask
  probability/risk/comparison questions, apply the exact cached action, reload from SQLite, and
  reject a stale recommendation after manual mutation.
- Initial-position HTTP sample covered all 596 root actions in 870 ms. With only two visits on the
  leading actions, the displayed 100% samples are correctly treated as low-confidence estimates.
- Persisted replay after applying a recommendation matched the returned state exactly.
- Desktop QA at 1280x720 covered Top 1/2/3 selection, four separate probability fields, map
  highlights, suggested questions, free-form questions, applying a recommendation, and loading the
  saved game.
- Mobile QA at 390x844 found and fixed document-wide overflow from the desktop card fan. Mobile
  hands now use an isolated horizontal scroller with complete card labels; the board, inspector,
  candidates, metrics, explanations, and question form have no horizontal page overflow.
- The supplied web-game Playwright client produced nonblank gameplay screenshots and matching
  `render_game_to_text` output. Browser console error count was zero.
- Verification: `cargo test --all-targets` passed 155 tests; `npm run build` passed. The only build
  output is the pre-existing Tailwind content warning and existing Rust unused-code warnings.

## Later

- Replace the random rollout policy with trained policy/value inference.
- Build self-play orchestration, evaluation gates, calibration, and model versioning.
- Add AI-vs-AI observer controls and model-strength comparison views.
- Add focused HTTP serialization/revision/replay tests around the analysis endpoints.
- Compile the Rust core to WASM Worker after the HTTP product path is stable.

## 2026-08-15 training schema and deterministic self-play

- Added versioned observer-relative training features: 2111 dense state values and 1615 sparse
  atomic-action features. Opponent hidden card identities are excluded.
- Added Python RL methods for the feature schema, training state, sparse legal actions, stable full
  search visit distributions, and official fractional shared-win outcomes.
- Added the Rust `export_self_play` command. JSONL shards lock the format, feature schema, crate
  version, engine revision, search settings, seed streams, and deterministic shortfall policy.
- Every position stores the full stable legal-action list, sparse features, visit policy target,
  selected action, and final shared-win/VP-margin target for the acting player.
- A one-simulation two-player smoke game produced 79 positions and 81 total records. Repeating the
  export with identical arguments produced byte-identical 2,841,865-byte files with SHA-256
  `575A5ADDB1F1220476CEADCB15A0CDB73CE6AD09D2E3B20225955AE71A37667A`.
- Verification: 35 focused library tests pass (including 2 deterministic self-play tests), all 155
  pre-existing/all-target tests pass, and all 43 Python-binding library tests pass.

## 2026-08-15 policy/value training and evaluation

- Added an offset-indexed JSONL dataset and variable-legal-action batching. Sparse action features
  use `EmbeddingBag`; the network has policy, shared-win, and VP-margin heads trained with policy
  cross-entropy/KL, BCE, and Huber losses.
- Added versioned atomic checkpoints plus training and Top-N inspection CLIs. Dataset loading rejects
  mixed engine revisions by default, and checkpoint loading validates the complete feature schema.
- Trained a real self-play shard for two smoke steps into `output/self-play-smoke-model.pt`; model
  inspection successfully emitted three legal actions with normalized policy probabilities.
- Added seat-rotated candidate-vs-champion evaluation with identical deck seeds. Promotion requires
  the two-sided 95% Student-t lower confidence bound on candidate score delta to exceed the configured
  margin. The default 40-game gate now uses `t(39) = 2.02269092`; a same-checkpoint smoke correctly
  refused promotion.

## 2026-08-15 AI-vs-AI browser observation

- Added persistent AI playback state and controls for play/resume, pause, single-step, stop, and
  0.5x/1x/2x pacing. Each new position keeps its Top 3 visible before Top 1 is applied.
- Added deterministic shortfall resolution plus replayable `ResolveShortfall` persistence events.
  Asking a question, selecting another candidate, or switching to the game inspector pauses playback.
- Browser QA started a clean two-player seed-777 game, advanced three automatic moves, paused with
  Top 3 intact, answered free-form and comparison questions without advancing, then single-stepped
  exactly once and refreshed the next position's 340 legal candidates.
- Desktop width was `clientWidth = scrollWidth = 1280`. At a 390x844 viewport, the scrollbar-adjusted
  width was `clientWidth = scrollWidth = 375`; Top 3, details, metrics, controls, answers, and the
  question form had no horizontal overflow or overlap. Browser console error count was zero.
- Final verification: 164 Rust all-target tests, 47 Rust tests with Python bindings, and 8 Python
  training/evaluation tests passed. The Svelte production build passed; only the existing Tailwind
  content warning and existing Rust unused-code warnings remain.

## Strength TODO

- Connect trained policy/value inference to Rust search and the browser analysis API. Current browser
  recommendations remain explicitly labeled random-rollout estimates, not trained policy probability
  or calibrated win rate.
- Replace root UCB with model-guided PUCT, generate a meaningful high-simulation self-play corpus,
  and run repeated candidate-vs-champion promotion cycles.
- Calibrate displayed win percentages on held-out games before presenting them as reliable odds.
- Measure strength against strong human or external reference play. Do not claim top-level strength
  from smoke training or random-rollout search.
- Compile the stable Rust core to a Web Worker WASM build if offline/local browser inference becomes
  more valuable than the current HTTP deployment path.

## 2026-08-15 model-guided root search and inference

- Added model-guided determinized root PUCT while preserving the random-rollout UCB baseline.
  Search validates the model ID, stable action-key order, probability mass, shared-win value, and
  VP-margin value; high-prior actions are visited first when simulations are scarce.
- Added strict checkpoint inference plus a persistent standard-library HTTP inference server.
  Checkpoints receive a stable SHA-256 model ID, and evaluation validates the complete state/action
  feature schema before returning keyed policy probabilities and both value-head outputs.
- Connected the Rust web backend to inference through `FAST_BRASS_INFERENCE_URL`. Configured model
  failures are surfaced instead of silently falling back; an unset URL retains the UCB baseline.
- Updated the browser analysis inspector to distinguish policy prior, PUCT visits, random-rollout
  evidence, root model value, and calibration. The value head is currently diagnostic and is not yet
  used at search leaves.
- Exposed policy-guided PUCT through the Python ABI. A focused binding regression proves that one
  simulation visits the unique highest-prior action, every legal-action row retains its policy
  probability, and an incorrect probability count is rejected.
- Verification so far: 49 Rust library tests with Python bindings, 5 focused search tests, and 9
  Python training/inference tests pass. A real 583-action smoke inference normalized to probability
  mass `1.0000000357`; its two-step checkpoint remains integration-only evidence, not strength
  evidence.
- Real HTTP verification created a fixed-seed two-player game, started the turn, and analyzed all
  596 legal actions with 800 model-guided simulations in 841 ms. The response carried the PUCT
  method, SHA-256 model ID, root values, per-candidate policy priors, and null calibrated win rates.
- Fixed Chinese comparison follow-ups for a selected Top 1 candidate. Questions such as
  `为什么这步比第二选择好？` now compare Top 1 against Top 2 by visits, sampled shared-win value,
  policy prior, and immediate resource deltas, while explaining that ranking is visit-first.
- Added `training.model_self_play`, which runs checkpoint inference at every decision, sends the
  complete stable prior through the Python Rust-PUCT binding, samples from visit targets, and writes
  atomic JSONL shards compatible with `SelfPlayDataset`. Model IDs, checkpoint step, root values,
  policy priors, visits, and deterministic seed streams are retained for auditability.
- Two real one-simulation smoke exports each completed 79 positions and were byte-identical with
  SHA-256 `F595CC0DDEA92C1B745E09B97433D4491700F2A1BDC820074D7CFFE178592AF2`.
  The resulting 2111-state/1615-action records load successfully and have normalized policy targets.

## 2026-08-15 first PUCT self-play promotion cycle

- Generated four parallel model-guided shards at 800 simulations per decision: 20 two-player games,
  1,580 positions, and about 1.26 million root simulations. All shards share engine revision
  `8fe8e3fa17b4331bb8a7478bd595bd52c898c193-dirty`; 12/20 games had nonzero VP, with mean
  player VP 7.475 and maximum 47. This is useful first-iteration data but still visibly weak play.
- Continued the smoke champion on the new corpus for 10 epochs on an RTX 5080, producing
  `output/puct-candidate-iter1.pt` at step 242. Validation loss moved from 4.903 to 4.438 and policy
  KL from 0.311 to 0.305; validation policy Top-1 remained weak at 17.0%.
- The existing 40-game seat-rotated gate promoted the candidate over the two-step smoke champion:
  mean shared win 0.9375, mean score delta +0.875, standard error 0.0639, and two-sided 95% lower
  bound +0.7457. This establishes improvement over the smoke baseline only, not strong-human play.
- Registered the byte-identical `output/champion-iter1.pt` with SHA-256
  `8E7180489A004CBDA2ACAD75D5FA724BB7018ADCBD9AFD6727C972367F9F636F` and switched both CUDA
  and container inference services to it. Health reports checkpoint step 242 and the matching model
  ID.
- Desktop and 390x844 browser QA verified full-action 800-simulation PUCT, Top 3, all four distinct
  probability fields, free-form Top-1-vs-Top-2 explanation, and one AI playback step that applied the
  recommendation and refreshed the next position. Desktop width was 1280/1280; mobile document
  width was 375/375; console error count remained zero. The supplied web-game client also produced
  a nonblank board screenshot and matching text state.
- Final regression: 168 Rust all-target tests, 50 Rust library tests with Python bindings, and 10
  Python training/inference/self-play tests passed. The Svelte production build passed. Remaining
  output is limited to pre-existing Rust unused-code warnings, the existing Tailwind content warning,
  and unrelated trailing whitespace already present in `ui/package-lock.json`.

## 2026-08-15 batched successor-value search foundation

- Added deterministic, stable root-action successor batches. Every action is evaluated across the
  same root-information-set determinizations; non-terminal states are encoded for the next actor,
  while terminal branches retain exact root-player shared-win and VP-margin values.
- Added value-only PyTorch batching and an `/evaluate-values` inference endpoint so hundreds of
  successor states can bypass policy action embeddings. Existing `/evaluate` behavior is unchanged.
- Focused verification: 7 Rust training-feature/successor tests and 5 Python training/inference tests
  pass. Next: aggregate next-actor predictions into two-player root-perspective action values and use
  those values for PUCT selection; three/four-player search must retain rollout fallback until the
  model exposes per-player values.

## 2026-08-15 two-player successor-value PUCT integration

- Added two-player root-perspective aggregation: a successor evaluated for the same actor keeps
  `(v, margin)`, while an opponent-to-act successor becomes `(1-v, -margin)`. Terminal branches use
  exact official outcomes. Three/four-player scalar conversion is rejected and retains rollout search.
- Added `determinized_root_puct_policy_batched_successor_value`. It uses every action's batched model
  Q value from the first PUCT visit instead of forcing one random terminal rollout per legal action.
  Reports distinguish valued actions from visited actions and leave rollout-only metrics null.
- The web backend now calls `/evaluate-values`; a fixed-seed live position evaluated all 603 actions
  with four determinizations, completed 800 PUCT visits, and visited 74 actions in 1,779 ms using the
  container CPU inference service. Top 1 received 695 visits. A Chinese Top-1-vs-Top-2 follow-up
  correctly described model values, perspective conversion, calibration limits, and one-ply limits;
  applying the cached exact action succeeded.
- Python self-play now records successor seeds, value sources, per-action model values, sample counts,
  and the new method. A real champion smoke game completed 79 positions in about 12 seconds at eight
  visits/two determinizations; the shard loaded through `SelfPlayDataset` with no `.partial` residue.
- Candidate evaluation now supports `--search-simulations` and runs the deployed PUCT agent. A real
  same-checkpoint two-game seat rotation at eight visits/two determinizations produced mirrored +1/-1
  outcomes, mean delta 0, and correctly refused promotion.

## 2026-08-15 successor-value explanation regression and repository boundary

- Value-mode risk explanations now describe the four hidden-information successor samples, model
  uncertainty, and the one-ply value limitation instead of claiming that later players use random
  policies. Candidate comparisons call the signal `successor-value feedback`; rollout modes retain
  their original rollout wording.
- A live 800-visit HTTP regression evaluated all 596 legal actions and returned the corrected Chinese
  comparison/risk answer. The supplied browser-game Playwright client produced a nonblank board,
  matching `render_game_to_text`, and no browser error artifact.
- Verification: 173 Rust all-target tests passed and the Svelte production build passed. Only the
  existing Rust unused warnings and Tailwind content warning remain.
- This prototype currently lives as a nested Git repository under the unrelated danmaku project.
  Before publishing, move it to a standalone checkout and create a public fork of
  `artyom-morozov/fast_brass`, preserving AGPL-3.0 history and attribution. Audit or replace the
  tracked board/card artwork separately because the repository does not document an asset license.

## 2026-08-15 batched multi-layer neural PUCT core

- Added a resumable Rust `BatchedNeuralPuctSearch` for two-player information-set approximation.
  It owns one tree per deterministic hidden-hand sample, exports stable encoded leaf batches, applies
  virtual visits inside a batch, converts next-actor values back to root perspective, and validates
  model/action identity before mutating the tree.
- Root reports aggregate visits and backed-up neural values across determinizations and expose the
  actual maximum search depth, leaf-evaluation count, and inference-batch count under method
  `determinized_batched_neural_puct`.
- Three focused integration tests prove depth-two expansion, distinct leaf reservation within one
  batch, root-value backup, visit conservation, and atomic rejection of an invalid leaf model.
- Next: add variable-action PyTorch batch inference and expose the search session through PyO3, then
  replace the one-ply browser/self-play path.

## 2026-08-15 end-to-end multi-layer neural search

- Connected the resumable Rust tree search to PyTorch variable-action batch inference through
  `/evaluate-batch`. Browser analysis, Python self-play, and candidate evaluation now share the same
  batched multi-layer neural PUCT implementation instead of stopping at one-ply successor values.
- Exposed real search work in the API and UI: maximum tree depth, neural leaf evaluations, and
  inference-batch count. Chinese follow-up answers now describe opponent responses, leaf evaluation,
  value backup, candidate comparisons, and the current lack of independent calibration.
- Canonicalized semantically equivalent legal actions caused by duplicate copies of the same card.
  Search and training no longer split visits or policy mass between indistinguishable moves.
- A fixed seed-424242 two-player position completed 800 visits in about 7.3 seconds on an RTX 5080,
  reached depth 5, evaluated 800 neural leaves in 13 inference batches, and returned three distinct
  semantic Top-N actions. The duplicate-action fix reduced the root action count from 582 to 518.
- Desktop 1280x720 and mobile 390x844 browser QA covered applying a recommendation, pausing after
  one AI step, refreshing analysis for the resulting state, long Chinese explanations, and overflow.
  The supplied Playwright game client produced a nonblank board and matching text state; browser
  console errors and warnings were empty.
- Deep neural analysis remains a two-player feature, so the setup screen now defaults to two players.
  The current champion is evidence of progress over a smoke baseline only, not top-level AI strength
  or calibrated practical win probability.

## 2026-08-15 public repository

- Created the public GitHub fork
  [`Yuiffy/brass-birmingham-daiyu-ai`](https://github.com/Yuiffy/brass-birmingham-daiyu-ai), keeping
  `artyom-morozov/fast_brass` as the upstream source and preserving the AGPL-3.0 history.
- Documented that the tracked board and card artwork was inherited from upstream and has no separate
  asset-license record in the repository. Model checkpoints, JSONL self-play shards, SQLite files,
  and other generated output remain excluded from version control.
- Publication regression passed 178 Rust all-target tests, 55 Rust library tests with Python
  bindings, 14 Python training/evaluation tests, and the Svelte static production build. Remaining
  compiler output is limited to existing unused-item and Tailwind content-configuration warnings.
- A fresh browser seed-424242 regression analyzed 518 semantic legal actions with 800 neural PUCT
  visits in 7.81 seconds. It reached depth 4 with 800 neural leaf evaluations in 13 inference
  batches; Top 3 visit shares were 41.5%, 25.5%, and 16.5%.
- The Chinese question `为什么这步比第二选择好？` compared visits, backed-up tree values, model
  priors, opponent responses, and immediate resource changes. Applying Top 1 then spent £3, built
  Road 33, advanced to the second player's turn, and wrote the exact action key to the game log.
- The standard web-game Playwright client produced setup and gameplay screenshots with matching
  `render_game_to_text` state and no console-error artifact. The in-app browser showed no document
  overflow at 1280x720.
- `npm audit --omit=dev` reports one moderate Svelte advisory; the full development tree reports 12
  advisories (1 low, 6 moderate, 5 high). The complete automated fix requires breaking Svelte/Vite
  upgrades, so dependency migration and its UI regression pass remain an explicit follow-up.

## 2026-08-15 independent validation partitions

- Replaced the position-level random validation split with explicit `--validation-shards`; all
  positions in training shards now remain training-only and all validation positions come from
  separately generated games.
- Training rejects overlapping shard paths, incompatible feature schemas, different engine
  revisions, and reused game seeds across training and validation when the seed is recorded.
- The deprecated `--validation-fraction` now defaults to zero and fails loudly when nonzero instead
  of reporting a leaked validation metric.
- Training and validation epoch metrics are weighted by position count, so a smaller final batch
  cannot change checkpoint selection merely because of `batch-size`; a focused regression compares
  uneven and single-batch validation loaders.
- Focused training-partition and model-pipeline verification passes 14 tests.

## 2026-08-15 Torch-free remote promotion evaluation

- Candidate and champion evaluation sources can now independently be local checkpoints or remote
  inference URLs. Both greedy policy play and multi-layer neural PUCT use the shared evaluator
  protocol, and reports retain each service's model ID and checkpoint step.
- `training.evaluate` no longer imports PyTorch at module load. The Docker development container
  reports `torch_available=False` and imports the evaluator successfully.
- A real two-game seat-rotated same-champion smoke used the host CUDA service from Docker at eight
  search simulations/two determinizations. The score deltas were `+1/-1`, mean delta was zero, and
  the gate correctly refused promotion.
- Focused evaluation, remote-inference, neural-search, and self-play verification passes 14 tests.

## 2026-08-15 iter2 independent-data cycle

- Generated 20 training games (1,580 positions) and four independent validation games (316
  positions) from champion step 242. Every decision used 256 multi-layer neural PUCT visits, four
  hidden-information determinizations, and batches of up to 64 leaves.
- Formal loading found 20/4 distinct game seeds with no overlap, identical schema and engine
  revision, exact 256-visit targets, zero policy-mass error, and no `.partial` residue. Training
  reached depth 6 across 390,036 neural leaves; 14/20 training games had nonzero VP.
- Resumed `champion-iter1.pt` for ten epochs. Independent validation loss improved from 5.1206 to a
  best 3.9554 at epoch 6 / step 417, and policy KL improved from 1.6879 to 0.9863. Later epochs
  regressed, so best-validation restoration selected step 417 instead of final step 492.
- Candidate `puct-candidate-iter2.pt` has SHA-256
  `931B61D94A1285434A13EA471DA546E84703C5370ACDF189BB2486C0AC23EA50`.
- Corrected promotion statistics to use the seat-rotated seed group as the independent confidence
  unit. The 40-game / 20-seed, 64-visit gate gave the candidate 60% shared wins, mean score delta
  `+0.20`, mean VP margin `+2.525`, standard error `0.1556`, and a 95% lower bound of `-0.1257`.
  It did not pass the strict `lower bound > 0` gate, so champion iter1 remains deployed.
- A transient Docker-to-host connection refusal aborted the first long gate despite both services
  remaining healthy. Remote inference now retries bounded idempotent requests with exponential
  backoff and retains model-ID checks; the complete rerun then finished all 40 games.
- Five separate CUDA inference services did not scale self-play linearly because large JSON feature
  batches and CUDA-context scheduling dominated. A shared dynamic batcher or local ABI/shared-memory
  inference path is the next throughput improvement before substantially larger self-play runs.

## 2026-08-15 end-to-end browser regression and percentage consistency

- Re-ran the complete two-player browser flow at seed 424242: create game, start turn, run an
  800-visit neural PUCT analysis, inspect Top 1/2/3, switch to Top 2, use the comparison shortcut,
  ask a free-form Chinese Top-1-vs-Top-2 question, apply Top 1, and single-step AI playback into a
  freshly analyzed next position.
- The first analysis reached depth 4 with 800 neural leaves in 13 inference batches. Applying Top 1
  spent £3, built Road 33, and advanced exactly one turn. AI single-step then applied one move,
  incremented the observer counter to one, paused, and retained a new Top 3 for the following state.
- Desktop 1280x720 and mobile 390x844 checks found no document-wide horizontal overflow. The mobile
  hand remains isolated in its own horizontal scroller; the analysis controls, Top 3, metrics, and
  apply action remain readable without overlap. Browser warnings and errors were empty.
- Fixed a user-visible half-tie rounding mismatch where a 162/800 visit share appeared as 20.3% in
  the Svelte panel but 20.2% in the Rust-generated explanation. Rust now rounds percentage tenths
  explicitly, with a focused regression proving 162/800 -> 20.3%.
- Restarted the release backend and repeated a real Top-3 analysis plus Chinese comparison question;
  panel and explanation percentages matched. The standard web-game client produced
  `output/web-game-rounding-regression/shot-0.png` and matching `state-0.json` with no error artifact.
- Final reviewed regression passes 179 Rust all-target tests, 56 Rust/PyO3 tests, 30 Python
  training/evaluation tests, Python bytecode compilation, and the Svelte production build. Remaining
  output is limited to the existing Rust unused-item and Tailwind content-configuration warnings.

## 2026-08-16 standalone checkout migration verification

- Verified the standalone checkout at `D:\workspace\myrepo\brass-birmingham-daiyu-ai` is on
  `main` at `01f168609201bfec1431bac6fafdb6966753f820` and matches `origin/main` with a clean worktree.
- Verified the former nested checkout contained only an empty `.git` directory, then removed that
  empty directory and its empty `fast-brass` parent without recursive deletion. The standalone
  checkout and its `.git` directory remain intact.
- Confirmed ports 3000, 5173, and 8765 were not listening before service recreation. The two old
  stopped containers that reference the former path remain preserved pending a separate data audit.
- Recreated `fast-brass-dev` from `fast-brass-dev:training-20260815` with the standalone checkout
  bound at `/work` and the existing Cargo registry/git volumes attached.
- Restarted host CUDA inference from the standalone checkout. `/health` reports champion step 242,
  RTX 5080 CUDA inference, and model ID
  `sha256:8e7180489a004cbda2acad75d5fa724bb7018adcbd9afd6727c972367f9f636f`.
- Recreated the release Rust API container with the new checkout, persistent SQLite path under
  `output/`, and host inference URL; also restarted the Svelte dev server. Ports 3000, 5173, and
  8765 all respond successfully, and the API retained the migrated saved-game records.
- The standard web-game client created a fresh two-player game from the standalone checkout and
  produced a nonblank board screenshot plus matching `render_game_to_text` state with no error
  artifact. The in-app browser then exercised 800-visit neural PUCT, Top 1/2/3 selection, a
  comparison shortcut, a free-form Chinese follow-up, applying Top 1, and AI single-step playback.
- The first search covered 428 semantic actions, reached depth 4, and produced Top-3 visit shares
  of 26.4%, 21.9%, and 18.1%. The observer step applied exactly one move, paused at one recorded AI
  move, and refreshed a new Top 3 from a depth-6 search. Reviewed desktop screenshots showed no
  overlap, and browser console warnings/errors were empty.
- Standalone-path regression passed 179 Rust all-target tests, 56 Rust/PyO3 tests, 30 Python tests,
  Python bytecode compilation, and the Svelte production build. Inference, API, UI, and both new
  containers remained healthy after the test run.

## 2026-08-16 shared dynamic batching experiment

- Measured the existing locked CUDA service with four concurrent 64-position requests built from
  real iter2 training records: mean throughput was 608.54 positions/s.
- Implemented and tested a bounded shared dynamic batcher, including result splitting, capacity
  limits, failure isolation, and health metrics. It correctly merged up to four HTTP requests into
  one 256-position model call, but real throughput regressed to 521.09 positions/s with a 25ms idle
  / 100ms hard window and 547.64 positions/s with a fixed 25ms window.
- Smaller 16-position client batches also regressed when deliberately delayed for merging: 440.12
  positions/s versus 505.28 positions/s with zero intentional wait. Large JSON parsing can overlap
  with serialized GPU work; waiting to merge requests destroys that pipeline, and the larger model
  batch is not faster enough to recover the delay.
- Removed the dynamic batching implementation and restored the original inference service. The next
  throughput path is local Linux PyO3 plus CUDA Torch in one GPU container, eliminating HTTP/JSON
  transport rather than trying to batch after JSON parsing.

## 2026-08-16 local CUDA ABI and parallel self-play

- Built a Rust 1.88 / Python 3.11 / Torch 2.11 cu128 image and rebuilt the current PyO3 extension
  against its Linux ABI. A fixed one-game, 64-visit comparison selected the same action and visit
  vector at all 79 decisions through local CUDA and remote HTTP; policy priors and backed-up values
  also matched exactly. Local ABI generation took 34.445 seconds versus 100.578 seconds over HTTP,
  a 2.92x speedup before process-level parallelism.
- Added NumPy 2.3.5 in a layer after the multi-gigabyte Torch install, plus a Compose stack with the
  repository mount, persistent Cargo caches, RTX GPU reservation, a correct
  `host.docker.internal:host-gateway` mapping, and a health-checked champion inference service on
  port 8765. Linux Cargo output is isolated under `target/linux`.
- Benchmarked identical fixed-seed local games at 64 visits. One process completed 79 positions in
  29.656 seconds (2.66 positions/s); two completed 158 in 33.706 seconds (4.69 positions/s, 1.76x);
  four completed 316 in 35.308 seconds (8.95 positions/s, 3.36x). Peak reported GPU memory was
  5,232 / 8,137 / 9,555 MiB respectively, and four-process utilization peaked at 77%.
- Every benchmark shard matched the prior local ABI shard byte-for-byte and no `.partial` file was
  left behind. This rules out concurrency-induced changes in Rust search, model inference, action
  sampling, or serialization for the fixed workload.
- Added `game_index_offset` to deterministic export and a `training.parallel_self_play` command that
  partitions one global index range across spawned processes, preflights every output, preserves
  per-shard atomic writes, checks model identity, and reports aggregate throughput. A real two-worker
  CUDA smoke produced 158 positions in two distinct game seeds; `SelfPlayDataset` loaded both shards
  under one engine revision with no residue.
- The next data cycle should use four local workers, separate train/validation seeds, 256 visits for
  the first larger candidate, and the existing seat-rotated confidence-bound promotion gate. Four
  workers are evidence-backed on the current 16 GB RTX 5080; higher concurrency remains unproven.

## 2026-08-16 iter3 data cycle and first-version model freeze

- Generated 40 training games (3,160 positions) and eight independent validation games (632
  positions) through four local CUDA workers. Training and validation seeds do not overlap; every
  position has exactly 256 visits, all shards share engine revision
  `01f168609201bfec1431bac6fafdb6966753f820`, and no `.partial` file remains.
- Fixed resumed training so checkpoint optimizer moments are restored while explicitly requested
  learning rate and weight decay still take effect. Also changed `--max-steps` to count steps in the
  current invocation rather than comparing against the checkpoint's lifetime global step. New
  checkpoint metadata records the starting step, steps in the current run, and training settings.
- The full iter3 update reached step 467 (`c3ac4198ee0a670f1c6ef1ec3d08751d352aedf9dcea58af42e6e0e8279309fc`)
  but scored 35% shared wins, mean score delta `-0.30`, and paired 95% lower bound `-0.5673` over 40
  games. A smaller policy-focused step-427 update
  (`ee8d3cf63d5c165a50ebd28bfb4a983fc181384de61dd7d5fe656a681401283c`) scored 48.75%, `-0.025`,
  and `-0.3768`. Neither passed promotion.
- Reproducing the iter2 gate at the deployed inference batch size of 64 produced 65% shared wins,
  mean score delta `+0.30`, and lower bound `-0.0512`. Repeating with the historical batch size of 32
  produced 45%, `-0.10`, and `-0.4446`. Both failed the predeclared strict `lower bound > 0` rule;
  the variance reinforces that a point estimate alone is not promotion evidence.
- First version is frozen on champion step 242,
  `8e7180489a004cbda2acad75d5fa724bb7018adcbd9afd6727c972367f9f636f`. Compose remains pointed at
  `output/champion-iter1.pt`; no iter2 or iter3 candidate is promoted.
- Later work should improve data efficiency and evaluation power before another training cycle, then
  add a clearer human-versus-AI seat assignment flow. No additional training is required for the
  first-version release.

## 2026-08-16 first-version final regression

- Final verification passed all 40 Python training/evaluation tests on both Windows and the Linux
  GPU container, all 179 Rust all-target tests, all 56 Rust library tests with Python bindings,
  Python bytecode compilation, the Svelte production build, Compose validation, and
  `git diff --check`. Existing Rust unused-item warnings remain unchanged.
- Restored the Compose inference service without rebuilding it. `/health` reports CUDA, checkpoint
  step 242, and model ID
  `sha256:8e7180489a004cbda2acad75d5fa724bb7018adcbd9afd6727c972367f9f636f`.
  The Rust API and Svelte UI respond on ports 3000 and 5173, and the backend points to the inference
  service on port 8765.
- The standard web-game client created a fresh two-player game, produced a nonblank board screenshot
  and matching `awaiting_start` state, and emitted no console-error artifact.
- A seed-20260816 browser regression started player 1, ran an 800-visit neural PUCT analysis, selected
  Top 2, asked both the comparison shortcut and a free-form Chinese question, switched back to Top 1,
  and applied the exact recommendation. The first Top 3 visit shares were 25.0%, 23.0%, and 14.4%.
- The opponent-side single-step control analyzed the new position, applied exactly one move, paused
  at one recorded AI move, and retained a fresh Top 3 for the following state. The game log showed
  both applied semantic action keys, the current player returned to player 1, desktop width remained
  `clientWidth = scrollWidth = 1280`, and browser warnings/errors were empty.
- First-version follow-up: add explicit human/AI seat assignment so human-vs-AI play no longer relies
  on the player choosing which turns to operate manually. The documented analysis/apply loop is the
  supported first-version workflow.

## 2026-08-16 explicit human-vs-AI seat control

- Added explicit `人机对练`, `AI 观战`, and `全部手动` modes plus a Coade/Brunel human-seat
  selector. The default two-player setup is human-vs-AI with Coade assigned to the human.
- Added a server-side observer seat. Only that seat's hand is serialized; opponent action sessions,
  choice sets, available manual actions, pending development choices, and private discard labels in
  AI analysis/explanations are hidden when the observer is not the acting player.
- Human hands remain visible and non-interactive through the AI turn. Manual controls lock on AI
  turns, the analysis inspector opens automatically, and AI playback stops only after the complete
  opponent turn returns control to the human seat.
- Browser QA used seed `20260820` to switch both seats and all three modes, analyze a human turn with
  800 neural PUCT visits, inspect Top 1/2/3, ask a free-form Chinese comparison question, inspect a
  redacted opponent analysis, single-step one AI move, and autoplay a later two-action AI turn back
  to the human. Desktop 1280x720 and mobile 390x844 had no horizontal overflow or overlap.
- HTTP regression verified an observer on player 1 sees 8 player-1 cards and zero player-0 cards
  while player 0 acts, receives no private choice session, sees redacted analysis and explanation
  text, and regains legal manual actions when player 1 becomes current. Invalid observer indices are
  rejected.
- Final verification passed 182 Rust all-target tests, the Svelte production build, `git diff
  --check`, and the supplied web-game client. Its final nonblank screenshot and matching state are
  under `output/web-game-seat-control-final-3/`, with no console-error artifact. Switching an
  already-started AI turn to the acting human seat also restored the cached legal-action controls.
- First-version champion remains checkpoint step 242 with model ID
  `sha256:8e7180489a004cbda2acad75d5fa724bb7018adcbd9afd6727c972367f9f636f` on CUDA. It is an
  improvement over the smoke baseline only; calibration and strong-human/external evaluation are
  still required before any top-level-strength claim.

## 2026-08-16 replay history

- Added replay persistence events for analyzed positions and applied moves. Each recorded position
  stores the observer-visible board state, action metadata, selections, and the saved AI Top 3
  analysis when available.
- Added `POST /api/replay` and a replay viewer reachable from the history icon beside each saved
  game. The viewer supports timeline/slider navigation, previous/next position controls, board
  snapshots, and inspection of the saved first/second/third recommendations.
- New games record replay snapshots automatically. Pre-existing database rows remain compatible;
  rows without the new events show an explicit no-snapshot message instead of failing to load.
- Browser replay regression passed for test game `49` with two analyzed/applied moves and three
  saved candidates at each position. Desktop and 390px mobile screenshots rendered the board and
  timeline correctly; next-position navigation changed the position counter to `2/2`, mobile
  document width stayed equal to the viewport, and browser errors were empty.

## 2026-08-16 automatic staged analysis

- Added an `自动分析玩家回合` toggle to the analysis inspector. When enabled, analysis starts
  automatically at the beginning of a human-controlled `choosing_action` turn; AI-controlled turns
  remain under the AI playback controls and do not create duplicate requests.
- Analysis now resumes one neural PUCT tree through progress stages `100`, `400`, `800`, and the
  configured final budget. Each completed stage refreshes Top 3 recommendations, while only the
  final stage is persisted to replay. Revision and generation checks prevent a stale response from
  overwriting a changed position.
- Browser verification on port `5173` observed all four staged requests, ending at `3000/3000`, with
  `render_game_to_text()` reporting `auto_analysis: true` and completed progress `3000`.
- The live API was also verified at `100`, `400`, `800`, and `3000` completed simulations. The Top 3
  ranking changed at an intermediate stage, confirming that recommendations refresh during search.

## 2026-08-16 iter4 training and promotion gate

- Stopped the CUDA inference service and generated new self-play data from champion step 242 using
  PUCT search with 256 simulations, four determinizations, and inference batches of 64. Training
  used base seed `2026081603` for 40 games / 3,160 positions; independent validation used
  `2026081703` for eight games / 632 positions. All eight shards were atomic, had no `.partial`
  residue, and passed the dataset schema, engine revision, seed, and visit-count checks.
- Trained `output/puct-candidate-iter4.pt` from `champion-iter1.pt` for 10 epochs with learning rate
  `1e-4`, policy-focused shared-win and VP-margin weights `0.05/0.05`, and batch size 64. The
  validation-selected checkpoint is step 742 with loss `3.68366`, policy KL `1.01438`, and policy
  Top-1 `18.99%`. Candidate SHA-256 is
  `744cfc258d711a4197c70482dfb2f9bb7dd6df5837a77405e8a104c1b66bc4f9`.
- A fresh seat-rotated candidate-vs-champion gate used seed `2026081803`, 20 independent seed
  groups / 40 games, and 64-search-simulation PUCT agents. Candidate mean shared win was `0.625`,
  mean score delta `+0.25`, standard error `0.12301`, and two-sided 95% lower bound `-0.00746`.
  The strict `lower bound > 0` promotion rule correctly refused promotion; champion step 242
  remains deployed. The full report is `output/iter4-promotion-gate.json`.

## 2026-08-16 human-strategy prior iteration 5

- Added the auditable `human-strategy-v1` prior from the RulesPal rulebook, the BGG strategy guide,
  and the open-source `npow/brass-birmingham` implementation. The public search found strategy
  guidance and a digital game's action-log UI, but no complete machine-readable human replay corpus;
  the BGG page itself is protected by Cloudflare. Human score references are metadata only and are
  not fabricated supervised labels.
- Generated 40 training games / 3,160 positions and eight independent validation games / 632
  positions at 256 searches, four determinizations, and strategy-prior strength `0.7`. The resulting
  48-game corpus has mean player VP `31.85`, median `31.5`, maximum `98`, and zero of 96 player
  scores at or above 100. Selected roots still contain 1,224 network actions and 484 passes, so this
  iteration is not human-strength evidence and the prior is not yet strong enough to correct the
  low-quality value distribution.
- Candidate `output/puct-candidate-iter5-strategy.pt` (step 742, model ID
  `sha256:1e041b8188a67235dc19a96559d8a06130b676ebf7437f53db7773b89d354666`) passed the declared
  40-game seat-rotated gate against `champion-iter1.pt` and was deployed as
  `output/champion-iter2-strategy.pt`. This is an improvement over the previous self-play champion,
  not evidence of 100+ VP human strength.
- Full verification passed 182 Rust all-target tests, 44 Python `unittest` training tests, Python
  bytecode compilation, Svelte build, and `svelte-check`. Port 5173 browser QA verified automatic
  analysis and staged `100 -> 400 -> 800 -> 3000` updates; final 3,000-search analysis completed
  in about 21.7 seconds with model step 742 and no browser console errors observed.
- On user request, GPU self-play/training processes were stopped. The 8765 CUDA inference service
  remains healthy for gameplay. Before the next training cycle, add an explicit guarded strategy
  policy for early Canal actions (especially no own industry -> no unproductive network) and
  evaluate it with a small same-seed score/action-distribution experiment before spending another
  full data-generation cycle.

## 2026-08-28 map-aware strategy prior iteration 11

- Read the complete AI implementation in `npow/brass-birmingham` at commit
  `2b1da2d41036f2afaafcab320b2175ba3fd9f877`. It is an open-source heuristic AI, not human replay
  data. Its useful network signals are newly reachable locations, merchant access, diminishing
  exploration value, and endpoint VP; it must not be represented as human training data.
- Added opt-in `human-strategy-v5-map-aware-lifecycle` while preserving v1-v4 and the default v3.
  V5 decodes the real road topology and merchant tiles from the observer state, simulates each
  candidate road (including double rail), measures shortest remaining phase-legal routes to a
  matching merchant, detects newly sellable industries with legal beer sources, and credits endpoint
  road VP plus useful industry-card access. Its audit ID is `network-route-v1`.
- Focused tests prove v4 still assigns equal scores to otherwise identical roads, while v5 strongly
  prefers a road that completes the correct merchant route and the first link on a shortest
  multi-link route. The complete container Python suite passes 67 tests.
- Same checkpoint, seed, 64-search, stable-argmax comparison: v4 strength 1 scored mean `4.75`,
  median `0`, 12/16 zero players, 0 players at 50+, 11 Sell actions, and 168 Canal Network actions.
  V5 strength 1 scored mean `36.875`, median `36.5`, 1/16 zero players, 7 players at 50+, 48 Sell,
  and 67 Canal Network actions. The exact first game improved from `0-0` to `37-52`.
- V5 strength 0.7 at 64 search further improved mean/median to `41.6875/45.0`, with one zero player,
  six players at 50+, and 47 Sell actions. This beat pure v5 on mean and median while using fewer
  loans, so strength 0.7 is the selected teacher blend.
- An eight-game 256-search pilot with the same seeds scored mean `54.5625`, median `52`, minimum
  `20`, maximum `88`, zero zero-score players, eight players at 50+, and 62 Sell actions. The same
  64-search games scored mean `41.6875`, median `45`, and minimum `0`, so 256 searches is selected
  for iter11 teacher generation. Report: `output/iter11-v5blend07-256-pilot-analysis.json`.
- Next: generate 40 independent training games plus eight validation games at the locked teacher
  settings, train a candidate without changing the deployed champion, then require both held-out
  quality and the existing 40-game seat-rotated promotion gate before deployment.

## 2026-08-28 map-aware strategy prior iteration 11 training and gate

- Generated 40 training games / 3,160 positions at base seed `2026082801` and eight independent
  validation games / 632 positions at base seed `2026082901`. The locked teacher used 256 searches,
  four determinizations, stable argmax, and v5 prior strength `0.7`. Training players averaged
  `50.125` VP with a maximum of `115`; validation players averaged `53.375` VP with a maximum of
  `80`. Quality-50 filtering retained 1,543 training and 473 validation positions.
- Full-model fine-tuning from the actor-VP-head-only iter8b checkpoint now deliberately restarts the
  optimizer while preserving model weights and the lifetime global step. The full-data candidate
  `output/puct-candidate-iter11-v5-full.pt` reached step `2558` with SHA-256
  `aa0beda57b7c38c8026901696c4c7f7ee7c7c2e1fd1b829d666b5c84deea8e2a`; held-out policy KL
  improved from `1.46221` to `1.24610` and Top-1 accuracy from `12.03%` to `16.77%`.
- The quality-filtered candidate `output/puct-candidate-iter11-v5-quality50.pt` reached step `2258`
  with SHA-256 `b06228bd3f1fbb674b230fab04816c4ad21e687a55e68b4780c56bcbfa4705f9`;
  held-out policy KL improved from `1.58510` to `1.43916` and Top-1 accuracy from `12.68%` to
  `15.01%`. Both full-model runs correctly recorded `optimizer_state_restored: false`.
- A fresh same-seed eight-game blind comparison used base seed `2026083101`, 256 searches, and the
  locked v5 teacher settings. The full-data candidate scored mean/median `66.8125/64.5`, minimum
  `34`, maximum `103`, zero zero-score players, 12/16 players at 50+, 68 Sell actions, and 87 Canal
  Network actions. The quality-filtered candidate scored `53.375/53`, minimum `0`, maximum `84`,
  one zero-score player, 10/16 at 50+, 50 Sell actions, and 79 Canal Network actions. The full-data
  candidate therefore advanced to the promotion gate. Reports are
  `output/iter11-fullcandidate-v5blend07-256-ab-analysis.json` and
  `output/iter11-quality50candidate-v5blend07-256-ab-analysis.json`.
- The predeclared 40-game / 20-seed seat-rotated gate at 64 PUCT searches gave the full-data
  candidate 55% shared wins, mean score delta `+0.10`, mean VP margin `+3.775`, standard error
  `0.12354`, and a two-sided 95% lower bound of `-0.15858`. Its absolute mean score was `50.925`
  versus champion `47.15`. The strict `lower bound > 0` rule refused promotion, so
  `output/champion-iter2-strategy.pt` remains deployed. Full report:
  `output/iter11-v5-full-promotion-gate.json`.
- Final verification passed 68 Python training/evaluation tests and 186 Rust all-target tests. The
  CUDA inference service was restored and reports champion step `742`, model ID
  `sha256:1e041b8188a67235dc19a96559d8a06130b676ebf7437f53db7773b89d354666`.

## 2026-08-28 iterations 12-14 policy/value isolation

- Iter12 tested larger policy models and a zero-initialized residual policy adapter without changing
  the deployed champion. The strongest adapter checkpoint,
  `output/puct-candidate-iter12-champion-value-adapter128.pt`, has step `1542` and SHA-256
  `7c215e47af9f67df350a061d4504b5a386a013dae71e4ba4a2611a1a75bf4e5f`. Its held-out policy KL
  was `0.8253` with Top-1 accuracy `26.90%`, but its four-game absolute-score pilot averaged only
  `25.75` VP. An eight-game no-prior pilot against the champion finished `4/8`, mean VP margin
  `+2.0`, and seed-group 95% lower bound `-1.837`; it was rejected before the formal gate.
- Added independent policy/value tower fusion in `training/model.py` plus
  `training/fuse_policy_value.py`. Iter13 fused the iter11 policy with the deployed champion value
  heads as `output/puct-candidate-iter13-iter11-policy-champion-value.pt` (SHA-256
  `7558a48103254e2b29a9506c999ce41504630d7c1859c1a8a145988b4945f1ab`). Exact tensor and
  inference checks proved the policy matched iter11 and the value outputs matched the champion.
  Its same-seed 256-search score averaged `65.81` VP, below iter11's `66.81`, and included a
  zero-score trajectory, so it was rejected.
- Added `--value-tower-only` training, which freezes every policy tensor and updates only the
  independent value encoder and three value heads. Focused regression coverage passed `23/23` tests.
  Iter14 generated 40 independent training games / 3,160 positions at seed `2026090601` and eight
  validation games / 632 positions at seed `2026090701`, using the iter11 teacher with 256 searches,
  four determinizations, stable argmax, and v5 prior strength `0.7`. Train/validation seed sets are
  disjoint, all shards share one engine revision, and no iter14 `.partial` file remains.
- Iter14 selected epoch 2 of 24 and saved
  `output/puct-candidate-iter14-value-tower.pt` at step `2708`, SHA-256
  `5fd2443029fccd7258cf726cec4f44d0f25dfa93598a115d57f77b89bcac3335`. All policy tensors
  remained byte-identical while 18 value tensors changed. Held-out shared-win BCE improved from
  `0.75749` to `0.64584`, VP-margin Huber from `0.03654` to `0.02872`, and actor-VP Huber from
  `0.04039` to `0.01250`; the last epoch had already overfit to BCE `1.00549`, but checkpoint
  rollback correctly retained epoch 2.
- Iter14's four-game 64-search pilot remained poor at mean/median `22.75/20`, with three zero-score
  players. The decisive same-seed eight-game 256-search pilot recovered to mean/median
  `63.4375/71`, range `28..80`, zero zero-score players, and 13/16 players at 50+, but remained below
  iter11's `66.8125` mean and `103` maximum. Report:
  `output/iter14-value-v5blend07-256-ab-analysis.json`.
- The eight-game no-prior direct pilot against the deployed champion improved to `6/8`, mean VP
  margin `+15.125`, and mean score delta `+0.5`, but its four independent seed groups still gave a
  95% lower bound of `-1.091`. Because it did not exceed iter11's absolute 256-search quality and
  did not establish a positive confidence bound, iter14 was rejected without spending a formal
  40-game gate. Full report: `output/iter14-value-tower-pilot-gate.json`.
- No iter12, iter13, or iter14 candidate was promoted. The online checkpoint remains
  `output/champion-iter2-strategy.pt`, step `742`, model ID
  `sha256:1e041b8188a67235dc19a96559d8a06130b676ebf7437f53db7773b89d354666`.

## Next training session

- Use the implemented validation early stopping for all multi-epoch candidates so value-only runs do
  not spend their full requested budget after sustained overfitting.
- Diagnose why lower held-out value loss improves direct no-prior play but reduces the locked
  256-search absolute-score benchmark. Keep both tests as independent candidate gates.
- Do not run a formal 40-game gate or replace the deployed champion unless a candidate first exceeds
  iter11's `66.8125` same-seed 256-search mean without zero scores and then clearly improves the
  eight-game direct pilot.
- Public-data research still has not found complete machine-readable human Brass: Birmingham action
  logs. `npow/brass-birmingham` is an open-source heuristic opponent, not human training data; do not
  label it as such.

## 2026-08-28 training pause and online runtime

- Paused at the user's request after iter14 evaluation. No self-play, training, or evaluation worker
  remains active; all iter14 checkpoints, JSONL shards, and reports are complete. The unrelated old
  `output/model-self-play-strategy-smoke.jsonl.partial` residue predates iter14 and was preserved.
- Restored the strict-gate champion inference service on CUDA. `/health` reports checkpoint step
  `742` and model ID
  `sha256:1e041b8188a67235dc19a96559d8a06130b676ebf7437f53db7773b89d354666`.
- Ports 3000 and 5173 were occupied by unrelated user projects, so those processes were left intact.
  Added an optional `FAST_BRASS_API_URL` Vite proxy override and a reusable local Nginx config, then
  published the Brass API on port 3010 and the current static UI on port 5174. Both runtime containers
  use `restart: unless-stopped`; the API connects to `http://inference:8765/evaluate` on the Compose
  network.
- Rebuilt the current Svelte UI and release Rust API successfully. Browser smoke at
  `http://127.0.0.1:5174` rendered the Brass setup screen, returned matching
  `render_game_to_text()` setup state, and emitted no console-error artifact. Final screenshot and
  state: `output/web-game-online-iter14-2/`.

## 2026-08-28 compute discipline and validation early stopping

- Added opt-in `--early-stopping-patience` and `--early-stopping-min-delta` to `training.train`.
  Early stopping requires independent validation shards. The exact lowest validation-loss state is
  still selected even when an improvement is smaller than `min_delta`; only patience reset uses the
  significant-improvement threshold.
- Training now emits a structured `training_early_stopped` event and records stop reason, epochs run,
  patience state, reference loss, and trigger status in checkpoint metadata. Invalid negative values
  and a min-delta without enabled patience are rejected before data loading.
- Added deterministic regression coverage proving that a requested 10-epoch run stops after three
  epochs under the configured synthetic validation sequence while retaining the exact best epoch.
  All 75 training tests passed before the final boundary test; the focused partition suite then
  passed `15/15`. Python bytecode compilation and CLI option inspection also passed.
- Added `docs/training-methodology.md`, which records measured throughput decisions, failed model
  routes, the six-stage compute funnel, hard promotion gates, top-human evidence requirements,
  three-failure hypothesis timeboxing, and the reusable contract for other hidden-information board
  games. README training examples now enable the initial patience/min-delta settings and link the
  methodology.
- This work used CPU-only test models. No self-play or CUDA training restarted, and the online
  champion inference/API/UI services remained available.

## 2026-08-28 reference heuristic and V6 resource-aware training

- The open-source `npow/brass-birmingham` heuristic was evaluated against the deployed champion
  for 16 seat-rotated games. The champion won 10, the heuristic won 5, and one game tied; the
  heuristic's mean score delta was `-0.3125` and mean VP margin was `-8.94`. It is a useful
  rules-based reference, not stronger training data or a human expert corpus.
- Added the independent `human-strategy-v6-resource-aware` prior. It prices coal and iron from
  legal choices, counts beer supply units, penalizes excess coal production, prepares Canal II+
  cotton/goods, scores Rail double links by cash/beer/route value, and gives targeted Develop,
  Loan, Scout, and card/industry/location bonuses. The V6 prior beat V5 in a paired 8-game
  256-search screen (`5-3`, mean VP margin `+12.25`) but its 95% score-delta lower bound was
  `-0.5455`, so that prior change was not promoted by itself.
- Generated V6 training data with 40 games / 3,160 positions and independent validation data with
  eight games / 632 positions. The training and validation means were `68.7875` and `71.625` VP;
  validation had no zero-score player. All 84 Python training tests passed.
- Full-model training from the champion at the default `3e-4` learning rate overfit immediately:
  validation loss rose from `4.75955` to `4.89234`, then `5.14087` and `5.16750`. Early stopping
  saved `output/puct-candidate-iter16-v6.pt` at the unchanged champion tensors (step `742`).
- A conservative `3e-5` learning-rate run selected epoch 2 / step `892`, reducing validation loss
  slightly to `4.75760`. Its locked V6 256-search quality screen passed at `73.25` mean VP with
  no zero-score players; report: `output/iter16-v6-lr3e-5-quality-analysis.json`.
- The same candidate then failed the no-prior, seat-rotated 8-game direct pilot: mean score delta
  `-0.5`, mean VP margin `-17.25`, and 95% lower bound `-1.4186`. The candidate was rejected before
  the formal 40-game gate. Reports: `output/iter16-v6-lr3e-5-direct-gate.json` and
  `output/iter16-v6-candidate-direct-gate.json` for the unchanged-weight control.
- `output/champion-iter2-strategy.pt` remains the deployed checkpoint at step `742`; inference was
  restored on CUDA and its health endpoint reports the original model ID. No candidate was promoted.

## 2026-08-28 V7 action-efficiency screen

- Added the unpromoted `human-strategy-v7-action-efficiency-route` prior on top of V6. It uses
  bounded tile-level VP/income/road output efficiency, public opponent resource supply, route
  concentration from the current hand and buildings, and marginal Develop/Network value. The
  strategy metadata now records the same web references used for the action-efficiency principles.
- The V7 prior passed its focused unit tests and the full Python suite (`86/86`). A fixed-seed,
  seat-rotated 8-game comparison against the same checkpoint using V6 at 256-search produced
  `2-6`, mean score delta `-0.5`, mean VP margin `-4.375`, and 95% lower bound `-2.091`; it was
  rejected. A two-game low-strength `0.25` diagnostic was also negative (`-37` and `-52` VP
  margins), so V7 was not used for training data or promotion.
- The V7 source remains available for future analysis, but V6 remains the best tested strategy
  prior and `output/champion-iter2-strategy.pt` remains the active checkpoint. No model, service,
  or historical self-play shard was replaced.

## 2026-08-28 iter17 mixed-data promotion

- Added `training/diagnose_strategy_match.py` and traced the same checkpoint under V6 and V7 on
  identical seeds. V7's selected action was the static-prior top action only 13/158 times in the
  trace, and several early network/develop choices had very low prior rank; this supported
  reducing the web-derived signal instead of adding more hand-tuned rules.
- Added the unpromoted V8 residual prior, which applies only 25% of the V7 score delta on top of
  V6. Its focused test passed, but the four-game 64-search screen was `2-2`, with mean score delta
  `0` and mean VP margin `-5.5`; V8 was rejected before training data generation.
- Mixed the existing independent V5 and V6 teacher shards 1:1: 80 training games / 6,320
  positions and 16 validation games / 1,264 positions. Low-learning-rate fine-tuning from the
  deployed champion selected step `1336`; validation loss fell from `4.804535` to `4.525704`,
  and policy KL fell to `1.395003`.
- The mixed candidate's locked 256-search absolute-quality screen averaged `80.3125 VP` across
  16 player trajectories, with median `85.5`, minimum `39`, and zero zero-score trajectories.
  Its eight-game no-prior direct pilot was `6/8`, mean score delta `+0.25`, mean VP margin
  `+8.5`, with a seed-group 95% lower bound of `-0.5455`.
- The predeclared 40-game / 20-seed-group no-prior formal gate passed: 70% first-place rate,
  mean score delta `+0.375`, mean VP margin `+22.475`, standard error `0.139901`, and strict
  95% lower bound `+0.082187`. The candidate was promoted as
  `output/champion-iter3-mixed-v5-v6.pt` at step `1336`, SHA-256
  `f2c71b677fff61e984ca2ef9ff398f98dbded59ccc024d9ed3cb43496e45c83c`.
- The inference service was recreated from the new checkpoint and health-checked on CUDA with
  feature version `1`, state dimension `2111`, and action dimension `1615`. The previous
  `output/champion-iter2-strategy.pt` remains available for rollback. Reports and quality shards
  are `output/iter17-mixed-v6-quality-analysis.json`,
  `output/iter17-mixed-no-prior-direct-gate.json`, and
  `output/iter17-mixed-formal-gate.json`.

## 2026-08-28 iter18 target and training follow-up

- Added explicit `policy_target_field` selection for training, including raw
  `search_policy_target` targets, and covered both target paths in the training pipeline tests.
  The complete Python suite passed (`87/87`) and `python -m compileall -q training` passed.
- A raw-search candidate trained for one epoch from the deployed champion reached step `1485`;
  validation loss improved from `4.468162` to `4.444427`, but its locked V6 256-search quality
  screen averaged only `76.1875 VP`, with minimum `0` and one zero-score trajectory. It was
  rejected before direct play. The checkpoint is `output/puct-candidate-iter18-raw-search.pt`.
- Compared the fixed deterministic port of `npow/brass-birmingham@2b1da2d` with the deployed
  champion for 8 seat-rotated 64-search games. The open-source heuristic had `25%` first-place
  rate, mean score delta `-0.625`, and mean VP margin `-21.875`; its action mix used much more
  `develop` and less `network`/`sell`, so it was not used as a teacher. Report:
  `output/reference-vs-champion-64.json`.
- A four-epoch V6-target candidate from the same 12-shard iter15/iter16/iter18 data reached step
  `1932`; validation loss reached `4.523008`. Its quality screen averaged `95.0 VP`, minimum
  `69`, and zero zero-score trajectories, but the 8-game no-prior 64-search direct pilot was
  neutral: `50%` first-place rate, mean score delta `0`, mean VP margin `+2.875`, and 95% lower
  bound `-1.299046`. It was retained as an unpromoted experiment at
  `output/puct-candidate-iter18-guided-4e.pt`; no formal gate was run.
- The inference service remains on `output/champion-iter3-mixed-v5-v6.pt`, model ID
  `f2c71b677fff61e984ca2ef9ff398f98dbded59ccc024d9ed3cb43496e45c83c`, and health-checked on
  CUDA after the experiments. Raw and guided candidates, reports, and all historical shards are
  preserved for the next training cycle.

## 2026-08-28 iter19 raw-target mix screen

- Trained a 20% raw `search_policy_target` plus 80% V6 `policy_target` candidate from the
  deployed champion using all 12 training and 12 validation shards, four epochs, and learning
  rate `1e-5`. The checkpoint reached step `1932`; validation loss was `4.506783` from a
  `4.543204` baseline. Its locked V6/256 quality screen averaged `88.125 VP`, median `89`,
  minimum `56`, and zero zero-score trajectories. The no-prior 8-game direct pilot was neutral:
  mean score delta `0`, mean VP margin `+8.125`, and 95% lower bound `-1.299046`, so it was
  rejected before the formal gate. Artifacts are `output/puct-candidate-iter19-mix20-v6.pt`,
  `output/iter19-mix20-v6-quality-analysis.json`, and
  `output/iter19-mix20-no-prior-direct.json`.
- Tested the second and final cheap variant, 10% raw plus 90% V6 target, with the same complete
  data partition and training settings. The valid full-validation checkpoint reached step
  `1932`; validation loss was `4.513585` from `4.552584`. Its quality screen averaged `94.875
  VP`, median `96.5`, minimum `74`, and zero zero-score trajectories. The no-prior direct pilot
  again had a neutral mean score delta `0`, mean VP margin `+5.875`, and 95% lower bound
  `-1.299046`, so it was rejected before promotion. Artifacts are
  `output/puct-candidate-iter19-mix10-v6-fullval.pt`,
  `output/iter19-mix10-v6-quality-analysis.json`, and
  `output/iter19-mix10-no-prior-direct.json`.
- An interrupted first 10% run continued after the client signal and wrote
  `output/puct-candidate-iter19-mix10-v6.pt` with only `1,738` validation positions. It is kept
  as an invalid audit artifact and was not evaluated; the complete rerun was saved under the
  `-fullval` name. Both target-mix variants were rejected by the direct-strength gate, closing
  this hypothesis family. The deployed champion and inference service remain unchanged at step
  `1336`; health reports model ID
  `f2c71b677fff61e984ca2ef9ff398f98dbded59ccc024d9ed3cb43496e45c83c` on CUDA.

## 2026-08-28 iter20 remaining-VP target

- Added the `remaining_vp` actor value target. Training now learns final victory points minus the
  observer's current victory points, while inference adds the current score back before search
  consumes the absolute final-VP value. The default `absolute_final_vp` path remains compatible;
  focused target-conversion and checkpoint tests cover both modes.
- Trained `output/puct-candidate-iter20-remaining-vp.pt` from the deployed champion with `9,480`
  training positions, `1,896` independent validation positions, eight epochs, and learning rate
  `1e-5`. The candidate reached step `2528`, model ID
  `sha256:270d8ce5c6f856d176caec0fd1f8377741f43f6a0c97d7e1656ba62ff03f18af`; total validation
  loss fell from `4.562906` to `4.503511`, and remaining-VP Huber loss fell from `0.021670` to
  `0.018095`.
- The locked V6 256-search quality screen averaged `82.125 VP` across eight games, with median
  `79`, minimum `34`, zero zero-score trajectories, no network-before-first-build cases, and no
  repeat-loan cases. The no-prior 64-search direct pilot placed the candidate first in `6/8`
  trajectories, with mean score delta `+0.5`, mean VP margin `+9.375`, and seed-group 95% lower
  bound `-0.418564`.
- The predeclared no-prior formal gate used 40 games, 20 independent seed groups, 64 searches,
  four determinizations, and seat rotation. The candidate reached `60%` first place, mean score
  delta `+0.2`, mean VP margin `-2.475`, standard error `0.116980`, and strict 95% lower bound
  `-0.044838`. Because the lower bound was not strictly above zero, promotion was rejected;
  report: `output/iter20-remaining-formal-gate.json`.
- The deployed checkpoint was left unchanged. The inference service was restored and health-checked
  on CUDA with step `1336`, model ID
  `sha256:f2c71b677fff61e984ca2ef9ff398f98dbded59ccc024d9ed3cb43496e45c83c`, feature version `1`,
  state dimension `2111`, and action dimension `1615`. Quality and pilot reports are
  `output/iter20-remaining-v6-quality-analysis.json` and
  `output/iter20-remaining-no-prior-direct.json`.
- Final verification passed the Python training suite (`89/89`), default Rust all-target tests,
  Rust library tests with `python-bindings` (`61/61`), Python bytecode compilation, and
  `git diff --check`. The bindings test required the temporary `libpython3.11-dev` package in the
  running training container because the checked-in image contains only the Python runtime; no
  Dockerfile or repository source change was made for that environment setup.

## 2026-08-28 iter21 V6 policy adapter isolation

- Tested a policy-only residual adapter against the deployed champion using the independent
  `iter18-champion-v6-256` teacher/validation shards. The base policy, shared trunk, and all value
  heads were frozen; only the zero-initialized adapter was trainable. This isolates the new V6
  teacher policy from the full-model regressions seen in iter20.
- Adapter width `32` reached step `1436`; validation loss moved only from `4.634486` to `4.633695`.
  Its locked V6/256 quality screen averaged `73.625 VP`, median `81.5`, and minimum `0`, with one
  zero-score trajectory, so it was rejected before direct play. Artifact:
  `output/puct-candidate-iter21-v6-policy-adapter32.pt`; quality report:
  `output/iter21-v6-policy-adapter32-quality-analysis.json`.
- The conservative width `8` variant also reached step `1436`; validation loss moved from
  `4.634486` to `4.634404`. Its quality screen averaged `76.6875 VP`, median `76.5`, minimum
  `36`, and zero zero-score trajectories; network-before-first-build and repeat-loan counts were
  both zero. It passed quality but was neutral in the 8-game, four-seed-group no-prior direct
  pilot: `62.5%` first place, mean score delta `0`, mean VP margin `0`, standard error `0`, and
  95% lower bound `0`. It was rejected before the formal gate. Artifacts are
  `output/puct-candidate-iter21-v6-policy-adapter8.pt`,
  `output/iter21-v6-policy-adapter8-quality-analysis.json`, and
  `output/iter21-v6-policy-adapter8-no-prior-direct.json`.
- Neither adapter was promoted. The CUDA inference service was restored on the unchanged
  `output/champion-iter3-mixed-v5-v6.pt` at step `1336`, model ID
  `sha256:f2c71b677fff61e984ca2ef9ff398f98dbded59ccc024d9ed3cb43496e45c83c`; health reports
  feature version `1`, state dimension `2111`, and action dimension `1615`.

## 2026-08-29 iter22 recovered V6 512 cycle

- Recovered the four interrupted V6/512 training prefixes instead of regenerating self-play. Each
  prefix contained six complete games and 474 positions; together they provided 24 games / 1,896
  positions, mean player VP `83.3125`, no zero-score player, no network-before-first-build case,
  and no repeat loan. `SelfPlayDataset` decoded every position successfully. The prefixes had
  distinct game seeds, the same feature schema and engine revision as the independent validation
  set, and no train/validation seed overlap.
- Fine-tuned the deployed champion with those 24 games plus the existing 40-game V6/256 training
  set: 5,056 training positions, 632 independent validation positions, learning rate `1e-5`, and
  four epochs. The selected checkpoint is `output/puct-candidate-iter22-v6-512-recovered.pt` at
  step `1652`, SHA-256
  `b6749474c448f6fdd01e3d1688251c5b910710da7b2461fec3afb29069bb8479`; validation loss fell from
  `4.634486` to `4.580637`.
- The locked V6/256 absolute-quality screen averaged `72.9375 VP`, with median `75`, minimum `3`,
  and no zero-score player. Network-before-first-build and repeat-loan counts were both zero.
  The eight-game no-prior, 64-search direct pilot placed the candidate first in `6/8` games, with
  mean score delta `+0.5` and mean VP margin `+14.625`; its 95% lower bound was `-0.418564`.
- The predeclared 20-seed / 40-game no-prior formal gate reached `55%` first place, mean score
  delta `+0.1`, mean VP margin `+1.625`, standard error `0.1`, and strict 95% lower bound
  `-0.1093`. Promotion was rejected; the deployed champion remains unchanged. Reports are
  `output/iter22-v6-512-recovered-quality-analysis.json`,
  `output/iter22-v6-512-recovered-no-prior-direct.json`, and
  `output/iter22-v6-512-recovered-formal-gate.json`.

## 2026-08-29 iter22 V5-anchored follow-up

- To test whether the recovered V6/512 candidate had forgotten useful V5 behavior, retained the
  original 40-game V5 and 40-game V6 training shards and added the 24 recovered V6/512 games. The
  resulting 104-game / 8,216-position training set and 16-game / 1,264-position validation set
  were fully decoded with disjoint game seeds.
- Low-learning-rate fine-tuning from the deployed champion selected
  `output/puct-candidate-iter22-v6-512-anchored.pt` at step `1852`, SHA-256
  `f697c766feab8d9fdb2279ad35e319eb840c573db19f9653f0c2b10330a4f3e7`. Validation loss fell from
  `4.525704` to `4.476158`, with policy KL `1.368131`.
- The locked V6/256 quality screen improved to mean `83.5 VP`, median `88.5`, minimum `44`, with
  no zero-score player, no network-before-first-build case, and no repeat loan. However, the
  eight-game no-prior, 64-search direct pilot was only `5/8` first place, mean score delta `+0.25`,
  and mean VP margin `+14.75`, so it did not clearly exceed the existing direct frontier and did
  not receive a formal 40-game gate.
- The deployed checkpoint remains `output/champion-iter3-mixed-v5-v6.pt` at step `1336`. Quality
  and pilot reports are `output/iter22-v6-512-anchored-quality-analysis.json` and
  `output/iter22-v6-512-anchored-no-prior-direct.json`.

## 2026-08-29 iter23 policy-target sharpening screen

- Added an opt-in `--policy-target-exponent` training parameter. The loader first normalizes the
  selected strategy/search target, raises it to the requested positive exponent, and renormalizes;
  the default `1.0` is unchanged. Checkpoint metadata records the exponent, and resuming with a
  changed policy-target configuration deliberately starts a fresh optimizer state. The full Python
  training suite passed (`92/92`), along with bytecode compilation and `git diff --check`.
- Tested the first V6/256-only sharpening candidate at exponent `2.0` from the deployed champion,
  using the existing four V6 training shards and four independent validation shards. It reached
  step `1536`; the transformed-target validation loss fell from `4.433221` to `4.380940`. The
  locked V6/256 quality screen produced mean/median `71.3125/73`, maximum `115`, minimum `0`, and
  one zero-score trajectory. Report: `output/iter23-policy-sharp2-quality-analysis.json`.
- Tested the predeclared softer exponent `1.5` with the same data, seed, and training settings. It
  reached step `1536`; transformed-target validation loss fell from `4.507955` to `4.458177`. The
  same-seed quality screen produced mean/median `70.4375/74.5`, maximum `111`, minimum `0`, and
  one zero-score trajectory. Report: `output/iter23-policy-sharp15-quality-analysis.json`.
- Both variants preserved legal lifecycle counts (`network_before_first_build=0` and
  `repeat_loans=0`) but failed the absolute-quality no-zero requirement. The sharpening hypothesis
  is closed after its two cheap variants; no direct or formal gate was run. Both checkpoints and
  all quality shards are retained as audit artifacts, while the deployed
  `output/champion-iter3-mixed-v5-v6.pt` remains unchanged at step `1336` and the CUDA inference
  service remains online.

## 2026-08-29 iter25 final-VP utility search pilot

- Completed the actor final-VP plumbing for successor action values and neural PUCT. Optional
  action-VP inputs remain backward-compatible, terminal successors carry exact root actor VP, and
  the bounded relative final-VP utility remains opt-in with default weight `0.0`. The rebuilt
  training-container extension was checked at schema version `1`; the deployed inference service
  was not restarted.
- Rust formatting and `git diff --check` passed. The Python suite passed `94/94`, and
  `cargo test --all-targets --features python-bindings` passed all library, binding, integration,
  and training-feature tests. Existing compiler warnings remain unrelated to this change.
- The four-game screening run used the iter24 candidate
  `output/puct-candidate-iter24-phase-remaining-vp.pt` against the unchanged champion, 64 searches,
  four determinizations, inference batch `64`, no strategy prior, and final-VP utility weight
  `0.15` for both agents. At base seed `2026085201`, it produced `3/4` candidate wins, mean score
  delta `+0.5`, and mean VP margin `+32.75`; candidate VP was `54/107/68/63` with no candidate
  zero score. Report: `output/iter25-final-vp-utility015-small-direct.json`.
- The predeclared eight-game direct confirmation used a new base seed `2026085301` with the same
  settings and seat rotation. The candidate reached `6/8` first place, mean score delta `+0.5`,
  mean VP margin `+8.75`, and four-seed-group 95% lower bound `-0.418564`; candidate VP was
  `47/49/52/63/70/57/90/62`, with no candidate zero score. The result is a healthy but
  inconclusive screening signal, so no formal 40-game gate was run and no checkpoint was promoted.
  Report: `output/iter25-final-vp-utility015-direct.json`.
- Replayed the same eight-game seed groups with final-VP utility weight `0.0` for both agents as
  the paired control. Every game result and selected action matched the `0.15` run; only the
  report metadata differed. The utility hook therefore had no behavioral effect at 64 searches
  under the current uncertainty gate, and this search-weight hypothesis is closed.
  Report: `output/iter25-final-vp-utility000-control.json`.
- The deployed checkpoint remains `output/champion-iter3-mixed-v5-v6.pt` at step `1336`, with the
  inference service still healthy on CUDA. The `0.15` utility weight is retained only as an
  unpromoted experiment; normal/default behavior remains weight `0.0` until a new independent
  hypothesis clears the quality and confidence gates.

## 2026-08-29 iter26 continuous final-VP quality weighting

- Added an opt-in `--final-vp-quality-weight` training loss coefficient. It gives positions from
  higher-final-VP trajectories a bounded, normalized weight while retaining every trajectory;
  the default `0.0` is unchanged, values are capped at `2.0`, and the checkpoint metadata records
  the setting. The option deliberately requires absolute final-VP targets so remaining-VP labels
  cannot be mistaken for trajectory quality. Focused training pipeline and partition tests pass.
- Trained the first variant at quality weight `1.0` from the deployed champion using the existing
  80-game V5+V6 training set and 16 independent validation games, learning rate `1e-5`, and eight
  epochs. The candidate is `output/puct-candidate-iter26-final-vp-quality1.pt` at step `2128`,
  model ID `sha256:d2ac288aadf9880081237820e028d96b936998a4453a9c28d8334af42d87d52d`. On an
  unweighted validation readout, policy CE improved from `3.908994` to `3.869094` and Top-1 from
  `13.8281%` to `14.6875%`, while actor-VP error worsened.
- The four-game V6/256 quality screen averaged `70.375 VP`, median `77.5`, minimum `25`, and
  zero zero-score trajectories. The independent eight-game confirmation averaged `87.625 VP`,
  median `81.5`, minimum `65`, and zero zero-score trajectories, with no network-before-first-
  build or repeat-loan violations.
- The no-prior 64-search direct pilot used four independent seed groups and seat rotation. The
  candidate reached `6/8` first place and mean score delta `+0.5`, but mean VP margin was
  `-4.625` with 95% lower bound `-0.418564`; it did not improve the direct frontier and was
  rejected before a formal gate. Reports are `output/iter26-final-vp-quality1-screen-analysis.json`,
  `output/iter26-final-vp-quality1-quality-confirm-analysis.json`, and
  `output/iter26-final-vp-quality1-direct.json`. The deployed champion and inference service
  remain unchanged.

## 2026-08-29 three-player human-versus-AI browser flow

- Enabled the existing `human-vs-ai` control mode for 3-player and 4-player browser games and
  made it the default after starting a multi-player game. Multi-player analysis and self-play now
  use the batched neural PUCT path for 2-4 players; the older root successor-value batch API remains
  intentionally two-player-only and is not used by the multi-player neural search path.
- Built the Svelte UI and verified a live 3-player game on port `5174`: the human seat reached its
  turn after one AI move, the AI control panel paused automatically on the human seat, and the
  AI move appeared in the UI log. Game `#6` contained one replay position and ten persisted events
  in `games.sqlite3`, confirming that later human actions use the same replay log.

## 2026-08-29 multiplayer neural PUCT and human replay audit

- Extended the Rust batched neural PUCT search, Python self-play, evaluation, and web analysis
  plumbing to support 2-, 3-, and 4-player games with stable observer/actor perspective handling.
  The current multiplayer backup is a scalar approximation: non-root win mass is distributed across
  the other players, so it is useful for search but is not yet a full per-player value vector.
  Extension verification used build/revision SHA
  `de00ad4e42ddf0a08182af2b828ca7daef9c4774f2b81ec263e0275fe23ff7dc`.
- Audited human replay game `#6` (`[183, 72, 80]` VP). Repeated `end_turn` and skipped-turn events
  make its terminal score unsafe as a value target. The cleaned artifact
  `output/human-replay-game6-audited.jsonl` therefore exports 44 confirmed human actions as
  policy-only examples (`value_loss_weight=0`): build 15, develop-double 5, loan 9, network 11,
  and sell 4.
- A policy adapter improved the demonstration-position expert NLL from about `4.821` to `4.121`
  (Top-1 `0/44` to `6/44`, Top-3 `9/44` to `17/44`) but collapsed to many zero-VP independent
  3-player games. Adapter, anchor, seat, and logit-scale variants were not promoted; the deployed
  champion remains `output/champion-iter3-mixed-v5-v6.pt` at step `1336`.
- Verification: Python training/analysis suite `106/106` passed; containerized
  `cargo test --all-targets --features python-bindings` passed 63 library tests plus all integration
  and feature-test groups. Existing compiler warnings are unrelated. The browser-only regression
  then passed separately (desktop bundled Playwright client; 390x844 mobile full-page check):
  `render_game_to_text()` reported a legal human turn after one AI move and both console-error
  buffers were empty. Playwright is release/UI verification only and is not part of Rust training.

## 2026-08-29 fast expert-iteration framework

- Reframed the next training cycle around an automated expert-iteration/league loop rather than
  serial ad-hoc hyperparameter trials. The detailed operating framework is now in
  `docs/training-methodology.md`: immutable champion, reservoir replay buffer, V6/V8 + PUCT teacher,
  search-policy distillation, fixed quality/direct suites, and early-stop promotion gates.
- Explicitly separated Rust/CUDA training validation from browser QA. Playwright remains a release
  smoke test for the Svelte UI only; it is not a dependency or step in local Rust self-play/training.

## 2026-08-29 source-mixture training loader

- Added optional `--replay-shards` and `--human-shards` inputs to `training.train`. Their
  replacement-sampling quotas are controlled by `--replay-fraction` and `--human-fraction`; the
  remaining epoch mass goes to current teacher shards. Source counts, paths, and fractions are
  recorded in checkpoint metadata, while audited human terminal targets remain value-masked.
- Added regression coverage for quota math and `WeightedRandomSampler`. This path uses only the
  local Rust/Python/CUDA stack; no browser or Playwright process is started.
- Trained `output/candidate-expert-iteration.pt` from the immutable champion with teacher/replay/
  human epoch fractions `0.65/0.30/0.05`. Independent validation loss improved from `4.634486`
  to `4.584551`; expert NLL improved from `4.821377` to `4.750230`, Top-1 from `0/44` to `1/44`,
  and Top-3 from `9/44` to `10/44`.
- The same-seed four-game/64-search quality screen averaged `60.25 VP` versus the champion's
  `53.375`, with no zero-score or lifecycle violations. The seat-rotated direct pilot then reached
  5/8 first places but averaged `-8.625 VP` margin and a score-delta 95% lower bound of `-0.5455`.
  It failed the direct gate, so no 256-search confirmation ran and the champion stayed unchanged.
- Added `training.promote`, which refuses failed/inconsistent reports, verifies candidate and
  champion SHA-256 identities, atomically replaces only an explicitly confirmed destination,
  preserves the previous champion, and writes a promotion audit manifest.

## 2026-08-30 quality-filtered expert data producer

- Added `training.produce_expert_data`, the local fast path for human-prior-guided self-play and
  expert iteration. It allocates deterministic batches across prior strengths and selection
  temperatures, reuses the parallel Rust/CUDA exporter, and writes raw plus filtered train/validation
  shards in one auditable run.
- Added trajectory-level rejection gates for incomplete outcomes, zero-score games, unhealthy ending
  income/cash, network-before-first-build, repeated loans, score/margin thresholds, and top-actor
  selection. Every rejection reason and accepted seed is recorded in `manifest.json`; train and
  validation seed namespaces are checked for overlap before publication.
- Added focused producer tests for recipe allocation, quality rejection/retention, dataset-compatible
  derived shards, and manifest seed separation. This command has no browser or Playwright dependency;
  Playwright remains UI release QA only.
- Re-audited 104 existing V5/V6/V6-512 games with the new gate at VP>=100: 22 games / 865 winner
  positions passed, while low-score and unhealthy-income games were rejected with explicit reasons.
  A smoke run of the producer then generated 12 fresh low-budget games, accepted 6 train and 2
  validation games, and verified the filtered shards load through `SelfPlayDataset`.
- Wired the audited human replay prior into model and parallel self-play. `--demonstration-shards`
  now learns context/action-intent frequencies once, blends them into root PUCT at a bounded,
  independently sweepable strength, preserves active lifecycle guards, and records prior provenance
  in every shard and expert-iteration manifest. A 2-game 3-player container smoke passed with 214
  positions accepted; the run was diagnostic only and did not alter the champion.
- Fine-tuning the deployed champion on the 865 high-score positions with 30% replay and 5% audited
  human policy data overfit the independent validation set in the first epoch; early stopping kept
  the original tensors. A policy-adapter-only follow-up improved validation loss from `4.634486` to
  `4.629441` but lost its 8-game no-prior direct pilot (`37.5%` first place, mean VP margin `-11.375`),
  so neither candidate was promoted. This negative result validates the quality/promotion gates.

## 2026-08-30 CPU rule-tree stabilization

- Kept the default policy on the shallow, explainable Rust rule tree: no neural inference, no
  terminal rollouts, and no Playwright dependency in training or benchmarking. The default
  benchmark configuration remains a one-ply decision with one continuation candidate.
- Reverted the experimental hard penalties for repeated Beer/Coal/Iron buildings and the extra
  saleable-industry bonus. On the fixed human-replay seed `1508972731975190517`, the restored
  policy returned the expected first-game result `[115,90,91]`; the eight-game three-player run
  averaged `86.92 VP`, minimum `64`, at `44.36 actions/s` (`rollouts=0`). The experiment had
  reduced the same-seed result to `[80,83,76]` and was rejected.
- Fixed incremental sale-route accounting when a build replaces a building at an occupied
  location: the old location contribution is removed before adding the replacement contribution.
  This prevents a stale route term from being counted twice during action ranking.
- Containerized `cargo test --all-targets` passed all library, integration, and example targets.
  Existing compiler warnings are unrelated unused imports/dead code. No candidate neural model
  or training checkpoint was changed or promoted in this run.
- A four-game diagnostic `depth=2, branching=3` comparison averaged `85.83 VP` at
  `18.50 actions/s` (minimum `57`), below the shallow default's `86.92 VP` at `44.36 actions/s`.
  The deeper profile remains an explicit diagnostic option and is not promoted as the live default.
- A single-variable trial reducing the network `new_build_sites` weight from `0.65` to `0.35`
  regressed the first fixed-seed game from `[115,90,91]` to `[106,67,74]`; it was stopped early
  and fully reverted. The live weight remains `0.65`.

## 2026-08-30 rule-tree tail-risk audit and pause

- The requested training direction remains the local, CPU-only, explainable rule tree. Neural
  training/checkpoint promotion and browser automation were intentionally not run in this cycle;
  Playwright remains release/UI QA only.
- The shallow live profile is unchanged: `lookahead_depth=1`, `lookahead_branching=1`, no rollout,
  and no neural inference. The fixed human-replay seed still reproduces the known strong opening
  result (`[115,90,91]` in the first game; paired 8-game checks were about `87.5 VP` mean with no
  zero-score player). Rust formatting and containerized `cargo test --all-targets` had passed before
  this audit.
- A broader 40-game diagnostic at base seed `20260830` exposed the remaining tail risk: shallow
  mean player VP `80.4667`, mean per-game minimum `66.25`, minimum `0`, and one zero-score player.
  The zero trajectory was seed `20260852` (`[94,95,0]`); another severe low trajectory was seed
  `20260859` (`[4,96,75]`). The traced zero player built several Canal-era level-one tiles and
  links, reached negative income, then lost all buildings at the era transition and entered a
  `Pass`/loan loop (`income=-10`, `cash=0`, `buildings=0`). This confirms that a broad network/build
  signal can outweigh the cost of leaving removable tiles unconverted.
- Several temporary lifecycle-risk penalties were tested only in isolated copies and rejected:
  fixed early-tile penalties improved one bad trajectory but reduced the human seed to `[70,46,87]`
  and the 40-game mean to about `74.42`; cumulative and risk-only variants also lowered the fixed
  seed and/or left a near-zero trajectory. None of these patches was applied to the live rule tree.
- The next viable experiment is a narrow, state-dependent gate: count existing unflipped
  `removed_after_phase1` tiles, increase pressure only near the Canal/Railroad transition, and
  discount the penalty when the candidate creates a *currently legal and affordable* sale/build
  frontier. The current `available_build_sites` metric is intentionally too permissive for this
  purpose and should be replaced or supplemented by `productive_options(after_runner)` checks.
  Any candidate must pass paired multi-seed checks on mean VP, worst-player VP, zero-score count,
  and action distributions before changing defaults.
- This cycle is paused at the user's request. Existing user and experiment changes in the dirty
  worktree are preserved; no reset, checkout, or unrelated revert was performed.

## 2026-08-30 saved checkpoint and stop

- This is the final checkpoint for the current work session. No further rule, model, dataset, or
  configuration changes were made after the tail-risk audit; the user requested that work stop.
- The currently usable CPU path is the explainable Rust rule tree in `src/game/rule_ai.rs` with
  `lookahead_depth=1`, `lookahead_branching=1`, no rollout, and no neural inference. Its strongest
  fixed human-replay result remains `[115,90,91]`; the paired eight-game check was about `87.5 VP`
  mean with no zero-score player.
- The neural champion remains `output/champion-iter3-mixed-v5-v6.pt` (step `1336`), and no
  unpromoted candidate was substituted. The expert-data producer, replay/human-prior plumbing,
  promotion gates, rule-tree benchmark, sweep, trace, and audit tooling are retained for the next
  session.
- Verification already recorded in this file remains the source of truth: containerized
  `cargo test --all-targets` passed for the rule-tree cycle, and the Python training/analysis
  suite passed `106/106` in the replay audit. Playwright was used only for browser release QA and
  is not part of Rust self-play, training, or benchmarking.
- The unresolved issue is tail stability rather than a missing basic policy: the 40-game diagnostic
  at base seed `20260830` had mean player VP `80.4667`, minimum `0`, and one zero-score trajectory
  (`20260852: [94,95,0]`). The proposed state-dependent Canal/Railroad transition-risk gate was
  investigated but not implemented or promoted. Any resumption should test it in an isolated
  variant and require multi-seed mean/worst-case/zero-score/action-distribution gates.
- The worktree is intentionally dirty and contains user and experiment artifacts; none were reset,
  checked out, deleted, or otherwise reverted. At checkpoint time there was no active cargo,
  training, or rule-tree benchmark process. The session is now stopped.

## 2026-08-30 final handoff (06:55 CST)

- The session is saved here as a handoff point. The live/default policy is still the shallow,
  explainable Rust rule tree in `src/game/rule_ai.rs`: `lookahead_depth=1`,
  `lookahead_branching=1`, `rollouts=0`, and no neural inference. No new code or model was
  promoted after the tail-risk audit.
- Reproducible reference results to use when resuming are: fixed human replay seed
  `1508972731975190517` -> `[115,90,91]`; diagnostic seeds `20260852` -> `[94,95,0]` and
  `20260859` -> `[4,96,75]`; the 40-game shallow diagnostic at base `20260830` averaged
  `80.4667 VP`, with per-game minimum mean `66.25`, minimum `0`, and one zero-score player.
- The remaining defect is transition tail risk: Canal-era unflipped `removed_after_phase1`
  buildings, negative income, no currently legal/affordable sale frontier, and insufficient
  cash through the Canal/Railroad settlement. Broad fixed penalties and deeper search were
  measured and rejected. The narrow state-dependent transition gate described above remains
  an isolated next experiment only; it is not in the live tree.
- The neural champion remains `output/champion-iter3-mixed-v5-v6.pt` (step `1336`). Expert-data,
  replay/human-prior, promotion-gate, benchmark, sweep, trace, and transition-probe artifacts
  remain available. `examples/transition_probe.rs` is an untracked diagnostic helper and has
  not been folded into the policy.
- Verification recorded in this log remains valid: the rule-tree cycle passed containerized
  `cargo test --all-targets`; the replay-audit Python suite passed `106/106`; formatting and
  `git diff --check` passed in the corresponding runs. Playwright was used only for browser
  release QA and is unrelated to local Rust training/benchmarking.
- The worktree remains intentionally dirty so all user and experiment changes are preserved.
  At handoff there was no project-local Cargo, Rust, training, or rule-tree benchmark process;
  unrelated services in other workspaces were left untouched. Resume from this section and
  validate any transition-risk change against the four reference seeds plus multi-seed mean,
  worst-player VP, zero-score count, and action-distribution gates.

## 2026-08-30 final stop snapshot (07:08 CST)

- The user requested that the current results be saved and work stop. No policy, model,
  dataset, or configuration change was made after the preceding handoff. The live path remains
  the explainable CPU Rust rule tree in `src/game/rule_ai.rs` with `lookahead_depth=1`,
  `lookahead_branching=1`, `rollouts=0`, and neural inference disabled.
- Latest direct reference runs from the unchanged tree are: human replay seed
  `1508972731975190517` -> `[115,90,91]` (mean `98.6667` VP); `20260852` -> `[94,95,0]`;
  `20260859` -> `[4,96,75]`; and `20260830` -> `[93,55,66]`. A 40-game shallow diagnostic
  at base `20260830` measured mean player VP `80.2667`, minimum VP `0`, one zero-score player,
  and about `33.97` actions/s. An earlier equivalent diagnostic recorded `80.4667` VP mean;
  both are retained as historical measurements and should be re-run with the exact command
  before comparing future changes.
- A four-game `depth=2, branching=3` diagnostic averaged `82.9167` VP at about `14.85`
  actions/s, while the shallow comparison was higher and materially faster; deeper search is
  therefore still diagnostic-only. The lifecycle sweep (`0.5/1.0/1.5`) likewise did not show a
  safe default improvement.
- Completed and retained work includes the loan/negative-income protections, corrected
  replacement-building sale-route accounting, live benchmark defaults, human-replay prior and
  quality-filtered expert-data tooling, promotion/evaluation gates, and rule-tree
  benchmark/sweep/trace/transition-probe helpers. The neural champion is unchanged at
  `output/champion-iter3-mixed-v5-v6.pt` (step `1336`); no candidate was promoted.
- Verification in this log remains the source of truth: containerized `cargo test --all-targets`
  passed, the Python replay audit passed `106/106`, and formatting/diff checks passed in the
  corresponding runs. Playwright is UI release QA only and is not a dependency of local Rust
  self-play, training, or benchmarking.
- The unresolved issue is tail stability at the Canal/Railroad transition. A player can enter
  negative income with no legal, affordable build/sell frontier, choose removable level-one
  tiles or links, lose those buildings at settlement, and fall into a pass/loan loop. Fixed
  penalties and deeper search were rejected. The next experiment, if resumed, is an isolated
  state-dependent gate based on actual `productive_options` before granting transition-network
  value; it must pass the four reference seeds plus multi-seed mean, worst-player VP, zero-score,
  and action-distribution gates before any default change.
- The worktree is intentionally dirty and all user/experiment artifacts are preserved. The
  transient container `cargo`, `transition_probe`, `rule_ai_sweep`, and candidate-evaluation
  commands were stopped at this checkpoint; the long-running `fast_brass` service was left
  running. No reset, checkout, deletion, or unrelated revert was performed. Parallel research
  agents were interrupted, and this session is now stopped.

## 2026-08-30 current-request final save (07:19 CST)

- This entry closes the current request to summarize and stop. No policy, model, dataset, or
  configuration change was made after the preceding `07:08 CST` stop snapshot.
- The live/default AI remains the explainable CPU Rust rule tree in `src/game/rule_ai.rs`:
  `lookahead_depth=1`, `lookahead_branching=1`, `rollouts=0`, and neural inference disabled.
  The fixed human-replay seed `1508972731975190517` still returns `[115,90,91]` (mean
  `98.6667 VP`). Diagnostic references remain `20260852 -> [94,95,0]`,
  `20260859 -> [4,96,75]`, and `20260830 -> [93,55,66]`. The latest 40-game shallow run is
  retained as `80.2667 VP` mean, minimum `0`, one zero-score player, and about `33.97 actions/s`;
  an earlier equivalent run measured `80.4667 VP`, so future comparisons must use one exact command.
- The neural champion is still `output/champion-iter3-mixed-v5-v6.pt` at step `1336`; no
  candidate was promoted. Retained reusable work includes lifecycle/loan protections, corrected
  replacement sale-route accounting, human replay and quality-filtered expert-data tooling,
  promotion and direct-quality gates, rule-tree benchmark/sweep/trace/probe helpers, and the
  documented expert-iteration framework.
- Verification recorded in this log remains valid: containerized `cargo test --all-targets`, the
  Python replay/training audit (`106/106`), formatting, and diff checks passed in their respective
  runs. Playwright is UI release QA only and is not a dependency of local Rust self-play, training,
  or benchmarking.
- The unresolved item is Canal-to-Railroad transition tail risk: negative income combined with no
  legal/affordable build or sell frontier can leave removable Canal assets to be cleared, followed
  by a pass/loan loop and a zero score. Fixed penalties, deeper search, and lifecycle sweeps were
  rejected. The next session should test the isolated state-dependent `productive_options` gate
  against the four reference seeds and multi-seed mean, worst-player VP, zero-score, and action
  distribution gates before any default change.
- At save time there was no project-local cargo, Rust, training, benchmark, or probe process. The
  long-running Docker game/API/inference services were intentionally left available; the inference
 service points at the unchanged champion. The dirty worktree and all user/experiment artifacts
 remain preserved; no reset, checkout, deletion, or unrelated revert was performed. Work is now
 stopped at the user's request.

## 2026-08-30 final stop handoff (07:38 CST)

- This entry is the authoritative save for the current request. Work is stopped; no new training,
  benchmark, model promotion, or browser/UI run was started after the previous snapshot.
- The usable/live default remains the explainable CPU Rust rule tree in `src/game/rule_ai.rs`:
  `lookahead_depth=1`, `lookahead_branching=1`, `rollouts=0`, and neural inference disabled.
  The fixed human-replay seed `1508972731975190517` remains `[115,90,91]` (mean `98.6667 VP`).
  Reference diagnostics are `20260852 -> [94,95,0]`, `20260859 -> [4,96,75]`, and
  `20260830 -> [93,55,66]`. The latest retained 40-game shallow run measured `80.2667 VP`
  mean, minimum `0`, one zero-score player, and about `33.97 actions/s`; an earlier run measured
  `80.4667 VP`, so future comparisons must use one exact command and seed protocol.
- Retained work includes the loan/negative-income and lifecycle protections, corrected
  replacement-building sale-route accounting, human-replay and quality-filtered expert-data
  tooling, replay/prior plumbing, evaluation and promotion gates, the expert-iteration
  documentation, and rule-tree benchmark/sweep/trace/probe helpers. The neural champion is still
  `output/champion-iter3-mixed-v5-v6.pt` at step `1336`; no candidate checkpoint was promoted.
- After the prior `07:19` save, `src/game/rule_ai.rs` was edited with an isolated,
  **unverified** transition-risk experiment. It adds legality-aware `saleable_builds`, counts
  unflipped `removed_after_phase1` buildings, and gates Canal liquidation penalties on actual
  sale/build frontiers and economic fragility. The latest pressure thresholds are deck `0..=4`
  -> `1.00`, `5..=7` -> `0.85`, `8..=10` -> `0.70`, `>10` -> `0.00`, with round `0..=4` ->
  `0.00`, round `5` -> `0.85`, and round `>=6` -> `1.00`. This change was not rebuilt, tested,
  probed, or benchmarked, and must not be considered live or promoted. The next session should
  build the release examples in `fast-brass-gpu-training`, probe the human action-17 case and
  failure-seed `20260852` action-19 case, then require the four reference seeds plus multi-seed
  mean, worst-player VP, zero-score count, and action-distribution gates.
- Historical verification recorded above remains valid: containerized `cargo test --all-targets`
  passed for the preceding rule-tree cycle, the Python replay/training audit passed `106/106`,
  and prior formatting/diff checks passed. The current post-edit compile was intentionally not
  run. `git diff --check` currently reports no whitespace errors (only normal line-ending
  warnings).
- The worktree is intentionally dirty, including user and experiment files; nothing was reset,
  checked out, deleted, or reverted. No local project process is active. The long-running Docker
  services (`fast-brass-gpu-training`, `fast-brass-inference`, `fast-brass-api-3010`, and
  `fast-brass-ui-5174`) remain available, with inference still serving the unchanged champion.
  This handoff is complete and the session is stopped at the user's request.

## 2026-08-30 final saved summary (07:44 CST)

- This section is the authoritative checkpoint for the current request. The user asked to
  preserve the current results and stop. No training, benchmark, model promotion, source
  rollback, or browser run was started after the preceding checkpoint.
- The usable/default policy is the explainable CPU Rust rule tree in `src/game/rule_ai.rs`:
  `lookahead_depth=1`, `lookahead_branching=1`, `rollouts=0`, and neural inference disabled.
  The fixed human-replay seed `1508972731975190517` reproduces `[115,90,91]` (mean
  `98.6667 VP`). Diagnostic references are `20260852 -> [94,95,0]`,
  `20260859 -> [4,96,75]`, and `20260830 -> [93,55,66]`. The retained 40-game shallow
  diagnostic measured `80.2667 VP` mean, minimum `0`, one zero-score player, and about
  `33.97 actions/s`; an earlier run measured `80.4667 VP` under a slightly different run.
- Stable work retained in the dirty worktree includes rules/turn-flow repairs, loan and
  negative-income protections, replacement-building sale-route accounting, deterministic
  rule-tree benchmark/sweep/trace/probe helpers, audited human replay and bounded
  demonstration priors, quality-filtered expert-data production, dataset partition checks,
  evaluation/direct-quality gates, promotion tooling, and the documented expert-iteration
  workflow. Local Rust training/self-play/benchmarking has no Playwright dependency;
  Playwright is UI release QA only.
- The neural champion is unchanged: `output/champion-iter3-mixed-v5-v6.pt`, step `1336`,
  feature version `1`, state dimension `2111`, action dimension `1615`. No candidate passed
  the promotion bar: `iter19-mix10-v6-fullval` had four-game mean VP margin `-2.75`,
  `candidate-iter27-quality100-adapter8` had eight-game mean margin `-11.375`, and
  `candidate-expert-iteration` had mean margin `-8.625` (its small score delta was not a
  statistically safe improvement). The audited expert producer retained `22/104` games and
  `865` winner positions at the VP>=100 gate, with a smoke publication of six train and two
  validation games.
- Verification that remains valid: the preceding rule-tree cycle passed containerized
  `cargo test --all-targets`; the Python replay/training audit passed `106/106`; formatting
  and prior diff checks passed. The current `git diff --check` reports no whitespace errors
  apart from normal CRLF conversion warnings.
- Important qualification: `src/game/rule_ai.rs` was edited at `07:36` with an isolated,
  legality-aware Canal/Railroad transition-risk gate (`saleable_builds`, unflipped
  `removed_after_phase1` counting, and economic-fragility thresholds). It has not been
  rebuilt, tested, probed, or benchmarked. `fast_brass.so` was built on `2026-08-29`, so this
  source experiment is not in the live binary and must not be described as an improvement.
- The unresolved problem is transition tail risk: a player can reach negative income with no
  legal/affordable sale or build frontier, retain removable Canal assets, lose them at the
  Canal/Railroad settlement, and fall into a pass/loan loop. Fixed penalties, deeper search,
  and lifecycle sweeps were rejected. A future run should first rebuild the extension, probe
  the human action-17 and failure-seed `20260852` action-19 cases, then require the four
  reference seeds plus multi-seed mean VP, worst-player VP, zero-score count, and action-
  distribution gates before changing the default.
- All user and experiment edits and artifacts are intentionally preserved; the worktree is
  dirty and nothing was reset, checked out, deleted, or reverted. At save time there was no
  project-local Cargo, Rust, training, benchmark, or probe job. The long-running Docker
  services (`fast-brass-gpu-training`, `fast-brass-inference`, `fast-brass-api-3010`, and
  `fast-brass-ui-5174`) remain available, with inference serving the unchanged champion.
  This session is now stopped.

## 2026-08-30 authoritative final checkpoint (07:57 CST)

- This is the latest save for the user's request to summarize and stop. The transient 40-game
  benchmark has completed; no further training, search, promotion, or browser work is running.
  Older stop entries above are historical; this section supersedes their statements about the
  narrow transition gate being unverified.
- The usable/default policy is the explainable CPU Rust rule tree in `src/game/rule_ai.rs` with
  `lookahead_depth=1`, `lookahead_branching=1`, no rollouts, and neural inference disabled. The
  source now contains the legality-aware Canal/Railroad transition guard: `productive_options`
  counts currently legal/affordable builds and sales, `unflipped_removed_buildings` counts
  removable level-one tiles, and the guard is active only in the narrowed Canal deck window
  (`deck=5..=7 -> 0.85`, `deck=8..=10 -> 0.70`, otherwise `0.00`; round pressure is applied only
  inside that window). The edited source was rebuilt into the container release examples and
  exercised by the formal benchmark.
- Latest exact command and result:
  `docker exec fast-brass-gpu-training sh -lc '/work/target/linux/release/examples/rule_ai_bench 40 3 20260830'`
  produced `40` games / `4,280` actions, mean player VP `81.025`, minimum player VP `2`, zero-score
  players `0`, and `45.0907 actions/s` (mean legal-action count `242.0047`). The retained pre-gate
  file `.tmp-sweep40-20260830.jsonl` measured mean `80.4667`, minimum `0`, one zero-score player,
  and `43.5306 actions/s`; these are separate historical runs, so future comparisons should use
  the exact command, seed, binary, and profile recorded here. The latest action totals were
  `BuildBuilding=1356`, `BuildDoubleRailroad=42`, `BuildRailroad=1283`, `Develop=34`,
  `DevelopDouble=578`, `Loan=600`, `Pass=45`, `Scout=32`, and `Sell=310`.
- Fixed-seed checks after narrowing the gate are: human replay seed `1508972731975190517` ->
  `[115,90,91]` (mean `98.6667 VP`); `20260852` -> `[100,85,65]` (the former zero trajectory is
  removed); `20260859` -> `[4,96,75]` (still unresolved); and `20260830` -> `[87,59,50]`.
  The human-strength result is therefore preserved, while the broad diagnostic no longer contains
  a zero-score player in this run; tail stability is not solved because the `20260859` trajectory
  remains very weak.
- The neural champion remains `output/champion-iter3-mixed-v5-v6.pt` at step `1336` (feature
  version `1`, state dimension `2111`, action dimension `1615`); no candidate checkpoint passed
  the promotion gates. Retained infrastructure includes human replay and bounded demonstration
  priors, quality-filtered expert-data production, evaluation/direct-quality and promotion gates,
  and deterministic rule-tree benchmark/sweep/trace/transition-probe helpers. Local Rust
  self-play, training, and benchmarking have no Playwright dependency; Playwright is UI release
  QA only.
- Verification status: the preceding rule-tree cycle's containerized `cargo test --all-targets`
  and the Python replay/training audit (`106/106`) remain valid; the latest narrow-gate release
  build and 40-game benchmark completed successfully. `git diff --check` reports no whitespace
  errors apart from normal CRLF conversion warnings. No additional broad test run was started
  after this benchmark because the user requested a stop.
- The worktree is intentionally dirty and all user/experiment edits and temporary artifacts remain
  preserved. No reset, checkout, deletion, or unrelated revert was performed. No project-local
  Cargo, Rust, training, benchmark, or probe process is active. The long-running Docker services
  (`fast-brass-gpu-training`, `fast-brass-inference`, `fast-brass-api-3010`, and `fast-brass-ui-5174`)
  remain available; inference still serves the unchanged neural champion. Work is now stopped.

## 2026-08-30 final diagnostic save and stop (08:08 Asia/Shanghai)

- This is the newest authoritative stopping point. It supersedes earlier stop notes where they
  describe the transition guard as fully diagnosed. No additional training, long benchmark,
  model promotion, or browser run was started after the user's stop request.
- The strongest immediately usable path remains the explainable Rust rule tree with
  `lookahead_depth=1`, `lookahead_branching=1`, no rollouts, and no neural inference. Its latest
  40-game release benchmark (base seed `20260830`) completed 40 games / 4,280 actions at
  `45.0907 actions/s`, with mean player score `81.025 VP`, minimum `2 VP`, and zero zero-score
  players. The fixed human replay seed `1508972731975190517` remains `[115,90,91]`; diagnostic
  seed `20260852` improved from `[94,95,0]` to `[100,85,65]`, while `20260859` remains
  `[4,96,75]`. These results show a meaningful tail-risk improvement, not a solved tail.
- The `20260859` trace now has a concrete failure chain. At Canal action 29, deck `0`, income
  `-2`, cash `17`, and zero removable exposure, player 0 builds a level-one Beer tile and creates
  its first removable exposure. The current guard only inspects pre-action exposure, so it does
  not fire. A loan on action 30 lowers income to `-5` but is spared because it opens an immediately
  affordable saleable build. Actions 35-36 then choose Coal and a canal link instead of converting
  a product; the boolean `saleable_builds` frontier still exempts them despite no legal Sell and
  inadequate debt runway. Later Scout/Pass/Loan choices drive income to `-8`. Only the Beer tile
  is removed at the era change; the final `4 VP` is caused by the subsequent Railroad cash/debt
  liquidation loop, not by Canal removal alone.
- The remaining guard defects are therefore narrow and testable: it misses an action that creates
  the first removable exposure; it skips a last-card action when the simulated successor already
  entered Railroad; an affordable saleable build does not prove that cash remains for the next
  debt settlement; and optimistic route distance can be mistaken for a repayment path. A future
  candidate should gate late-Canal negative-income actions on both post-action exposure and real
  conversion legality, and grant a build/network exemption only when a legal Sell exists or the
  build can be paid while retaining a debt payment plus fixed reserve.
- The prepared human-seed action 31-34 diagnostic was intentionally not run after the stop request.
  On resumption, it should first confirm that the action-31 loan at income `-6`, cash `2` genuinely
  opens a funded saleable build with sufficient reserve. This is the key false-positive check
  before changing the guard. Any candidate must then pass the four fixed seeds plus a multi-seed
  gate on mean VP, worst-player VP, zero-score count, and action distribution.
- Retained reusable work includes deterministic rule-tree benchmark/sweep/trace/probe tools,
  legality-aware lifecycle and loan protections, audited human replay and bounded demonstration
  priors, quality-filtered Best-of-N expert-data production, independent data partitions,
  source-mixture training, direct-quality evaluation, and statistically gated promotion. This is
  the intended fast framework: use an engineered teacher for immediate strength, generate verified
  high-score trajectories, distill them with replay/human policy data, and promote only against a
  frozen champion and blind evaluation seeds.
- The neural champion remains unchanged at `output/champion-iter3-mixed-v5-v6.pt`, step `1336`
  (feature version `1`, state dimension `2111`, action dimension `1615`). No later neural candidate
  passed promotion, so none was deployed. Playwright remains browser/UI QA only; local Rust
  simulation, benchmarking, expert-data generation, and training do not depend on it.
- Latest verification: containerized `cargo check --example transition_probe` passed. The earlier
  rule-tree cycle's `cargo test --all-targets` and Python replay/training audit (`106/106`) remain
  recorded as passing. `cargo fmt --check` is not currently clean because the untracked transition
  probe and one `rule_ai.rs` site need formatting; this is saved as known cleanup rather than being
  silently claimed green.
- The dirty worktree and all user/experiment artifacts are intentionally preserved. No reset,
  checkout, deletion, or unrelated revert was performed. Process inspection at this checkpoint
  found no project-local Cargo, Rust, Python training, benchmark, sweep, or probe process. The
  Docker game/API/inference services remain available and inference still serves the unchanged
  champion. Work is stopped here.

## 2026-08-30 authoritative AI-improvement checkpoint (08:27 Asia/Shanghai)

- This is the latest and authoritative handoff for the user's request to save the current
  results and stop. All three parallel research tasks completed, the final 40-game benchmark
  exited successfully, and no further build, training, search, promotion, or browser work was
  started. Older checkpoints above remain experiment history; this section supersedes them
  where results or source defaults differ.
- The selected source default remains the explainable, CPU-friendly Rust decision tree in
  `src/game/rule_ai.rs`: `lookahead_depth=1`, `lookahead_branching=1`, no rollouts, and no neural
  inference. The narrowed transition gate remains `deck=5..=7 -> 0.85`, `deck=8..=10 -> 0.70`,
  and zero elsewhere. The newest addition is `ProductiveOptions.immediate_income_builds`, which
  distinguishes a merely affordable product build from a currently legal Coal/Iron build that
  will be fully absorbed by the market, flip immediately, and actually raise income.
- The retained late-Canal loan rule is deliberately narrow. It applies only when the Canal deck
  is empty, income was already negative before the loan, no legal Sell exists after it, the only
  claimed conversion is an affordable saleable build, no immediate market-flip income recovery
  exists, and there is no real route conversion. That case receives an `-8.0` soft penalty. This
  keeps ordinary opening loans and the strong human-reference loan legal and competitive while
  discouraging the observed build-without-selling debt spiral.
- Formal 40-game comparison, using the same release benchmark profile and base seed `20260830`:
  the prior narrowed-gate baseline scored `81.025 VP` mean, `66.975` mean per-game minimum,
  `2 VP` global minimum, zero zero-score players, and `45.0907 actions/s`. The selected `-8.0`
  candidate scored `82.425 VP` mean (`+1.400`), `71.10` mean per-game minimum (`+4.125`),
  `14 VP` global minimum, zero zero-score players, and `45.165 actions/s`. It preserved the
  fixed human-reference seed at `[115,90,91]`; failure seed `20260838` became `[73,79,89]` and
  `20260859` became `[14,88,68]`.
- The completed `-6.0` confirmation run produced 40 games / 4,280 actions at
  `44.6355 actions/s`, with `82.15 VP` mean, `70.9` mean per-game minimum, `14 VP` global
  minimum, and zero zero-score players. Its action totals were `BuildBuilding=1366`,
  `BuildDoubleRailroad=29`, `BuildRailroad=1285`, `Develop=29`, `DevelopDouble=587`,
  `Loan=590`, `Pass=37`, `Scout=43`, and `Sell=314`. Fixed checks were human reference
  `[115,90,91]`, `20260852 -> [100,85,65]`, `20260830 -> [87,59,50]`,
  `20260838 -> [80,71,77]`, and `20260859 -> [14,88,68]`. Because `-6.0` was lower on both
  aggregate mean and tail mean, the source constant was restored to the already benchmarked
  `-8.0` before stopping.
- The gain is real but not yet a claim of human-level play. The `-8.0` candidate regressed total
  table VP in 6 of 40 paired games versus baseline, with a worst paired table-total drop of
  `65 VP`; `20260859` still has a weak `14 VP` seat. The current change should therefore be
  treated as the best validated rule-tree candidate, not a finished strongest AI. The final
  `-8.0` source was not rebuilt after restoring the constant; the container release example's
  last binary was the `-6.0` run. Rebuild before any resumed evaluation or deployment.
- Root-cause evidence is now specific: both original tail failures took four loans, never sold,
  ended at income `-10` and cash `0`, and entered asset liquidation. Their nominal saleable
  builds produced zero immediate income and never formed a legal Sell loop. By contrast, the
  human-reference action-31 loan unlocked Iron level 0; its legal build was immediately consumed
  by the market, flipped, and raised income by `+3`. This is why the new signal uses successor
  economics rather than simply asking whether a product tile can be built.
- Rejected experiments remain rejected: globally shortening the remaining-turn estimate raised
  the minimum but reduced the 40-game mean to `78.833`; globally replacing saleable-build logic
  with immediate flips caused Pass behavior and reduced the human seed to `[52,72,113]`; and a
  hard deck-empty ban on false conversion preserved the human seed but reduced `20260852` to
  `[69,73,17]`. These failures support the retained soft, local penalty rather than a global rule.
- The neural champion is unchanged: `output/champion-iter3-mixed-v5-v6.pt`, step `1336`. On the
  audited 183-VP human replay it has 44 legal policy-only decisions, with champion Top-1 match
  `0%` and Top-3 match `20.45%`; the replay must not supply value targets because its terminal
  log is incomplete. Prior self-distillation candidates failed promotion, so no checkpoint was
  promoted and no neural training was run in this cycle.
- The retained fast-improvement framework is: use the engineered Rust policy as the immediate
  teacher; generate Best-of-N/self-play trajectories; reject whole games below absolute quality
  and lifecycle gates; mix only audited human policy labels and verified high-score trajectories;
  and promote against a frozen champion on blind seeds with a positive 95% lower confidence
  bound. The next cheapest neural test is a search-time A/B on exact failure seeds `20260838`
  and `20260859`, comparing champion+V6 with and without a seat-0 human demonstration prior of
  `0.25`; it is diagnostic only and must not train or promote a model unless both seeds improve.
- Verification retained from the cycle: the earlier containerized `cargo test --all-targets`
  passed, the Python replay/training audit passed `106/106`, the selected `-8.0` candidate
  completed its formal release benchmark, and the final `-6.0` comparison exited with code 0.
  Tests were not rerun after the one-line restore to `-8.0`, in accordance with the stop request.
  The worktree remains intentionally dirty; no reset, checkout, deletion, or unrelated revert
  was performed. All experiment artifacts are preserved for resumption.

## 2026-08-30 recovery-policy review save and stop (13:46 Asia/Shanghai)

- This is the newest authoritative stopping point. The current pass was a read-only design review
  of `RecoveryPriorityInputs`, `recovery_priority_from_inputs`,
  `lifecycle_priority_adjustment`, `strategic_priority_adjustment`, and `loan_plan_value` in
  `src/game/rule_ai.rs`. No policy edit, neural training, self-play, benchmark, browser run, model
  promotion, or deployment was performed during this review.
- The last formally measured default remains the shallow, CPU-friendly Rust decision tree from the
  08:27 checkpoint: depth `1`, branching `1`, no rollouts, and no neural inference. Its selected
  `-8.0` late-Canal loan candidate measured `82.425 VP` mean, `71.10` mean per-game minimum,
  `14 VP` global minimum, zero zero-score players, and about `45.165 actions/s` over 40 games.
  It preserved the human-reference seed at `[115,90,91]`; `20260838` was `[73,79,89]` and
  `20260859` was `[14,88,68]`. The source was restored to `-8.0` after the `-6.0` comparison but
  was not rebuilt afterward, so a resumed run must rebuild before treating the binary as current.
- The source now also contains an initial severe-recovery layer that is newer than that benchmark.
  Severe Railroad recovery is currently defined as Railroad phase, VP below `35`, at most three
  buildings, and income at or below `-5`. Late Canal recovery is deck-empty Canal with the same
  VP/building limits and negative income. The current layer rewards a first Scout that opens
  several legal product builds, penalizes road spending without a conversion, penalizes unflipped
  Coal/Iron/Beer, and penalizes destruction of the final product-build frontier. These newer rules
  and their in-file unit tests were inspected but were not run or benchmarked in this pass; they
  must not be described as a validated strength gain.
- Legality evidence is solid: `ProductiveOptions.saleable_builds` comes from the real build
  validator, so it already includes card matching, resource availability, and current
  affordability. `immediate_income_builds` specifically counts legal Coal/Iron builds whose
  complete output can enter the market immediately, flip the tile, and raise income.
- The review found four remaining defects in the recovery layer. First, `concrete_conversion` is
  too broad because the mere existence of a future immediate-income build is treated as conversion
  by the current action. Second, a product Build can consume the last card/cash option and be
  mistaken for destroying the product frontier even though it just placed the desired product.
  Third, Pass has no recovery-specific penalty beyond the generic `-8`. Fourth,
  `loan_plan_value` calls a product funded at `cost + 8` without also reserving the next negative
  income settlement. These defects can still allow pass/loan loops or can punish the correct move.
- The next implementation should add `reserve_safe_saleable_builds` to `ProductiveOptions`, defined
  from legal Cotton/Goods/Pottery builds for which current cash covers
  `total_money_cost + max(0, -income) + 4`. `RecoveryPriorityInputs` should separately carry
  `before_sell`, before/after safe product counts, `before_immediate_income_builds`, and
  `converted_now`. `converted_now` should mean only VP secured now, a building flipped now, or
  income raised now; a newly legal Sell should be represented separately rather than pretending
  the present action already converted value.
- The proposed near-hard product window is deliberately narrow:
  `severe_railroad_recovery && before_sell == 0 && before_sellable_units == 0 &&
  before_safe_saleable_builds > 0`. Inside it, use one mutually exclusive adjustment: about `+28`
  for a safe Cotton/Goods/Pottery Build; `-30` for Pass; `-24` for another Loan; and `-22` for an
  unconverted Road, Develop, Scout, or resource Build. An immediately flipping Coal/Iron Build gets
  a `+12` exemption/recovery value. A product Build that leaves less than debt plus GBP 4 reserve
  gets `-8` rather than the strong reward. Do not stack several near-hard penalties on one action.
- Correct final-frontier destruction to require a pre-action product frontier, no post-action
  frontier, no legal Sell after the action, no conversion now, and a non-product selected action.
  Suggested values are `-32` in severe Railroad recovery and `-16` in late Canal. A product Build
  must always be exempt from this particular penalty. Pass should receive `-30` only when a legal
  Sell or safe product Build exists, `-18` when the sole real recovery is an immediate-income
  Coal/Iron Build, and no extra recovery penalty when none of those options exists.
- Make severe-recovery loan scoring reserve-aware as well. If a safe product is already buildable,
  another Loan receives `-24`; if the Loan first opens a product Build that still leaves
  `max(0, -after_income) + 4` after its cost, it may receive about `+14`. Merely opening a nominally
  affordable product without this reserve is not funded conversion. Keep this branch Railroad-only:
  the human-reference loan occurs in Canal at income `-6`, cash `2`, and opens Iron level 0 that
  immediately flips and raises income, so it should retain the existing opening/Canal treatment.
- Required focused tests on resumption are: inactive healthy-state behavior; all four severe-state
  activation boundaries; safe product reward and exact reserve boundary; product-Build exemption
  from frontier destruction; tiered Pass behavior with and without real recovery options; Loan
  penalty when already funded and reward only when it first creates a reserve-safe product; unsafe
  nominal funding remaining negative; non-product final-frontier destruction in Railroad and late
  Canal; immediate-flip Coal/Iron exemption; Road opening a legal Sell exemption; first-Scout reward
  without repeat-wild reward; and preservation of the human-reference Canal loan.
- `git diff --check` at this checkpoint reports no whitespace errors, only the repository's normal
  LF-to-CRLF warnings. No project-local Cargo, Rust, Python training, benchmark, sweep, or probe
  process was found. The intentionally dirty worktree and all experiment artifacts were preserved;
  no reset, checkout, deletion, cleanup, or unrelated revert was performed. Work stops here at the
  user's request.

## 2026-08-30 authoritative saved state and stop (14:20 Asia/Shanghai)

- This is the newest authoritative handoff. The user requested a summary and stop. This save changed
  only `progress.md`: no AI source, checkpoint, dataset, configuration, service, or deployment was
  changed, and no training, self-play, benchmark, browser, or Playwright job was started.
- The project now contains two complementary AI paths. The immediately usable path is the local,
  explainable Rust rule tree with a shallow default (`lookahead_depth=1`,
  `lookahead_branching=1`, no rollout, no neural inference). The longer-term path includes
  deterministic legal-action/state export, hidden-information determinizations, batched multi-layer
  neural PUCT for two to four players, CUDA self-play, human policy replay, quality-filtered
  Best-of-N trajectory production, source-mixture training, frozen-champion evaluation, and an
  atomic hash-checked promotion gate. The browser analysis/observer UI is separate from local Rust
  training; Playwright is only optional UI release QA.
- The last formally measured rule-tree default remains the `-8.0` late-Canal loan configuration
  from the 08:27 checkpoint. Across its 40-game release benchmark it measured `82.425 VP` mean,
  `71.10` mean per-game minimum, `14 VP` global minimum, zero zero-score players, and about
  `45.165 actions/s`. It preserved the human-reference seed `1508972731975190517` at
  `[115,90,91]`; failure references were `20260838 -> [73,79,89]` and
  `20260859 -> [14,88,68]`. These measurements do not validate later source edits.
- The current `src/game/rule_ai.rs` is uncommitted/untracked and has SHA-256
  `228a0530f1c5bb363a1cb81e7acec0eeea73de35c89c65efba5dace358b11aff`. It contains the newer
  severe-recovery layer described in the 13:46 review plus a loan-maturity surcharge. This exact
  source was not rebuilt, unit-tested, or benchmarked in the final review, so it is an experiment,
  not a promoted strength result. In particular, the recovery layer still lacks the proposed
  reserve-safe product frontier and still treats some merely available future actions as a current
  conversion; the detailed recovery defects and focused-test list remain in the preceding section.
- The final read-only loan audit confirmed that scoring uses the actual displayed-income change:
  `after.income_amount - before.income_amount`. A loan limits the forecast horizon to `2.5`, but
  the main signal is clamped to `[-10,10]` and still adds a small hidden income-marker delta. The
  new maturity surcharge is zero through displayed income `+1`, then rises monotonically to a
  maximum of `11.75`. It therefore implements "borrowing against high income is worse", but not
  "an unusually large displayed-income drop is clearly worse": early drops of 4, 5, 6, or 7 are
  mostly flattened by the clamp. Low-income loans are also charged again in `safety_score`,
  `loan_plan_value`, and `economic_guard_penalty`, partly from nonlinear marker position rather
  than displayed income and a concrete recovery path.
- The loan rule itself gives GBP 30 and calls `decrease_income_level(3)`. The legacy nonlinear
  decrement can turn the nominal three-level cost into very different displayed-income losses
  (normally about 3, sometimes 4-7, and pathological at the top marker). Existing parity tests lock
  part of that behavior. Treat a rule-engine correction as a separate change from AI tuning so a
  policy gain is not confused with a rules change.
- The next narrow AI experiment should remove loan marker-delta scoring and compute one bounded cost
  from displayed-income drop, remaining horizon, and pre-loan maturity. The reviewed candidate is:
  `base = drop * min(rounds_remaining, 2.5)`,
  `maturity = min(0.75 * max(before_income - 1, 0), 12)`,
  `large_drop = 1.5 * max(drop - 3, 0)`, and
  `income_score = -min(base + maturity + large_drop, 22)`. Add pure tests for low/high income,
  drop-3 versus drop-5, the `-9 -> -10` boundary, the 22-point cap, and independence from hidden
  marker delta before running a single fixed-seed smoke. Keep the human seed near `[115,90,91]`,
  then check `20260838`, `20260859`, `20260852`, and `20260830`; run a 40-game gate only if those
  diagnostics improve without a new zero-score player.
- The neural champion is unchanged at `output/champion-iter3-mixed-v5-v6.pt` (step `1336`). No
  candidate was trained or promoted in this final cycle. Historical verification remains as
  recorded above (`cargo test --all-targets` for the earlier rule-tree state and `106/106` Python
  replay/training tests); it must not be attributed to the current post-benchmark source.
- `git diff --check` still reports no whitespace error, only the repository's normal LF-to-CRLF
  warnings. The worktree intentionally remains dirty (39 tracked files changed plus untracked code,
  diagnostics, and experiment artifacts); nothing was reset, deleted, cleaned, committed, or
  reverted. No project-local Cargo, Rust, Python training, benchmark, sweep, or probe process was
  active at this save point. Work is stopped here.

## 2026-08-30 validated loan-cost and tail-recovery result (14:53 Asia/Shanghai)

- This section supersedes the unvalidated source status in the preceding 14:20 handoff. The current
  Rust rule-tree default was rebuilt, tested, and evaluated without Playwright or neural training.
  The evaluated `src/game/rule_ai.rs` has SHA-256
  `78ABADC8E3F619FBC3EA2F9FDBEE14AA872B3A029530886D802ED07189F06EEC`.
- Loan scoring now separates two costs. `loan_maturity_surcharge` remains zero through displayed
  income `+1` and rises smoothly after that, preserving useful opening loans while making borrowing
  against a mature income engine less attractive. `loan_excess_income_loss_surcharge` measures the
  actual displayed-income loss from the before/after states and charges only the portion above the
  normal three points, capped at twelve. The selected independent default weights are `0.5` for
  maturity and `0.5` for excess loss; this is a policy change, not a loan-rule-engine change.
- A separate narrowly gated Railroad recovery signal rewards a Cotton, Goods, or Pottery build only
  when VP is below `35`, buildings are at most three, income is at most `-5`, no saleable product is
  already present, one remaining legal network action provably exposes a legal Sell, and cash after
  that network action still covers the next negative-income settlement. Its selected weight is
  `12`. The broader build-to-sale-frontier reward remains default-off because it regressed the
  human-reference seed.
- On the fixed 40-game, three-player benchmark beginning at seed `20260830`, the previous baseline
  measured `82.425` mean VP, `69.40` mean per-game minimum, `30` global minimum, `42` Pass actions,
  `292` Sell actions, and `575` Loan actions. The validated default measured `82.825` mean VP,
  `70.65` mean per-game minimum, `43` global minimum, `28` Pass actions, `304` Sell actions, and
  `578` Loan actions at about `42.835 actions/s`. Loans losing more than three displayed-income
  points fell from `237` to `231`, and mean displayed-income loss per loan fell from `3.866` to
  `3.846`.
- The loan-only hybrid measured `82.633` mean VP, `70.10` mean per-game minimum, `30` global minimum,
  and `40` Pass actions. The narrow recovery plan supplied the remaining tail gain. It changed only
  two games in the 40-game set: `20260858` improved from `[88,30,88]` to `[97,63,91]`; in
  `20260833`, the rewarded seat improved from `57` to `59` while opponent interaction changed the
  table from `[97,57,68]` to `[95,59,46]`.
- Fixed default release results are: human-reference seed `1508972731975190517 -> [115,90,91]`,
  `20260852 -> [90,92,77]`, `20260838 -> [73,70,104]`, `20260859 -> [63,98,47]`,
  `20260830 -> [95,47,66]`, and former Pass-chain seed `20260858 -> [97,63,91]`. On the last seed,
  Pass fell from `13` to `2`.
- Verification is complete: `cargo test --all-targets` passed, including `72/72` library tests,
  `22/22` focused rule-AI tests, `3/3` benchmark-example tests, and `2/2` sweep-example tests. The
  release benchmark and sweep examples were rebuilt successfully. Only pre-existing unused
  import/variable warnings remain. No project Cargo, Rust benchmark, sweep, probe, or neural
  training process was left active; unrelated long-lived Python services were not touched.

## 2026-08-30 stalled-product stockpile tail fix

- The `43 VP` tail at seed `20260862` was traced to seat 0, which developed a Goods plan but then
  built four unflipped Canal Coal mines, never established a legal Sell, entered Railroad at severe
  negative income, and finished with `0` Sell actions. A deck-empty conversion gate only moved the
  result from `[43,78,86]` to `[45,77,85]`; increasing its weight had no further effect and changed
  seed `20260859` from the prior `[63,98,47]` to `[57,97,42]`, so its default is now restored to `0`.
- The promoted signal is intentionally narrower than a generic repeated-resource penalty. It only
  applies in Canal when the player has negative income, no flipped building or saleable product,
  existing unspent Coal/Iron, at least two product tiles and two support tiles removed specifically
  by Develop rather than Build, a legal product Build/Develop follow-up, and a candidate Coal/Iron
  build that neither flips nor improves income. Beer is excluded. This distinction preserves the
  valid human-reference resource-first route that had regressed under broad stockpile penalties.
- `resource_stockpile_weight=1.0` is promoted as the default. On the paired 40-game benchmark from
  seed `20260830`, the prior default (`conversion=0`, stockpile `0`) scored `82.825` mean VP,
  `70.650` mean per-game minimum, and `43` global minimum. The promoted candidate scored `83.125`
  mean VP, `71.525` mean per-game minimum, and `46` global minimum. Pass fell `28 -> 25`, Loan fell
  `578 -> 577`, Sell stayed `304`, and zero-score players stayed at zero. The target game became
  `[78,85,80]`, raising its weak seat by `35 VP` and its minimum from `43` to `78`.
- The human-reference seed remains exactly `[115,90,91]`. Fixed checks `20260830`, `20260833`,
  `20260838`, `20260852`, `20260858`, and `20260859` retained the same terminal scores and
  action-family totals. In the paired 40-game VP output, only the intended `20260862` row changed,
  and the aggregate action delta exactly matched that game's delta. A softer weight `0.5` produced a
  slightly higher mean (`83.183`) but one fewer Sell and five more high-cost loans; weight `1.0`
  was selected for the stronger tail mean, preserved Sell count, and lower total loan count.
- Focused verification passed `26/26` rule-AI tests, `3/3` benchmark-example tests, and `2/2`
  sweep-example tests before promotion. After promotion, `cargo test --all-targets` passed in the
  Rust container (`76/76` library tests plus every integration/example target), `rustfmt --check`
  passed for all three changed Rust files, and `git diff --check` reported no whitespace errors.
  The release benchmark/sweep binaries were rebuilt, and no-override smoke runs reproduced
  `20260862 -> [78,85,80]`, `20260859 -> [63,98,47]`, and the human reference
  `1508972731975190517 -> [115,90,91]`.

## 2026-08-30 post-sale product-cycle tail fix (16:38 Asia/Shanghai)

- The former `43 VP` global tail is no longer present. It was the seed `20260862` zero-Sell
  stockpile failure fixed in the preceding section; the promoted stockpile default reproduces
  `[78,85,80]`. The next worst game was `20260833 -> [95,59,46]`, so this cycle diagnosed the new
  tail rather than continuing to tune the already-fixed seed.
- In `20260833`, seat 2 completed its only Pottery Sell at action 81. At action 82 it had `9 VP`,
  income `+3`, GBP 17, eight cards, and one card left in the draw deck. A reachable Goods build
  scored `7.8847`, only `0.5983` below a road at `8.4830`. The policy chose two roads and a Loan,
  did not place Goods until action 94, and could no longer convert the product. It finished at
  `46 VP`, income `-5`, with 15 road actions but only one Sell.
- The new `post_sale_product_cycle` signal is deliberately narrow. It rewards only a Railroad
  Cotton/Goods/Pottery build when VP is below 15, displayed income is non-negative, at most two
  draw-deck cards remain, no unsold product is already present, and a flipped owned product proves
  that the player completed a real earlier Sell. The exact successor must be sellable within at
  most three future Route/Sell actions, retain those cards plus one financing/Beer action, and
  retain cash for the next negative-income settlement. It does not globally penalize roads or
  reward generic product builds.
- The selected default is `post_sale_product_cycle_weight=0.75`. At the target decision this moves
  Goods to `8.6347`, just above the unchanged road at `8.4830`, and the action switches exactly as
  intended. Weights below `0.75` did not switch the decision. Values through `2.0` preserved the
  human reference, but `0.75` was chosen as the smallest effective value.
- On the paired 40-game benchmark beginning at seed `20260830`, the previous default measured
  `83.125` mean VP, `71.525` mean per-game minimum, `46` global minimum, `25` Pass, `577` Loan,
  and `304` Sell actions. The promoted candidate measured `83.267` mean VP, `72.075` mean
  per-game minimum, `47` global minimum, `26` Pass, `578` Loan, and `305` Sell actions. It changed
  only two games: `20260833` improved from `[95,59,46]` to `[93,61,55]`, and `20260868` improved
  from `[75,93,107]` to `[88,94,101]`. Total VP rose by 17 across 120 seats and zero-score players
  remained zero. The aggregate action mix added one Build and one Sell while removing six net
  road actions; the single added Pass and Loan are noted rather than hidden.
- Fixed-seed gates `20260830`, `20260837`, `20260838`, `20260852`, `20260858`, `20260859`, and
  `20260862` remained exactly unchanged. The human-reference seed remained
  `1508972731975190517 -> [115,90,91]`. No-override release smoke after rebuilding reproduced
  `20260833 -> [93,61,55]`, `20260868 -> [88,94,101]`, and the human reference. The evaluated
  `src/game/rule_ai.rs` SHA-256 is
  `42C117C7649F4FB7686190C40A30ADF8CDD19385133ADBB2BD1BA2E30677EA8D`.
- Verification passed `28/28` focused rule-AI tests and full `cargo test --all-targets` (`78/78`
  library tests plus every integration/example target). `cargo fmt --all -- --check` and
  `git diff --check` passed; only pre-existing unused import/variable and LF-to-CRLF warnings
  remain. The release benchmark and sweep examples were rebuilt successfully.
- The new global minimum consists of two `47 VP` seats and needs two separate follow-through rules.
  Seed `20260830` seat 1 placed a late Goods tile after one earlier Sell but lost the only shared
  Beer before its next turn. Seed `20260859` seat 2 did not make two Railroad product attempts;
  its decisive error was a final-Canal Cotton build at action 51 that secured no VP, flip, income,
  legal Sell, or route progress and was then cleared at the era transition. The earlier description
  incorrectly treated industry index `i1` as a product; `i1` is Iron.

## 2026-08-30 cross-turn Beer and late-Canal tail fixes

- The `20260830 -> [95,47,66]` tail was a cross-turn external-Beer race, not a generic failure to
  build products. At action 81, seat 1 had `14 VP`, income `+11`, GBP 37, and two draw-deck cards.
  It built Goods at `b14`, then used its second action to build route `r0`, exposing a legal Sell
  that depended on merchant Beer. Before seat 1 acted again, seat 0 built Brewery `b3` and sold its
  own Goods using `beerb3 + beerm5`; seat 1's Goods became unsellable and never flipped.
- Added `contested_external_beer_plan`. It applies only to a low-score, late-Railroad first-action
  product Build that cannot Sell immediately, whose remaining route action ends the turn, whose own
  Beer cannot cover the sale, and whose external Beer is also usable by an opponent's connected,
  unflipped product. The selected weight is `3.5`, the smallest screened value that changes the
  target decision from the exposed Goods Build to route preparation.
- After the protected route-first sequence, seat 1 successfully built and sold the Goods but then
  locally preferred another road over a second product. A global increase of
  `post_sale_product_cycle_weight` to `2.5` fixed that continuation but was rejected: it regressed
  seed `20260835` from a minimum of `82` to `76` and seed `20260865` from `63` to `58`. The stable
  default remains `0.75`. Instead, `mature_product_cycle_weight=1.625` adds the needed follow-up only
  when displayed income is at least `+10` and at least five cards remain after the Build. The two
  regression seeds then return exactly to baseline.
- Restored the validated simple semantics for `late_canal_stall` and selected weight `10.5`. In the
  final two-card Canal window for a player below `35 VP` and at income `-8` or worse, a non-Pass
  action is penalized only when it gains no actor VP, actor-owned flip, income, newly legal Sell, or
  sale-route progress. The later experimental spend/development/frontier exemptions were discarded
  because they changed `20260859` to `[16,94,72]` with ten Pass actions. The restored rule changes it
  from `[63,98,47]` to `[63,87,66]` with only one additional Pass.
- The final defaults are `contested_external_beer_plan_weight=3.5`,
  `mature_product_cycle_weight=1.625`, and `late_canal_stall_weight=10.5`. Human-reference seed
  `1508972731975190517` remains exactly `[115,90,91]`; fixed regressions `20260835`, `20260865`,
  and `20260867` also remain exactly at baseline. Target `20260830` becomes `[85,73,55]`, adding two
  Sell actions without adding a Pass, while `20260859` becomes `[63,87,66]`.
- On the paired 40-game benchmark beginning at seed `20260830`, the prior default measured `83.267`
  mean VP, `72.075` mean per-game minimum, `47` global minimum, `26` Pass, `578` Loan, and `305`
  Sell actions. The promoted candidate measured `83.375` mean VP, `72.675` mean per-game minimum,
  `48` global minimum, `27` Pass, `579` Loan, and `306` Sell actions. Zero-score players stayed at
  zero. Exactly two of 40 games changed, the two intended tails; the other 38 three-seat score rows
  were byte-for-byte identical. Throughput changed from `37.82` to `36.19 actions/s` in the paired
  local runs, a `4.3%` decrease within the five-percent guard.
- Final verification passed `cargo test --all-targets`, including `84/84` library tests, every
  integration target, `3/3` benchmark-example tests, and `2/2` sweep-example tests.
  `cargo fmt --all -- --check` and `git diff --check` passed; only pre-existing unused-code and
  LF-to-CRLF warnings remain. Rebuilt no-override release smokes reproduced the human reference,
  both repaired tails, and `20260862 -> [78,85,80]`. The evaluated `src/game/rule_ai.rs` SHA-256 is
  `5F592A5BB788CBB82463D220C3D6EFF597069EA2A256136DEC08A8FB12872EE0`.

## 2026-08-30 Beer-funded late product backlog fix (18:25 Asia/Shanghai)

- The historical `43 VP` tail at seed `20260862` remains fixed at `[78,85,80]`. The current
  baseline tail was `20260837 -> [69,82,48]`, where seat 2 took 15 Build, 9 Route, 7 Loan, and only
  1 Sell action. It finished with five Goods, roughly four unflipped products, income `-2`, and no
  Railroad Sell. This was conversion failure rather than a Pass chain.
- The first narrow experiment penalized a third late product with no legal Sell, but the user
  correctly identified that three Goods can be an efficient plan when one later Sell flips the
  whole batch. The promoted `late_railroad_backlog` signal therefore exempts a batch whenever the
  actor's own unflipped Beer covers the Beer demand of every pending product. It only prices a
  Railroad product Build when the deck is empty, the actor is below `35 VP`, has at most four cards,
  already owns at least two unsold products, has no legal Sell before or after the Build, lacks Beer
  for the batch, and gains neither an actor-owned flip nor displayed income.
- At target action 95, the original third Goods scored `6.4834` against DevelopDouble at `5.4758`.
  Weight `1.00` did not switch the decision; `1.01` is the smallest effective value and moves the
  Goods score to `5.4734`. The actor develops, then builds a Goods that opens a real sale frontier,
  sells it at action 105, and finishes at `56 VP`, income `+3`, with four Goods and two Sell actions.
  Its Loan count falls from seven to six. Weight `1.25` produced byte-identical 40-game behavior, so
  the lighter `1.01` value was selected to preserve legitimate batching.
- Fixed checks preserve human reference `1508972731975190517 -> [115,90,91]` and seeds
  `20260830 -> [85,73,55]`, `20260833 -> [93,61,55]`, `20260835 -> [84,88,82]`,
  `20260862 -> [78,85,80]`, `20260865 -> [74,63,66]`, `20260867 -> [74,81,83]`, and
  `20260868 -> [88,94,101]`. The target becomes `20260837 -> [69,77,56]`; a second matching tail,
  `20260859`, improves from `[63,87,66]` to `[64,96,69]`.
- On the paired 40-game benchmark beginning at seed `20260830`, the previous default measured
  `83.375` mean VP, `72.675` mean per-game minimum, `48` global minimum, `306` Sell, `579` Loan,
  `27` Pass, and `38.6245 actions/s`. The promoted Beer-aware `1.01` candidate measured `83.5083`
  mean VP, `72.900` mean per-game minimum, `55` global minimum, `308` Sell, `578` Loan, `27` Pass,
  and `38.6288 actions/s`. Exactly the two intended games changed; zero-score players remain zero.
  Aggregate Build falls `1367 -> 1363`, Route rises `1347 -> 1349`, and Develop rises `599 -> 600`.
- Verification passed the two focused Beer/backlog boundary tests and full
  `cargo test --all-targets`: `86/86` library tests plus every integration and example target,
  including `3/3` benchmark and `2/2` sweep tests. `cargo fmt --all -- --check` passed. Rebuilt
  no-override release smokes reproduce `[115,90,91]`, `[69,77,56]`, and `[64,96,69]` with
  `late_railroad_backlog_weight=1.01`. The evaluated `src/game/rule_ai.rs` SHA-256 is
  `EEB6ABC7487FFD3308F18EF00732B8D6BC7DC7F2189DEE4C4C7942398442B3FC`.
- This work used only the local Rust decision tree, release benchmark, and trace tools. No neural
  training, GPU workload, browser, or Playwright process was started. The existing dirty worktree
  and unrelated user changes remain preserved.

## 2026-08-30 final `20260833` diagnosis and stop (18:30 Asia/Shanghai)

- This is the newest authoritative stop handoff. The validated default from the preceding section
  is unchanged: `late_railroad_backlog_weight=1.01`, source SHA-256
  `EEB6ABC7487FFD3308F18EF00732B8D6BC7DC7F2189DEE4C4C7942398442B3FC`, 40-game mean
  `83.5083 VP`, mean per-game minimum `72.900`, global minimum `55`, and human-reference result
  `[115,90,91]`. This final pass was diagnosis only; it did not edit AI code or run neural
  training, GPU work, browser automation, or Playwright.
- Read-only traces were collected with `rule_ai_bench 1 3 20260833 trace trace_all` and
  `transition_probe 20260833 1 1 none frontier inspect=51`. In the current default game, seat 2
  finishes at `55 VP`, displayed income `0`, and GBP 9 after 9 Build, 12 Route, 8 Develop, 7 Loan,
  1 Sell, and 0 Pass actions. Six of its seven buildings flip; one `GoodsL4` remains unsold.
- The late chain is mostly locally defensible. Action 72 builds Pottery and action 81 sells it.
  At action 82, with `9 VP`, income `+3`, GBP 17, and a nearly empty draw deck, the repaired Goods
  build scores `8.6347` against route at `8.4830`, so the policy now takes the intended product
  action. Cash later forces a Loan at action 87. Action 88 prefers Beer (`7.7397`) over Goods
  (`7.1808`) and route (`5.3717`). Action 93's Coal is valuable because it flips immediately and
  raises income, scoring `27.7827` against route at `9.1907`; action 94 then builds a route.
- The first high-confidence low-conversion choice is action 100. Railroad is active, the draw deck
  is empty, only three hand cards remain, the actor has `9 VP`, income `0`, GBP 38, an unsold Goods,
  and no legal Sell. `DevelopDouble` scores `6.94125`, another Goods `6.84791`, and route `6.05920`.
  The selected Develop leads by only `0.88205`; actions 101 and 102 then build two routes without
  ever opening a Sell, and the Goods remains unflipped at the end.
- One earlier suspicion is explicitly closed: the Coal built at Canal action 51 is `CoalL2`, has
  `removed_after_phase1=false`, survives into Railroad, and ultimately flips. It is not a doomed
  phase-transition build and must not be used as evidence for a broad late-Canal penalty.
- The narrow next experiment, not yet implemented, is
  `late_unsold_product_setup_feasibility`. Gate it to Railroad, empty draw deck, low VP, an existing
  unsold product, no currently legal Sell, and at most three hand cards. Apply it only to Develop or
  DevelopDouble when an exact lower bound for the remaining Build/Route/Beer/Sell actions exceeds
  the remaining card budget and the candidate itself gains no VP, flip, income, or legal Sell.
  Emit a `-1` signal and screen weights `0.9`, `1.0`, and `1.25`; `0.9` is already large enough to
  cross action 100's observed `0.88205` gap. Do not globally raise conversion-feasibility scoring
  or penalize product Builds and Loans, because that would threaten valid batch-product sales,
  multi-route setups, and the human-reference line.
- Promotion gates for that experiment are: improve `20260833` above `55` and reduce or sell its
  stranded Goods; preserve the human reference and seeds `20260830`, `20260835`, `20260837`,
  `20260859`, `20260862`, `20260865`, and `20260867`; add neither Pass nor Loan actions; keep the
  40-game mean at least `83.375`, mean per-game minimum at least `72.675`, and global minimum no
  lower than the current `55`; and keep throughput regression within five percent.
- No project-local training, benchmark, sweep, or probe process remains active. The existing
  `fast_brass` application server on port 3001 and its container were left running because they are
  not training jobs and may be in use. The dirty worktree and all diagnostic artifacts were
  preserved without reset, cleanup, deletion, commit, or revert. Work stops here at the user's
  request.

## 2026-08-30 three-Goods Beer-aware batch planning (20:32 Asia/Shanghai)

- A third Goods tile is no longer treated as intrinsically bad. The preserved rule is action
  efficiency: several products are a valid batch when one eventual Sell can flip all of them and
  the remaining cards can secure every required Beer unit. The allocator continues to assign Beer
  per product and per source, so neither a merchant barrel nor a Brewery cube can be reused.
- `batch_sale_plan_is_feasible` now expands real actor-owned Brewery builds as well as real single
  and double railway actions. It reserves one card for Sell and searches only within the remaining
  hand. A new boundary test uses Goods II + III + V, whose Beer demand is `1 + 0 + 2 = 3`: one
  existing Brewery plus two legal Brewery builds and one Sell succeeds with three cards, while the
  same position correctly fails with only two cards. Existing tests still prove that three
  zero-Beer Goods behind one shared route are preserved and that separate route branches are not
  collapsed into a false one-route plan.
- The opt-in `financed_product_cycle` experiment exposed two runner-boundary bugs. `LegalAction::apply`
  automatically ends a completed turn and opens the next player's turn, so checking for zero
  remaining actions rejected every valid Loan. Also, an actor can move from first in one turn
  order to last in the next, requiring up to two sets of opponent turns before it acts again. The
  projection now handles both cases, with regression tests, and does not apply income settlement a
  second time.
- The corrected external-Beer experiment at weight `0.28` changed seed `20260830` from
  `[87,73,55]` to `[92,65,66]`, but it was rejected. Across 40 games it produced mean VP `83.7000`,
  mean per-game minimum `73.200`, global minimum `56`, Sell `312`, Loan `586`, and Pass `24`; the
  eight additional Loans and human-reference regression to `[107,91,74]` fail promotion gates.
  Requiring actor-owned Beer was still insufficient as a generic positive reward: the human
  reference became `[101,79,78]`. Therefore `financed_product_cycle_weight` remains `0.0`.
- With only the Beer-aware batch feasibility correction active, the paired 40-game benchmark is
  byte-for-byte identical in aggregate behavior to the retained default: mean VP `83.6833`, mean
  per-game minimum `73.050`, global minimum `55`, Sell `309`, Loan `578`, Pass `27`, and no zero-VP
  players. Measured throughput was `36.2646 actions/s`. Human reference remains `[115,90,91]`;
  fixed checks include `20260830 -> [87,73,55]`, `20260833 -> [93,61,56]`,
  `20260837 -> [69,77,56]`, `20260859 -> [64,96,69]`, and `20260862 -> [78,85,80]`.
- Verification passed `cargo test --all-targets`: `97/97` library tests plus every integration and
  example target. `cargo fmt --all -- --check` and `git diff --check` passed; only pre-existing
  unused-code and LF-to-CRLF warnings remain. The evaluated `src/game/rule_ai.rs` SHA-256 is
  `291B96E6E857AE873A36DB2FD34CB7A07274961F80B4E238B85D69C629E49B61`.
- This cycle used only the local Rust decision tree, release benchmark, and transition probe. It did
  not start neural training, GPU work, a browser, Playwright, or a new application server.

## 2026-08-30 low-cost loan and Beer-backlog recovery (22:46 Asia/Shanghai)

- Promoted the narrow `low_cost_loan_runway_weight=14.8` rescue. It applies only in Railroad after
  the deck is empty, below 20 VP, with at most four buildings, displayed income `0..=1`, an actual
  loan loss of at most three, GBP `15..=20`, at least five cards after the loan, and a conservatively
  funded single-rail-plus-product follow-up. Seed `20260830` changes from `[87,73,55]` to
  `[85,70,63]`; the mature six-building false positive is excluded, so `20260864` remains
  `[77,60,84]`. The human reference remains `[115,90,91]`.
- Preserved the user's action-efficient three-Goods strategy. Beer demand is accumulated per
  product and allocated against shared Brewery/merchant capacities, so no cube or merchant barrel
  can be reused. Route and Brewery actions are really applied, the final Sell card is reserved, and
  multiple products behind one shared route can still be completed by one Sell.
- Fixed a batch-planner pruning bug: discard-invariant route deduplication could keep the route
  variant that discarded the only Beer card, incorrectly rejecting a valid
  `Route -> Brewery -> Sell` plan. The small late-game planner now retains exact discard-card
  branches. The regression position proves `r16` with a Coal discard, `Beer b47`, then selling
  `Goods b36`; the old pruning returned false and the fixed planner returns true.
- Added and promoted `beer_backlog_recovery_weight=2.18`, the smallest effective fixed-seed value.
  It does not reward Beer development generically. The actor must be in Railroad below 35 VP, own
  at least two unsold products with no legal Sell and a Beer shortfall of at least three, retain at
  least five cards, and the exact Beer development must change an unavailable Brewery build into an
  affordable one. Zero-Beer Goods and already Beer-funded three-Goods batches are unchanged.
- In seed `20260837`, two old Goods need four Beer. The old policy double-developed Cotton; the new
  policy unlocks Beer II, builds the required Breweries, and converts the backlog. The result changes
  from `[69,77,56]` to `[72,77,78]`. A second beneficial trigger changes `20260869` from
  `[81,82,77]` to `[77,96,78]`. Seed `20260833` remains `[93,61,56]`; its remaining 56-point tail
  is a cross-turn Beer/route commitment problem involving one two-Beer Goods, not excessive Goods
  count.
- On the paired 40-game benchmark beginning at `20260830`, the post-loan baseline measured mean VP
  `83.7083`, mean per-game minimum `73.25`, global minimum `56`, Sell `309`, Loan `578`, and Pass
  `27`. The promoted Beer recovery measured mean VP `84.0083`, mean per-game minimum `73.65`, global
  minimum `56`, Sell `308`, Loan `576`, and Pass `27`. Exactly two games changed, total actions
  remained `4280`, zero-score players remained zero, and throughput did not regress.
- Final verification passed `cargo test --all-targets`: `100/100` library tests plus every
  integration and example target. Rebuilt no-override release smokes reproduced `[115,90,91]`,
  `[85,70,63]`, `[72,77,78]`, and `[77,96,78]` with the promoted default. The evaluated
  `src/game/rule_ai.rs` SHA-256 is
  `A69B0DF3DA88FB870618BA28B67BA764E6A42914473359AF8E54B2E6AF840390`.
- This work used only the local Rust decision tree, tests, release benchmarks, and probes. Docker
  was limited to five of twenty logical CPUs and the simulations used roughly one core. No neural
  training, GPU workload, browser, Playwright, or application-server restart was used.

## 2026-08-30 project closure and BrassForge assessment

- Active AI development is paused and this repository is the final handoff. The implementation,
  tests, benchmark tools, expert-data pipeline, promotion gates, and full experiment record remain
  available for inspection or a future restart. No local experiment data was deleted; root-level
  `.tmp-*` traces and sweep directories are now ignored so they cannot enter a commit accidentally.
- A public-source check found no BrassForge repository or matching published package on GitHub,
  GitLab, npm, or crates.io. Its approximately 1.5 MB minified browser bundle did not expose source
  maps. The observed WebSocket traffic provides game state and action records, while bot selection
  is performed by the service. The implementation should therefore be treated as closed source
  unless its authors publish it later.
- The supplied completed-game log is useful as one strategic example, but it does not contain the
  bot's rejected candidates, evaluations, search statistics, or internal state representation.
  Reproducing that policy from behavior alone would require many consented games, a normalized
  state/action corpus, behavior cloning or hand-authored imitation, and repeated head-to-head
  evaluation. That is a longer research project, not a quick transfer of BrassForge's playbook.
- The retained deterministic rule AI is the best reproducible local result: a paired 40-game,
  three-player benchmark beginning at seed `20260830` measured `84.0083 VP` mean, `73.65` mean
  per-game minimum, `56 VP` global minimum, `308` Sell, `576` Loan, and `27` Pass actions. The human
  reference seed `1508972731975190517` remains `[115,90,91]`. These numbers measure regressions
  inside this engine and are not directly comparable with BrassForge's rules, seeds, or opponents.
- The final promoted rules preserve low-cost late loans only when their actual income loss is small,
  plan multi-product sales with non-reusable Beer capacity, retain exact discard-card branches for
  short route/Beer/Sell plans, and recover Beer-starved product backlogs without broadly rewarding
  extra loans or surplus Goods. The corresponding source SHA-256 is
  `A69B0DF3DA88FB870618BA28B67BA764E6A42914473359AF8E54B2E6AF840390`.
- Neural and expert-iteration infrastructure is preserved but no neural candidate is claimed to be
  strong-human or BrassForge level. A future restart should begin with a legally obtained external
  action corpus and a fixed cross-engine evaluation protocol; more blind self-play or isolated
  heuristic tuning is not justified by the evidence collected here.
- Closure verification passed `cargo fmt --all -- --check`, `cargo test --all-targets` (100 library
  tests plus every integration and example target), `python3 -m unittest discover -s training/tests
  -v` (126 tests), `npm run build`, and `git diff --check`. Rust and Python checks reused the
  existing training container with bounded CPU threads and CUDA disabled for Python; no training,
  benchmark sweep, browser, Playwright, or new service was started.

## 2026-09-06 economic rule AI restart

- Replaced the default August action-bonus evaluator with `rule_economic_conversion_v1`.
  Completed VP, Canal carryover, route/Beer conversion work, shared Beer capacities, resource
  demand, diminishing cash utility and debt reserves now determine the investment score.
  `RuleDecisionConfig::legacy()` preserves the August policy for comparison.
- Development checks and optional same-turn continuations exclude unobserved draws. New tests
  verify invariance to hidden-card permutation, two-era accounting, terminal double-counting,
  merchant/owned Beer allocation, batch sales and disconnected resource risk.
- Added `rule_ai_match`, which rotates one candidate against legacy opponents through every seat,
  reports official wins and VP, and groups confidence intervals by seed. A six-game all-legacy
  control reproduced identical per-seed score vectors and exactly 1/3 win share.
- Frozen final policy on untouched seeds `2026092600..2026092623`: 2P 47/48 wins, 134.4375 mean
  VP, +46.25 margin; 3P 63/72 wins, 118.3889 VP, +26.7083 margin; 4P 64/96 wins, 100.875 VP,
  +8.8646 margin. Every grouped 95% margin interval is above zero. No games were excluded.
  The remaining four-player zero at seed `2026092610`, seat 2 is explicitly retained.
- Economic module SHA-256 is `2A759829AFD6B72ACBEA1ED077DFF5BB7AA505E0A6073A3EB336B8D37AD38395`.
  Complete reports remain in `output/economy-final-{2,3,4}p.json`; methodology and limitations are
  in `docs/rule-ai-economy.md`. Final release rebuild reproduced the first six 3P evaluation games.
- The CPU implementation needs no checkpoint or GPU. Neural weights and rules-engine behavior
  were not changed. The result demonstrates improvement over the retained policy in this engine;
  it does not establish expert-human strength or certify official-rule completeness.
- Local preview is running at `http://127.0.0.1:5175/` with API port 3011 and an independent
  playtest database. Browser/API smoke passed five analyzed and applied moves in each player
  count, the economic method label renders correctly, and browser console errors are zero.

## 2026-09-06 rulebook audit and economic planning v2

- User rejected weak-opponent win rates as sufficient evidence of skill. Reproduced the four-player
  zero at 2026092610/seat 2: isolated Coal/Brewery investments, repeated loans, no conversions,
  then Railroad passing and liquidation. Read the official rulebook, including its scoring diagram.
- Corrected loan income-band arithmetic and the -7 loan eligibility floor, end-of-turn card refill,
  one setup discard per player, Scout with any held Wild, printed merchant link icons, and displayed
  income tiebreaks. Shared scoring/ranking helpers also serve Python and training consumers.
  New passing-only tests verify 10/9/8 rounds per era and exact action budgets for 2/3/4 players.
- V2 combines pending income on one nonlinear track, prices credit exhaustion, discounts disconnected
  resources, anticipates unflipped link icons and merchant Beer after a planned route. Default search
  now considers the second action for eight distinct root intents, with strict turn/era boundaries
  and hidden-card invariance. V1 remains available as RuleDecisionConfig::economic_v1().
- Independent direct matches, 24 new seeds 2026092800..2026092823 per player count: mean candidate
  VP 150.0625 / 136.6944 / 120.9583; wins 38/48, 49/72, 41/96; minima 113/101/91; zero shortfall
  sessions. Four-player margin versus the best of three opponents is -0.125 with CI crossing zero.
  Five top-score ties were replayed after the income tiebreak correction; scores reproduced, and
  2026092813/seat 0 correctly changed from a win to a loss. The JSON and documentation reflect it.
- Independent all-seat self-play, eight new paired seeds 2026092900..2026092907: V1 mean VP
  120.6875 / 121.75 / 103.875; V2 151.4375 / 133.125 / 116.6875. V2 minima 126/113/81;
  below-100 counts 0/16, 0/24, 2/32. No zero scores. The three-player paired mean-gain CI crosses
  zero on eight games; this and the weak four-player tail are explicit limits, not excluded games.
- The complete Rust suite with Python bindings passed: 128 library tests plus all integration and
  example targets. API smoke applied ten analyzed actions at each player count; browser smoke
  created a game, executed an AI turn, and displayed v2 without console warnings/errors.
- Preview remains on 5175 with API 3011. Use new games: old action-log saves and checkpoints have
  not been migrated to the corrected engine. No neural training or checkpoint promotion occurred.
  Details and raw evidence paths are in docs/rule-ai-economy.md and docs/official-rules-audit.md.

## 2026-09-06 city card map preview

- Added location-card hover and keyboard-focus previews, resolving the two compact Rust city
  names to their board coordinate keys. The canvas outlines the whole city's slot group in
  cyan and labels the city above it, including while action or analysis highlights are visible.
- Previews clear on pointer leave, focus loss, window blur, hand replacement, and teardown.
  Industry and wild cards do not point at a single city. Card-face lift preserves the button's
  hit area, and the preview respects reduced-motion preferences.
- Exposed card_preview_town through the existing render_game_to_text hook.
- Screenshot QA caught the existing selection overlay erasing the board along with its mask.
  It now cuts windows in a separate reusable mask canvas, preserving tiles and overlapping
  selectable areas underneath the green borders and the new city preview.
- Validation passed: production build; all 20 city mappings; pointer, keyboard, non-city cards,
  hand replacement, blur cleanup, legal/illegal card clicks, animation, and reduced motion.
  Pixel QA confirmed the preview only changes the matching city area and that selection windows
  preserve the original board pixels, including overlapping slots. Screenshots reviewed at
  1440x1000, 390x844, and 320x740; no horizontal page overflow or browser console errors.
- Playwright fixtures intercept game mutations; the running backend game was not changed.
  Evidence and the bounded browser script are in output/city-card-preview/. The standard
  develop-web-game client also passed with its state/screenshot artifacts in client-final/.
- The updated local frontend is available on http://127.0.0.1:5176/ with API port 3011.
  No remaining TODOs for this hover-preview change.

## 2026-09-06 gameplay UX audit

- Reproduced a cancelled industry animation submitting into a newly started Build action,
  approximately 972 ms of fixed input delay, missing modal focus/Escape handling, invisible
  industry art on phones, and a discard Next button outside the 390 px viewport.
- Industry choices now submit immediately and invalidate continuations on close. Chained
  development keeps the panel available; explicit projected-tile dependencies prevent a
  Svelte update error when a development changes the displayed level and stack count.
- Added a shared native-dialog action for focus, Escape, backdrop dismissal, and body scroll
  restoration. Industry browsing uses the entire card, with an explicit back control and no
  nested buttons. Both main and expanded industry views have bounded responsive scrolling.
- The discard viewer now uses card assets and friendly names, a position counter, boundary
  states, working keyboard/wheel navigation, and a layout that fits 320 px and 390 px screens.
- Actual UI/API verification on independent port 3012 passed Build, Undo, Loan, End Turn,
  Develop x2, slow-response cancellation/restart, retry after a deliberate failure, focus,
  and mobile layout checks. Final local request dispatch: 2 ms; browser console errors: zero.
- Evidence is in output/ux-audit/. See docs/ui-playtest-2026-09-06.md for reproduced issues
  and the remaining design opportunities: board details, touch zoom, mobile action placement,
  and consistent language.
- Final production build passed without warnings; the 20-city hover regression and standard
  web-game client startup check passed. Cross-level pending development counts and live
  expanded-view updates were also verified. Temporary test backend 3012 is stopped; the
  updated preview remains available at http://127.0.0.1:5176/ using the normal API on 3011.

## 2026-09-06 board inspection and navigation

- Added board-object details for built industries, links, merchants, and legal empty slots.
  Details use current game state and existing industry data; they do not estimate link scoring.
- Added a bounded 1x-4x camera, wheel and button zoom, pointer drag, pinch gestures, fit reset,
  and keyboard target selection. Pointer release submits only a stationary, single-pointer
  gesture in the same choice state. Replay and AI-controlled views explicitly lock mutation.
- Canvas rendering, highlight masks, hit testing, and the game text hook now share camera
  coordinates. City-card previews reveal off-screen towns while preserving the zoom level.
- Mobile board height follows available width and action panels precede player/market details.
- Browser verification passed: zoomed mouse/keyboard/touch selection, drag and pinch
  suppression, detail pin/close, pointer-anchored zoom, reset, zoomed city reveal, replay and
  AI locks, desktop/mobile canvas pixels, and action placement. Native Chromium touch input
  covered the two-finger gesture. Screenshots at 1440x1000, 390x844, and 320x740 were reviewed.
- The actual-game regression passed Build, Undo, Loan, End Turn, and Develop x2 with the new
  camera mapping. The 20-city preview regression also passed. All browser runs had zero
  console errors. Evidence: output/map-navigation/, output/ux-audit/, and output/city-card-preview/.
- Fixed explicit Svelte dependency ordering after card-driven camera movement so the canvas,
  detail anchors, and render_game_to_text camera state update in the same render.
- Full industry data also loads when inspecting a replay directly. Keyboard target details
  are announced through the existing live status region; closing pinned details restores focus.
- Final production build and standard web-game startup client passed. Temporary backend 3012
  is stopped, and the normal preview remains on http://127.0.0.1:5176/ with API port 3011.

## 2026-10-10 next training round (active)

- PR #1 is merged at 8c69940; next work isolated on codex/brass-next-training.
- Public native replays: six four-player human-vs-AI games fully legal and terminal, 372 human action-intent rows (248/62/62 whole-game train/validation/test). Expert status is unverified. Foreign final scores never become JS value targets.
- Human public-feature MLP trained (best epoch 11); classification is not playing strength. Bounded optional root prior and settlement diagnostics being integrated and validated.
- Old prediction holdout is now development diagnosis: next-player accuracy ~86%; new independent seeds required after candidate freeze.
- Local private SQLite audit found no clean, complete usable trajectory. No private raw DB published.
- Remaining: audit integrity/privacy/parity tests; development prior ablation; critical-field world training; freeze and independent games/predictions; document artifacts, push draft PR. No default promotion without evidence.

## 2026-10-10 candidate freeze and independent evaluation

- Public human prior selected weight .5 on four development seed groups. Critical categorical controls trained 100 epochs; prior prediction test is now development evidence (~86% -> 95% current-player exactness).
- Frozen 53 source files and candidate models before fresh final seeds 217122949 (strategy), 218122949 (world), 219122949 (prediction). Protocol and frozen source archive saved in public-human-20261010.
- Running 360 strategy games, 40 world comparison games, 40 new prediction trajectories. No tuning after final freeze. Results pending, no default promotion.
- Tests: full 83 Node tests passed, then 2 new critical-control parity/pure-world tests passed; all 15 Python tests passed. Browser skill client plus teacher controls smoke passed; preview/execution screenshots inspected, no console errors.
- Reference native replay audit includes exact source commit/tree and compiled JS module hashes. Private SQLite audit publishes aggregate anomalies only; zero accepted complete training trajectories.

## 2026-10-10 next round complete (delivery pending)

- 400 frozen games completed, plus 40 new independent synthetic prediction games / 4,960 transitions. All 400 recorded game traces replay-audited: 220 distinct seed/lineups, 27,280 legal actions, every terminal score and pre-settlement statistic matched.
- Human intent vs three current teacher+cards: 131.433 VP, paired gain +3.667 CI [.10,7.117]. Selfplay 129.025 vs 127.767; gain +1.258 CI [-1.675,4.183], unconfirmed. Selfplay under100 2.5% vs 6.67%; >=140/150 fractions fell. Not stable140/150, no default promotion.
- World control new/old mixed means 48.7/50.3, gain -1.6 CI [-12.625,8.225]; no playing-strength improvement. Independent next-player exactness 86.45% -> 95.04%, round 85.28% -> 97.02%. One-step control accuracy does not establish game improvement.
- Public provenance, derived corpus, frozen models/sources, all compressed development/final games, new prediction data and validation manifests published in public-human-20261010. No private raw records or foreign implementation copied.
- Tests: 85 unique Node tests (full suite 83 + new control 2), 15 Python tests; browser standard client and teacher controls, actual screenshot inspection, zero console errors.
- Documentation: docs/public-human-training-results.md. Optional prior remains CLI-only; browser still previous teacher. Next investigations: action-specific board/resource/supply heads, stochastic future-card representation, more verified human games; use new holdouts if tuning again.
- Remaining delivery: commit/push codex/brass-next-training, create draft PR, verify remote. Original Manila checkout untouched; PR #1 already merged.
