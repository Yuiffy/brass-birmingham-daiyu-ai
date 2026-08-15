Original prompt: 我想找一个工业革命伯明翰的顶级AI进行人机对练或者观察ai局，来提升我的伯明翰实力。你帮我找找有没有现成的，没有的话找找有没有开源游戏我们拿来自己训练，还没有的话我们看规则书开发一个游戏然后练ai。我们项目里有ai训练经验。

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
