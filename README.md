# Brass Birmingham Daiyu AI

> **仓库整合（2026-10-09）：** 原 `brass-birmingham` 与 `brass-birmingham-sim` 的
> JavaScript 游戏、自动对局、世界模型、训练权重和实验报告已合入
> [brass-sim/](brass-sim/README.md)，并保留原提交历史。运行
> `npm --prefix brass-sim start -- --port 8086`，打开 `http://127.0.0.1:8086`；
> 经济规则校准版入口为 `http://127.0.0.1:8086/?rules=economy-v2`。
> 详见[整合说明](docs/repository-consolidation.md)。

`Brass Birmingham Daiyu AI` is an unofficial browser game and AI analysis project for
**Brass: Birmingham**. It is a public AGPL-3.0 fork of
[`artyom-morozov/fast_brass`](https://github.com/artyom-morozov/fast_brass), extending the Rust
rules engine with model-guided search, self-play training, and an interactive Svelte interface.

The project provides transparent analysis for two-, three-, and four-player games: each position
can show several candidate moves, the search share and model signals behind each move, and
contextual answers to Chinese follow-up questions such as why the first choice is preferred over
the second.

> **AI update (2026-09-06): economic planning v2.** The default CPU opponent plans both actions
> of its turn, budgets shared future income and loan capacity, and considers merchant Beer and
> future link scoring. The rulebook audit also fixes loans, card refill/setup, scouting, merchant
> link icons and income tiebreaks. Start a fresh game: historical replays used different rules.
> Both `RuleDecisionConfig::economic_v1()` and `RuleDecisionConfig::legacy()` remain available
> for comparison. See [strategy and evaluation](docs/rule-ai-economy.md) and the
> [rules audit](docs/official-rules-audit.md). Self-play scores are not a human skill rating.

## What This Repo Contains

- A Rust game engine for the main Brass: Birmingham action system
- Validation logic for legal builds, links, sales, development, loans, scouts, and passes
- A local web server and browser client for manual play, AI observation, and debugging
- Batched, multi-layer neural PUCT search over hidden-information determinizations
- Top-N move analysis with visits, policy priors, backed-up values, uncertainty, and search depth
- Contextual Chinese explanations that compare concrete candidate moves
- Deterministic self-play, PyTorch policy/value training, and candidate promotion tooling
- SQLite-backed persistence for saved games
- Optional Python bindings via `PyO3` / `maturin`
- Integration and regression tests for gameplay edge cases


![Board UI view](docs/screenshots/board-overview.png)


## Quick Start

### Run the Rust backend

```bash
cargo run
```

The backend serves on `http://localhost:3000` by default.

### Run the full local playtest stack

```bash
make dev
```

This starts:

- the Rust backend on port `3000`
- the Svelte frontend in `ui/` on port `5173`

You can also run them separately:

```bash
make server
make client
```

### Run tests

```bash
cargo test
```

### Build Python bindings

```bash
maturin develop
```

This exposes the `fast_brass` Python module when the `python-bindings` feature is enabled through `maturin`.

### Run the CUDA training stack on Windows

The checked-in Docker stack provides Rust 1.88, Python 3.11, Torch 2.11/cu128, NumPy, the local
training workspace, and an optional CUDA inference service. Docker Desktop must have NVIDIA GPU
support enabled.

Create the reusable Cargo cache volumes once, then build and start the training container:

```powershell
docker volume create brass-cargo-registry
docker volume create brass-cargo-git
docker compose --file docker/training-gpu.compose.yaml up --detach --build training-gpu
```

Build the Linux PyO3 extension inside that container. Linux Cargo artifacts are isolated under
`target/linux` so they do not replace native Windows builds.

```powershell
docker compose --file docker/training-gpu.compose.yaml exec training-gpu cargo build --release --features python-extension
docker compose --file docker/training-gpu.compose.yaml exec training-gpu cp target/linux/release/libfast_brass.so fast_brass.so
```

Start the checkpoint inference service used by the browser backend:

```powershell
docker compose --file docker/training-gpu.compose.yaml up --detach inference
```

It publishes `http://localhost:8765`, includes a health check, and defaults to
`output/champion-iter1.pt`. Change the checkpoint in
`docker/training-gpu.compose.yaml` when promoting a new champion.

### Play against the local AI on Windows

With the local first-version checkpoint available at `output/champion-iter1.pt` (a generated
artifact that is not committed), start the inference service first:

```powershell
docker compose --file docker/training-gpu.compose.yaml up --detach inference
```

In a second PowerShell terminal, start the Rust API with neural inference enabled:

```powershell
$env:FAST_BRASS_INFERENCE_URL = 'http://127.0.0.1:8765'
$env:WEB_NEURAL_BATCH_SIZE = '64'
cargo run --release
```

In a third terminal, start the browser client:

```powershell
Set-Location ui
npm run dev
```

Open `http://127.0.0.1:5173`, create a two-, three-, or four-player game, and enter the `AI 分析` tab. The analysis
panel can show the Top 3 candidate moves, answer follow-up questions about the selected move, apply
one recommendation, or advance the AI side one move at a time.

The control bar defaults to `人机对练`. Choose `Coade` or `Brunel` under `你的席位`; the other
seat is then controlled by the AI. Your hand stays visible during the opponent's turn, while the
AI's private cards remain hidden.

On your turn:

1. Use `对局` to make the move manually, or open `AI 分析` for coaching.
2. Choose `快速`, `标准`, or `深入`, then click `分析局面`.
3. Inspect the first, second, and third choices. Select any row to see its exact action and search
   statistics.
4. Ask `选择依据`, `胜率可信度`, `主要风险`, or enter a temporary free-form question such as
   `为什么一选比二选好？`.
5. Click `采用此步` if you want the AI to play that recommendation for you.

On the AI's turn, the interface opens `AI 分析` and starts the AI automatically. Use the controls under `AI 对局` when you want to intervene:

- Play resumes the complete AI turn and pauses as soon as control returns to your seat.
- Pause keeps the current Top 3 visible so you can ask a question before the move is applied.
- Step analyzes without moving when no report exists; press it again to apply the current first
  choice exactly once.
- Stop clears the playback session.

Saved games also have a history icon in `Join Game`. Open it to enter `棋谱回放`, where you can
move through every recorded position with the timeline, previous/next controls, or the slider.
Each position keeps the board state, the applied action and its selections, and any AI analysis
that was available at that moment, including the first, second, and third recommendations. New
games record these snapshots automatically as actions are analyzed or applied. Saves created
before replay recording was added may not contain snapshots and will show an empty-history notice.

Use `AI 观战` to let the AI control both seats, or `全部手动` to operate every seat yourself.
Search visit share, backed-up model value, and policy prior are separate estimates; the displayed
model value is not yet a calibrated real-world win probability.

## AI Status

- The default rule policy uses economic conversion evaluation, with exact legal actions and
  explainable score components. Its development and optional same-turn planning use only known
  hand cards. No model checkpoint or GPU is required. The old rule policy and neural pipeline
  remain available for comparison. See [the rule-AI evaluation protocol](docs/rule-ai-economy.md).
- Deep neural search uses hidden-information determinizations and batched multi-layer PUCT for
  two-, three-, and four-player games. In multiplayer positions, the value head's non-root win
  mass is distributed across the other seats as an explicit approximation.
- Search visit share, policy prior, and backed-up model value are different signals. The UI keeps
  them separate instead of presenting them as one probability.
- Training now uses separate self-play shards for optimization and validation. This prevents
  positions from the same game leaking into both sets, but the value head has not yet been
  probability-calibrated or benchmarked against strong human play. Displayed model values must not
  be interpreted as reliable real-world win probabilities.
- The deployed champion has beaten only the project's initial smoke baseline. A later candidate won
  60% of a 40-game, seat-rotated PUCT match, but its paired 95% confidence lower bound remained below
  zero, so it was correctly not promoted. The project does not yet claim top-level strength.
- Model checkpoints and generated self-play data are local artifacts under `output/` and are not
  included in this repository. Set `FAST_BRASS_INFERENCE_URL` to a compatible inference service to
  enable neural analysis; an unset URL retains the non-neural search path.

### Generate model-guided self-play

After installing the Python extension and a training checkpoint, generate a deterministic JSONL
shard with checkpoint priors and Rust root PUCT search:

```bash
python -m training.model_self_play \
  --output output/model-self-play.jsonl \
  --checkpoint output/champion.pt \
  --games 10 \
  --players 2 \
  --simulations 800 \
  --device auto
```

The exporter refuses to overwrite an existing shard. It writes to a `.partial` file and only
renames it after every game and terminal value target has been written successfully.

When PyTorch runs on a different host or outside the rules-engine container, replace `--checkpoint`
with `--inference-url http://HOST:PORT`. Remote inference is schema-checked, model-ID-checked, and
retried on bounded transient transport failures.

For CUDA generation, use independent local processes so Rust search and Torch inference overlap
without HTTP/JSON transport. Stop the optional inference service first so its CUDA allocator does
not retain training memory, then restore it after generation. The Rust/CUDA training path does not
start or require the browser, Svelte, or Playwright:

```powershell
docker compose --file docker/training-gpu.compose.yaml stop inference
docker compose --file docker/training-gpu.compose.yaml exec training-gpu python3 -m training.parallel_self_play --output-prefix output/iter3-train --checkpoint output/champion-iter1.pt --games 100 --workers 4 --players 2 --simulations 256 --search-determinizations 4 --inference-batch-size 64 --device cuda --seed 2026081601
docker compose --file docker/training-gpu.compose.yaml up --detach inference
```

This writes one atomic JSONL shard per worker, such as `iter3-train-000.jsonl`. Global game indices
are assigned before work is split, so changing `--workers` does not change which deterministic games
the requested index range represents. Use a different `--seed` for validation data. Four workers
are the measured default for an RTX 5080 with 16 GB VRAM; lower the count on smaller GPUs.

### Produce high-quality expert data (fast path)

The faster route than repeatedly fine-tuning on a tiny human replay is **human-prior-guided
self-play plus expert iteration**. The checkpoint (already adapted to audited human choices) and the
versioned strategy prior act as a teacher. The producer runs several temperatures/prior strengths,
then applies rejection sampling: incomplete games, zero-score games, unhealthy income/cash, and
known lifecycle violations are discarded; only top actors above the declared VP/margin threshold
become teacher trajectories. This is also called quality-filtered trajectory generation or
Best-of-N self-play.

One command creates raw shards, filtered teacher shards, an untouched-seed validation partition, and
an audit manifest that can be passed directly to `training.train`:

```bash
python -m training.produce_expert_data \
  --checkpoint output/champion-iter3-mixed-v5-v6.pt \
  --output-dir output/iter19-expert \
  --train-games 128 \
  --validation-games 16 \
  --workers 4 \
  --players 2 \
  --simulations 256 \
  --search-determinizations 4 \
  --strategy-prior-version human-strategy-v6-resource-aware \
  --strategy-prior-strengths 0.65 0.85 \
  --selection-temperatures 0.35 0.70 \
  --demonstration-shards output/human-replay-game6-audited.jsonl \
  --demonstration-prior-strengths 0.25 0.40 \
  --minimum-actor-vp 80 \
  --minimum-vp-margin 0 \
  --device cuda
```

The command is local Rust/Python/CUDA only. It never starts the inference service, browser, Svelte,
or Playwright. Inspect `output/iter19-expert/manifest.json` before training; use its `train` shards
as the teacher pool, its `validation` shards only for validation, and keep the old champion as the
replay pool. Raise `--minimum-actor-vp` only after a pilot shows that the acceptance rate leaves
enough diverse positions; a single spectacular score is not a reliable label.

`--demonstration-shards` enables the optional human-demonstration prior. It learns smoothed
phase/round/action-intent frequencies from confirmed policy-only positions, blends them into the
checkpoint root prior, and never bypasses Rust legality or search. Use a small strength (for example
`0.25`--`0.40`) first; include `0.0` as a control recipe when comparing the effect. The shard's
terminal value labels are not used by this prior, so a replay with contaminated turn markers can
still provide policy guidance while remaining value-masked during learner training.

### Fast expert-iteration fine-tune

Use the current PUCT teacher as `--shards`, retain older champion trajectories with
`--replay-shards`, and add audited human-policy positions with `--human-shards`. The loader uses
replacement sampling to keep the requested source fractions visible in every epoch; contaminated
human replay automatically contributes policy loss only. This command runs entirely in the local
Python/Rust/CUDA training container:

```bash
python -m training.train \
  --shards output/iter18-guided4e-v6-quality-000.jsonl output/iter18-guided4e-v6-quality-001.jsonl output/iter18-guided4e-v6-quality-002.jsonl output/iter18-guided4e-v6-quality-003.jsonl \
  --replay-shards output/iter15-champion-v5-train-000.jsonl output/iter15-champion-v5-train-001.jsonl output/iter15-champion-v5-train-002.jsonl output/iter15-champion-v5-train-003.jsonl \
  --human-shards output/human-replay-game6-audited.jsonl \
  --replay-fraction 0.30 \
  --human-fraction 0.05 \
  --validation-shards output/iter18-champion-v6-256-val-000.jsonl output/iter18-champion-v6-256-val-001.jsonl output/iter18-champion-v6-256-val-002.jsonl output/iter18-champion-v6-256-val-003.jsonl \
  --resume output/champion-iter3-mixed-v5-v6.pt \
  --output output/candidate-expert-iteration.pt \
  --epochs 4 \
  --early-stopping-patience 2 \
  --learning-rate 1e-5 \
  --device cuda
```

Keep the champion immutable until the staged Rust evaluation gate passes. Playwright is reserved
for an optional UI release smoke test after a model is selected; it is never a training dependency.

### Train with an independent validation partition

```bash
python -m training.train \
  --shards output/train-*.jsonl \
  --validation-shards output/validation-*.jsonl \
  --output output/candidate.pt \
  --resume output/champion.pt \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.001 \
  --device cuda
```

Training rejects overlapping paths, incompatible schemas or engine revisions, and reused recorded
game seeds across the two partitions. Validation metrics are weighted by position rather than by
batch, and training saves the state with the lowest independent validation loss (including the
resumed baseline) rather than blindly keeping the final epoch. The old position-level
`--validation-fraction` split is rejected because it leaks positions from the same game.
Early stopping is opt-in, requires independent validation shards, and records its stop reason and
patience state in checkpoint metadata. See
[`docs/training-methodology.md`](docs/training-methodology.md) for the staged compute funnel,
promotion standards, top-human evidence requirements, and the reusable cross-game method.

### Run a seat-rotated promotion gate

```bash
python -m training.parallel_evaluate \
  --candidate output/candidate.pt \
  --champion output/champion.pt \
  --players 2 \
  --rounds 20 \
  --minimum-games 40 \
  --search-simulations 64 \
  --workers 4 \
  --device cuda \
  --output output/candidate-formal-gate.json
```

Every seed rotates the candidate through both seats. The Student-t confidence interval treats each
seat-rotated seed group, rather than each correlated game, as one independent observation.
After all earlier quality gates also pass, promote through the checked atomic path:

```bash
python -m training.promote \
  --candidate output/candidate.pt \
  --champion output/champion.pt \
  --report output/candidate-formal-gate.json \
  --confirm
```

The promotion command recomputes both checkpoint hashes, rejects a failed or inconsistent report,
keeps a pre-promotion backup, and writes an audit manifest. It cannot turn the current failed pilot
into a champion merely because validation loss or one score statistic improved.

## Repo Layout

- `src/core/` - basic game types, locations, static data, player mats
- `src/board/` - board state, resources, connectivity, board-level operations
- `src/actions/` - action execution logic
- `src/validation/` - legal move generation and rule validation
- `src/game/` - high-level action/session flow and turn runner
- `src/web/` - HTTP API and state serialization
- `src/python/` - optional Python API for RL / simulation workflows
- `tests/` - integration and regression tests
- `ui/` - browser UI used for manual play and debugging

## Brass: Birmingham Rules In Brief

Brass: Birmingham is a two-era economic network game about building industries and transport links in the English Midlands.

### Objective

Score the most victory points by the end of the Rail era. Points come mainly from:

- flipped industry tiles
- links connected to valuable developed locations
- strong positioning across both eras

### Turn structure

- The game is played across the **Canal** era and the **Rail** era.
- In the first round of the Canal era, each player takes **1 action**.
- In all later rounds, each player takes **2 actions** on their turn.
- **Every action requires discarding a card**, including `Pass`.

### Main action types

- `Build`: place an industry tile, pay its cost, and consume required resources
- `Network`: build canal or rail links to expand your network
- `Develop`: remove lower-level tiles from your mat to reach stronger industries
- `Sell`: flip Cotton, Goods, or Pottery by connecting to merchants and paying any beer cost
- `Loan`: gain money in exchange for dropping income
- `Scout`: trade extra cards for wild cards
- `Pass`: skip an action, but still discard a card

### Important rules to remember

- Building usually depends on both your network and the card you discard.
- Coal generally requires a connected source; iron does not.
- Beer is critical for selling and for some rail-era network actions.
- Level 1 tiles are removed from the board at the end of the Canal era.
- Turn order for the next round depends on how much money each player spent this round.

### Full rules

For the complete official rules, see the [Brass: Birmingham rulebook](https://cdn.1j1ju.com/medias/60/39/64-brass-birmingham-rulebook.pdf).

## Project Notes

- This is an **unofficial** implementation intended for engine development, testing, AI experimentation, and browser play.
- The focus of this repo is correctness and debuggability of the game state.
- The public API, training schema, and module structure may continue to evolve.

## License, Attribution, and Assets

- Source code is distributed under the [GNU AGPL-3.0 license](LICENSE), preserving the upstream
  repository history and attribution.
- This fan project is not affiliated with or endorsed by Roxley Games or the creators and
  publishers of Brass: Birmingham.
- Board, card, tile, and related artwork in the fork was inherited from the upstream repository.
  No separate asset-license record was found there, so its redistribution and reuse rights remain
  unverified. Replace or independently clear those assets before any use that requires confirmed
  artwork rights.
