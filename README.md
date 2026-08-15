# Brass Birmingham Daiyu AI

`Brass Birmingham Daiyu AI` is an unofficial browser game and AI analysis project for
**Brass: Birmingham**. It is a public AGPL-3.0 fork of
[`artyom-morozov/fast_brass`](https://github.com/artyom-morozov/fast_brass), extending the Rust
rules engine with model-guided search, self-play training, and an interactive Svelte interface.

The project currently focuses on transparent two-player analysis: each position can show several
candidate moves, the search share and model signals behind each move, and contextual answers to
Chinese follow-up questions such as why the first choice is preferred over the second.

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

Open `http://127.0.0.1:5173`, create a two-player game, and enter the `AI 分析` tab. The analysis
panel can show the Top 3 candidate moves, answer follow-up questions about the selected move, apply
one recommendation, or advance the AI side one move at a time.

The first version does not yet assign human and AI seats automatically. Pick one seat as your own
(for example player 1) and use this loop:

1. Play your seat normally in the `对局` tab.
2. When the other seat becomes current, open `AI 分析`, choose `快速`, `标准`, or `深入`, and click
   `分析局面`.
3. Select any Top 3 row to inspect it. Use `选择依据`, `胜率可信度`, `主要风险`, `对比一选`, or the
   free-form question box to ask about that exact move.
4. Click `采用此步` to play the selected move for the AI seat. Repeat the analysis/apply cycle if
   that seat still has another action, then return to `对局` when your seat becomes current again.

The play, pause, step, and stop icons under `AI 对局` are intended for AI-vs-AI observation. In a
human-vs-AI game, `采用此步` is the safer control because it returns to the game view after exactly
one selected action.

## AI Status

- Deep neural search is currently implemented for two-player games. New browser games therefore
  default to two players.
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
without HTTP/JSON transport. Stop the browser inference service first so its CUDA allocator does
not retain training memory, then restore it after generation:

```powershell
docker compose --file docker/training-gpu.compose.yaml stop inference
docker compose --file docker/training-gpu.compose.yaml exec training-gpu python3 -m training.parallel_self_play --output-prefix output/iter3-train --checkpoint output/champion-iter1.pt --games 100 --workers 4 --players 2 --simulations 256 --search-determinizations 4 --inference-batch-size 64 --device cuda --seed 2026081601
docker compose --file docker/training-gpu.compose.yaml up --detach inference
```

This writes one atomic JSONL shard per worker, such as `iter3-train-000.jsonl`. Global game indices
are assigned before work is split, so changing `--workers` does not change which deterministic games
the requested index range represents. Use a different `--seed` for validation data. Four workers
are the measured default for an RTX 5080 with 16 GB VRAM; lower the count on smaller GPUs.

### Train with an independent validation partition

```bash
python -m training.train \
  --shards output/train-*.jsonl \
  --validation-shards output/validation-*.jsonl \
  --output output/candidate.pt \
  --resume output/champion.pt \
  --device cuda
```

Training rejects overlapping paths, incompatible schemas or engine revisions, and reused recorded
game seeds across the two partitions. Validation metrics are weighted by position rather than by
batch, and training saves the state with the lowest independent validation loss (including the
resumed baseline) rather than blindly keeping the final epoch. The old position-level
`--validation-fraction` split is rejected because it leaks positions from the same game.

### Run a seat-rotated promotion gate

```bash
python -m training.evaluate \
  --candidate-inference-url http://HOST:8766 \
  --champion-inference-url http://HOST:8765 \
  --players 2 \
  --rounds 20 \
  --minimum-games 40 \
  --search-simulations 64
```

Every seed rotates the candidate through both seats. The Student-t confidence interval treats each
seat-rotated seed group, rather than each correlated game, as one independent observation.

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
