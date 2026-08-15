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
