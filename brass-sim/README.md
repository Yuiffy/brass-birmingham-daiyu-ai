# Brass: Birmingham

> **维护位置（2026-10-09）：** 本项目已从 `Yuiffy/brass-birmingham-sim` 整合至
> [`Yuiffy/brass-birmingham-daiyu-ai`](https://github.com/Yuiffy/brass-birmingham-daiyu-ai)
> 的 `brass-sim/` 目录。原 `Yuiffy/brass-birmingham` 的全部提交也已包含在内。
> 以下命令均在 `brass-sim/` 中执行；在仓库根目录可运行
> `npm --prefix brass-sim start -- --port 8086` 或 `npm --prefix brass-sim test`。
> 此 JavaScript 引擎与根目录 Rust/Svelte 引擎独立，训练权重不能直接互换。
> 迁移范围和验证记录见[仓库整合说明](../docs/repository-consolidation.md)。

## Economy calibration experiment

The optional `/?rules=economy-v2` game and `/arena.html?rules=economy-v2` arena separate corrected income, loans, era flow, market supply and multi-sale handling from historical training. They use a distinct model schema and fresh data. This remains a partial rules calibration, not a complete official-rules implementation. See [scope and limitations](docs/economy-calibration.md) and [independent score comparisons](docs/economy-training-results.md). The default URL preserves the historical rules and mature checkpoints.

## World Model / AI Imagination MVP

This checkout includes five selectable AI types, trained world-model checkpoints,
an imagination panel, and an AI tournament/evaluation page. Run:

```sh
npm run worldmodel:demo -- --port 8086
```

Open `http://127.0.0.1:8086` for the game or `/arena.html` for comparisons.
See [the implementation, results and reproduction guide](docs/world-model.md).
See [continued training and the matched-seed comparison](docs/continued-training.md) for model versions and their measured differences.
The current model was trained with geographic scoring features and additional games against a mixed league of agents, and ranks up to 32 candidates before deeper planning. In 400 fresh games per agent/version, world-model mean engine VP rose from 98.97 to 112.64; guided search rose from 106.67 to 126.33. Gains remain positive after removing the engine-only final income bonus. See [human score references, rules audit and independent results](docs/higher-scores.md). Earlier experiments remain in [score-focused training](docs/score-training.md) and [architecture improvement](docs/improved-world-model.md).
The latest replay continuation and search ablations are recorded in [next-round training](docs/next-training.md); candidates are kept separate until they pass independent seed groups.
The game now selects trained dynamics for 2, 3, or 4 players automatically. The demo uses only guided/world agents, and the current strong lineup in the arena supports all three player counts. The 2P/3P models gained 19.54 / 12.13 mean VP with fixed guided opponents in independent paired tournaments; 4P retains the existing weights. See [small-player training and runtime integration](docs/small-player-dynamics.md).

Git includes the current 2P/3P/4P checkpoints, the historical checkpoints selectable in the arena, and their evaluation reports. Raw replay datasets and other local candidate checkpoints are not included; older experiment documents also reference those local artifacts. Playing the game requires only Node.js 18 or newer. Training additionally requires Python and `world_model/requirements.txt`.

The original game and autorun instructions follow below.

[![Docs](https://img.shields.io/badge/docs-mintlify-18a34a?style=flat-square)](https://mintlify.com/npow/brass-birmingham)

A digital adaptation of the award-winning board game by Roxley Games. Build your industrial empire across the English Midlands during the height of the Industrial Revolution (1770-1870).

> **Note:** This is an unofficial fan project. [Brass: Birmingham](https://roxley.com/products/brass-birmingham) is designed by Gavan Brown, Matt Tolman, and Martin Wallace, published by Roxley Games. Please support the original by purchasing the physical game.

![Title Screen](screenshots/title.png)

---

## The Game

Compete as rival entrepreneurs in Birmingham and its surrounding towns. Establish canals and railways, develop industries, and outmaneuver your opponents across two distinct eras.

**2-4 Players** · **Canal Era + Rail Era** · **Hotseat Multiplayer**

### Two Eras of Industry

**Canal Era (1770-1830)** — Build canals and establish your first industries. Only one tile per location. At the end, all canals and Level I tiles are removed — but Level II+ tiles carry over and score again.

**Rail Era (1830-1870)** — Build railways (requiring coal), expand aggressively with multiple tiles per location, and push for the highest score.

![Canal Era — early board with industries and canal links](screenshots/canal_era.png)

<p align="center"><em>Canal Era — industries built across the Midlands, canal links connecting the network</em></p>

![Rail Era — dense network of rail links and high-level industries](screenshots/rail_era.png)

<p align="center"><em>Rail Era — railways criss-cross the map as players race toward final scoring</em></p>

---

### Build Your Empire

Choose from six industry types, each with a unique strategic role:

| Industry | Role | Key Trait |
|----------|------|-----------|
| **Cotton Mills** | Sell goods for VP | Expensive but high-scoring |
| **Coal Mines** | Fuel rail links and industries | Cheap, strong income |
| **Iron Works** | Supply iron for building and development | Efficient, denies opponents |
| **Manufacturers** | 8 unique levels with varied rewards | Versatile and complex |
| **Potteries** | Massive VP potential (up to 20 VP) | High cost, requires planning |
| **Breweries** | Supply beer for selling goods | The most important industry |

Select what to build, where, and for how much — all costs including coal and iron sourcing are calculated automatically.

![Build modal — choose an industry, see the cost breakdown](screenshots/build_modal.png)

<p align="center"><em>Build action — choose from available industries at locations matching your cards</em></p>

---

### Clear Action Workflow

A persistent **action phase bar** guides you through each turn:

1. **Choose Action** — Pick from Build, Network, Develop, Sell, Loan, Scout, or Pass
2. **Select Target** — Choose what/where to build, which connection, etc.
3. **Discard Card** — Valid cards glow gold; invalid cards are dimmed

The phase bar shows your current step, contextual instructions, and a cancel button. Press **Escape** at any time to abort.

When waiting for card selection, action buttons and your player mat dim to focus attention on your hand. Only valid discard options are clickable.

---

### Game Log & Turn Transitions

A **scrollable game log** in the right panel records every action: builds, network links, sales, loans, and more — color-coded by player. No more relying on fleeting toast notifications.

Between turns, a **brief animated overlay** shows the next player's name in their color, giving a clear visual break.

![Players panel and game log](screenshots/player_markets.png)

<p align="center"><em>Right panel — player stats, markets with current prices, and the game log</em></p>

---

### Develop to Unlock Higher Tiers

Spend iron to remove low-level tiles from your player mat and access the powerful high-level industries underneath. Optionally develop two tiles at once.

![Develop modal — remove tiles from your mat to unlock stronger ones](screenshots/develop_modal.png)

<p align="center"><em>Develop action — skip past weak tiles to access Level III+ industries</em></p>

---

### Your Hand, Your Strategy

Play location cards to build anywhere on the map, or industry cards to build within your network. Manage your hand carefully — every action costs a card.

Cards show **inline SVG icons** matching each industry type, with colored top borders indicating card type (green for location, red for industry, gold gradient for wild).

![Hand, actions, and industry mat](screenshots/hand_actions.png)

<p align="center"><em>Bottom panel — your hand of cards, the seven available actions, and your remaining industry tiles</em></p>

---

### Atmospheric Board

The game board features:
- **Parchment texture** via SVG noise filters and a radial vignette
- **Double-line canal** connections (translucent blue water effect)
- **Rail connections** with track-tie patterns
- **Geometric SVG icons** on industry slots and built tiles (factory, diamond, gear, crate, vase, barrel)
- **City nodes** with dark label backdrops, inner shadow depth, and increased region color opacity
- **VP badges** on flipped tiles, player color strips on built tiles

---

### Era Scoring

At the end of each era, score VP from your flipped industry tiles and the links connecting them. Higher-level tiles placed in the Canal Era score in *both* eras.

![Scoring screen at end of Canal Era](screenshots/scoring.png)

<p align="center"><em>Canal Era scoring — link VP and industry VP tallied for each player</em></p>

---

## How to Play

### Quick Start

```
python3 -m http.server 8080
```

Open [http://localhost:8080](http://localhost:8080) in your browser. No build step or dependencies.

1. Choose 2-4 players and enter names
2. Click **Begin Game**

![Setup screen](screenshots/board_start.png)

### Node CLI Auto-Run

The core game logic can also run headless in Node.js with a simple bot:

```
node scripts/autorun.js --games 200 --players 4 --seed 1
```

Parameters:

- `--games` number of full games to simulate
- `--players` player count from 2 to 4
- `--seed` base seed for deterministic shuffling and bot tie-breaking
- `--log-file` optional JSON output path for the full per-turn log bundle

The CLI prints one summary line per game and a final aggregate report with:

- each game's final VP totals
- average final score
- action distribution across the whole run
- action distribution split by canal era and rail era

The simulator reuses the existing `js/gameData.js`, `js/gameState.js`, and `js/gameLogic.js` files directly, so no browser globals or bundler are required.

To capture everything for downstream AI analysis, run:

```
node scripts/autorun.js --games 200 --players 4 --seed 1 --log-file logs/autorun-200.json
```

That file contains one JSON object with all games, each game's action-by-action timeline, pre/post state snapshots, and final score table.

### On Your Turn

1. **Select an action** from the action panel (Build, Network, Develop, Sell, Loan, Scout, or Pass)
2. **Follow the phase bar** — it shows your current step and what to do next
3. **Choose your target** from the modal or board
4. **Click a valid card** (highlighted with a gold pulse) to discard

Disabled action buttons show **tooltips explaining why** they can't be used (e.g. "No iron available", "Need at least 3 cards").

The game handles all resource sourcing (coal, iron, beer), market pricing, turn order, and scoring automatically.

### Tips for New Players

- **Build breweries early** — Beer is the most contested resource in the game
- **Watch the turn order** — Spending less money means going first next round
- **Develop to skip weak tiles** — Accessing Level III+ tiles is worth the iron cost
- **Build in the Canal Era for double scoring** — Level II+ tiles persist into the Rail Era
- **Don't ignore income** — It compounds every round and scores VP at the end

---

## Rules Reference

This implementation follows the official Brass: Birmingham rulebook:

- **Coal** requires a connected source (mine or market via merchant link)
- **Iron** can be taken from any iron works on the board — no connection needed
- **Beer** for selling comes from your own breweries (anywhere), connected opponent breweries, or merchant barrels
- **Overbuilding** your own tiles requires same type + higher level; opponent coal/iron tiles can only be overbuilt when that resource is globally depleted
- **Pottery I and III** have the lightbulb icon and cannot be developed — they must be built
- **Turn order** each round goes to whoever spent the least money

---

## Architecture

Pure HTML/CSS/JS — no build tools, no frameworks, no external images.

| File | Purpose |
|------|---------|
| `index.html` | Layout: setup screen, game screen with phase bar, game log, overlays |
| `css/style.css` | Dark Victorian theme with CSS animations, card-select mode, transitions |
| `js/gameData.js` | All constants: tiles, cities, connections, cards, merchants, markets |
| `js/gameState.js` | Game state, market mechanics, network pathfinding, turn management |
| `js/gameLogic.js` | All 7 actions + `getDisabledReason()` for tooltip feedback |
| `js/boardRenderer.js` | SVG board with texture filters, styled connections, geometric icons |
| `js/uiManager.js` | Phase bar, card selection mode, game log, turn transitions, modals |
| `js/main.js` | Entry point, setup screen |
| `scripts/autorun.js` | Node CLI bot runner for headless simulation |

---

## Credits

- **Brass: Birmingham** designed by Gavan Brown, Matt Tolman, and Martin Wallace
- Published by [Roxley Games](https://roxley.com)
- Tile data verified against the [Tabletop Simulator](https://github.com/ikegami/tts_brass) implementation by Kini/ikegami
- Board geography and card data referenced from community implementations

---

*Fan-made digital adaptation for personal and educational use. All game design credit belongs to the original designers and publisher.*
