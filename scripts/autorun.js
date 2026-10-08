#!/usr/bin/env node

(() => {
const isNode = typeof module !== 'undefined' && module.exports;
const path = isNode ? require('path') : null;
const fs = isNode ? require('fs') : null;

const gameData = isNode ? require('../js/gameData.js') : globalThis;
const GameState = isNode ? require('../js/gameState.js') : globalThis.GameState;
const GameLogic = isNode ? require('../js/gameLogic.js') : globalThis.GameLogic;

const {
    ACTIONS,
    CARD_TYPES,
    ERA,
    LOAN_AMOUNT,
    INDUSTRY_TYPES,
    INDUSTRY_DISPLAY,
    CITIES,
    MERCHANTS,
    isSellableIndustry,
} = gameData;

function parseArgs(argv) {
    const out = { games: 1, players: 4, seed: 1, logFile: null };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const [flag, inlineValue] = arg.split('=');
        const key = flag.slice(2);
        const value = inlineValue !== undefined ? inlineValue : argv[i + 1];
        if (inlineValue === undefined && value !== undefined && !value.startsWith('--')) {
            i++;
        }
        switch (key) {
            case 'games':
                out.games = parseInt(value, 10);
                break;
            case 'players':
                out.players = parseInt(value, 10);
                break;
            case 'seed':
                out.seed = parseInt(value, 10);
                break;
            case 'log-file':
                out.logFile = value;
                break;
        }
    }
    return out;
}

function mulberry32(seed) {
    let a = seed >>> 0;
    return function next() {
        a |= 0;
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function makeRng(seed) {
    const next = mulberry32(seed);
    return {
        next,
        int(max) {
            return Math.floor(next() * max);
        },
        pick(list) {
            return list[this.int(list.length)];
        },
    };
}

function cardValue(card) {
    if (!card) return -1;
    if (card.type === CARD_TYPES.WILD_LOCATION || card.type === CARD_TYPES.WILD_INDUSTRY) return 100;
    if (card.type === CARD_TYPES.LOCATION) return 40;
    if (card.type === CARD_TYPES.INDUSTRY) return 30;
    return 0;
}

function chooseLowestValueCard(hand, validIndices) {
    if (!validIndices || validIndices.length === 0) return null;
    return validIndices
        .slice()
        .sort((a, b) => {
            const diff = cardValue(hand[a]) - cardValue(hand[b]);
            return diff !== 0 ? diff : a - b;
        })[0];
}

function chooseScoutCards(hand) {
    const ranked = hand
        .map((card, idx) => ({ card, idx, value: cardValue(card) }))
        .filter(item => item.card.type !== CARD_TYPES.WILD_LOCATION && item.card.type !== CARD_TYPES.WILD_INDUSTRY)
        .sort((a, b) => {
            const diff = a.value - b.value;
            return diff !== 0 ? diff : a.idx - b.idx;
        });
    return ranked.slice(0, 3).map(item => item.idx);
}

function buildScore(state, target) {
    const tile = target.tileData;
    let score = tile.vp * 2 + tile.income * 1.5 + tile.linkVP * 1.25;
    score -= target.cost.total * 0.7;
    score -= (target.cost.coalCost || 0) * 0.4;
    score -= (target.cost.ironCost || 0) * 0.4;
    if (state.era === ERA.CANAL && tile.level >= 2) score += 4;
    if (tile.level >= 3) score += 2;
    if (tile.type === INDUSTRY_TYPES.POTTERY) score += 3;
    if (tile.type === INDUSTRY_TYPES.BREWERY) score += 1;
    return score;
}

function networkScore(state, playerId, target) {
    const player = state.players[playerId];
    const end1InNetwork = state.isInNetwork(playerId, target.cities[0]);
    const end2InNetwork = state.isInNetwork(playerId, target.cities[1]);
    let score = 0;
    if (end1InNetwork || end2InNetwork) score += 6;
    if (target.type === 'rail') score += 2;
    score -= target.cost * 0.45;
    if (player.money - target.cost < 5) score -= 2;
    return score;
}

function developScore(state, playerId, type1, type2) {
    const player = state.players[playerId];
    const tiles = [];
    const t1 = state.getNextTile(playerId, type1);
    if (t1) tiles.push(t1);
    if (type2) {
        const t2 = state.getNextTile(playerId, type2);
        if (t2) tiles.push(t2);
    }
    let score = 0;
    for (const tile of tiles) {
        score += tile.level * 2 + tile.income * 0.5 + tile.vp * 0.25;
        if (tile.level >= 3) score += 3;
    }
    if (player.money < 5) score -= 2;
    return score;
}

function sellScore(state, playerId, target) {
    const tile = target.tile;
    const player = state.players[playerId];
    let score = tile.tileData.vp * 2 + tile.tileData.income * 1.5;
    score += (tile.tileData.beersToSell || 0) * 1.5;
    score += 3;

    const connected = state.getConnectedLocations(target.cityId);
    for (const mt of state.merchantTiles) {
        if (connected.has(mt.location) && (mt.buys === null || mt.buys === tile.type)) {
            const merch = MERCHANTS[mt.location];
            if (merch) {
                if (merch.bonusType === 'vp') score += merch.bonusAmount * 2;
                if (merch.bonusType === 'money') score += merch.bonusAmount * 0.75;
                if (merch.bonusType === 'income') score += merch.bonusAmount * 1.5;
                if (merch.bonusType === 'develop') score += merch.bonusAmount * 2;
            }
            break;
        }
    }

    if (player.money < 8) score += 1;
    return score;
}

function loanScore(state, playerId) {
    const player = state.players[playerId];
    if (player.money < 0) return 12;
    if (player.money < 5) return 6;
    if (player.money < 10) return 2;
    return -4;
}

function scoutScore(state, playerId) {
    const player = state.players[playerId];
    if (player.hand.length >= 6) return 5;
    if (player.hand.length >= 4) return 2;
    return -3;
}

function passScore() {
    return 0;
}

function collectCandidates(state, logic, playerId) {
    const player = state.players[playerId];
    const candidates = [];

    for (const target of logic.getValidBuildTargets(playerId)) {
        const validCards = logic.getValidCardsForAction(playerId, ACTIONS.BUILD, target);
        const cardIndex = chooseLowestValueCard(player.hand, validCards);
        if (cardIndex === null) continue;
        candidates.push({
            action: ACTIONS.BUILD,
            target,
            cardIndex,
            score: buildScore(state, target),
        });
    }

    for (const target of logic.getValidNetworkTargets(playerId)) {
        const cardIndex = chooseLowestValueCard(player.hand, logic.getValidCardsForAction(playerId, ACTIONS.NETWORK));
        if (cardIndex === null) continue;
        candidates.push({
            action: ACTIONS.NETWORK,
            target,
            cardIndex,
            score: networkScore(state, playerId, target),
        });
    }

    const developable = logic.getDevelopableTypes(playerId);
    const ironSources = state.findIronSource(playerId);
    for (const first of developable) {
        const cardIndex = chooseLowestValueCard(player.hand, logic.getValidCardsForAction(playerId, ACTIONS.DEVELOP));
        if (cardIndex === null) continue;
        candidates.push({
            action: ACTIONS.DEVELOP,
            target: { type1: first.type, type2: null },
            cardIndex,
            score: developScore(state, playerId, first.type, null),
        });
    }
    if (ironSources.length >= 2) {
        for (let i = 0; i < developable.length; i++) {
            for (let j = i; j < developable.length; j++) {
                const first = developable[i];
                const second = developable[j];
                if (first.type === second.type) {
                    const remaining = state.getRemainingTiles(playerId);
                    if (remaining[first.type].count < 2) continue;
                }
                const cardIndex = chooseLowestValueCard(player.hand, logic.getValidCardsForAction(playerId, ACTIONS.DEVELOP));
                if (cardIndex === null) continue;
                candidates.push({
                    action: ACTIONS.DEVELOP,
                    target: { type1: first.type, type2: second.type },
                    cardIndex,
                    score: developScore(state, playerId, first.type, second.type) + 2,
                });
            }
        }
    }

    for (const target of logic.getValidSellTargets(playerId)) {
        const cardIndex = chooseLowestValueCard(player.hand, logic.getValidCardsForAction(playerId, ACTIONS.SELL));
        if (cardIndex === null) continue;
        candidates.push({
            action: ACTIONS.SELL,
            target,
            cardIndex,
            score: sellScore(state, playerId, target),
        });
    }

    if(state.rulesVersion==='economy-v2')for(const target of logic.getValidSellBundles(playerId)){
        const cardIndex=chooseLowestValueCard(player.hand,logic.getValidCardsForAction(playerId,ACTIONS.SELL));
        if(cardIndex!==null)candidates.push({action:ACTIONS.SELL,target,cardIndex,score:target.keys.reduce((sum,key)=>sum+sellScore(state,playerId,{tile:state.boardIndustries[key],cityId:key.slice(0,key.lastIndexOf('_'))}),0)});
    }
    if (player.hand.length > 0) {
        const cardIndex = chooseLowestValueCard(player.hand, logic.getValidCardsForAction(playerId, ACTIONS.LOAN));
        if(state.canTakeLoan(playerId))candidates.push({
            action: ACTIONS.LOAN,
            cardIndex,
            score: loanScore(state, playerId),
        });

        candidates.push({
            action: ACTIONS.PASS,
            cardIndex: chooseLowestValueCard(player.hand, logic.getValidCardsForAction(playerId, ACTIONS.PASS)),
            score: passScore(),
        });
    }

    if (logic.canScout(playerId)) {
        candidates.push({
            action: ACTIONS.SCOUT,
            cardIndices: chooseScoutCards(player.hand),
            score: scoutScore(state, playerId),
        });
    }

    return candidates.filter(candidate => {
        if (candidate.action === ACTIONS.SCOUT) {
            return candidate.cardIndices && candidate.cardIndices.length === 3;
        }
        return candidate.cardIndex !== null && candidate.cardIndex !== undefined;
    });
}

function chooseCandidate(candidates, rng) {
    if (candidates.length === 0) return null;
    let bestScore = -Infinity;
    let best = [];
    for (const candidate of candidates) {
        if (candidate.score > bestScore) {
            bestScore = candidate.score;
            best = [candidate];
        } else if (candidate.score === bestScore) {
            best.push(candidate);
        }
    }
    return best[rng.int(best.length)];
}

function candidateKey(candidate) {
    if (candidate.action === ACTIONS.SCOUT) {
        return `${candidate.action}:${candidate.cardIndices.join(',')}`;
    }
    const targetBits = candidate.target ? JSON.stringify(candidate.target) : '';
    return `${candidate.action}:${targetBits}:${candidate.cardIndex}`;
}

function actionLabel(candidate) {
    switch (candidate.action) {
        case ACTIONS.BUILD: {
            const tile = candidate.target.tileData;
            return `build ${INDUSTRY_DISPLAY[candidate.target.industryType].name} Lv${tile.level} in ${CITIES[candidate.target.cityId].name}`;
        }
        case ACTIONS.NETWORK: {
            return `network ${candidate.target.connectionId}`;
        }
        case ACTIONS.DEVELOP:
            return `develop ${candidate.target.type1}${candidate.target.type2 ? ` + ${candidate.target.type2}` : ''}`;
        case ACTIONS.SELL:
            return `sell ${(candidate.target.keys||[candidate.target.key]).join(", ")}`;
        case ACTIONS.LOAN:
            return `loan`;
        case ACTIONS.SCOUT:
            return `scout`;
        case ACTIONS.PASS:
            return `pass`;
        default:
            return candidate.action;
    }
}

function clonePlayerSummary(player) {
    return {
        id: player.id,
        name: player.name,
        money: player.money,
        income: player.income,
        vp: player.vp,
        handSize: player.hand.length,
        linksRemaining: { ...player.linksRemaining },
    };
}

function summarizeState(state) {
    return {
        era: state.era,
        round: state.round,
        currentPlayerId: state.currentPlayerId,
        currentPlayerName: state.currentPlayer.name,
        actionsThisTurn: state.actionsThisTurn,
        actionsPerTurn: state.actionsPerTurn,
        drawDeckSize: state.drawDeck.length,
        coalMarket: state.coalMarket,
        ironMarket: state.ironMarket,
        players: state.players.map(clonePlayerSummary),
        boardIndustries: Object.keys(state.boardIndustries).length,
        boardLinks: Object.keys(state.boardLinks).length,
        breweryFarmTiles: Object.keys(state.breweryFarmTiles).length,
    };
}

function applyTurnResult(state, result) {
    if (result === 'endCanalEra') {
        const scores = state.endCanalEra();
        return { type: 'era', scores };
    }
    if (result === 'endGame') {
        const scores = state.endGame();
        return { type: 'game', scores };
    }
    return { type: 'continue' };
}

function runGame({ players, seed }) {
    const rng = makeRng(seed);
    const originalRandom = Math.random;
    Math.random = rng.next;

    try {
        const names = ['Red', 'Yellow', 'Purple', 'White'].slice(0, players).map((name, idx) => `${name}-${idx + 1}`);
        const state = new GameState(players, names);
        const logic = new GameLogic(state);

        const stats = {
            actions: Object.fromEntries(Object.values(ACTIONS).map(action => [action, 0])),
            eraActions: {
                [ERA.CANAL]: Object.fromEntries(Object.values(ACTIONS).map(action => [action, 0])),
                [ERA.RAIL]: Object.fromEntries(Object.values(ACTIONS).map(action => [action, 0])),
            },
            turns: 0,
        };
        const actionLog = [];

        let safety = 0;
        while (!state.gameOver && safety < 10000) {
            safety++;

            if (state.currentPlayer.hand.length === 0) {
                const skipped = applyTurnResult(state, state.advanceTurn());
                if (skipped.type === 'game') break;
                continue;
            }

            let candidates = collectCandidates(state, logic, state.currentPlayerId);
            if (candidates.length === 0) {
                throw new Error(`No legal action found for player ${state.currentPlayerId}`);
            }

            let result = null;
            let chosenCandidate = null;
            const beforeState = summarizeState(state);
            while (candidates.length > 0) {
                const candidate = chooseCandidate(candidates, rng);
                if (!candidate) break;
                chosenCandidate = candidate;

                switch (candidate.action) {
                    case ACTIONS.BUILD:
                        result = logic.executeBuild(
                            state.currentPlayerId,
                            candidate.target.cityId,
                            candidate.target.slotIndex,
                            candidate.target.industryType,
                            candidate.cardIndex
                        );
                        break;
                    case ACTIONS.NETWORK:
                        result = logic.executeNetwork(state.currentPlayerId, candidate.target.connectionId, candidate.cardIndex);
                        break;
                    case ACTIONS.DEVELOP:
                        result = logic.executeDevelop(state.currentPlayerId, candidate.target.type1, candidate.target.type2, candidate.cardIndex);
                        break;
                    case ACTIONS.SELL:
                        result = logic.executeSell(state.currentPlayerId, [candidate.target.key], candidate.cardIndex);
                        break;
                    case ACTIONS.LOAN:
                        result = logic.executeLoan(state.currentPlayerId, candidate.cardIndex);
                        break;
                    case ACTIONS.SCOUT:
                        result = logic.executeScout(state.currentPlayerId, candidate.cardIndices);
                        break;
                    case ACTIONS.PASS:
                        result = logic.executePass(state.currentPlayerId, candidate.cardIndex);
                        break;
                    default:
                        throw new Error(`Unsupported action ${candidate.action}`);
                }

                if (result && result.success) {
                    break;
                }

                const failedKey = candidateKey(candidate);
                candidates = candidates.filter(item => candidateKey(item) !== failedKey);
                result = null;
                chosenCandidate = null;
            }

            if (!chosenCandidate || !result || !result.success) {
                throw new Error(`No executable candidate found for player ${state.currentPlayerId}`);
            }

            stats.actions[chosenCandidate.action]++;
            stats.eraActions[state.era][chosenCandidate.action]++;
            stats.turns++;

            const turnResult = applyTurnResult(state, state.advanceTurn());
            const afterState = summarizeState(state);
            actionLog.push({
                turn: stats.turns,
                playerId: beforeState.currentPlayerId,
                playerName: beforeState.currentPlayerName,
                era: beforeState.era,
                round: beforeState.round,
                action: chosenCandidate.action,
                actionLabel: actionLabel(chosenCandidate),
                target: chosenCandidate.target ? chosenCandidate.target : null,
                cardIndex: chosenCandidate.cardIndex ?? null,
                cardIndices: chosenCandidate.cardIndices ?? null,
                result: {
                    success: true,
                    message: result.message,
                },
                before: beforeState,
                after: afterState,
                turnResult: turnResult.type,
            });

            if (turnResult.type === 'game') break;
        }

        if (safety >= 10000) {
            throw new Error('Safety limit exceeded while simulating game');
        }

        const finalScores = [...state.players]
            .map(player => ({
                id: player.id,
                name: player.name,
                vp: player.vp,
                income: player.income,
                money: player.money,
            }))
            .sort((a, b) => b.vp - a.vp);

        return { state, logic, stats, finalScores, actionLog };
    } finally {
        Math.random = originalRandom;
    }
}

function printGameSummary(index, total, seed, result) {
    const scores = result.finalScores.map(player => `${player.name}:${player.vp}`).join(' ');
    const winner = result.finalScores[0];
    console.log(`Game ${index}/${total} seed=${seed} winner=${winner.name}(${winner.vp}) scores=${scores}`);
}

function summarize(results, config) {
    const totals = results.reduce((acc, result) => {
        for (const player of result.finalScores) {
            acc.scoreSum += player.vp;
            acc.playerCount++;
            acc.bestScore = Math.max(acc.bestScore, player.vp);
        }
        for (const [action, count] of Object.entries(result.stats.actions)) {
            acc.actions[action] = (acc.actions[action] || 0) + count;
        }
        for (const era of [ERA.CANAL, ERA.RAIL]) {
            for (const [action, count] of Object.entries(result.stats.eraActions[era])) {
                acc.eraActions[era][action] = (acc.eraActions[era][action] || 0) + count;
            }
        }
        acc.games++;
        return acc;
    }, {
        scoreSum: 0,
        playerCount: 0,
        bestScore: -Infinity,
        actions: Object.fromEntries(Object.values(ACTIONS).map(action => [action, 0])),
        eraActions: {
            [ERA.CANAL]: Object.fromEntries(Object.values(ACTIONS).map(action => [action, 0])),
            [ERA.RAIL]: Object.fromEntries(Object.values(ACTIONS).map(action => [action, 0])),
        },
        games: 0,
    });

    const averageScore = totals.scoreSum / totals.playerCount;
    console.log('');
    console.log('=== Summary ===');
    console.log(`Games: ${totals.games}, Players/Game: ${config.players}, Seed: ${config.seed}`);
    console.log(`Average final VP per player: ${averageScore.toFixed(2)}`);
    console.log(`Best single-player score: ${totals.bestScore}`);
    console.log('Action distribution:');
    for (const action of Object.values(ACTIONS)) {
        const count = totals.actions[action] || 0;
        const pct = ((count / Object.values(totals.actions).reduce((a, b) => a + b, 0)) * 100).toFixed(2);
        console.log(`  ${action}: ${count} (${pct}%)`);
    }
    console.log('Era action distribution:');
    for (const era of [ERA.CANAL, ERA.RAIL]) {
        const eraTotal = Object.values(totals.eraActions[era]).reduce((a, b) => a + b, 0);
        console.log(`  ${era}: ${eraTotal}`);
        for (const action of Object.values(ACTIONS)) {
            const count = totals.eraActions[era][action] || 0;
            const pct = eraTotal === 0 ? '0.00' : ((count / eraTotal) * 100).toFixed(2);
            console.log(`    ${action}: ${count} (${pct}%)`);
        }
    }
}

function writeLogFile(results, config) {
    if (!config.logFile) return;

    const payload = {
        generatedAt: new Date().toISOString(),
        source: 'brass-birmingham/scripts/autorun.js',
        config: {
            games: config.games,
            players: config.players,
            seed: config.seed,
        },
        games: results.map((result, index) => ({
            gameIndex: index + 1,
            seed: config.seed + index * 9973,
            finalScores: result.finalScores,
            stats: result.stats,
            actions: result.actionLog,
            finalState: summarizeState(result.state),
        })),
    };

    const resolved = path.isAbsolute(config.logFile)
        ? config.logFile
        : path.join(process.cwd(), config.logFile);

    let finalPath = resolved;
    const safeFallbackDir = path.join(
        require('os').homedir(),
        'Documents',
        'Codex',
        '2026-05-28',
        'fork-clone-https-github-com-npow'
    );
    try {
        const dir = path.dirname(resolved);
        if (dir && dir !== '.' && dir !== process.cwd()) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(finalPath, JSON.stringify(payload, null, 2), 'utf8');
    } catch (err) {
        try {
            fs.mkdirSync(safeFallbackDir, { recursive: true });
            finalPath = path.join(safeFallbackDir, path.basename(resolved));
            fs.writeFileSync(finalPath, JSON.stringify(payload, null, 2), 'utf8');
            console.log(`Wrote log file fallback: ${finalPath}`);
            return;
        } catch (fallbackErr) {
            throw new Error(`Failed to write log file to ${resolved} or fallback ${safeFallbackDir}: ${fallbackErr.message}`);
        }
    }

    console.log(`Wrote log file: ${finalPath}`);
}

function main() {
    const config = parseArgs(process.argv.slice(2));
    if (!Number.isInteger(config.games) || config.games <= 0) {
        throw new Error('--games must be a positive integer');
    }
    if (!Number.isInteger(config.players) || config.players < 2 || config.players > 4) {
        throw new Error('--players must be 2, 3, or 4');
    }
    if (!Number.isInteger(config.seed)) {
        throw new Error('--seed must be an integer');
    }

    console.log(`Running ${config.games} games with ${config.players} players, seed ${config.seed}`);

    const results = [];
    for (let i = 0; i < config.games; i++) {
        const seed = config.seed + i * 9973;
        const result = runGame({ players: config.players, seed });
        results.push(result);
        printGameSummary(i + 1, config.games, seed, result);
    }

    summarize(results, config);
    writeLogFile(results, config);
}

const api = { collectCandidates, chooseCandidate, makeRng, actionLabel, applyTurnResult, runGame };
if (isNode) module.exports = api;
else globalThis.BrassBaseline = api;
if (isNode && require.main === module) {
    main();
}
})();
