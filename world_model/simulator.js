/* Thin adapter: every real transition delegates to the original engine. */
(function (root) {
    'use strict';
    const node = typeof module !== 'undefined' && module.exports;
    const State = node ? require('../js/gameState') : root.GameState;
    const Logic = node ? require('../js/gameLogic') : root.GameLogic;
    const bot = node ? require('../scripts/autorun') : root.BrassBaseline;
    const snapshot = s => structuredClone(Object.fromEntries(Object.entries(s)));
    const clone = s => Object.assign(Object.create(State.prototype), snapshot(s));

    function withRandom(holder, fn) {
        const original = Math.random;
        Math.random = () => {
            let a = holder._rngState = ((holder._rngState || 1) + 0x6d2b79f5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        try { return fn(); } finally { Math.random = original; }
    }

    function create(players = 4, seed = 1, names) {
        const holder = { _rngState: seed | 0 };
        const s = withRandom(holder, () => new State(players,
            names || Array.from({ length: players }, (_, i) => `Player ${i + 1}`)));
        s._rngState = holder._rngState;
        return s;
    }

    function execute(state, a) {
        const l = new Logic(state), p = state.currentPlayerId, t = a.target || {};
        switch (a.action) {
            case 'build': return l.executeBuild(p, t.cityId, t.slotIndex, t.industryType, a.cardIndex);
            case 'network': return l.executeNetwork(p, t.connectionId, a.cardIndex);
            case 'develop': return l.executeDevelop(p, t.type1, t.type2, a.cardIndex);
            case 'sell': return l.executeSell(p, [t.key], a.cardIndex);
            case 'loan': return l.executeLoan(p, a.cardIndex);
            case 'scout': return l.executeScout(p, a.cardIndices);
            case 'pass': return l.executePass(p, a.cardIndex);
            default: throw Error(`Unknown action: ${a.action}`);
        }
    }

    function candidates(state) {
        if (state.gameOver || !state.currentPlayer?.hand.length) return [];
        const list = bot.collectCandidates(state, new Logic(state), state.currentPlayerId);
        // The legacy bot retries a Develop when executeDevelop reports insufficient
        // iron. Remove those known failures without altering its ranking or rules.
        const iron = state.findIronSource(state.currentPlayerId).length;
        return list.filter(a => a.action !== 'develop' || iron >= (a.target.type2 ? 2 : 1));
    }

    // Membership is checked using the engine's candidate generator. Failed
    // executions cannot partially mutate the live state because we use a clone.
    const key = a => JSON.stringify([a.action, a.cardIndex, a.cardIndices,
        a.target?.cityId, a.target?.slotIndex, a.target?.industryType,
        a.target?.connectionId, a.target?.type1, a.target?.type2, a.target?.key]);
    function step(state, action, { validate = true } = {}) {
        if (state.gameOver) throw Error('Game is over');
        if (validate && !candidates(state).some(a => key(a) === key(action))) {
            throw Error('Action is no longer an available candidate');
        }
        const next = clone(state);
        return withRandom(next, () => {
            const result = execute(next, action);
            if (!result.success) throw Error(result.message || 'Action failed');
            const event = bot.applyTurnResult(next, next.advanceTurn());
            return { state: next, result, event };
        });
    }
    const api = { create, snapshot, clone, candidates, step, key, withRandom,
        label: bot.actionLabel, rng: bot.makeRng };
    if (node) module.exports = api; else root.BrassSimulator = api;
})(globalThis);
