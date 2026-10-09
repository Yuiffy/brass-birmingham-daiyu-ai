/* Thin adapter: every real transition delegates to the original engine. */
(function (root) {
    'use strict';
    const node = typeof module !== 'undefined' && module.exports;
    const State = node ? require('../js/gameState') : root.GameState;
    const Logic = node ? require('../js/gameLogic') : root.GameLogic;
    const bot = node ? require('../scripts/autorun') : root.BrassBaseline;
    const D = node ? require('../js/gameData') : root;
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
            case 'network': return t.connectionIds ? l.executeNetworkLinks(p,t.connectionIds,a.cardIndex,t.beerKey) : l.executeNetwork(p, t.connectionId, a.cardIndex);
            case 'develop': return l.executeDevelop(p, t.type1, t.type2, a.cardIndex);
            case 'sell': return l.executeSell(p, t.keys||[t.key], a.cardIndex);
            case 'loan': return l.executeLoan(p, a.cardIndex);
            case 'scout': return l.executeScout(p, a.cardIndices);
            case 'pass': return l.executePass(p, a.cardIndex);
            default: throw Error(`Unknown action: ${a.action}`);
        }
    }

    function cardUtility(state,player,card) {
        if(!card)return Infinity;
        if(card.type.startsWith('wild'))return 100;
        const typeValue=type=>{
            const tile=state.getNextTile(player,type);if(!tile)return 0;
            const product=['cottonMill','manufacturer','pottery'].includes(type);
            if(product&&!state.merchantTiles.some(m=>m.buys===null||m.buys===type))return 0;
            return tile.vp*(state.era==='canal'&&tile.level>=2?1.9:1)+(type==='brewery'?6:0);
        };
        if(card.type==='industry')return 8+typeValue(card.industryType);
        if(card.type==='location') {
            const city=D.CITIES[card.location];if(!city)return 0;
            const types=city.slots.flatMap(s=>Array.isArray(s)?s:[s]);
            return 12+Math.max(0,...types.map(typeValue));
        }
        return 0;
    }
    function candidates(state, {doubleRail = false,retainCards=false,cardChoices=false} = {}) {
        if (state.gameOver || !state.currentPlayer?.hand.length) return [];
        const list = bot.collectCandidates(state, new Logic(state), state.currentPlayerId);
        if (doubleRail) {
            const discard = list.find(a => a.action === 'network' || a.action === 'pass')?.cardIndex;
            if (discard !== undefined) for (const target of new Logic(state).getValidDoubleNetworkTargets(state.currentPlayerId))
                list.push({action:'network',target,cardIndex:discard,score:0});
        }
        // The legacy bot retries a Develop when executeDevelop reports insufficient
        // iron. Remove those known failures without altering its ranking or rules.
        const iron = state.findIronSource(state.currentPlayerId).length;
        const legal=list.filter(a => {
            if(a.action!=='develop')return true;
            const count=a.target.type2?2:1;if(iron<count)return false;
            if(state.rulesVersion!=='economy-v2')return true;
            const sources=state.findIronSource(state.currentPlayerId).slice(0,count);
            if(sources.reduce((sum,s)=>sum+(s.free?0:s.price),0)>state.currentPlayer.money)return false;
            const used=new Map();return [a.target.type1,a.target.type2].filter(Boolean).every(type=>{
                const offset=used.get(type)||0,tile=state.currentPlayer.industryTiles[type].filter(t=>!t.used)[offset];used.set(type,offset+1);return tile?.canDevelop;
            });
        });
        if(!retainCards&&!cardChoices)return legal;
        const logic=new Logic(state),actor=state.currentPlayerId;
        return legal.flatMap(a=>{
            if(a.action==='scout')return [a];
            const indices=logic.getValidCardsForAction(actor,a.action,a.target);
            if(cardChoices)return indices.map(cardIndex=>({...a,cardIndex}));
            const cardIndex=indices.slice().sort((x,y)=>cardUtility(state,actor,state.currentPlayer.hand[x])-cardUtility(state,actor,state.currentPlayer.hand[y])||x-y)[0];
            return cardIndex===undefined?[]:[{...a,cardIndex}];
        });
    }

    // Membership is checked using the engine's candidate generator. Failed
    // executions cannot partially mutate the live state because we use a clone.
    const key = a => JSON.stringify([a.action, a.cardIndex, a.cardIndices,
        a.target?.cityId, a.target?.slotIndex, a.target?.industryType,
        a.target?.connectionId, a.target?.type1, a.target?.type2, a.target?.key, a.target?.keys,
        a.target?.connectionIds, a.target?.beerKey]);
    function step(state, action, { validate = true,captureSettlement=false } = {}) {
        if (state.gameOver) throw Error('Game is over');
        if (validate && !candidates(state,{doubleRail:!!action.target?.connectionIds,cardChoices:true}).some(a => key(a) === key(action))) {
            throw Error('Action is no longer an available candidate');
        }
        const next = clone(state);
        return withRandom(next, () => {
            const result = execute(next, action);
            if (!result.success) throw Error(result.message || 'Action failed');
            const turnResult=next.advanceTurn();
            const beforeSettlement=captureSettlement&&['endCanalEra','endGame'].includes(turnResult)?snapshot(next):null;
            const event = bot.applyTurnResult(next, turnResult);
            if(beforeSettlement)event.beforeSettlement=beforeSettlement;
            return { state: next, result, event };
        });
    }
    const api = { create, snapshot, clone, candidates, step, key, withRandom,cardUtility,
        label: bot.actionLabel, rng: bot.makeRng };
    if (node) module.exports = api; else root.BrassSimulator = api;
})(globalThis);
