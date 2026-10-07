import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, clone, rng, capacity, observation, transition } from '../engine.mjs';
import { reserveBid, policy } from '../ai.mjs';
import { AUCTION_FORMULAS, formulaFeatures, formulaLimit, continuationPolicy } from '../auction-formulas.mjs';
import { chooseStrategy } from '../strategies.mjs';
import { playMatch } from '../tournament.mjs';

test('legacy formula is an exact policy control over complete matches', () => {
  const a = playMatch({ lineup: ['bid+3', 'bid+6', 'bid-liquid+3', 'bid+3'], seed: 321, validate: true });
  const b = playMatch({ lineup: ['formula-legacy', 'bid+6', 'bid-liquid+3', 'bid+3'], seed: 321, validate: true });
  assert.deepEqual(a.results, b.results.map(r => ({ ...r, strategy: r.identity === 0 ? 'bid+3' : r.strategy })));
  assert.deepEqual(a.auctionHistory, b.auctionHistory);
});

test('all deterministic formulas honor financing and change only auction decisions', () => {
  const s = createGame(4, 654), before = clone(s);
  for (const [id, definition] of Object.entries(AUCTION_FORMULAS).filter(([, d]) => d.kind !== 'mc')) {
    const limit = formulaLimit(s, id);
    assert.ok(limit >= 0 && limit <= capacity(s.players[s.actor]));
    if (definition.buffer !== undefined) assert.ok(limit <= capacity(s.players[s.actor]) - definition.buffer);
  }
  assert.deepEqual(s, before);
  s.bid = capacity(s.players[s.actor]);
  for (const id of Object.keys(AUCTION_FORMULAS).filter(id => !id.includes('-mc'))) assert.equal(chooseStrategy(s, id).type, 'pass');
});

test('auction features use public purchases rather than hidden rival initial cards', () => {
  const a = createGame(4, 99), b = clone(a);
  [b.players[1].shares, b.players[2].shares] = [b.players[2].shares, b.players[1].shares];
  assert.deepEqual(observation(a, 0), observation(b, 0));
  assert.deepEqual(formulaFeatures(a, 0), formulaFeatures(b, 0));
  b.players[1].bought[0] += 3;
  assert.notEqual(formulaFeatures(a, 0).relativeSpread, formulaFeatures(b, 0).relativeSpread);
});

test('sampled quotes hide rival cards and actual future dice and preserve live state', () => {
  const a = createGame(4, 99), b = clone(a), before = clone(a);
  [b.players[1].shares, b.players[2].shares] = [b.players[2].shares, b.players[1].shares];
  b.seed = 999999; b.diceIndex = 100;
  for (const id of ['formula-mcround-wealth', 'formula-mcfull-relative']) {
    assert.equal(formulaLimit(a, id, rng(123)), formulaLimit(b, id, rng(123)));
    assert.ok(formulaLimit(a, id, rng(456)) <= capacity(a.players[0]) - 6);
  }
  assert.deepEqual(a, before);
});

test('new model quotes terminate in real legal matches', () => {
  const match = playMatch({ lineup: ['formula-mcround-wealth', 'formula-b7-o70-c75', 'bid+3', 'bid-liquid+3'], seed: 456, validate: true });
  assert.equal(Math.max(...match.market), 5);
  assert.equal(match.results.reduce((v, r) => v + r.win, 0), 1);
});

test('cached continuation preserves exact decisions across financing, shares, and cargo markets', () => {
  for (let seed = 1; seed <= 80; seed++) {
    const s = createGame(4, seed), random = rng(seed);
    s.phase = 'setup'; s.captain = s.actor = seed % 4;
    s.market = Array.from({ length: 4 }, () => Math.floor(random() * 5));
    s.players[s.actor].cash = Math.floor(random() * 40);
    s.players[s.actor].mortgages = s.players[s.actor].shares.map(n => Math.floor(random() * (n + 1)));
    s.supply = s.supply.map(n => random() > .3 ? n : 0);
    for (const tieSeed of [1, 2, 123, 999]) assert.deepEqual(continuationPolicy(s, rng(tieSeed)), policy(s, 'aggressive', rng(tieSeed)));
  }
});
