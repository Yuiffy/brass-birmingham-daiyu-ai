import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { AUCTION_FORMULAS } from '../auction-formulas.mjs';
import { runLeague } from './league-batch.mjs';

const root = path.resolve(process.argv[2] || 'output/formulas');
const workers = Number(process.argv[3] || 8), stage = process.argv[4] || 'screen';
const read = async name => JSON.parse(await readFile(path.join(root, name, 'selection.json'), 'utf8'));
const run = async spec => {
  await mkdir(path.join(root, 'specs'), { recursive: true });
  await writeFile(path.join(root, 'specs', spec.stage + '.json'), JSON.stringify(spec, null, 2) + '\n');
  return runLeague(spec, root, workers);
};
if (stage === 'screen') {
  const challengers = ['bid+3', 'bid-liquid+3', ...Object.keys(AUCTION_FORMULAS).filter(id => AUCTION_FORMULAS[id].kind !== 'mc')];
  await run({ stage: '01-resident-screen', format: 'challenge', seeds: 64, base: 101700001, challengers, opponents: Array(3).fill('bid+3'), resident: 'bid+3', selectionReason: 'All deterministic formulas; frozen +3 residents and matched self-play controls' });
  await run({ stage: '02-mixed-screen', format: 'challenge', seeds: 64, base: 102700001, challengers, opponents: ['bid+3', 'bid-liquid+3', 'bid+6'], selectionReason: 'All deterministic formulas; distinct fresh seeds and mixed aggressive opponents' });
} else if (stage === 'pair') {
  const a = await read('01-resident-screen'), b = await read('02-mixed-screen');
  const scores = a.challenge.filter(r => r.strategy.startsWith('formula-') && r.strategy !== 'formula-legacy').map(r => ({ id: r.strategy, win: (r.winRate + b.challenge.find(v => v.strategy === r.strategy).winRate) / 2 }));
  scores.sort((a, b) => b.win - a.win || a.id.localeCompare(b.id));
  await run({ stage: '03-survivor-pair', format: 'pair', seeds: 48, base: 103700001, strategies: ['bid+3', 'bid-liquid+3', ...scores.slice(0, 6).map(r => r.id)], selectionReason: 'Top six average challenger win rates across both screens; now directly compete in all four compositions' });
} else if (stage === 'models') {
  const prior = await read('03-survivor-pair'), top = prior.summary.filter(r => r.strategy.startsWith('formula-')).slice(0, 2).map(r => r.strategy);
  await run({ stage: '04-model-qualification', format: 'challenge', seeds: 48, base: 104700001, challengers: ['bid+3', 'bid-liquid+3', ...top, ...Object.keys(AUCTION_FORMULAS).filter(id => AUCTION_FORMULAS[id].kind === 'mc')], opponents: ['bid+3', ...top], selectionReason: 'Terminal/one-voyage counterfactual quotes; identical seed/hand controls against selected strong formulas' });
} else if (stage === 'confirm') {
  const pair = await read('03-survivor-pair');
  const formulas = pair.summary.filter(r => r.strategy.startsWith('formula-'));
  const finalists = ['bid+3', 'bid-liquid+3', ...formulas.slice(0, 2).map(r => r.strategy)];
  await run({ stage: '05-independent-confirmation', format: 'mixed', seeds: 128, base: 105700001, finalists, selectionReason: 'Top two deterministic formulas from direct pair league; 128 fresh seeds x all 24 policy-to-hand/seat assignments. Separate model confirmation uses the same original and deterministic winner controls.' });
} else if (stage === 'model-confirm') {
  const pair = await read('03-survivor-pair'), model = await read('04-model-qualification');
  const best = pair.summary.find(r => r.strategy.startsWith('formula-')).strategy;
  const models = model.challenge.filter(r => AUCTION_FORMULAS[r.strategy]?.kind === 'mc').slice(0, 2).map(r => r.strategy);
  await run({ stage: '06-model-confirmation', format: 'mixed', seeds: 48, base: 106700001, finalists: ['bid+3', best, ...models], selectionReason: 'Top two sampled formulas versus original +3 and best deterministic formula; 48 fresh seeds x all 24 assignments. Smaller predeclared compute budget than deterministic confirmation; never pool their ranks.' });
} else if (stage === 'linear-confirm') {
  const pair = await read('03-survivor-pair');
  const best = pair.summary.filter(r => AUCTION_FORMULAS[r.strategy]?.kind === 'linear').slice(0, 2).map(r => r.strategy);
  await run({ stage: '07-linear-confirmation', format: 'mixed', seeds: 128, base: 107700001, finalists: ['bid+3', ...best, 'formula-b7-o70-c0'], selectionReason: 'Independent extension: top two linear candidates from the pair league and direct control-term ablation, separately from fixed-cap finalists. Freeze before reading new seeds; preserve all earlier stages.' });
} else if (stage === 'final-audit') {
  const model = await read('06-model-confirmation'), linear = await read('07-linear-confirmation');
  const bestModel = model.summary.find(r => AUCTION_FORMULAS[r.strategy]?.kind === 'mc').strategy;
  const bestLinear = linear.summary.find(r => AUCTION_FORMULAS[r.strategy]?.kind === 'linear').strategy;
  await run({ stage: '10-cross-confirmation', format: 'mixed', seeds: 64, base: 110700001, finalists: ['bid+3', 'bid-liquid+3', bestLinear, bestModel], selectionReason: 'Freeze winners of both formula families; 64 unseen seeds x all 24 assignments. Directly compare model winner, linear winner and both original references in one strong opponent pool.' });
  await run({ stage: '11-model-residents', format: 'challenge', seeds: 32, base: 111700001, resident: bestModel, challengers: [bestModel, bestLinear, 'bid+3'], opponents: Array(3).fill(bestModel), selectionReason: 'Selected model is copied into all three opponents; matched self-play control, 32 fresh seed groups x four seats. Predeclared smaller compute budget; intervals may be broad.' });
} else throw Error('Use screen, pair, models, confirm, model-confirm, linear-confirm, or final-audit');
