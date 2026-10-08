"""Compare strong-only lineups with paired, complete-seed-group uncertainty."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from model import read_json, write_json


def compare(baseline, candidate):
    for key in ('games', 'seed', 'depth', 'width', 'types'):
        if baseline['config'][key] != candidate['config'][key]:
            raise ValueError(f'Mismatched {key}')
    if baseline['config'].get('rulesVersion','legacy-v1')!=candidate['config'].get('rulesVersion','legacy-v1'):raise ValueError('Different rules versions')
    players = len(baseline['config']['types'])
    if set(baseline['config']['types']) != {'guided', 'world'}:
        raise ValueError('Expected guided/world entrants only')
    for report in (baseline, candidate):
        if report['completedGames'] != report['config']['games'] or report['cancelled']:
            raise ValueError('Incomplete tournament')
        if len(report['games']) != report['completedGames']:
            raise ValueError('Missing game records')
    for before, after in zip(baseline['games'], candidate['games']):
        if (before['game'], before['seed'], before['seats']) != (after['game'], after['seed'], after['seats']):
            raise ValueError('Unpaired games or rotations')
    seeds = sorted({g['seed'] for g in baseline['games']})
    for seed in seeds:
        if sum(g['seed'] == seed for g in baseline['games']) != players:
            raise ValueError('Incomplete seed rotation')
    indices = np.random.default_rng(92017).integers(0, len(seeds), size=(20000, len(seeds)))

    def scores(report, actor, field, seed=None):
        return np.array([s[field] for g in report['games'] if seed is None or g['seed'] == seed
                         for s in g['scores'] if actor == 'all' or s['type'] == actor], dtype=float)

    def describe(report, actor):
        vp = scores(report, actor, 'vp')
        result = dict(appearances=len(vp), averageVP=float(vp.mean()),
                      earnedVP=float(scores(report, actor, 'vpWithoutIncomeBonus').mean()),
                      medianVP=float(np.median(vp)), p90VP=float(np.quantile(vp, .9, method='lower')),
                      maxVP=float(vp.max()), atLeast100=float((vp >= 100).mean()),
                      atLeast150=float((vp >= 150).mean()))
        if actor != 'all':
            result['winRate'] = next(row['winRate'] for row in report['rows'] if row['type'] == actor)
        return result

    results = {}
    for actor in ('guided', 'world', 'all'):
        gains = {}
        for field in ('vp', 'vpWithoutIncomeBonus'):
            delta = np.array([scores(candidate, actor, field, seed).mean() -
                              scores(baseline, actor, field, seed).mean() for seed in seeds])
            bounds = np.quantile(delta[indices].mean(1), [.025, .975])
            gains[field] = dict(mean=float(delta.mean()), ci95=bounds.tolist())
        results[actor] = dict(baseline=describe(baseline, actor), candidate=describe(candidate, actor), gain=gains)
    return dict(generatedAt=datetime.now(timezone.utc).isoformat(), config=baseline['config'],
                independentSeedGroups=len(seeds), bootstrapSamples=20000, bootstrapSeed=92017,
                artifacts=dict(baseline=baseline.get('artifacts'), candidate=candidate.get('artifacts')),
                results=results,
                notes=['95% percentile bootstrap over paired seed groups, retaining all seats and rotations together.',
                       'Repeated entrants and identical lineups are not independent observations.',
                       'When both agents change weights, gains describe the whole lineup; a weaker opponent can raise another agent\'s score.',
                       'VP without the terminal income bonus is still this simulator\'s score, not an official-rules correction.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(read_json(args.baseline), read_json(args.candidate))
    result['sources'] = dict(baseline=args.baseline, candidate=args.candidate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    for actor, values in result['results'].items():
        print(actor, values['baseline']['averageVP'], '->', values['candidate']['averageVP'], values['gain']['vp'])
