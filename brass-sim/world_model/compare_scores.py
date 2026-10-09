"""Compare absolute final VP with identical opponents, seeds and seat rotations."""
import argparse
from pathlib import Path
import numpy as np
from model import read_json,write_json
from compare_improvement import grouped
from compare_runs import interval


def compare_pair(old,new,actor,indices):
    if old['config']!=new['config']:raise ValueError('Paired score configurations differ')
    if old['artifacts']['neural-policy.json']['sha256']!=new['artifacts']['neural-policy.json']['sha256']:raise ValueError('Opponent weights differ')
    a,aw,av=grouped(old);b,bw,bv=grouped(new)
    if a!=b:raise ValueError('Paired seeds differ')
    extract=lambda r:next(x for x in r['rows'] if x['type']==actor)
    opponent_mean=lambda values:np.mean([v for t,v in values.items() if t!=actor],0)
    return dict(type=actor,games=new['completedGames'],seedGroups=len(a),config=new['config'],
        baseline=extract(old),candidate=extract(new),scoreGain=interval(bv[actor]-av[actor],indices),
        opponentAverageGain=interval(opponent_mean(bv)-opponent_mean(av),indices),
        winRateChange=interval(bw[actor]-aw[actor],indices),baselineArtifacts=old['artifacts'],candidateArtifacts=new['artifacts'])


def compare(args):
    baseline=Path(args.baseline);candidate=Path(args.candidate);pairs=[];seeds=None
    for actor in ['world','guided']:
        old=read_json(baseline/f'baseline-{actor}-final.json');new=read_json(candidate/f'{actor}-final.json')
        current,_,_=grouped(new)
        if seeds is not None and current!=seeds:raise ValueError('Experimental seed sets differ')
        seeds=current;pairs.append((old,new,actor))
    source=read_json(Path(args.data)/'games.json')+read_json(Path(args.replay)/'games.json')
    if set(seeds).intersection(g['seed'] for g in source):raise ValueError('Final evaluation overlaps learning data')
    indices=np.random.default_rng(92017).integers(0,len(seeds),size=(20000,len(seeds)))
    results=[compare_pair(*pair,indices) for pair in pairs]
    result=dict(primaryMetric='mean final victory points',runs=results,bootstrapSamples=20000,bootstrapSeed=92017,
        method='Paired 95% percentile bootstrap over four-game seed groups; fixed heuristic, search and neural opponents.',
        notes=['World and guided search are separately substituted into the same fourth slot against the same three opponent algorithms.',
            'Opponent behavior changes in response to changed game states; their weights and algorithms remain fixed.',
            'Absolute VP is the primary criterion requested by the user. Win rate and opponent scores are secondary.',
            'No candidate was chosen using final evaluation scores; development seed 48000001, final seed 68000001.',
            'Intervals cover game-seed sampling for these weights and this opponent lineup, not training randomness or all playing conditions.'])
    write_json(args.out,result)
    for r in results:print(r['type'],r['baseline']['averageVP'],'->',r['candidate']['averageVP'],'gain',r['scoreGain'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='world_model/experiments/score-study')
    p.add_argument('--data',default='world_model/data-10k');p.add_argument('--replay',default='world_model/data-score-league')
    p.add_argument('--candidate',required=True);p.add_argument('--out',default='world_model/reports/score-training.json');compare(p.parse_args())
