"""Independent final evaluation, with paired bootstrap over complete seat rotations."""
import argparse
from pathlib import Path
import numpy as np
from model import read_json,write_json
from compare_runs import interval


def grouped(report):
    if report['cancelled'] or not report['balancedSeats']:raise ValueError('Complete balanced tournaments required')
    types=report['config']['types'];groups={}
    for g in report['games']:
        scores={p['type']:p['vp'] for p in g['scores']};top=max(scores.values());ties=sum(v==top for v in scores.values())
        if set(scores)!=set(types):raise ValueError('Missing entrant')
        groups.setdefault(g['seed'],[]).append((g['seats'],[float(scores[t]==top)/ties for t in types]+[scores[t] for t in types]))
    for rows in groups.values():
        if len(rows)!=4 or any({r[0].index(t) for r in rows}!={0,1,2,3} for t in types):raise ValueError('Incomplete seat rotation')
    seeds=sorted(groups);matrix=np.array([np.mean([r[1] for r in groups[s]],0) for s in seeds])
    return seeds,{t:matrix[:,i] for i,t in enumerate(types)},{t:matrix[:,i+4] for i,t in enumerate(types)}


def main(args):
    base=Path('world_model/experiments/expanded-10k');new=Path('world_model/experiments/structured-value-10k')
    paths=[base/'reports/final-400.json',new/'reports/final-400.json',new/'reports/guided-final-400.json']
    reports=[read_json(p) for p in paths];groups=[grouped(r) for r in reports]
    if any(g[0]!=groups[0][0] for g in groups):raise ValueError('Seeds differ')
    if reports[0]['config']!=reports[1]['config']:raise ValueError('Baseline and upgraded tournament configurations differ')
    if len({r['artifacts']['neural-policy.json']['sha256'] for r in reports})!=1:raise ValueError('Neural opponent was not frozen')
    seeds=groups[0][0];train_games=read_json('world_model/data-10k/games.json')
    if set(seeds).intersection(g['seed'] for g in train_games):raise ValueError('Final tournament overlaps source data')
    old_eval=read_json(base/'reports/fresh-test/evaluation.json');new_eval=read_json(new/'reports/evaluation.json')
    if old_eval['testGameSeeds']!=new_eval['testGameSeeds']:raise ValueError('Prediction test sets differ')
    fitted={g['seed'] for g in train_games if g['split']=='train'}
    if fitted.intersection(new_eval['testGameSeeds']):raise ValueError('Prediction-test leakage')
    rng=np.random.default_rng(92016);indices=rng.integers(0,len(seeds),size=(20000,len(seeds)))
    ci=lambda x:interval(x,indices)
    names=['旧版：10k 普通差分模型','新版：结构化预测＋终局价值','新版：加入学习增强搜索']
    result=dict(gamesPerRun=400,independentSeedGroups=len(seeds),bootstrapSamples=20000,bootstrapSeed=92016,
        method='Percentile 95% intervals, bootstrap over four-game seed groups. Paired sampling across model versions. Ties share wins.',
        runs=[dict(name=n,path=p.as_posix(),rows=r['rows'],config=r['config'],artifacts=r['artifacts']) for n,p,r in zip(names,paths,reports)],
        worldMinusNeural=ci(groups[1][1]['world']-groups[1][1]['neural']),
        worldWinGain=ci(groups[1][1]['world']-groups[0][1]['world']),
        worldVPGain=ci(groups[1][2]['world']-groups[0][2]['world']),
        worldMinusSearch=ci(groups[1][1]['world']-groups[1][1]['search']),
        guidedMinusSearch=ci(groups[2][1]['guided']-groups[2][1]['search']),
        guidedMinusSearchVP=ci(groups[2][2]['guided']-groups[2][2]['search']),
        guidedMinusWorld=ci(groups[2][1]['guided']-groups[2][1]['world']),
        prediction=dict(testGames=new_eval['testGames'],testTransitions=new_eval['testTransitions'],testStart=new_eval['testStart'],
            testGameSeeds=new_eval['testGameSeeds'],baseline=old_eval,upgraded=new_eval),
        notes=['Weights and architecture were frozen before these final tournaments; development games used seed 800001.',
            'A four-player win rate is not a head-to-head probability. Different lineups cannot be directly ranked.',
            'Confidence intervals cover game-seed sampling, not training-seed variance or all opponent policies.',
            'The neural baseline distills heuristic action scores. This does not compare against a strong policy trained by reinforcement learning.',
            'Four-player, all-hands-public observation setting; future deck order and game RNG hidden. Same depth/width, unequal computation time.'])
    write_json(args.out,result)
    for k in ['worldMinusNeural','worldWinGain','worldVPGain','guidedMinusSearch','guidedMinusSearchVP','guidedMinusWorld']:print(k,result[k])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='world_model/reports/improvement.json');main(p.parse_args())
