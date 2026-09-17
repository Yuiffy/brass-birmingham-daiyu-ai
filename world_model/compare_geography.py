"""Audit the frozen higher-score experiment against matched fresh games."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from model import read_json,write_json
from compare_scores import compare_pair
from compare_runs import interval
from compare_improvement import grouped


def describe(report,actor):
    scores=[next(p for p in g['scores'] if p['type']==actor) for g in report['games']]
    vp=np.array([p['vp'] for p in scores]);earned=np.array([p['vpWithoutIncomeBonus'] for p in scores])
    components={k:float(np.mean([p['components'][k] for p in scores])) for k in ['canalIndustry','canalLinks','railIndustry','railLinks']}
    return dict(mean=float(vp.mean()),median=float(np.median(vp)),p10=float(np.quantile(vp,.1)),p90=float(np.quantile(vp,.9)),maximum=int(vp.max()),
        atLeast150=float((vp>=150).mean()),withoutIncomeBonus=float(earned.mean()),withoutBonusAtLeast150=float((earned>=150).mean()),
        incomeBonus=float(np.mean([p['incomeBonus'] for p in scores])),components=components,
        actions={k:float(np.mean([p['components']['actions'].get(k,0) for p in scores])) for k in ['build','network','sell','develop','loan','scout','pass']})


def compare(args):
    base=Path(args.baseline);candidate=Path(args.candidate);protocol=read_json(base/'protocol.json');selection=read_json(base/'selection.json')
    expected=hashlib.sha256((candidate/'models/world-model.json').read_bytes()).hexdigest()
    if selection['worldModelSHA256']!=expected:raise ValueError('Checkpoint differs from frozen selection')
    heldout=set();source_dirs=['world_model/data-10k','world_model/data-geography-league']
    for source in source_dirs:heldout.update(g['seed'] for g in read_json(Path(source)/'games.json'))
    records=[]
    for actor in ['world','guided']:
        old=read_json(base/f'baseline-{actor}-final.json');new=read_json(candidate/f'{actor}-final.json')
        seeds,_,_=grouped(new)
        if new['config']['seed']!=protocol['finalSeed'] or new['completedGames']!=protocol['finalGamesPerVersionAndAgent']:raise ValueError('Final protocol differs')
        if set(seeds)&heldout:raise ValueError('Final matches overlap learning data')
        if new['artifacts']['world-model.json']['sha256']!=expected:raise ValueError('Final matches used another checkpoint')
        indices=np.random.default_rng(92017).integers(0,len(seeds),size=(20000,len(seeds)))
        result=compare_pair(old,new,actor,indices)
        def values(report):
            return np.array([np.mean([p['vpWithoutIncomeBonus'] for g in report['games'] if g['seed']==s for p in g['scores'] if p['type']==actor]) for s in seeds])
        result.update(baselineScores=describe(old,actor),candidateScores=describe(new,actor),withoutIncomeBonusGain=interval(values(new)-values(old),indices))
        records.append(result)
    result=dict(runs=records,protocol=protocol,selection=selection,bootstrapSamples=20000,bootstrapSeed=92017,
        sources=[dict(title='Four-player firsthand scores',url='https://www.reddit.com/r/boardgames/comments/i7cy6f/brass_birmingham_4_player_typical_scores/'),
                 dict(title='Roxley rulebook, mirrored PDF, pp. 6–7 and 12',url='https://bghub.org/r/brassbirmingham.pdf')],
        limitation='Engine scoring is preserved. Removing its final income bonus alone does not make these games equivalent to official tabletop play. Human accounts are not a representative sample.')
    write_json(args.out,result)
    for r in records:print(r['type'],r['baselineScores'],'->',r['candidateScores'],'gain',r['scoreGain'],'excluding bonus',r['withoutIncomeBonusGain'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='world_model/experiments/geography-study')
    p.add_argument('--candidate',required=True);p.add_argument('--out',default='world_model/reports/higher-scores.json');compare(p.parse_args())
