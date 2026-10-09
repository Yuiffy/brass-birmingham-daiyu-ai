"""Freeze development selection, then audit a new paired tournament."""
import argparse,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from model import read_json,write_json
from compare_scores import compare_pair
from compare_geography import describe
from compare_improvement import grouped
from compare_runs import interval

STUDY=Path('world_model/experiments/dynamics-study')
EXPERIMENTS=STUDY.parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def select():
    target=STUDY/'selection.json'
    if target.exists():raise ValueError('Selection already frozen')
    protocol=read_json(STUDY/'protocol.json');records=[]
    for name,file in [('geography-pool-v2',STUDY/'baseline-world-dev.json')]+[(n,EXPERIMENTS/n/'world-dev.json') for n in ['dynamics-wide-baseline','dynamics-v1','dynamics-wide-v1']]:
        report=read_json(file)
        if report['config']['seed']!=protocol['developmentSeed'] or report['completedGames']!=protocol['developmentGames']:raise ValueError('Wrong development games')
        artifact=EXPERIMENTS/name/'models/world-model.json'
        if report['artifacts']['world-model.json']['sha256']!=digest(artifact):raise ValueError('Development artifact changed')
        row=next(r for r in report['rows'] if r['type']=='world')
        records.append(dict(name=name,**describe(report,'world'),winRate=row['winRate'],sha256=digest(artifact)))
    baseline=records[0]
    eligible=[r for r in records[1:] if r['mean']>baseline['mean'] and r['withoutIncomeBonus']>baseline['withoutIncomeBonus'] and r['winRate']>=baseline['winRate']-.05]
    chosen=max(eligible,key=lambda r:r['withoutIncomeBonus']) if eligible else baseline
    result=dict(selected=chosen['name'],worldModelSHA256=chosen['sha256'],frozenAt=datetime.now(timezone.utc).isoformat(),development=records,criterion=protocol['selection'])
    write_json(target,result)
    print([(r['name'],r['mean'],r['withoutIncomeBonus'],r['winRate']) for r in records]);print('Frozen:',chosen['name'])

def paired(old,new,actor,indices):
    seeds,_,_=grouped(new);result=compare_pair(old,new,actor,indices)
    def values(report):return np.array([np.mean([p['vpWithoutIncomeBonus'] for g in report['games'] if g['seed']==s for p in g['scores'] if p['type']==actor]) for s in seeds])
    result.update(baselineScores=describe(old,actor),candidateScores=describe(new,actor),withoutIncomeBonusGain=interval(values(new)-values(old),indices))
    return result

def compare(out):
    protocol=read_json(STUDY/'protocol.json');selection=read_json(STUDY/'selection.json')
    candidate=EXPERIMENTS/selection['selected'];expected=selection['worldModelSHA256']
    if expected!=digest(candidate/'models/world-model.json'):raise ValueError('Frozen checkpoint changed')
    source=set()
    for name in ['data-10k','data-geography-league','data-dynamics-league']:
        source.update(g['seed'] for g in read_json(Path('world_model')/name/'games.json'))
    records=[]
    for actor in ['world','guided']:
        old=read_json(STUDY/f'baseline-{actor}-final.json');new=read_json(candidate/f'{actor}-final.json')
        seeds,_,_=grouped(new)
        if new['config']['seed']!=protocol['finalSeed'] or new['completedGames']!=protocol['finalGames']:raise ValueError('Wrong final games')
        if set(seeds)&source:raise ValueError('Final games overlap learning data')
        if new['artifacts']['world-model.json']['sha256']!=expected or old['artifacts']['world-model.json']['sha256']!=protocol['baselineSHA256']:raise ValueError('Wrong final checkpoint')
        indices=np.random.default_rng(92017).integers(0,len(seeds),size=(20000,len(seeds)))
        records.append(paired(old,new,actor,indices))
    wide=read_json(EXPERIMENTS/'dynamics-wide-baseline/world-final.json')
    new=read_json(candidate/'world-final.json')
    if wide['artifacts']['world-model.json']['sha256']!=digest(EXPERIMENTS/'dynamics-wide-baseline/models/world-model.json'):raise ValueError('Wrong ablation checkpoint')
    ablation=paired(wide,new,'world',indices)
    prediction=[]
    for label,old_dir,new_dir in [('stronger-games','baseline-prediction-new','prediction-new'),('original-games','baseline-prediction-old','prediction-old')]:
        old=read_json(STUDY/old_dir/'evaluation.json');new=read_json(candidate/new_dir/'evaluation.json')
        if old['testGameSeeds']!=new['testGameSeeds']:raise ValueError('Prediction samples differ')
        prediction.append(dict(distribution=label,before=old,after=new))
    result=dict(protocol=protocol,selection=selection,runs=records,dynamicsAtSameSearchBudget=ablation,prediction=prediction,
        bootstrapSamples=20000,bootstrapSeed=92017,
        notes=['Paired 95% percentile bootstrap over 100 complete four-seat seed groups.',
               'The value head and three opponent algorithms/weights are frozen; new dynamics and continuation candidate width are the only playing changes.',
               'Prediction improvement does not guarantee better decisions. The same-width ablation separates dynamics retraining from extra search work.',
               'The existing engine and its income bonus are unchanged. Bonus-excluded VP is not full official-rule scoring.'])
    write_json(out,result)
    for r in records+[ablation]:print(r['type'],r['baselineScores']['mean'],'->',r['candidateScores']['mean'],'VP',r['scoreGain'],'earned',r['withoutIncomeBonusGain'],'wins',r['winRateChange'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');p.add_argument('--out',default='world_model/reports/dynamics-training.json');args=p.parse_args()
    select() if args.select else compare(args.out)
