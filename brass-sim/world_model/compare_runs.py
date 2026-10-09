"""Compare fixed-seed tournaments with bootstrap intervals clustered by seed."""
import argparse
from pathlib import Path
import numpy as np
from model import read_json, write_json


def seed_metrics(report):
    if report['cancelled'] or not report['balancedSeats']:
        raise ValueError('Use complete, balanced tournaments')
    grouped = {}
    for game in report['games']:
        scores = {p['type']: p['vp'] for p in game['scores']}
        if len(scores) != 4: raise ValueError('Expected four distinct AI entrants')
        top = max(scores.values()); ties = sum(v==top for v in scores.values())
        row = grouped.setdefault(game['seed'],[])
        row.append([(int(scores['world']==top)-int(scores['neural']==top))/ties,
                    scores['world']-scores['neural'], scores['world'], scores['neural']])
    if any(len(v)!=4 for v in grouped.values()): raise ValueError('Each seed needs four seat rotations')
    return sorted(grouped),np.array([np.mean(grouped[k],axis=0) for k in sorted(grouped)])


def interval(values, indices):
    means=values[indices].mean(axis=1)
    return dict(estimate=float(values.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)))


def compare(args):
    runs=[]; seeds=None; matrices=[]; common_config=None
    for spec in args.run:
        name,directory=spec.split('=',1); directory=Path(directory)
        report=read_json(directory/'reports/tournament-400.json')
        config={k:report['config'][k] for k in ['games','seed','depth','width','types']}
        if common_config is not None and config!=common_config: raise ValueError('Tournament configurations differ')
        common_config=config
        sk,m=seed_metrics(report)
        if seeds is not None and sk!=seeds: raise ValueError('Tournaments must use identical seeds')
        seeds=sk; matrices.append(m)
        evaluation=read_json(directory/'reports/evaluation.json')
        training=read_json(directory/'models/training.json')
        wm=read_json(directory/'models/world-model.json')
        policy=read_json(directory/'models/neural-policy.json')
        runs.append(dict(name=name,directory=directory.as_posix(),completedGames=report['completedGames'],
            datasetGames=wm['datasetGames'],trainingGames=wm['trainingGames'],trainingRows=wm['trainingRows'],
            worldEpoch=wm['epoch'],policyEpoch=policy['epoch'],continuationConfig=training['config'],
            testGames=evaluation['testGames'],normalizedMSE=evaluation['normalizedMSE'],
            currentPlayerAccuracy=evaluation['currentPlayerAccuracy'],changeF1=evaluation['overall']['changeF1'],
            moneyMAE=evaluation['groups']['money']['mae'],rollout=evaluation['rollout'],
            rows=report['rows']))
    rng=np.random.default_rng(args.seed)
    indices=rng.integers(0,len(seeds),size=(args.bootstrap,len(seeds)))
    for run,m in zip(runs,matrices):
        run['worldMinusNeuralWinRate']=interval(m[:,0],indices)
        run['worldMinusNeuralVP']=interval(m[:,1],indices)
        run['worldVPGainVsOriginal']=interval(m[:,2]-matrices[0][:,2],indices)
        run['gapChangeVsOriginal']=interval(m[:,0]-matrices[0][:,0],indices)
    result=dict(runs=runs,independentSeedGroups=len(seeds),bootstrapSamples=args.bootstrap,bootstrapSeed=args.seed,
        method='Paired bootstrap over four-game seed groups, percentile 95% intervals. Ties share one win.',
        notes=['Both networks are continued in each continuation run; common engine/search/shortlist/evaluator are unchanged.',
               'The two continuation schedules use different numbers of gradient updates; this is not a compute-matched causal ablation.',
               'Four-player tournament win rates are not head-to-head probabilities.',
               'Tournaments differ in both learned opponents, so VP gains are not isolated causal effects.',
               'Intervals measure seed sampling uncertainty for these weights, not variance across training random seeds.',
               'Prediction metrics use the original 100 held-out test games; checkpoints are selected only by validation loss.'])
    write_json(args.out,result)
    if args.markdown:
        latest=runs[-1]; gap=latest['worldMinusNeuralWinRate']; change=latest['gapChangeVsOriginal']
        signed=lambda v:f'{v*100:+.2f}'
        ci=lambda v:f"{signed(v['estimate'])} [{signed(v['low'])}, {signed(v['high'])}]"
        conclusion='本批对局支持世界模型领先。' if gap['low']>0 else '本批对局支持神经网络领先。' if gap['high']<0 else '95% 区间包含 0，尚未拉开可靠差距。'
        lines=['# 世界模型续训实验','',
            f"扩展数据版的世界模型与神经网络胜率相差 {signed(gap['estimate'])} 个百分点。{conclusion}",
            f"相对原版，胜率差变化 {ci(change)} 个百分点；方括号为 95% 区间。",'',
            f"相对原版，单步测试 MSE 下降 {(1-latest['normalizedMSE']/runs[0]['normalizedMSE'])*100:.1f}%；五步 MAE 变化 {(latest['rollout'][4]['normalizedMAE']/runs[0]['rollout'][4]['normalizedMAE']-1)*100:+.1f}%。单步拟合的改善没有保证多步推演和棋力改善。",'',
            '## 实际完成的训练','',
            '两条续训分支均从原版权重开始，保留 836 → 96 → 689 的世界模型与 836 → 96 → 1 的动作网络。',
            '世界模型学习状态差分；动作网络继续拟合原启发式评分。没有新增奖励强化学习，也没有更换搜索或评价函数。','',
            '| 版本 | 数据池局数 | 实际训练局数 / 行数 | 本轮训练轮数 | 保存的世界 / 动作网络累计轮数 |',
            '|---|---:|---:|---:|---:|']
        for i,r in enumerate(runs):
            lines.append(f"| {r['name']} | {r['datasetGames']:,} | {r['trainingGames']:,} / {r['trainingRows']:,} | {r['continuationConfig']['epochs']} | {r['worldEpoch']} / {r['policyEpoch']} |")
        lines += ['',
            '续训 batch 512，学习率从 0.0003 余弦下降到 0.00003。两条分支只按原来 100 局验证集的指标各自保存最佳权重；测试结果和比赛成绩不参与选轮。',
            '原版未保存 Adam 状态，因此这次两条分支加载原权重后重置优化器；本次已保存与最佳权重匹配的优化器状态，后续可以完整恢复。差分尺度沿用原训练集拟合值。',
            '1,000 局再训 72 轮与 10,000 局再训 24 轮的更新次数不同。这是两种续训方案比较，不能把差异全部归因于数据量，也不是等计算量消融。','',
            '## 相同 400 局对战','',
            '每版使用相同 100 个种子组，每组 4 局轮换座位。四种 AI 同场，深度 2、根候选宽度 8；并列冠军平分一次胜利。',
            '这些胜率是四人比赛中夺冠的比例，不是两种 AI 一对一对战的胜率。','',
            '| 版本 | 搜索树胜率 | 世界模型胜率 | 神经网络胜率 | 原启发式胜率 | 世界 − 神经 / 百分点（95% 区间） |',
            '|---|---:|---:|---:|---:|---:|']
        for r in runs:
            row={x['type']:x for x in r['rows']}
            rates=' | '.join(f"{row[k]['winRate']:.2%}" for k in ['search','world','neural','heuristic'])
            lines.append(f"| {r['name']} | {rates} | {ci(r['worldMinusNeuralWinRate'])} |")
        lines += ['', '| 版本 | 搜索树平均 VP | 世界模型平均 VP | 神经网络平均 VP | 原启发式平均 VP |',
                  '|---|---:|---:|---:|---:|']
        for r in runs:
            row={x['type']:x for x in r['rows']}
            lines.append(f"| {r['name']} | "+' | '.join(f"{row[k]['averageVP']:.2f}" for k in ['search','world','neural','heuristic'])+' |')
        lines += ['',
            f"区间使用 {args.bootstrap:,} 次 bootstrap，以同一种子的四局为一个整体重采样，版本间采用配对采样。它只反映这组权重的棋局抽样波动，不覆盖训练随机种子之间的差异。",
            '两种网络在续训版本中同时变化，对手也随之变化，因此平均 VP 增益不是某一个模型升级的独立因果效应。并行比赛按整组分配，测试已核对与串行产生完全相同的逐局成绩；耗时会受到 CPU 争用影响。','',
            '## 相同留出测试集的预测误差','',
            '所有模型都在最初 100 局 / 12,800 条留出 transition 上评估。10,000 局数据的前 1,000 局保持原种子与划分；其训练集合不含原验证或测试局。','',
            '| 版本 | 测试 MSE | 下一行动玩家准确率 | 金钱 MAE / 英镑 | 变化检测 F1 | 1 步 MAE | 5 步 MAE |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in runs:
            lines.append(f"| {r['name']} | {r['normalizedMSE']:.6f} | {r['currentPlayerAccuracy']:.2%} | {r['moneyMAE']:.3f} | {r['changeF1']:.2%} | {r['rollout'][0]['normalizedMAE']:.5f} | {r['rollout'][4]['normalizedMAE']:.5f} |")
        lines += ['',
            f"同一 rollout 窗口的状态不变基线：1 步 MAE {latest['rollout'][0]['persistenceMAE']:.5f}，5 步 {latest['rollout'][4]['persistenceMAE']:.5f}。多数棋盘字段不变，不能只用总体准确率评价模型。",
            '固定动作序列包含真实合法候选的描述与成本，但所有中间状态输入均来自前一步网络预测。实际自由推演还会受到离散状态投影与候选生成误差影响。','',
            '## 文件与复现','',
            '对战实验室可切换原版、只增加轮数版、扩展数据版，查看已保存结果或运行新的比赛。','',
            '本次交付的游戏默认加载 10,000 局续训版，刷新游戏页即可载入新权重。它作为最新实验版本提供，当前对战结果没有显示其更强。原版仍保存在 experiments/v1。','',
            '- `world_model/experiments/v1/`：原模型与原始报告、扩充后的 400 局报告。',
            '- `world_model/experiments/longer-1k/`：同一数据续训 72 轮的权重、优化器、日志及比赛。',
            '- `world_model/experiments/expanded-10k/`：10,000 局续训权重、优化器、日志及比赛。',
            '- `world_model/reports/continuation.json`：完整汇总与置信区间。',
            '- `world_model/data-10k/`：本地生成的 1,280,000 条数据，已从 Git 排除。','',
            '以下复现命令使用新目录，以免覆盖已完成的结果。','',
            '```sh',
            'node world_model/cli.js generate --games 10000 --players 4 --seed 1701 --ai mixed --out world_model/data-repro-10k',
            'node world_model/cli.js train --data world_model/data-repro-10k --validation-data world_model/data --resume world_model/experiments/v1/models --epochs 24 --hidden 96 --lr 0.0003 --final-lr 0.00003 --out world_model/repro-10k/models',
            'node world_model/cli.js evaluate --data world_model/data --models world_model/repro-10k/models --out world_model/repro-10k/reports',
            'node world_model/cli.js benchmark --games 400 --seed 800001 --depth 2 --width 8 --workers 4 --models world_model/repro-10k/models --out world_model/repro-10k/reports/tournament-400.json',
            '```','',
            '再次续训可把 `--resume` 指向本次模型目录，并指定新的 `--out`；存在优化器文件时会自动恢复。模型仍是实验性的小网络，没有学会可靠的完整游戏规则。']
        Path(args.markdown).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for run in runs:
        print(run['name'], 'MSE',run['normalizedMSE'],'world-neural win gap',run['worldMinusNeuralWinRate'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True)
    p.add_argument('--out',default='world_model/reports/continuation.json')
    p.add_argument('--markdown')
    p.add_argument('--bootstrap',type=int,default=20000);p.add_argument('--seed',type=int,default=92015)
    compare(p.parse_args())
