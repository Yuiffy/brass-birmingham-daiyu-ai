"""Render the published evolutionary league report. Requires matplotlib >=3.10."""
import sys
import json
import gzip
from pathlib import Path
local = Path(__file__).resolve().parents[1] / 'output' / 'python-deps'
if local.exists(): sys.path.insert(0, str(local))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

root = Path(sys.argv[1] if len(sys.argv)>1 else 'reports/2026-10-07-evolution')
report = json.loads((root/'summary.json').read_text(encoding='utf-8'))
stages = {s['stage']:s for s in report['stages']}
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
screen=stages['01-screen']
order=sorted(screen['summary'],key=lambda r:int(r['strategy'].split('+')[1]))
ids=[r['strategy'] for r in order]
matrix=np.full((len(ids),len(ids)),np.nan)
for pair in screen['pairDetails']:
    for trial in pair['invasions']:
        matrix[ids.index(trial['challenger']),ids.index(trial['resident'])]=100*trial['win']['mean']
fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=0,vcenter=25,vmax=100))
labels=['+'+i.split('+')[1] for i in ids]
ax.set_xticks(range(len(ids)),labels);ax.set_yticks(range(len(ids)),labels)
ax.set_xlabel('三个居民对手的竞价上限增量');ax.set_ylabel('单个挑战者的竞价上限增量')
ax.set_title('积极策略普及后的入侵胜率（%）\n每格 32 种子 × 4 座位，25% 为对称参考；对角线未测试',pad=16)
for i in range(len(ids)):
    for j in range(len(ids)):
        value=matrix[i,j]
        ax.text(j,i,'—' if np.isnan(value) else f'{value:.1f}',ha='center',va='center',color='white' if not np.isnan(value) and (value>65 or value<12) else '#142c34')
fig.colorbar(im,ax=ax,label='挑战者共享胜率（%）')
fig.savefig(root/'invasion-matrix.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
x=np.arange(len(order));means=[100*r['winRate'] for r in order]
errors=[[100*(r['winRate']-r['win95'][0]) for r in order],[100*(r['win95'][1]-r['winRate']) for r in order]]
axes[0].bar(x,means,color='#b67641',alpha=.8,yerr=errors,capsize=3)
axes[0].axhline(25,color='#365963',linestyle='--',linewidth=1)
axes[0].set_xticks(x,labels);axes[0].set_xlabel('竞价上限增量（比索）');axes[0].set_ylabel('共享胜率（%）');axes[0].set_title('均衡配比的两策略联赛');axes[0].grid(axis='y',alpha=.2)
auctions={r['strategy']:r['prices'] for r in screen['auction']}
prices=[auctions[i]['mean'] for i in ids]
axes[1].plot(x,prices,'o-',color='#486a82')
axes[1].set_xticks(x,labels);axes[1].set_xlabel('竞价上限增量（比索）');axes[1].set_ylabel('赢得船长时的成交均价（比索）');axes[1].set_title('实际成交金额随竞争上升');axes[1].grid(axis='y',alpha=.2)
fig.suptitle('九种积极竞价：18,432 局直接竞争',fontsize=15)
fig.savefig(root/'auction-arms-race.png',dpi=160);plt.close(fig)

if '09-critical-pair' in stages:
    critical=stages['09-critical-pair'];rows=critical['summary']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,key,ci_key,title in [(axes[0],'winRate','win95','共享胜率（%）'),(axes[1],'mean','mean95','平均终局财富（比索）')]:
        scale=100 if key=='winRate' else 1
        means=[scale*r[key] for r in rows]
        errors=[[scale*(r[key]-r[ci_key][0]) for r in rows],[scale*(r[ci_key][1]-r[key]) for r in rows]]
        ax.barh(range(len(rows)),means,xerr=errors,capsize=4,color=['#b67641','#486a82','#92799d'],alpha=.8)
        ax.set_yticks(range(len(rows)),[r['name'] for r in rows]);ax.invert_yaxis();ax.set_xlabel(title+'，按种子分组的 95% 区间');ax.grid(axis='x',alpha=.2)
        for i,v in enumerate(means):ax.text(rows[i][ci_key][1]*scale+(.3 if scale==100 else .5),i,f'{v:.2f}',va='center')
        ax.set_xlim(0,max(r[ci_key][1]*scale for r in rows)+(5 if scale==100 else 12))
    fig.suptitle('关键对抗复核：256 个新种子 · 12,288 局 · 全部人数配比与座位轮换',fontsize=14)
    fig.savefig(root/'critical-contest.png',dpi=160);plt.close(fig)

finals=[s for s in report['stages'] if s['manifest']['config'].get('finalists')]
final=finals[-1]
with gzip.open(root/(final['stage']+'.jsonl.gz'),'rt',encoding='utf-8') as f: matches=[json.loads(line) for line in f]
rows=final['summary'];ids=[r['strategy'] for r in rows]
values=[[p['score'] for m in matches for p in m['results'] if p['strategy']==i] for i in ids]
fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
boxes=axes[0].boxplot(values,orientation='horizontal',whis=(0,100),showmeans=True,patch_artist=True,meanprops={'marker':'o','markersize':4,'markerfacecolor':'#16323a'},medianprops={'color':'#16323a','linewidth':1.5})
colors=['#b67641','#236c6b','#486a82','#92799d']
for box,color in zip(boxes['boxes'],colors):box.set_facecolor(color);box.set_alpha(.65)
axes[0].set_yticks(range(1,len(rows)+1),[r['name'] for r in rows]);axes[0].invert_yaxis();axes[0].set_xlabel('终局财富（比索）');axes[0].set_title('P25–P75 / 中位数 / 均值 / 全部观测范围');axes[0].grid(axis='x',alpha=.2)
win=[100*r['winRate'] for r in rows]
errors=[[100*(r['winRate']-r['win95'][0]) for r in rows],[100*(r['win95'][1]-r['winRate']) for r in rows]]
axes[1].barh(range(len(rows)),win,xerr=errors,color=colors,capsize=4,alpha=.8)
axes[1].set_yticks(range(len(rows)),[r['name'] for r in rows]);axes[1].invert_yaxis();axes[1].axvline(25,linestyle='--',color='#365963',linewidth=1);axes[1].set_xlim(0,max(55,max(r['win95'][1]*100 for r in rows)+12));axes[1].set_xlabel('共享胜率（%），按种子分组的 95% 区间');axes[1].grid(axis='x',alpha=.2)
for i,w in enumerate(win):axes[1].text(100*rows[i]['win95'][1]+1,i,f'{w:.1f}%',va='center')
fig.suptitle(f"强策略确认赛 · {final['manifest']['completedGames']:,} 局 · {final['manifest']['config']['seeds']} 新种子",fontsize=15)
fig.savefig(root/'final-distributions.png',dpi=160);plt.close(fig)

audits=[s for s in report['stages'] if s.get('auditPaired')]
if audits:
    audit=audits[-1];data=sorted(audit['auditPaired'],key=lambda p:p['win']['mean'],reverse=True)
    names={r['strategy']:r['name'] for r in audit['challenge']}
    fig,ax=plt.subplots(figsize=(11,max(4,len(data)*.55+1.5)),layout='constrained')
    for i,p in enumerate(data):
        mean=p['win']['mean']*100;low,high=[v*100 for v in p['win']['ci95']]
        ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=4,color='#236c6b' if low>0 else '#92799d')
    ax.set_yticks(range(len(data)),[names[p['challenger']] for p in data]);ax.invert_yaxis();ax.axvline(0,color='#364b51',linewidth=1);ax.grid(axis='x',alpha=.2);ax.set_xlabel('相对居民自我对战的配对胜率差（百分点）与 95% 区间')
    ax.set_title('候选最强策略普及后，谁还能入侵？\n居民：'+names[audit['manifest']['config']['resident']],pad=16)
    fig.savefig(root/'resident-invasion.png',dpi=160);plt.close(fig)
print('Rendered evolutionary league figures to',root)
