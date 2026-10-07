"""python tools/plot-tournament.py reports/2026-10-07
Requires matplotlib; only used to render research figures, not to play the game.
"""
import gzip
import json
import sys
from pathlib import Path

# Optional isolated install, used for the recorded run on Windows.
local_dependencies = Path(__file__).resolve().parents[1] / 'output' / 'python-deps'
if local_dependencies.exists():
    sys.path.insert(0, str(local_dependencies))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'reports/2026-10-07')
report = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
matches = {}
for run in report['runs']:
    with gzip.open(root / (run['source'] + '.jsonl.gz'), 'rt', encoding='utf-8') as stream:
        matches[run['source']] = [json.loads(line) for line in stream]
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'figure.facecolor': '#faf9f5', 'axes.facecolor': '#faf9f5', 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
ids = ['balanced', 'cautious', 'aggressive', 'greedy', 'random', 'search']
names = {r['strategy']: r['name'] for r in report['focal']}
colors = dict(zip(ids, ['#486a82', '#78998b', '#b67641', '#92799d', '#999999', '#236c6b']))
def scores(mode, players, strategy):
    return [r['score'] for run in report['runs'] if run['config']['mode'] == mode for m in matches[run['source']] if m['players'] == players for r in m['results'] if r['strategy'] == strategy and (mode != 'focal' or r['identity'] == 0)]
def boxes(ax, mode, players, order):
    data = [scores(mode, players, strategy) for strategy in order]
    result = ax.boxplot(data, orientation='horizontal', patch_artist=True, whis=(0,100), showmeans=True, meanprops={'marker':'o','markersize':4,'markerfacecolor':'#15333a','markeredgecolor':'#15333a'}, medianprops={'color':'#172c33','linewidth':1.5})
    for box, strategy in zip(result['boxes'], order):
        box.set_facecolor(colors[strategy]); box.set_alpha(.65)
    ax.set_yticks(range(1,len(order)+1), [f"{names[s]}  n={len(v)}" for s,v in zip(order,data)])
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.2); ax.set_xlim(-60,280 if mode == 'focal' else 240)
    ax.set_xlabel('终局财富（比索）')
fig, axes = plt.subplots(1,3, figsize=(17,5.5), layout='constrained', sharex=True)
for ax, players in zip(axes,[3,4,5]):
    boxes(ax,'focal',players,ids); ax.set_title(f'{players} 人局 · 其余对手均衡',pad=14)
fig.suptitle('马尼拉：六种 AI 终局财富分布\n箱体 P25–P75，竖线中位数，圆点均值，须线为观测最低 / 最高',fontsize=15)
fig.savefig(root / 'focal-distributions.png', dpi=160)
plt.close(fig)
fig, axes = plt.subplots(2,3,figsize=(13,8),layout='constrained',sharex=True,sharey=True)
for ax,strategy in zip(axes.flat,ids):
    values=scores('focal',4,strategy)
    ax.hist(values,bins=list(range(-60,241,15)),weights=[100/len(values)]*len(values),color=colors[strategy],alpha=.85,edgecolor='#faf9f5')
    ax.axvline(sum(values)/len(values),color='#203d48',linestyle='--',linewidth=1)
    ax.set_title(f'{names[strategy]} · n={len(values)}'); ax.set_xlabel('终局财富（比索）'); ax.set_ylabel('每个区间占比（%）'); ax.grid(axis='y',alpha=.2)
fig.suptitle('四人挑战赛：完整财富直方图（箱宽 15 比索，虚线为均值）',fontsize=15)
fig.savefig(root / 'four-player-histograms.png',dpi=160)
plt.close(fig)
order=['balanced','cautious','aggressive','greedy']
fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
boxes(axes[0],'mixed',4,order);axes[0].set_title('四种策略同桌：财富分布')
rows={r['strategy']:r for r in report['mixed']}
means=[100*rows[s]['winRate'] for s in order]
errors=[[100*(rows[s]['winRate']-rows[s]['win95'][0]) for s in order],[100*(rows[s]['win95'][1]-rows[s]['winRate']) for s in order]]
axes[1].barh(range(4),means,xerr=errors,color=[colors[s] for s in order],alpha=.8,capsize=4)
axes[1].set_yticks(range(4),[names[s] for s in order]);axes[1].invert_yaxis();axes[1].set_xlim(0,60);axes[1].set_xlabel('共享胜率（%），误差线为按种子分组的 95% 区间');axes[1].set_title('胜率：128 种子 × 24 排列');axes[1].grid(axis='x',alpha=.2)
for i,v in enumerate(means):axes[1].text(v+5,i,f'{v:.1f}%',va='center')
fig.suptitle('四人混合赛：3,072 局完整终局',fontsize=15)
fig.savefig(root / 'mixed-distributions.png',dpi=160)
plt.close(fig)
print('Rendered 3 distribution figures to',root)
