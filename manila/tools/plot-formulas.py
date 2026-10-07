"""Plot frozen auction-formula comparisons; standard scientific figures."""
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

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'reports/2026-10-07-formulas')
report = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
stages = {s['stage']: s for s in report['stages']}
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'figure.facecolor': '#faf9f5', 'axes.facecolor': '#faf9f5', 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})

screen = [stages[name] for name in ['01-resident-screen', '02-mixed-screen']]
def average_delta(strategy):
    return np.mean([100 * (next(r['winRate'] for r in s['challenge'] if r['strategy'] == strategy) - next(r['winRate'] for r in s['challenge'] if r['strategy'] == 'bid+3')) for s in screen])

fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout='constrained')
norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
for ax, control in zip(axes, [0, 25, 75]):
    matrix = np.array([[average_delta(f'formula-b{base}-o{option}-c{control}') for option in [35, 70, 100]] for base in [4, 7, 10]])
    im = ax.imshow(matrix, cmap='RdBu', norm=norm)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:+.1f}', ha='center', va='center', color='white' if abs(matrix[i,j]) > 9 else '#142c34')
    ax.set_xticks(range(3), ['0.35', '0.70', '1.00']); ax.set_yticks(range(3), ['4', '7', '10'])
    ax.set_xlabel('购股上涨空间权重'); ax.set_ylabel('固定项')
    ax.set_title(f'持股控制权重 = {control / 100:g}')
fig.colorbar(im, ax=axes, label='相对原 +3 的胜率差（百分点）', shrink=.8)
fig.suptitle('替换基础公式：两组筛选的平均胜率差\n每组 64 种子 × 4 座位；仅用于筛选，不代表独立确认结果', fontsize=13)
fig.savefig(root / 'formula-screen.png', dpi=180); plt.close(fig)

final = stages['05-independent-confirmation']
rows = final['summary']; ids = [r['strategy'] for r in rows]
matches = [json.loads(line) for line in gzip.open(root / '05-independent-confirmation.jsonl.gz', 'rt', encoding='utf-8')]
values = [[r['score'] for m in matches for r in m['results'] if r['strategy'] == strategy] for strategy in ids]
labels = [r['name'] for r in rows]
colors = ['#297c70' if s.startswith('formula-') else '#a18459' for s in ids]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained', gridspec_kw={'width_ratios': [1.15, 1]})
parts = axes[0].violinplot(values, positions=np.arange(len(ids)), orientation='horizontal', showmeans=True, showextrema=False)
for body, color in zip(parts['bodies'], colors): body.set_facecolor(color); body.set_alpha(.55)
axes[0].set_yticks(range(len(ids)), labels); axes[0].invert_yaxis(); axes[0].set_xlabel('终局财富分数'); axes[0].set_title('完整分布；线标记均值')
for i, r in enumerate(rows):
    axes[1].errorbar(100 * r['winRate'], i, xerr=[[100 * (r['winRate'] - r['win95'][0])], [100 * (r['win95'][1] - r['winRate'])]], fmt='o', color=colors[i], capsize=4)
    axes[1].annotate(f"{100*r['winRate']:.2f}%", (100*r['winRate'], i), xytext=(5, 9), textcoords='offset points', fontsize=9)
axes[1].set_yticks(range(len(ids)), labels); axes[1].invert_yaxis(); axes[1].axvline(25, color='#999', ls='--', lw=1)
axes[1].set_xlabel('共享胜率（%）'); axes[1].set_title('按种子分组的近似 95% 区间')
fig.suptitle(f"冻结四强的独立确认：{final['manifest']['config']['seeds']} 新种子 × 全部 24 种手牌 / 座位分配", fontsize=13)
fig.savefig(root / 'formula-confirmation.png', dpi=180); plt.close(fig)

def plot_extra_confirmation(stage_name, filename, title):
    if stage_name not in stages: return
    stage = stages[stage_name]; rows = stage['summary']
    matches = [json.loads(line) for line in gzip.open(root / (stage_name + '.jsonl.gz'), 'rt', encoding='utf-8')]
    values = [[r['score'] for m in matches for r in m['results'] if r['strategy'] == row['strategy']] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained', gridspec_kw={'width_ratios': [1.15, 1]})
    parts = axes[0].violinplot(values, positions=np.arange(len(rows)), orientation='horizontal', showmeans=True, showextrema=False)
    colors = ['#297c70' if r['strategy'].startswith('formula-') else '#a18459' for r in rows]
    for body, color in zip(parts['bodies'], colors): body.set_facecolor(color); body.set_alpha(.55)
    labels = [r['name'] for r in rows]
    axes[0].set_yticks(range(len(rows)), labels); axes[0].invert_yaxis(); axes[0].set_xlabel('终局财富分数')
    axes[0].set_title('完整分布；线标记均值')
    for i, r in enumerate(rows):
        axes[1].errorbar(100*r['winRate'], i, xerr=[[100*(r['winRate']-r['win95'][0])], [100*(r['win95'][1]-r['winRate'])]], fmt='o', color=colors[i], capsize=4)
        axes[1].annotate(f"{100*r['winRate']:.2f}%", (100*r['winRate'], i), xytext=(5, 9), textcoords='offset points', fontsize=9)
    axes[1].set_yticks(range(len(rows)), labels); axes[1].invert_yaxis(); axes[1].axvline(25, color='#999', ls='--', lw=1)
    axes[1].set_xlabel('共享胜率（%）'); axes[1].set_title('按种子分组的近似 95% 区间')
    fig.suptitle(f"{title}：{stage['manifest']['config']['seeds']} 新种子 × 全部 24 种分配", fontsize=13)
    fig.savefig(root / filename, dpi=180); plt.close(fig)

plot_extra_confirmation('06-model-confirmation', 'formula-models.png', '模拟报价的独立确认')
plot_extra_confirmation('07-linear-confirmation', 'formula-linear.png', '线性权重与持股控制项的独立确认')
plot_extra_confirmation('10-cross-confirmation', 'formula-cross.png', '模型、线性胜者与原策略直接竞争')

audits = [s for s in report['stages'] if s.get('auditPaired') and s['stage'] != '01-resident-screen']
if audits:
    fig, axes = plt.subplots(1, len(audits), figsize=(6 * len(audits), 4.8), layout='constrained', squeeze=False)
    for ax, stage in zip(axes[0], audits):
        for i, comparison in enumerate(stage['auditPaired']):
            mean = comparison['win']['mean'] * 100; low, high = np.array(comparison['win']['ci95']) * 100
            ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt='o', capsize=4, color='#297c70')
            ax.annotate(f'{mean:+.2f}', (mean, i), xytext=(4, 8), textcoords='offset points', fontsize=9)
        ax.set_yticks(range(len(stage['auditPaired'])), [c['challenger'].replace('formula-', '') for c in stage['auditPaired']]); ax.invert_yaxis()
        ax.axvline(0, color='#999', ls='--'); ax.set_xlabel('相对居民自我对战的胜率差（百分点）')
        ax.set_title('居民：' + stage['manifest']['config']['resident'].replace('formula-', ''))
    fig.suptitle('公式普及后，谁还能挑战它？\n配对控制；种子分组的近似 95% 区间', fontsize=13)
    fig.savefig(root / 'formula-residents.png', dpi=180); plt.close(fig)
print('Saved auction formula figures to', root)
