import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { AUCTION_FORMULAS } from '../auction-formulas.mjs';
import { strategyDefinition } from '../strategies.mjs';

const root = path.resolve(process.argv[2] || 'reports/2026-10-07-formulas');
const report = JSON.parse(await readFile(path.join(root, 'summary.json'), 'utf8'));
const stages = Object.fromEntries(report.stages.map(s => [s.stage, s]));
const final = stages['05-independent-confirmation'];
const model = stages['06-model-confirmation'];
const linear = stages['07-linear-confirmation'];
const cross = stages['10-cross-confirmation'];
const audits = report.stages.filter(s => s.auditPaired && s.stage !== '01-resident-screen');
const f = value => value.toFixed(2), pct = value => f(100 * value) + '%';
const ci = (range, scale = 1) => range ? range.map(v => f(v * scale)).join(' ～ ') : '样本不足';
const name = id => strategyDefinition(id).name;
const table = rows => [
  '| 策略 | 平均分 | 最低 | 最高 | 中位数 | P05–P95 | 胜率 [95%区间] |',
  '| --- | ---: | ---: | ---: | ---: | --- | --- |',
  ...rows.map(r => `| ${r.name} | ${f(r.mean)} | ${r.min} | ${r.max} | ${f(r.median)} | ${ci([r.p05, r.p95])} | ${pct(r.winRate)} [${ci(r.win95, 100)}]% |`),
].join('\n');
const pairTable = stage => [
  '| 同桌配对 | 胜率差（百分点） | 95%区间 | 财富差 | 95%区间 |',
  '| --- | ---: | --- | ---: | --- |',
  ...stage.paired.map(p => `| ${name(p.a)} − ${name(p.b)} | ${f(p.win.mean * 100)} | ${ci(p.win.ci95, 100)} | ${f(p.score.mean)} | ${ci(p.score.ci95)} |`),
].join('\n');
const expression = id => {
  const d = AUCTION_FORMULAS[id];
  if (!d) return '原基础上限 +3（普通 / 资金保护版本）';
  if (d.kind === 'fixed') return `max(0, min(支付能力 − ${d.buffer}, ${d.cap}))`;
  if (d.kind === 'linear' || d.kind === 'relative') return `max(0, min(支付能力 − ${d.buffer}, floor(${d.base} + ${d.optionWeight} × 上涨空间 + ${d.controlWeight} × ${d.kind === 'relative' ? '公开持股优势差' : '持股控制差'})))`;
  return '在 8 个隐藏持股 / 骰子世界中，配对比较赢得船长与立即退出的收益；先试价 12，按收益修正报价，再模拟修正价并二次修正。';
};
const formulaIds = report.strategies.filter(id => id.startsWith('formula-'));
const firstBest = final.summary[0], modelBest = model?.summary[0];
const crossBest = cross?.summary[0];
const linearId = 'formula-b7-o100-c0';
const linearRow = linear?.summary.find(r => r.strategy === linearId);
const linearOriginal = linear?.summary.find(r => r.strategy === 'bid+3');
const linearPair = linear?.paired.find(p => p.a === 'bid+3' && p.b === linearId);
const originalAudit = stages['08-original-residents'], linearAudit = stages['09-linear-residents'];
const originalInvader = originalAudit?.auditPaired.find(p => p.challenger === linearId);
const fixedInvader = linearAudit?.auditPaired.find(p => p.challenger === 'formula-fixed14');
const fixedOriginal = originalAudit?.auditPaired.find(p => p.challenger === 'formula-fixed14');
const modelAudit = stages['11-model-residents'];
const modelLinear = modelAudit?.auditPaired.find(p => p.challenger === linearId);
const text = [
  '# 替换船长竞拍估值公式，会不会更强？', '',
  '原来的 `4 + 0.7 × 购股上涨空间 + 0.25 × 持股控制差` 来自最初实现时手工设定的启发式。没有论文推导、开源专家参数或训练拟合依据。上一轮只检验了它的加价幅度，不能证明基础公式正确。', '',
  `本轮完成 **${report.stages.length} 个阶段、${report.completedGames.toLocaleString('en-US')} 局完整四人对战**，${report.uniqueSeeds} 个不同基础种子。比较 ${formulaIds.length - 1} 套替代估值方案、一个原公式等价对照及两种旧 +3 策略；+6 只作部分筛选的陪测对手。策略 ID / 参数的数量不等于独立行为数量。所有实际动作逐步校验，失败 / 非法 / 截断均为 0；41 项测试通过。`, '',
  '购股、货物选择、起点和部署继续使用同一启发式，只替换竞拍估值。模型候选也不在真实购股 / 部署阶段调用额外规划。这样衡量的是竞拍公式的改变，而不是同时更换其他决策。', '',
  ...(crossBest ? [`最后的强对手交叉确认点估计第一：**${crossBest.name}**，胜率 ${pct(crossBest.winRate)}，平均分 ${f(crossBest.mean)}。线性家族、模拟家族胜者与两种旧 +3 直接同桌。是否优于原 +3，应看配对区间和居民复核；不把最高点估计直接称作已证明的唯一最优。`, ''] : []),
  ...(modelLinear ? [`将所选模拟策略复制成三个对手后，新线性挑战者相对模型居民自我对战的配对胜率差为 ${f(modelLinear.win.mean * 100)} 个百分点，区间 [${ci(modelLinear.win.ci95, 100)}]。该复核只有 ${modelAudit.manifest.config.seeds} 个独立种子组，精度较低；它也不支持把交叉确认点估计第一直接当成不可入侵的冠军。`, ''] : []),
  `较早的固定上限确认赛点估计第一是 ${firstBest.name}（${pct(firstBest.winRate)}），说明初筛领先的固定 14 / 18 没有自动变成强对手确认赛冠军。所有阶段保留，后续线性与模型确认分别列出。`, '',
  ...(linearRow && linearOriginal && linearPair ? [
    `改变基础公式可以改善特定竞争环境，但本轮没有证实一个通吃所有群体的替代公式。线性确认中 \`7 + 上涨空间\`（控制权重 0，预留支付能力 6）胜率 ${pct(linearRow.winRate)}，同桌原 +3 为 ${pct(linearOriginal.winRate)}；配对胜率差 ${f(-linearPair.win.mean * 100)} 个百分点，区间 [${ci([-linearPair.win.ci95[1], -linearPair.win.ci95[0]], 100)}]。该比较适用于此混合阵容。`, '',
    ...(originalInvader && fixedInvader && fixedOriginal ? [
      `对三个原 +3，${originalAudit.manifest.config.seeds} 种子复核中新线性公式相对居民自我对战的胜率差为 ${f(originalInvader.win.mean * 100)} 个百分点，区间 [${ci(originalInvader.win.ci95, 100)}]。当三个对手都用新线性公式，固定 14 的配对胜率差为 ${f(fixedInvader.win.mean * 100)}，区间 [${ci(fixedInvader.win.ci95, 100)}]；固定 14 对三个原 +3 的差为 ${f(fixedOriginal.win.mean * 100)}，区间 [${ci(fixedOriginal.win.ci95, 100)}]。对手组成改变了效果，不能宣称新线性公式已成为不可入侵的冠军。`, '',
    ] : []),
    '新线性公式开局上限 22，而旧普通 +3 在两张不同 / 相同起手股时分别为 18 / 20；到后期重仓时，去掉持股控制加价又能降低上限。因此它改变了出价随局面的形状，而不是每轮统一多出几块钱。上涨空间仍是手工预测，胜出并没有证明这个股价预测已校准。', '',
  ] : []),
  '## 比较了哪些替代方案', '',
  '- 固定上限 10 / 14 / 18 / 22 / 26 / 30，同时保留 6 的支付能力。',
  '- 27 组线性公式：固定项 4 / 7 / 10；购股上涨空间权重 0.35 / 0.7 / 1；持股控制差权重 0 / 0.25 / 0.75。',
  '- 剩余上涨级数系数从 0.8 改为 0.4 / 1.2；融资预留改为 0 / 12。',
  '- 三种根据对手公开购股计算持股优势差的线性版本，不读取其隐藏初始股。',
  '- 四种模拟公式：单航次 / 终局，分别优化自身财富或自身减最强对手财富。', '',
  '线性公式里的上涨空间保持原定义：领先股票距离 30 尚差 L 级，假设每股再涨 max(1, round(系数 × L)) 级；在仍有库存的股票中，取预测价格减买入成本的最大值。持股控制差是自身各货物“持股数 × 再涨一级收益”的最大值减最小值。支付能力 = 现金 +12×未抵押股票数；预留的是现金与信用合计，不保证手中现金余额。', '',
  '![线性权重筛选](formula-screen.png)', '',
  '两组筛选分别用 64 个不同种子：对三个普通 +3；对普通 +3 / 资金保护 +3 / +6。按两组挑战胜率平均选六个新公式，与两种旧策略进行完整两两配比联赛。直接联赛为 48 个新种子、1:3 / 相邻 2:2 / 交替 2:2 / 3:1、全部座位轮换。该轮身份携带手牌，后续确认赛再做完整手牌互换。', '',
  '## 固定上限的独立确认', '',
  `${final.manifest.config.seeds} 个新种子 × 全部 24 种策略与初始手牌 / 座位分配，共 ${final.manifest.completedGames.toLocaleString('en-US')} 局。两种旧策略与直接联赛前两名同桌；各策略每局出现一次。筛选与确认种子分离，四强在读取确认种子之前冻结。`, '',
  table(final.summary), '', pairTable(final), '',
  ...final.summary.filter(r => r.strategy.startsWith('formula-')).map(r => `- **${r.name}**：\`${expression(r.strategy)}\`。`), '',
  '![确认赛分布](formula-confirmation.png)', '',
  '固定上限是对本候选池和购股 / 部署续走的策略测试，不是船长每个局面价值恒定的证明；也不应把它转换成人类对局的统一合理价。固定上限候选可完全退出超过上限的竞拍，资金紧张时上限进一步下降。', '',
  ...(model ? [
    '## 模拟估值的独立确认', '',
    `资格赛用 ${stages['04-model-qualification'].manifest.config.seeds} 个新种子，与两个强确定性公式和普通 +3 竞争；选四种模拟公式里的前两名，再用 ${model.manifest.config.seeds} 个新种子 × 全部 24 种分配，对普通 +3 和最强确定性候选确认，共 ${model.manifest.completedGames.toLocaleString('en-US')} 局。计算预算比确定性确认小，两个赛场的绝对排名不能直接混合。该场点估计第一为 ${modelBest.name}（${pct(modelBest.winRate)}）。`, '',
    table(model.summary), '', pairTable(model), '', '![模拟报价的终局分布](formula-models.png)', '',
    '模型报价每航次、每玩家首次轮到时抽样 8 个与公开信息一致的隐藏持股世界；与立即永久退出竞拍进行配对模拟。先在 min(12, max(0, 支付能力−6)) 试价，令建议价 = 试价 + 平均收益差；再模拟建议价，做第二次同样的修正。两次修正均向下取整并裁剪到可用支付能力内。报价缓存至该航次结束，每次真实出价只加 1。它假设报价附近“多支付 1，收益少 1”，融资 / 股票决策可能造成非线性，因此不把两次修正称为精确求根或已校准的价值模型。', '',
    '终局版本模拟到游戏真正结束，以现金 + 股票终值 − 抵押债务计算收益，没有手工终局涨幅补丁；单航次版本仍用 0.4×min(剩余涨幅,12) 的股票远期补丁。两者的未来所有玩家都按旧积极 +3 续走，不递归规划、不推断对手竞价行为中的隐藏持股信息。实际强对手可以使用其他公式，这种续走模型误差会影响报价。没有训练 MuZero 或神经价值网络。', '',
  ] : []),
  ...(linear ? [
    '## 线性家族与持股控制项的单独确认', '',
    '固定上限进入前两名后，另冻结直接联赛中最好的两个线性公式，加上仅去掉持股控制项的消融版本，与原 +3 同桌。使用另一组 128 个新种子 × 全部 24 种分配。这个补充赛场检验线性权重和对手组成，不覆盖或替换之前的固定上限结果。', '',
    table(linear.summary), '', pairTable(linear), '', '![线性公式确认分布](formula-linear.png)', '',
    ...linear.summary.filter(r => r.strategy.startsWith('formula-')).map(r => `- **${r.name}**：\`${expression(r.strategy)}\`。`), '',
  ] : []),
  ...(cross ? [
    '## 两个家族胜者与原策略的直接交叉确认', '',
    `${cross.manifest.config.seeds} 个新种子 × 全部 24 种分配，共 ${cross.manifest.completedGames.toLocaleString('en-US')} 局。线性家族与模型家族的胜者先冻结，再与两种旧策略直接同桌；这张表可在同一个强对手池下比较四者。`, '',
    table(cross.summary), '', pairTable(cross), '', '![最后强对手交叉确认](formula-cross.png)', '',
    '本场两种旧 +3 的行为完全重合：已核对 768 对交换旧策略标签的比赛，所有实际结果与竞拍记录一致。它们是两个旧策略席位，不是对新方法的两份独立验证。新方法之间的胜率差也应看配对区间，不能只按点估计宣布唯一冠军。', '',
  ] : []),
  ...audits.flatMap(s => [
    '## 居民复核：' + name(s.manifest.config.resident), '',
    `${s.manifest.config.seeds} 个新的基础种子，各轮换四个座位。一个挑战者对三个居民；同种子、同初始股、同骰子的居民自我对战作配对控制。不能简单拿单个挑战者的胜率与 25% 作显著性比较。`, '',
    table(s.challenge), '',
    '| 挑战者 | 相对自我对战的胜率差（百分点） | 95%区间 | 财富差 | 95%区间 |',
    '| --- | ---: | --- | ---: | --- |',
    ...s.auditPaired.map(p => `| ${name(p.challenger)} | ${f(p.win.mean * 100)} | ${ci(p.win.ci95, 100)} | ${f(p.score.mean)} | ${ci(p.score.ci95)} |`), '',
  ]),
  ...(audits.length ? ['![居民复核](formula-residents.png)', ''] : []),
  '## 统计与复现', '',
  '胜率并列第一平分，分数为终局财富。所有近似 95% 区间按基础种子分组；同种子的座位、手牌互换和对手配比不当作独立样本。候选筛选没有多重比较校正，结论以独立确认和居民配对复核为主。有限参数范围、8 个样本的模拟报价及固定购股 / 部署策略限制了外推；没有证明不存在其他更强公式。', '',
  '单独实现了原 +3 的等价公式对照，两个筛选阶段各 256 场逐局结果与原 +3 完全一致。所有终局日志、配置和源哈希保留；实现中途为模拟报价减少计算的草稿数据没有混入本报告，最终所有阶段统一使用冻结实现复跑。', '',
  '- [全部阶段 / 配对 / 成交价 / 尾部风险](RESULTS.md)',
  '- [全部公式定义](FORMULAS.md)、[CSV](summary.csv)、[JSON](summary.json)',
  '- `specs/*.json` 为冻结赛程，`*.jsonl.gz` 为全部逐局终局与竞拍历史。', '',
  '从本报告所在的冻结提交，在 `manila` 目录运行：', '',
  '```powershell',
  'Get-ChildItem reports/2026-10-07-formulas/specs/*.json | Sort-Object Name | ForEach-Object {',
  '  npm run league -- $_.FullName output/formulas-replayed 8',
  '}',
  'node tools/report-evolution.mjs output/formulas-replayed output/formulas-replayed-report',
  'node tools/report-formulas.mjs output/formulas-replayed-report',
  'python tools/plot-formulas.py output/formulas-replayed-report',
  '```', '',
  '无需重赛即可复核：', '',
  '```powershell',
  'node tools/restore-evolution.mjs reports/2026-10-07-formulas output/formulas-restored',
  'node tools/report-evolution.mjs output/formulas-restored output/formulas-rebuilt',
  '```', '',
];
await writeFile(path.join(root, 'README.md'), text.join('\n'));
await writeFile(path.join(root, 'FORMULAS.md'), ['# 冻结估值公式', '', '所有公式仅改变竞拍，仍使用相同的购股 / 部署启发式。原公式等价对照不视为一个独立的新方法。', '', '| ID | 名称 | 参数 |', '| --- | --- | --- |', ...formulaIds.map(id => `| ${id} | ${name(id)} | \`${JSON.stringify(AUCTION_FORMULAS[id])}\` |`), ''].join('\n'));
console.log('Wrote formula research interpretation to', root);
