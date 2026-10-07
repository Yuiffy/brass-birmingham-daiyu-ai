# 马尼拉 · 船长竞价实验室

一个原版《Manila》基础规则的本地浏览器人机版，以及研究港口船长竞拍价格的可复现实验。纯 JavaScript，无运行时依赖，不需要 API key 或模型下载。

```powershell
cd D:\workspace\myrepo\brass-birmingham-daiyu-ai\manila
npm start
```

打开 http://127.0.0.1:5180 。默认四人，你控制座位 1，其余为规则基线 AI。三人和五人模式可选。可以 AI 代走一步或观察整局，自动保存，也支持 JSON 存档导入导出。只监听本机，不发布到外网。

自己的竞拍回合可运行价格分析。默认模拟至终局；短期模式可以查看股票未来收益被截断后估值有何变化。设置样本数、最高测试价格、对手竞价风格；输出每个价格的配对财富增益、近似 95% 均值区间、相对最强对手领先变化，以及完整终局的胜率。可以导出原始分析 JSON。

```powershell
npm test
npm run benchmark -- --samples 64 --max-bid 40 --horizon 0
```

基准结果写入 `manila/output/bid-benchmark.json`（Git 忽略）。可传 `--players 3`、`--seed 123`、`--styles balanced,cautious,aggressive`。比较多个初始股票、人数和对手，才能形成可迁移的出价建议。界面默认的少量样本只用于快速探索。

## 定价问题

对每个候选价格 b，计算：

`Δ(b) = E[终局财富 | 现在以 b 赢得船长] − E[终局财富 | 现在永久退出本轮竞拍]`

这是一项条件于赢得拍卖的价值实验，**没有估计该价能否成交**。退出分支的其他玩家继续按基线策略竞价；若全部退出，前任船长免费续任。赢得分支支付真实出价，并重新模拟现金限制、自动抵押、购股、装载、起点、后续部署及未来竞拍。因此 Δ 不必严格线性，不能直接从某一个低价格结果减去出价就当作所有价格的结果。

每个样本从公开信息与自己的股票重采样对手初始股票，保持供应、公开购股、各人股数、抵押股数。配对分支共用按航次 / 掷骰次数 / 货物编码的骰子随机流。估计不读取对手实际初始股票，也不读取游戏未来骰子种子。均值置信区间用样本差的标准误计算，描述有限模拟误差，不包括错误策略假设、不完美对手建模或多价格选择偏差。它不是单局盈亏的预测区间。

均值可接受集合：Δ(b) ≥ 0。谨慎集合：Δ(b) 的近似 95% 下界 ≥ 0。它们可能不连续，JSON 返回完整价格集合。测试范围顶部仍为正时，只能报告“至少测试到这里仍为正”，不能宣称已经找到真正盈亏平衡价。

还报告相对领先变化和终局共享胜率：多人竞争中，最大化自身财富、扩大领先、最大化胜率是不同目标。不要把财富盈亏平衡价当作唯一最优竞争价格。

## 为什么不用 MuZero 学规则

2026-10-07 用户确认以前的世界模型调研可能在另一台未上传电脑上，并允许自行决定。这里所有状态转移和骰子分布可明确编程，先用精确前向规则模型做反事实搜索，省去训练近似动力学带来的规则误差。`ruleWorldModel.observe/step` 提供可替换模型接口；当前没有声称训练过 MuZero，也没有跨游戏复用伯明翰权重。

AI 的装载、部署、领航和后续竞拍使用可解释启发式策略。对手风格设置只影响其他玩家的竞价，我方后续策略固定为均衡基线。竞拍的股票期权和工人部署的未来分摊属于策略假设，不是精确最优解。已支持完整游戏模拟与多竞价风格比较，**尚未证明达到高手水平或纳什均衡**。若继续强化，应优先训练策略 / 价值函数并用精确转移做信息集随机树搜索；通过冻结对手池和未见种子验证后，再推广其竞拍区间。

## 规则范围

- 原版 3–5 人、12 张初始抽样股、其余公开供应；每人初始 30 比索和 2 股。
- 轮流竞拍，一次退出不得重入；全员放弃则前任免费续任。
- 船长可购 1 股（最低 5）、选四货中的三种，0–5 起点总和 9，先手部署。
- 三人局 4 次部署，前两次都在首次掷骰前。四 / 五人各 3 次部署。停止后本航次不再部署。
- 船上最低价空位、船员平分货物收益；港口 / 船厂 A/B/C 费用 4/3/2、收益 6/8/15。
- 第二次恰停 13：海盗可登有空位的船。第三次恰停 13：海盗劫掠，海盗船长决定港口 / 船厂；无海盗则到港。
- 小领航员先、大领航员后；已到港船不可领航；超过 13 立即到港，领航移至 13 不触发海盗。
- 保险立即收 10，空船厂也赔维修，自己占据船厂不用付自己；不足强制贷款，耗尽贷款能力后银行承担缺口。
- 抵押借 12、赎回付 15；最终股票市值仍计入，抵押每股再减 15。贫困且无可借股票可作免费乘客，不能投保。
- 市价 0 / 5 / 10 / 15 / 20 / 30，首次货物达到 30 终局。基础海盗规则；不启用挤人变体。

支付不足的自动抵押优先选择最低市值股票，是本实现的支付便利策略。可通过界面逐张赎回。没有主动抵押按钮：任何需支付的合法动作自动调用贷款，无需为了竞价先手工借钱。

## 现成项目调研

截至 2026-10-07，未确认到符合原版规则、可直接运行且能研究船长竞价的完整 AI 版。并不声称网络上绝对不存在。

| 来源 | 找到的内容 | 是否适合核心问题 |
| --- | --- | --- |
| [framist/manila](https://github.com/framist/manila) / [在线计算器](https://framist.github.io/manila/) | MIT Vue 期望计算器，作者自己提示准确性待改善 | 可辅助参考，但没有完整人机对局与船长反事实定价；README 对到港 / 海盗有简化，规则以出版商书为准 |
| [Nanyang Trade Port](https://apps.apple.com/au/app/nanyang-trade-port/id6772065301) | App Store 描述有三个离线 AI 角色 | 致敬游戏，五种角色拍卖、固定七轮，规则不适合替代原版船长定价；未下载安装验证 |
| [密歇根大学项目报告](https://lwlxy.github.io/Portfolio/pdfs/CSE_592_Final_Report.pdf) | 马尼拉 Q-learning / DQN 实验 | 十格航线、无船长竞拍 / 股票的简化环境，不能把论文胜率用于原版 |
| [Zoch 英文规则书镜像](https://raw.githubusercontent.com/framist/manila/master/SR-Manila-en.pdf) | 出版商原版基础规则 | 本实现规则依据；下载审计，不把规则书或美术作为本项目资产分发 |

所有程序、UI 和画布图形从零编写，没有复制以上项目源代码或商业游戏美术。保留所在仓库 AGPL-3.0 授权；非官方粉丝与研究项目。

模块：`engine.mjs` 规则、合法动作、观察 / 隐藏信息采样；`ai.mjs` 策略、骰子动态规划和反事实模拟；`worker.mjs` 后台分析；`app.mjs` UI / 存档 / 画布；`tests/rules.test.mjs` 边界与完整对局回归。

## 完整 AI 对战比较

[竞价升级与强策略联赛](reports/2026-10-07-evolution/README.md)是后续研究：淘汰原均衡 / 即时收益 / 随机候选，直接比较不同积极上限、支付能力保留和财富 / 领先前瞻；逐轮筛选、独立种子确认，再测试策略普及后的入侵表现。该报告替代“单个积极策略对均衡对手”的强度排名解释。

[2026-10-07 对战报告](reports/2026-10-07/README.md)包含三、四、五人局的终局财富分布、均值 / 最低 / 最高 / 分位数、共享胜率、船长成交价及抵押统计。原始逐局日志压缩后随报告提交，可由种子复现。

六种策略：均衡、竞价上限减 3 的保守策略、加 3 的积极策略、即时收益部署、随机合法部署，以及精确规则模型前瞻（默认最多 4 个候选 × 8 个重采样，模拟到本航次结算）。随机策略仅部署随机；前瞻策略竞价沿用均衡，当前没有 MuZero 训练。策略模块可独立扩展，尚未接入游戏 UI 的对手选择。

在 `manila` 目录运行：

```powershell
npm run tournament -- --seeds 128 --strategies balanced,cautious,aggressive,greedy,random --players 3,4,5 --workers 4 --validate --output output/tournament-focal
npm run tournament -- --seeds 64 --strategies search --players 3,4,5 --workers 4 --validate --output output/tournament-search
npm run tournament -- --mode mixed --seeds 128 --strategies balanced,cautious,aggressive,greedy --players 4 --workers 4 --validate --output output/tournament-mixed
node tools/report-tournament.mjs reports/2026-10-07
python tools/plot-tournament.py reports/2026-10-07
```

仅制图需要 Python / matplotlib，可安装到独立虚拟环境。比赛本身只有 Node 标准库依赖。`--validate` 在每步核验合法动作及资产不变量；对局必须到终局，超限会报错，不能把截断财富当最终分数。输出目录写入 manifest（配置、Node 版本及模拟源文件 SHA256）、原始 JSONL、JSON / CSV 汇总。报告脚本核验全部局数、编号和源哈希后才汇总；如果规则或策略已改动，请先重新跑比赛。

挑战赛按全部座位轮换，其他玩家固定均衡；混合赛跑四种策略的全部 24 种座位 / 手牌分配排列。骰子按航次、掷骰轮、货物编码，独立于决策随机流。均值与胜率区间按种子分组，同种子的轮换不当作独立样本；不同样本量只用共同种子做配对比较。绝对财富分数依赖人数、对手与游戏长度，应结合胜率阅读。

## 强策略的参数与联赛工具

`bid+N` 将原竞价上限增加 N（支持 0–96）；`bid-liquid+N` 还要求支付后剩余支付能力至少 6，比单纯现金余额多计可抵押信用。`look+N` 用财富前瞻部署，`relative+N` 用“自身减最强对手”的叶估值；它们也支持 `-liquid`，例如 `relative-liquid+6`。其他阶段仍使用精确规则和基础启发式，不把参数筛选称作 MuZero 训练。

```powershell
npm run evolve -- --stage all --workers 8
npm run league -- reports/2026-10-07-evolution/specs/06-higher-models.json output/evolution 8
node tools/report-evolution.mjs output/evolution reports/2026-10-07-evolution
python tools/plot-evolution.py reports/2026-10-07-evolution
```

`evolve` 运行初始五轮；后续扩展与确认赛的完整参数保存在报告的 `specs/` 中，按文件顺序运行 `league` 即可复现。报告说明了自适应筛选依据。比赛默认八个 Node worker，前瞻预算为三个候选、四个样本；所有比赛都到真实终局并逐动作校验。

只想复核统计、无需重新比赛时：

```powershell
node tools/restore-evolution.mjs reports/2026-10-07-evolution output/evolution-restored
node tools/report-evolution.mjs output/evolution-restored output/evolution-rebuilt
python tools/plot-evolution.py output/evolution-rebuilt
```

恢复工具验证压缩日志及源代码哈希，重新计算各阶段统计。旧的第一轮报告重建需使用其冻结提交 `0f3f47a`；本轮扩展了策略接口与比赛日志，历史数据不冒充当前代码生成的结果。
