# 向更高的产业与道路分数提升

## 桌游分数参考与适用范围

2026-09-15 检索的[四人局玩家记录](https://www.reddit.com/r/boardgames/comments/i7cy6f/brass_birmingham_4_player_typical_scores/)中，发帖者描述常见成绩约为 115 / 125 / 135 / 155；另有人报告赢家约 160 以上，熟练组有人报告接近或超过 200。也有人记录过赢家仅 117–121 的互相牵制对局。这些自述年代、玩家和竞争环境不同，不是代表性调查，也不是官方段位门槛。150–160 可以作为高分方向，不能把 200 当作所有四人局的正常平均值。尤其不能直接比较“人类赢家”与“AI 所有场次平均”。

[Roxley 官方规则书的镜像 PDF](https://bghub.org/r/brassbirmingham.pdf)第 6–7 页说明两个时代的产业与道路计分；第 12 页明确完整游戏中收入等级不值 VP，二级及以上产业可跨时代保留、获得两次计分。原出版社[规则文件地址](https://files.roxley.com/Brass-Birmingham-Rulebook-2018.11.20-highlights.pdf)在本次检索时返回 502，因此读取了同一出版社规则书的镜像。

## 原引擎核对

本轮沿用原项目规则，保证新旧 AI 对比所处环境一致。核对发现：

| 项目实现 | 与完整桌游规则的差异 / 影响 |
|---|---|
| `gameState.endGame()` 最后把 `player.income` 加到 VP | 官方完整游戏没有这项终局加分；本轮逐局记录并单独扣除 |
| `getIncomeAmount(income)` 直接返回收入数值，产业直接增加该数值 | 没有区分收入轨格数与实际收入等级，经济节奏不同 |
| 铁路时代也设 `isFirstRound=true`、首轮一个行动 | 官方只有游戏最开始的运河首轮采用一个行动 |
| 初始手牌之外未建立每人的额外弃牌；AI 每次只卖一个产业、铺一条路 | 行动数量与效率不能直接对标真人规则与完整动作空间 |

“扣除收入加分”只是明确消除一项额外计分，不等于整套规则已经校准。报告同时展示原项目 VP 和扣除后的 VP，避免把收入奖励误当成产业、道路策略的进步。本轮没有改写规则或提高产业分值来抬高成绩。

## 实际改进

1. 价值网络从 109 项输入扩展到 125 项。增加道路连接的已翻面产业图标分、尚未翻面产业的潜在道路收益、自己的产业对道路的贡献、可跨时代保留产业的已实现和待实现分值、待售产业分值。全部来自当前可观测棋盘；终局结果仅作为训练标签。
2. 从已有价值网络和 Adam 状态继续训练；新增输入权重从零开始，旧输入顺序及权重保留。旧模型仍使用旧版特征。
3. 修正后续搜索候选排序：以前宽度为 3 时常被固定动作顺序截成建造 / 铺路 / 研发，出售和贷款即使更有价值也可能不进入搜索。
4. 按动作类别和产业种类轮流取最多 32 个根动作，由学习到的终局价值先比较一步，再保留 8 个进行深入推演。世界模型的预筛选与后续推演都使用学习到的状态转移；学习增强搜索使用规则引擎。旧搜索、原启发式和神经网络对手继续使用旧算法。
5. 收集更强候选棋手的 800 局新经验，与上一轮 800 局经验合并，保留整局训练 / 验证 / 测试划分。新增续训尝试以“项目 VP 减终局收入加分”为目标，使学习方向更接近产业与道路收益。

状态转移网络和神经网络对手权重冻结；本轮主要改进价值学习和动作规划。32 个动作的预比较会增加计算量，因此这是整体棋手升级实验，不是相同计算预算下的架构优劣证明。

## 数据与复测协议

- 原始动力学数据：10,000 局，训练 / 验证 / 测试按整局划分为 8,000 / 1,000 / 1,000。本轮价值训练的基础样本来自其中 8,000 局，验证沿用初版 100 局。
- 两批附加真实对局：各 800 局，每批 640 / 80 / 80，合并为 1,280 / 160 / 160。新一批起始种子 280000001，探索率 0.08，每 4 步记录一个状态并记录终局；所有终局都保留，未只筛高分胜局。
- 开发赛：种子 88000001，每种 AI 每版 80 局。探索特征、候选排序与预筛选后，在预筛选版本和两种新经验续训版本间选择；条件见 `world_model/experiments/geography-study/protocol.json`。排除收入奖励的续训降低了世界模型开发赛分数，因此在模型冻结和候选最终复测之前，加做保留原得分目标的同数据对照，区分数据变化与目标变化。选择按开发集调整，最终测试集始终不参与选择。
- 最终复测：冻结模型后，种子 108000001，每种 AI 每版 400 局，共四组 1,600 局。每个种子轮换四个座位，共 100 个独立种子组。三名对手固定为原启发式、原搜索、原神经网络；世界模型与学习增强搜索分别放入第四席。
- 报告按四局种子组做 20,000 次配对 bootstrap，给出 95% 区间；不把同一发牌的四个座位当作四个独立样本。置信区间不包含训练随机种子、其他对手、其他规则环境的不确定性。

## 复现命令

在仓库根目录运行。输出目录须为空；已保存的实验请换新目录复现。

```powershell
node world_model/cli.js train-value --resume world_model/experiments/score-v2/models --feature-version brass-value-v2 --replay world_model/data-score-league --replay-weight 4 --stride 8 --epochs 64 --lr 0.0003 --out world_model/experiments/value-geography-v1
node world_model/cli.js compose --dynamics world_model/experiments/score-v2/models --value world_model/experiments/value-geography-v1 --shortlist score-diverse-v2 --candidates learned-pool-v1 --out world_model/experiments/geography-pool-v1/models
node world_model/cli.js collect-value --games 800 --seed 280000001 --models world_model/experiments/geography-pool-v1/models --out world_model/data-geography-new --workers 6 --depth 2 --stride 4 --exploration 0.08
node world_model/cli.js merge-value --sources world_model/data-score-league world_model/data-geography-new --out world_model/data-geography-league
node world_model/cli.js train-value --resume world_model/experiments/geography-pool-v1/models --replay world_model/data-geography-league --replay-weight 4 --income-bonus-weight 1 --stride 8 --epochs 64 --lr 0.0003 --out world_model/experiments/value-geography-v2
node world_model/cli.js compose --dynamics world_model/experiments/geography-pool-v1/models --value world_model/experiments/value-geography-v2 --out world_model/experiments/geography-pool-v2/models
node world_model/cli.js train-value --resume world_model/experiments/geography-pool-v1/models --replay world_model/data-geography-league --replay-weight 4 --income-bonus-weight 0 --stride 8 --epochs 64 --lr 0.0003 --out world_model/experiments/value-geography-v3
node world_model/cli.js compose --dynamics world_model/experiments/geography-pool-v1/models --value world_model/experiments/value-geography-v3 --out world_model/experiments/geography-pool-v3/models
```



## 最终结果

最终采用 `geography-pool-v2`，价值头第 184 轮，9,280 局训练对局。完整 SHA-256：`4ebb5e7cf5ead3f30aa6ff102d7d885796f97c32c44cd134a838c009f00de05c`。三组各 64 轮训练尝试共执行 192 轮；最终所选模型来自前两阶段路径，使用验证集选出的检查点，未使用最后一轮替代最佳检查点。

开发集候选选择（每项 80 局）：

| 候选 | 世界模型项目 VP / 扣除后 | 学习增强搜索项目 VP / 扣除后 | 两者扣除后的平均 |
|---|---:|---:|---:|
| geography-pool-v1 | 117.58 / 88.10 | 126.81 / 96.90 | 92.50 |
| geography-pool-v2 | 112.54 / 89.17 | 127.40 / 100.83 | 95.00 |
| geography-pool-v3 | 119.12 / 89.78 | 129.96 / 100.11 | 94.94 |

v2 与 v3 的开发均值仅差约 0.06 分，不能据此证明训练目标普遍优劣。依据冻结前制定的选择规则采用 v2；最终测试只复测选定候选与上轮基线，没有在测试成绩上重新挑模型。

独立复测结果（每种 AI 每版 400 局）：

| AI | 项目平均 VP：前 → 后 | 项目增分 95% 区间 | 扣除收入后：前 → 后 | 扣除后增分 95% 区间 | 新版胜率 |
|---|---:|---:|---:|---:|---:|
| 世界模型 | 98.97 → 112.64 | +13.68 [11.19, 16.15] | 70.05 → 89.08 | +19.03 [16.94, 21.11] | 56.2% |
| 学习增强搜索 | 106.67 → 126.33 | +19.66 [17.28, 22.06] | 77.14 → 100.83 | +23.68 [21.58, 25.83] | 78.1% |

平均分提高同时出现在原项目计分和扣除收入的口径下。原始逐局成绩保存在候选目录的 `world-final.json` / `guided-final.json`；汇总与置信区间见 `world_model/reports/higher-scores.json`。

| AI | 项目 VP 中位数 / P90 / 最高 | 项目 VP ≥150 比例 | 扣除后 ≥150 比例 | 运河产业：前 → 后 | 运河道路：前 → 后 | 铁路产业：前 → 后 | 铁路道路：前 → 后 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 世界模型 | 115.00 / 135.00 / 160 | 1.0% | 0.0% | 13.41 → 8.08 | 22.45 → 31.90 | 23.00 → 24.34 | 9.98 → 23.90 |
| 学习增强搜索 | 127.00 / 147.00 / 171 | 7.8% | 0.0% | 15.15 → 8.06 | 25.87 → 35.52 | 24.69 → 27.46 | 9.79 → 28.89 |

单局最高值不等于平均实力；时代分解之外还可能有商人奖励或扣分。这里没有将项目分数解释为达到同分真人的水平。

世界模型胜率从 59.25% 到 56.25%，变化为 −3.00 个百分点，95% 区间 [−9.50, 3.50]，不能确认胜率提升。学习增强搜索从 67.25% 到 78.125%，增加 10.875 个百分点，95% 区间 [4.875, 16.625]。两套阵容中，对手的平均项目 VP 也分别增加 12.27 和 13.68 分；更活跃的公共产业、道路环境可能同时帮助其他玩家，不能把自身绝对增分全部解释为相对对手优势。

世界模型每局平均空过从 6.21 次降至 1.51 次，铺路从 10.74 次增至 16.14 次；学习增强搜索空过从 5.78 次降至 0.94 次，铺路从 11.71 次增至 17.96 次。运河产业分有所下降，主要增加的是道路得分。下一步若要对标真实桌游高手，仍需先完整校准规则，再训练跨时代高等级产业和批量行动的策略；本轮结果仅证明当前环境中的已测改进。

```powershell
node world_model/cli.js benchmark --games 400 --seed 108000001 --workers 6 --models world_model/experiments/geography-pool-v2/models --types heuristic,search,neural,world --out world_model/experiments/geography-pool-v2/world-final.json
node world_model/cli.js benchmark --games 400 --seed 108000001 --workers 6 --models world_model/experiments/geography-pool-v2/models --types heuristic,search,neural,guided --out world_model/experiments/geography-pool-v2/guided-final.json
node world_model/cli.js compare-geography --candidate world_model/experiments/geography-pool-v2
```

## 验证

- 21 项 JavaScript 测试与 7 项 Python 测试通过：包括新版 Python / 浏览器特征及推理一致、当前规则计分与道路特征对齐、世界模型的预筛选与推演不调用真实状态转移、旧模型兼容、整局座位轮换和串并行一致、收入扣除标签正确。
- 1,600 局附加数据的 52,800 个状态全部有限值，终局标签逐局匹配真实终局；训练、开发、最终复测种子无重叠。两版基线在增加计分记录前后逐局得分完全相同。审计见 `world_model/experiments/geography-study/data-audit.json`。
- 新版独立价值测试：原数据 100 局 RMSE 12.52 VP；附加对局测试 160 局 RMSE 13.71 VP。它们使用扣除收入后的新目标，不能直接与旧目标的 RMSE 比较。
- 状态转移网络的全部参数与原 `score-v2` 完全一致；神经网络对手的 SHA-256 未改变。默认模型目录与冻结复测模型的 SHA-256 完全一致。
- Chrome 实际完成 4 局世界模型阵容比赛（种子 108000001），四种 AI 的项目平均 VP 为 77.50 / 98.00 / 94.75 / 101.75，扣除收入后为 47.50 / 74.00 / 64.75 / 78.50，与 Node 最终复测的同一组四局完全一致。
- 游戏加载显示价值学习 9,280 局、产业与道路分数优先；世界模型、学习增强搜索、神经网络可以选择。推演面板实际预比较 32 个动作、深入 8 个候选，并标明估值不含终局收入加分。通过面板执行了一步真实动作，执行结果与预览一致，轮到世界模型玩家。
- 对战实验室的新版 / 旧版、两套阵容、收入扣除列、95% 区间与胜率说明均已核对。页面无浏览器控制台错误；截图见 [高分实验](../screenshots/world-model/higher-scores.png) 和 [模型推演](../screenshots/world-model/higher-score-imagination.png)。
