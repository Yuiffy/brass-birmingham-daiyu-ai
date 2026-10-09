> 历史版本。当前模型与新增训练结果见 [更高分数实验](higher-scores.md)。

# 以终局分数为目标的续训

本轮以**自身终局平均VP**作为主要棋力指标，胜率只作辅助。两个AI分别对阵相同的原启发式、原搜索、固定动作评分神经网络，在同一位置轮换座位；真实规则、计分方式、推演深度2和根候选宽度8保持不变。

独立复测已完成，当前游戏已加载选中的第二阶段权重。

| AI | 续训前平均VP | 续训后平均VP | 分数增益 | 增益95%区间 |
|---|---:|---:|---:|---:|
| 世界模型 | 87.6425 | **97.3700** | **+9.7275** | [7.5349, 11.8550] |
| 学习增强搜索 | 89.1125 | **106.3150** | **+17.2025** | [14.7375, 19.6951] |

相对提升分别约11.1%和19.3%，区间均支持平均分提高。两种AI的对手平均分也分别提高5.16和10.08，因此部分变化表现为整局得分提升；报告保留这一结果，不把绝对分数增益等同于同等幅度的相对优势。胜率作为辅助指标：世界模型47.625%→54.750%，学习增强搜索54.750%→64.625%。

结果仅适用于这套4人公开手牌设置、固定三类对手与预算。学习增强搜索在这组分别对战的基准中平均分更高，仍不能据此断言面对所有对手或直接同场时都更强。[完整统计JSON](../world_model/reports/score-training.json)包含逐项增益和文件哈希。

## 训练做了什么

此前价值网络学习 `自身终局VP − 0.25×最高对手终局VP`。这轮把目标改为纯粹的 `自身终局VP`，继续训练已有价值网络。世界模型的变化量、变化门和离散控制权重冻结，搜索和规则引擎不变。这样可以把变化归因到本轮价值目标及数据训练组合，而不会混入新计分规则或更大搜索预算。

第一阶段从原价值网络第23轮的权重与Adam状态恢复，使用原10,000局数据中的8,000局训练，64轮、batch1024、学习率0.0003余弦降至0.00003。每8步采样，加入终局、四个玩家视角，共544,000条训练样本，选择原100局验证集上最好的第86轮。

第二阶段新增800局真实比赛：第一阶段AI与原启发式、搜索、神经网络混合对局，并有10%合法动作探索。起始种子180000001，每4步记录公开状态，加上终局，共26,400个状态。按整局划分640训练 / 80验证 / 80测试；记录的终局标签已与真实终局状态逐项核对。

原10,000局数据的所有玩家平均终局得分为42.72；这批新数据为82.41，补充了原数据中较少出现的较高分局面。两批数据的策略与分布不同，这个均值差不能单独当作AI升级的对照证据。

第二阶段从第一阶段最佳权重恢复，再训64轮。640局新训练游戏产生84,480个玩家视角样本，每轮采样权重为旧样本的4倍；与原样本合计881,920条呈现样本，来自8,640个不同训练游戏。权重只依据原100局验证MSE与新80局验证MSE的等权均值选择，最佳累计第137轮。测试游戏没有参与训练或选轮。

价值网络结构仍为109→64→1，输入仅使用当前公开状态。终局分数是标签，不是推理输入；未知未来抽牌序列及游戏随机数状态不在输入中。真实终局直接返回自身已结算VP，旧版相对价值仍按原模型元数据解释。

## 开发赛与方案选择

每个AI、每个版本80局，种子48000001。比较同一批20个种子组、完整四座位轮换。

| 开发版本 | 世界模型平均VP | 学习增强搜索平均VP |
|---|---:|---:|
| 本轮续训前 | 83.01 | 85.45 |
| 第一阶段：纯分数目标、原数据续训 | 86.95 | 86.98 |
| 第二阶段：加入新对局数据续训 | **94.40** | **105.53** |

选择规则为两种AI平均VP的等权均值最大，且两者都不得低于各自开发赛基线；胜率不参与选择。第二阶段满足条件，选中后将文件哈希写入 `experiments/score-study/selection.json`，然后才运行候选最终比赛。上述80局用于选择方案，不能作为独立成功证据。

## 独立复测

最终复测使用新种子68000001，每版每种AI400局，100个种子组；两种AI各有旧版/新版对照，共1,600局。统计以四局种子组为整体，20,000次配对bootstrap计算平均VP增益的95%区间。比赛种子与原始数据、新对局数据、开发比赛均不重叠。

最终数值见本目录报告前部和 `world_model/reports/score-training.json`。对手的策略及神经网络权重固定，但会自然响应改变后的棋局；同时报告对手平均分变化，避免把游戏整体分数上升误解为压制对手。区间只反映这组权重的棋局抽样波动，不覆盖训练随机种子、其他对手或隐藏手牌设置。

价值预测另在原数据的第201–300个测试游戏和新对局数据80个测试游戏上检验。第二阶段的终局价值RMSE分别为17.97和15.01 VP单位。它是预测误差指标，不能代替实际平均终局得分。动力学权重逐项核对未变化，因此原动力学评估报告继续适用。

## 游戏与复现

游戏支持选择世界模型或学习增强搜索。实验室按终局平均VP排序，并提供“分数续训前”版本与当前版本的相同种子对照。原架构改进和旧续训报告作为历史记录保留。

```sh
npm run worldmodel:demo -- --port 8086
```

以下使用新目录，避免覆盖本次实验。Python需要NumPy；浏览器推理没有Python服务依赖。

```sh
node world_model/cli.js train-value --resume world_model/experiments/value-v1 --opponent-weight 0 --epochs 64 --lr 0.0003 --stride 8 --out world_model/repro-score-value1
node world_model/cli.js compose --dynamics world_model/experiments/structured-10k/models --value world_model/repro-score-value1 --out world_model/repro-score1/models
node world_model/cli.js collect-value --games 800 --seed 180000001 --models world_model/repro-score1/models --workers 4 --out world_model/data-repro-score
node world_model/cli.js train-value --resume world_model/repro-score-value1 --opponent-weight 0 --epochs 64 --lr 0.0003 --stride 8 --replay world_model/data-repro-score --replay-weight 4 --out world_model/repro-score-value2
node world_model/cli.js compose --dynamics world_model/experiments/structured-10k/models --value world_model/repro-score-value2 --out world_model/repro-score2/models
node world_model/cli.js benchmark --models world_model/repro-score2/models --games 400 --seed 68000001 --depth 2 --width 8 --workers 4 --out world_model/repro-score2/world-final.json
node world_model/cli.js benchmark --models world_model/repro-score2/models --types heuristic,search,neural,guided --games 400 --seed 68000001 --depth 2 --width 8 --workers 4 --out world_model/repro-score2/guided-final.json
node world_model/cli.js compare-scores --candidate world_model/repro-score2 --replay world_model/data-repro-score --out world_model/repro-score2/score-report.json
node world_model/cli.js evaluate-value --models world_model/repro-score2/models --replay world_model/data-repro-score --out world_model/repro-score2/reports
```

再次继续训练已组合的当前价值头，可用 `train-value --resume world_model/models --opponent-weight 0 ... --out 新目录`，再用 `compose` 把新价值头与动力学组合。续训会恢复匹配的Adam状态；目标相同情况下，续训前权重也纳入验证选择。省略 `--opponent-weight` 时会继承恢复模型的目标；从旧模型切换为纯分数目标需显式提供0。

所有样本、权重、配置和报告均来自本地实际运行。新数据约73MB，存于 `world_model/data-score-league/` 并从Git排除；审计摘要和来源模型哈希保存在 `experiments/score-study/replay-audit.json`。代码与实验目录保留，尚未推送远程。

## 验证记录

- 18项JavaScript测试通过：包括原规则回归、世界模型规划不调用真实转移、学习增强搜索正确调用规则与价值网络、Python/JS多阶段价值输出一致、纯分数终局语义、旧模型兼容、串行与并行逐局一致。
- 5项Python测试通过：包括纯分数标签不受对手终局分变化影响、原相对分数目标兼容，以及Adam恢复一致性。
- 实际800局新数据全部终局，终局标签与原规则结算状态一致，640/80/80划分且种子无重复；最终比赛种子与所有拟合数据隔离。
- 当前权重SHA-256与开发赛结束时冻结的选择清单一致，动作神经网络对手哈希不变。世界模型动力学参数逐项不变。
- 两次1轮续训烟雾检查从已组合模型恢复；优化器恢复成功，较差续训权重未替换最佳版本。省略目标参数时正确继承纯分数目标。
- Chrome实测新版本、固定三对手的学习增强搜索阵容：4局、种子68000001、深度2。学习增强搜索/原搜索/原启发式/神经网络的平均VP为100.50/84.00/73.50/67.25，与Node完全一致。该4局仅检验浏览器一致性，不参与上述独立统计。
- 游戏加载状态显示“价值学习8,640局、终局分数优先”；可以选择新AI并打开包含8个候选的3步预测面板。实验室可切换本轮前后版本，按平均VP排序。

截图：[分数对照](../screenshots/world-model/score-comparison.png)、[新模型推演](../screenshots/world-model/score-imagination.png)。
