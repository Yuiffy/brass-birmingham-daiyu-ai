# Brass: Birmingham — World Model MVP

最新版本与结果：[以终局分数为目标的续训](score-training.md)。下文保留MVP分析与基础操作说明。

本项目在原游戏上增加了真实训练的轻量世界模型、直接动作评分网络、规则搜索 AI，以及可交互的预测对照与对战实验室。所有真实行棋和计分复用原来的规则引擎。

已附浏览器可用权重和真实实验报告。没有调用 LLM，没有预置虚构的预测结果，也没有把规则引擎包装成神经网络。

**当前版本**：[结构化世界模型、价值学习与独立复测](improved-world-model.md)。已加入学习增强搜索，默认权重更新为结构化模型。**历史续训记录**：[训练配置、版本对照与结果](continued-training.md)。本文中的 1,000 局 / 24 轮数值及普通差分架构保留为初版记录；对应权重与报告存放在 `world_model/experiments/v1/`。当前游戏读取 `world_model/models/`，实验室可以切换不同版本。

## 快速运行

需要 Node.js 18+。游戏和模型推理没有 npm 依赖，无需构建，也不需要 Python 服务。

```sh
npm run worldmodel:demo -- --port 8086
```

打开 http://127.0.0.1:8086 。Windows 也可以双击项目根目录的 `start-demo.cmd`。

- 开局时，每位玩家可选人类、原启发式 AI、搜索树逻辑型、神经网络型、世界模型型、学习增强搜索。
- 点击“四种 AI 同场演示”设置四种不同 AI 并开始游戏。
- “暂停 AI”“AI 走一步”用于逐步观察。
- “AI 推演 / 预测对照”暂停自动行棋并打开世界模型面板。选择候选查看预测与真实规则副本的差异，再点击“执行真实动作并比较”。执行后该按钮禁用，防止重复落子；局面变化后必须重新推演。
- 时代结束仍使用原来的结算弹窗，点击 Continue 进入下一时代。
- `arena.html` 展示所选版本的已保存结果和测试指标，也能在 Web Worker 中运行新的对战并导出 JSON。续训前后表提供相同种子下的对照与置信区间。
- 取消“启用 AI 实验模块”会禁用 AI 选项，按原人工游戏流程运行。权重加载失败会明确提示，神经网络 / 世界模型不能静默回退冒充已加载。

## 原项目分析与复用

基线来自 `Yuiffy/brass-birmingham-sim` 的 `main`，提交 `cdd9ac458106fa34e626168cf36c82d9f3a38cf7`。完整 Git 仓库已克隆到本地。

| 职责 | 原模块 / 接口 | 本次处理 |
|---|---|---|
| 固定规则、地图、牌、产业数据 | `js/gameData.js` | 原样复用 |
| 状态和资源 | `GameState`，`js/gameState.js` | 原样复用规则；增加模块作用域封装 |
| 合法目标、弃牌选择 | `GameLogic.getValid*` | 直接复用 |
| 七类动作执行 | `GameLogic.executeBuild/Network/Develop/Sell/Loan/Scout/Pass` | 直接复用 |
| 回合、时代、计分 | `advanceTurn/endCanalEra/endGame/calculateEraScore` | 直接复用 |
| 原 AI | `scripts/autorun.js` 的 `collectCandidates/chooseCandidate` | 保留策略与 CLI，增加 Node / 浏览器导出 |
| 棋盘与人工动作 UI | `boardRenderer.js`、`uiManager.js`、`main.js` | 加入独立 AI 控制入口 |

原仓库只有命令行启发式 AI，没有现成的神经网络 AI 或 MCTS。本次新增搜索树与两种学习型 AI；搜索采用小宽度 beam search，不称作 MCTS。

原仓库的 CommonJS 兼容改动在浏览器中产生重复的顶层 `const` 声明，已用 IIFE 修复，没有改写规则。新局开始会取消旧 UI 的事件监听，避免多次开局重复处理一个动作。原 AI 在三组基线种子下的所有玩家终局 VP 均保持一致，见测试。

## 什么是 Learned World Model

真实规则引擎直接执行程序员编写的规则。世界模型是用真实 transition 训练的函数：

```text
next_vector = current_vector + MLP(current_vector, action_vector) * delta_scale
```

它可能预测错误的金额、回合、地图资源与下一行动玩家。模型输出不会写回真实棋局；真正落子时仍使用原规则。

## 文件结构

```text
world_model/
  simulator.js       原引擎适配、完整状态副本、可复现随机数
  encoding.js        固定状态/动作编码、假想状态投影、共同评价函数
  generate.js        完整自动对局与训练数据写出
  dataset.py         NumPy memmap 数据读取
  model.py           MLP 前向、反向传播、Adam、JSON 权重导出
  train.py           世界模型 + 独立动作评分网络训练
  evaluate.py        测试指标、连续 rollout、真实样本差异
  inference.js       浏览器与 Node 共用推理（不调用 Python）
  planner.js         原启发式 / 规则搜索 / 直接评分 / 世界模型规划
  tournament.js      共用对战、座位轮换、终局统计
  benchmark_parallel.js 按完整种子组并行比赛，保留全部座位轮换
  compare_runs.py     配对版本比较、按种子组 bootstrap 区间与报告
  arena-worker.js    网页批量对战，避免阻塞主界面
  arena.js           对战与评估页面展示
  cli.js / server.js 命令行与仅监听本机的静态服务
  models/            已训练权重和训练日志
  reports/           当前评估、逐字段样本、对战、学习曲线、续训对比
  experiments/       原版、相同数据续训版、扩展数据版的权重和结果
  data/              本次数据，已在 .gitignore 中排除
js/aiController.js   原游戏内 AI 控制与预测对照面板
arena.html           对战实验室
tests/               规则回归、推理一致性、不变性、完整对局测试
```

`GameState.toJSON()` 仅返回摘要，不能用来复制规则状态。适配器复制自身字段后恢复 `GameState.prototype`，保留具体手牌、产业板、商人、牌堆和随机数游标，且不改变原对象。

## 数据生成

```sh
npm run worldmodel:generate -- --games 1000 --players 4 --seed 1701 --ai mixed --out world_model/data
```

生成目录已经存在时不会覆盖；复现实验请指定新目录，例如 `world_model/data-repro`。支持 `mixed`、`heuristic`、`random`。默认 mixed 在 70% 的选择中使用原 AI 排名，30% 按动作类型均匀探索后选择该类候选，避免稀有动作完全没有样本。

每次动作均在状态副本上执行。原 AI 可能提出缺铁的 Develop；数据生成保持原来的“执行失败后重选”行为，失败动作不入库，失败副本被丢弃。实际交互候选预先去掉这类已知失败。每条记录边界是“一次动作 + 回合推进 + 必要的时代 / 终局结算”。

输出：

- `transitions.f32`：小端 float32，每行 `[state, action, next_state, teacher_score/100]`。
- `transitions.jsonl`：行号、game id、种子、split、回合、行动玩家、时代、动作名称、完整候选参数、VP 变化与时代事件。
- `games.json`：每局起止行、完整终局成绩。
- `schema.json`：字段名、单位尺度、固定顺序、维度、生成配置和各动作样本数。
- `splits.json`：按整局 80/10/10 切分。每十局中的第九局验证，第十局测试。邻接 transition 不跨集合。

本次实际生成 1,000 局、128,000 条 transition：Build 45,718，Network 19,784，Develop 17,488，Sell 2,897，Loan 16,870，Scout 6,654，Pass 18,589。800 局 / 102,400 条用于训练，100 局验证，100 局测试。二进制数据约 745 MiB，另有 JSONL 元数据。

## 状态编码：689 维

编码完全独立于 DOM。城市、产业、连接及卡牌词表稳定排序；支持 2–4 人，缺席玩家补零。

- 全局：时代、回合、人数、当前玩家 one-hot、行动进度、首轮标志、结束标志、牌堆长度、百搭牌池、市场、回合顺序。
- 各玩家：金钱、收入、VP、手牌数量、当轮支出、剩余运河/铁路连接、百搭标志、各类产业已使用数量、手牌类型/城市计数。
- 每个城市槽位和两座农场：所属玩家、产业类型、等级、翻面、资源数、牌面 VP / 收入 / 连接 VP。
- 每条连接：所属玩家和时代。
- 商人：存在、啤酒、奖励领取状态，以固定的地点/交易类型键排列，不受数组洗牌影响。

标量按固定尺度归一化，例如钱 / 100、收入 / 30、煤市场 / 14。尺度并非截断，真实钱超过 100 仍可编码。产业板用已使用前缀长度表示，因为当前引擎按顺序取用/开发牌。

**信息边界**：这是公开棋盘和全部当前手牌的实验，不是严格隐藏信息对战。模型不接收未来牌序和内部随机数状态，也不编码 UI 选中状态。其输入不是完整 Markov 状态，抽牌和时代重新洗牌具有未建模的不确定性。原始完整状态只供真实引擎和安全复制使用，不作为全部模型输入。

## 动作编码：147 维

包含七类动作 one-hot、行动玩家、城市槽位、连接、第一/第二产业类型、实际弃牌的类别计数，以及合法候选中的成本、等级、牌面 VP、收入、煤铁需求、资源产量、出售啤酒需求。

成本来自当前引擎公开的合法候选描述；因此这是“给定丰富合法动作描述”的预测实验。固定动作序列评估也使用记录中的动作描述，包含当时的成本。不能把它当作完全不依赖规则元数据的环境学习结果。

原动作 API 自动选择商人、煤铁和啤酒来源，候选里没有独立的商人/资源选择参数，故没有伪造这些动作维度；相应的环境状态已编码。

## 网络和训练

依赖 Python 3.10+ 与 NumPy 2.x。选择 NumPy 是为了在本机 CPU 上完成小型模型训练，并让权重直接在原生 JS 中推理，无需 PyTorch、ONNX 或远程服务。

```sh
python -m pip install -r world_model/requirements.txt
npm run worldmodel:train -- --data world_model/data --epochs 24 --hidden 96 --seed 42 --out world_model/initial-repro
```

如 Python 不在 PATH，设置 `BRASS_PYTHON` 为 Python 可执行文件路径。CLI 会探测包含 NumPy 的 Python，也兼容此 Codex 环境的已配置 Python 运行时。

世界模型结构为 `836 → 96 ReLU → 689`，约 147k 参数，预测归一化状态差分。差分尺度只在训练集拟合，下限 0.03；非零变化字段损失权重为 5，未变化字段权重为 1。使用 Adam、batch 512、学习率 0.001，训练 24 轮，按验证 MSE 保存最佳权重。

独立的动作网络为 `836 → 96 ReLU → 1`，拟合原启发式评分。它是监督蒸馏基线，既没有自我强化学习，也不调用世界模型。按验证评分 MAE 保存最佳权重。浏览器根据元数据验证权重维度与编码版本，缺权重会报错。

## 评估与真实结果

```sh
npm run worldmodel:evaluate -- --data world_model/data --models world_model/models --out world_model/reports
```

原始机器可读结果为 `reports/evaluation.json`，易读表格为 `reports/evaluation.md`。评估包括当前玩家 argmax 准确率，各字段组原单位 MAE/取整准确率，仅真实变化字段上的误差，变化 precision/recall/F1，按动作类型分组的误差，以及留出 transition 的逐字段样例。

下述初版数值现保存在 `world_model/experiments/v1/reports/`；根 `reports/` 随当前游戏模型更新。

本次测试：下一行动玩家准确率 21.34%，金钱 MAE £6.057，收入 MAE 0.810，VP MAE 0.575，煤市场取整准确率 89.87%，铁市场 93.80%；变化检测 precision 32.66%，recall 80.07%，F1 46.39%。总体高比例的未变化字段会掩盖错误，因此同时展示“状态不变”的 persistence baseline。本模型整体误差高于该基线。

出售样本少（测试 290 条），其标准化 MAE 约 0.039，高于常见的 Build（约 0.020）。回合轮换、收入支付、稀有翻面/奖励与大范围结算尤其困难。当前玩家准确率很低，不能据此声称模型已经学会完整规则。

## 多步 imagination 与两种规划

`Network.imagine(vector, actionVectors)` 支持连续预测。测试窗口不会跨局，递归输入严格使用上一轮模型输出，绝不喂回真实中间状态。

| 预测深度 | 模型标准化 MAE | 状态不变基线 MAE |
|---:|---:|---:|
| 1 | 0.01980 | 0.00459 |
| 2 | 0.02986 | 0.00962 |
| 3 | 0.03964 | 0.01368 |
| 4 | 0.05184 | 0.01767 |
| 5 | 0.06065 | 0.02137 |

面板中的误差条使用“固定真实合法动作序列”；左侧规划则会从假想状态继续选择候选。两者用途不同，已分别标注。

世界模型规划首先获得最多 8 个真实合法候选（优先保留不同动作类型，再按原评分补齐）；第一步以后只用网络预测下一向量，再投影成近似 GameState，复用合法目标生成函数提出后续动作。原始连续预测向量用于下一次神经网络推理；投影只用于候选生成。自己的分支保留宽度 2，对手从最多 3 个候选里选择对自己评价最高的分支。无后续候选时提前停止。**该规划路径不调用 `Sim.step`**，单元测试用抛错替身验证这一点。

离散投影会取整、截断资源、重新选取匹配等级的产业数据，并保留基底状态中未重建的牌堆与部分回合信息。这是粗糙的候选提议器，不能用于实际行棋或声称预测出一致的完整状态。

搜索树型使用相同根候选、深度、beam 宽度和评价函数，但分支转移调用原引擎。它先用编码状态推导可复现种子，从固定普通卡词表独立采样未来抽牌，不读取真实牌序/内部 RNG。该隐藏牌采样并非精确的剩余牌分布。

共同评价函数综合当前 VP、收入、现金、产业牌面 VP（翻面较高权重）、连接数，并扣除最强对手的一部分评价。它是启发式，不是胜率概率；例如 89.01 表示评价值，不表示 89% 胜率。

## AI 对战比较

```sh
npm run worldmodel:benchmark -- --games 100 --seed 800001 --depth 2 --width 8
```

默认四种 AI 同场。每四局循环座位并复用相同初始种子，随后换种子。并列最高 VP 平分一场胜利。每局必须到达 `gameOver` 才记录终局成绩，安全上限触发会报错，不会把截断局当完整局。浏览器的 Worker 使用相同实现。

本次 100 局真实结果：

| AI | 胜率 | 平均终局 VP |
|---|---:|---:|
| 搜索树逻辑型 | 37.5% | 85.13 |
| 世界模型型 | 24.5% | 78.99 |
| 神经网络型 | 21.0% | 78.18 |
| 原启发式 AI | 17.0% | 75.31 |

完整逐局记录位于 `reports/tournament.json`。计时范围是共同候选集和编码准备完成后的评分/搜索，不包括 UI、公共候选准备和真实执行。相同深度并不等于相同计算时间。这些结果只适用于本次配置和模型，不能推出世界模型普遍强于神经网络或模型已经准确理解规则。

## 数据量对照

另外完成了相同 100 局验证集、24 轮训练、隐藏层 96 的训练子集实验，分别使用 80 / 320 / 800 个训练局（总数据池 1,000 局）。`reports/learning-curve.json` 记录最佳验证 MSE 和训练样本数。

```sh
npm run worldmodel:train -- --max-games 80 --epochs 24 --out world_model/curve80
npm run worldmodel:train -- --max-games 320 --epochs 24 --out world_model/curve320
```

80 / 320 / 800 局的最佳验证 MSE 分别约 0.004125 / 0.003739 / 0.003632。这只是一次种子下的观测。10k 续训实验另见续训记录；50k 尚未运行。更大数据集会消耗更多磁盘，当前固定 float32 数据布局每条记录约 6 KiB。

## 验证与后续工作

```sh
npm test
node scripts/autorun.js --games 3 --players 4 --seed 1
```

测试覆盖原 AI 回归、浏览器/Worker 全局作用域、完整状态深拷贝、失败动作不改变棋局、2/3/4 人完整对局、七类动作和时代结算、跨 Python/JS 推理一致性、权重/输入检查、递归 prediction、世界模型规划不调用真实转移、隐藏牌隔离和座位轮换。还在真实 Chrome 中检查了开局、模型加载、自动 AI、时代结算、预测面板、真实执行和评估页面。

当前 MVP 的主要局限是预测精度，尤其是金钱、回合与当前玩家；离散棋盘用回归损失而非专门分类头，不保证合法性；网络针对 4 人训练；原候选没有穷举所有弃牌、连卖和双铁路组合；原引擎自身的规则简化被保留，指标衡量的是其实现而非官方规则完备性。

下一步优先考虑离散变量分类头与变化 mask、稀有事件采样、显式合法性/不确定性预测和对手建模。之后才考虑 representation encoder + latent dynamics + policy/value head、MuZero 风格搜索，或隐藏信息 belief state。进行这些升级时仍保留真实规则引擎作独立对照，重新做按整局划分的训练与评估。
