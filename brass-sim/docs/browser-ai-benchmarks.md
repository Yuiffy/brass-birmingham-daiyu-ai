# 浏览器 AI：2～4 人适配与统一测分

教师＋保牌版的学习增强 AI 现在支持 2、3、4 人局。所有网页 AI 选项的括号均分使用本轮统一实测结果，切换人数、规则或版本会显示对应数据。

线上入口为 `/ai-lab/`，使用本目录的 JavaScript 引擎和 AI。网站首页的原版 Rust/Svelte 游戏使用另一套独立 AI，本报告不用于标注它的棋力。

此前教师训练、模型编码和验证都围绕四人局，因此网页直接限制四人。本轮为二、三人学习增强搜索迁移已冻结的四人估值网络，继续用真实规则引擎展开动作，并沿用保牌策略。这是估值迁移，**没有重新训练二、三人神经网络**。二、三人纯世界模型 AI 继续使用各自原有动态模型；四人教师版保留原模型。原权重、默认版本均未升级替换。

直接选择：[两人教师版](../?rules=economy-v2&strategy=teacher-trained-v2&players=2)、[三人教师版](../?rules=economy-v2&strategy=teacher-trained-v2&players=3)、[四人教师版](../?rules=economy-v2&strategy=teacher-trained-v2&players=4)。

## 如何测试

- 60 个网页组合：历史规则 15 个，经济规则 45 个。每组包含基础、搜索、神经网络、世界模型、学习增强五种 AI，经济规则另有标准、攻略、教师＋保牌三种版本。
- 实际行为完全相同的选项复用同一报告，共 38 个不同配置；每配置 32 个独立种子的完整同类 AI 自对弈，总计 1,216 局。
- 全部座位使用同一种 AI，搜索深度 2、宽度 8，与浏览器自动落子一致。种子为 `620260101 + 人数 * 100000 + 局号 * 9973`，局号 0～31；相同人数各配置使用相同起局种子。
- 每局先求各座位 VP 平均，再跨局求平均；95% 区间按完整对局进行 4,000 次 bootstrap。各座位不是独立样本，同策略轮换座位没有重复计数。
- 每局保存全部真实动作，再单独从初始种子重放；不调用规划器或模型，校验动作合法、对局结束及每座位最终 VP、现金、收入。
- 执行前冻结源代码、模型和配置 SHA-256，完成每组时检查文件没有变化。网页均分是同类自对弈数据，不能等同于对战真人的得分或赛事水平。

## 完整结果

下表由冻结结果生成。括号为按整局重采样的 95% 区间；网页只显示四舍五入后的平均分。

### 经济规则 · 标准版

| AI | 两人局 | 三人局 | 四人局 |
|---|---|---|---|
| 基础 AI | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-2-heuristic-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-3-heuristic-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-4-heuristic-standard.json.gz)（0.00～0.00） |
| 搜索 AI | [56.38](../world_model/experiments/browser-options-20261010/economy-v2-2-search-standard.json.gz)（48.63～63.98） | [59.79](../world_model/experiments/browser-options-20261010/economy-v2-3-search-standard.json.gz)（54.06～65.42） | [53.63](../world_model/experiments/browser-options-20261010/economy-v2-4-search-standard.json.gz)（48.48～58.73） |
| 神经网络 AI | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-2-neural-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-3-neural-standard.json.gz)（0.00～0.00） | [0.24](../world_model/experiments/browser-options-20261010/economy-v2-4-neural-standard.json.gz)（0.00～0.69） |
| 世界模型 AI | [14.22](../world_model/experiments/browser-options-20261010/economy-v2-2-world-standard.json.gz)（9.70～19.03） | [13.82](../world_model/experiments/browser-options-20261010/economy-v2-3-world-standard.json.gz)（9.17～18.68） | [22.06](../world_model/experiments/browser-options-20261010/economy-v2-4-world-standard.json.gz)（17.76～26.38） |
| 学习增强 AI | [92.22](../world_model/experiments/browser-options-20261010/economy-v2-2-guided-standard.json.gz)（85.92～98.03） | [72.77](../world_model/experiments/browser-options-20261010/economy-v2-3-guided-standard.json.gz)（68.41～77.14） | [77.55](../world_model/experiments/browser-options-20261010/economy-v2-4-guided-standard.json.gz)（75.04～79.80） |

### 经济规则 · 攻略版

| AI | 两人局 | 三人局 | 四人局 |
|---|---|---|---|
| 基础 AI | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-2-heuristic-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-3-heuristic-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-4-heuristic-standard.json.gz)（0.00～0.00） |
| 搜索 AI | [56.38](../world_model/experiments/browser-options-20261010/economy-v2-2-search-standard.json.gz)（48.63～63.98） | [59.79](../world_model/experiments/browser-options-20261010/economy-v2-3-search-standard.json.gz)（54.06～65.42） | [53.63](../world_model/experiments/browser-options-20261010/economy-v2-4-search-standard.json.gz)（48.48～58.73） |
| 神经网络 AI | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-2-neural-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-3-neural-standard.json.gz)（0.00～0.00） | [0.24](../world_model/experiments/browser-options-20261010/economy-v2-4-neural-standard.json.gz)（0.00～0.69） |
| 世界模型 AI | [14.22](../world_model/experiments/browser-options-20261010/economy-v2-2-world-standard.json.gz)（9.70～19.03） | [13.82](../world_model/experiments/browser-options-20261010/economy-v2-3-world-standard.json.gz)（9.17～18.68） | [22.06](../world_model/experiments/browser-options-20261010/economy-v2-4-world-standard.json.gz)（17.76～26.38） |
| 学习增强 AI | [156.20](../world_model/experiments/browser-options-20261010/economy-v2-2-guided-human-guide-v1.json.gz)（153.75～158.55） | [140.73](../world_model/experiments/browser-options-20261010/economy-v2-3-guided-human-guide-v1.json.gz)（138.16～143.50） | [120.41](../world_model/experiments/browser-options-20261010/economy-v2-4-guided-human-guide-v1.json.gz)（117.73～123.11） |

### 经济规则 · 教师＋保牌版

| AI | 两人局 | 三人局 | 四人局 |
|---|---|---|---|
| 基础 AI | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-2-heuristic-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-3-heuristic-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-4-heuristic-standard.json.gz)（0.00～0.00） |
| 搜索 AI | [56.38](../world_model/experiments/browser-options-20261010/economy-v2-2-search-standard.json.gz)（48.63～63.98） | [59.79](../world_model/experiments/browser-options-20261010/economy-v2-3-search-standard.json.gz)（54.06～65.42） | [57.81](../world_model/experiments/browser-options-20261010/economy-v2-4-search-teacher-trained-v2.json.gz)（51.85～63.36） |
| 神经网络 AI | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-2-neural-standard.json.gz)（0.00～0.00） | [0.00](../world_model/experiments/browser-options-20261010/economy-v2-3-neural-standard.json.gz)（0.00～0.00） | [0.24](../world_model/experiments/browser-options-20261010/economy-v2-4-neural-standard.json.gz)（0.00～0.69） |
| 世界模型 AI | [14.22](../world_model/experiments/browser-options-20261010/economy-v2-2-world-standard.json.gz)（9.70～19.03） | [13.82](../world_model/experiments/browser-options-20261010/economy-v2-3-world-standard.json.gz)（9.17～18.68） | [10.30](../world_model/experiments/browser-options-20261010/economy-v2-4-world-teacher-trained-v2.json.gz)（6.32～14.78） |
| 学习增强 AI | [163.41](../world_model/experiments/browser-options-20261010/economy-v2-2-guided-teacher-trained-v2.json.gz)（156.42～169.42） | [151.92](../world_model/experiments/browser-options-20261010/economy-v2-3-guided-teacher-trained-v2.json.gz)（148.67～154.98） | [128.25](../world_model/experiments/browser-options-20261010/economy-v2-4-guided-teacher-trained-v2.json.gz)（125.88～130.52） |

### 历史规则（含额外收入分）

| AI | 两人局 | 三人局 | 四人局 |
|---|---|---|---|
| 基础 AI | [68.03](../world_model/experiments/browser-options-20261010/legacy-v1-2-heuristic-standard.json.gz)（62.39～73.69） | [63.09](../world_model/experiments/browser-options-20261010/legacy-v1-3-heuristic-standard.json.gz)（57.27～69.02） | [63.42](../world_model/experiments/browser-options-20261010/legacy-v1-4-heuristic-standard.json.gz)（57.50～69.38） |
| 搜索 AI | [40.95](../world_model/experiments/browser-options-20261010/legacy-v1-2-search-standard.json.gz)（38.53～43.73） | [63.78](../world_model/experiments/browser-options-20261010/legacy-v1-3-search-standard.json.gz)（61.52～66.23） | [72.30](../world_model/experiments/browser-options-20261010/legacy-v1-4-search-standard.json.gz)（69.26～75.77） |
| 神经网络 AI | [64.94](../world_model/experiments/browser-options-20261010/legacy-v1-2-neural-standard.json.gz)（59.67～70.13） | [64.84](../world_model/experiments/browser-options-20261010/legacy-v1-3-neural-standard.json.gz)（59.98～70.14） | [73.97](../world_model/experiments/browser-options-20261010/legacy-v1-4-neural-standard.json.gz)（67.17～81.15） |
| 世界模型 AI | [120.81](../world_model/experiments/browser-options-20261010/legacy-v1-2-world-standard.json.gz)（116.05～125.94） | [123.80](../world_model/experiments/browser-options-20261010/legacy-v1-3-world-standard.json.gz)（118.82～128.67） | [121.83](../world_model/experiments/browser-options-20261010/legacy-v1-4-world-standard.json.gz)（119.50～124.09） |
| 学习增强 AI | [127.91](../world_model/experiments/browser-options-20261010/legacy-v1-2-guided-standard.json.gz)（120.36～134.95） | [126.38](../world_model/experiments/browser-options-20261010/legacy-v1-3-guided-standard.json.gz)（119.92～132.77） | [129.80](../world_model/experiments/browser-options-20261010/legacy-v1-4-guided-standard.json.gz)（126.38～133.09） |

全部 1,216 局、127,264 个动作均通过重放核验。

## 如何解读

历史规则含额外终局收入分，与经济规则的 VP **不能横向比较**。当前经济规则仍是部分官方规则校准，AI 还可看到所有当前手牌（不能读取未来牌序）。

基础 AI 在经济规则下的 0 分是实际弱策略结果，非缺失评分：32 局两人测试中，它反复发展、贷款、侦察与跳过，没有建造或出售；三、四人虽有少量建造，同样没有出售。神经网络基线也有类似局限。重放审计确认这些对局合法且完整，网页如实保留低分。

版本影响学习增强的策略；教师版还改变四人世界模型权重和搜索的双铁路候选。基础与神经网络不随版本改变；二、三人世界模型及搜索也不随版本改变，因此对应选项共享实测报告。跨配置的自对弈均分高低不等于互相对战胜率，迁移版也不保证所有人数都最强。

## 原始证据与复现

- [冻结协议与逐文件 SHA-256](../world_model/experiments/browser-options-20261010/protocol.json)
- [全部 60 个选项到实测报告的映射、统计和审计总数](../world_model/experiments/browser-options-20261010/summary.json)
- [网页实际加载的均分数据](../world_model/experiments/browser-options-20261010/browser-ai-scores.js)
- [冻结执行源代码原始字节](../world_model/experiments/browser-options-20261010/frozen-sources.zip)：解压到仓库根目录后可恢复测试时的源文件行尾。
- 每个配置的 `.json.gz` 报告位于同目录，含 32 局的种子、完整动作、逐座位终局分数、估值/策略模型身份和重放校验结果。具体文件链接在表中。

模型文件继续使用仓库中原有 checkpoint；其 Git blob 字节与协议的 11 个模型哈希一致。源文件可能被 Git 自动转换行尾，精确复现时使用上述压缩包。

在 `brass-sim` 目录运行：

```sh
node world_model/browser_benchmark.js --games 32 --workers 16 --seed 620260101 --out world_model/.local-experiments/browser-reproduction
```

同一冻结协议的已有报告可复用，源代码或权重变化会拒绝复用。更改策略后必须重新测分。自动化验证另覆盖全部 60 个选项标签，以及二、三、四人教师版的开始、暂停、单步、推演、重新推演和真实执行。
