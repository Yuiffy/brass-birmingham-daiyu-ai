# Brass 仓库整合 · 2026-10-09

统一维护仓库为 [`Yuiffy/brass-birmingham-daiyu-ai`](https://github.com/Yuiffy/brass-birmingham-daiyu-ai)。

| 原仓库 | 来源提交 | 整合方式 |
| --- | --- | --- |
| `Yuiffy/brass-birmingham` | `fa9d928dfd5c3008f32925630ae60513c5115a01` | 已是模拟器历史的祖先，随模拟器一并保留 |
| `Yuiffy/brass-birmingham-sim` | `621a552748a6297e9a279060d8b7a307b0a1a90e` | 完整历史通过 Git subtree 合入 `brass-sim/`，未 squash |

两份来源仓库在整合时均只有 `main` 分支，没有待处理的 PR。
`brass-birmingham` 来自 `npow/brass-birmingham`；当前仓库的 Rust 引擎来自
`artyom-morozov/fast_brass`，二者是独立实现。

## 包含内容

`brass-sim/` 保留原 JavaScript 游戏、自动对局脚本、2/3/4 人模型、训练代码、
检查点、测试、截图和实验报告。原项目作者及桌游设计者的署名保留在其 README 中。
原始回放数据和本地候选模型本来就未被来源仓库跟踪，不在此次迁移范围内。

根目录的 Rust/Svelte 规则引擎及其训练流程继续使用原有路径。
JavaScript 模拟器拥有自己的规则版本和模型格式，不能将其权重直接载入 Rust 引擎。
默认模拟器入口仍使用历史训练规则；`?rules=economy-v2` 使用部分校准后的规则及独立权重。
规则限制见 [economy-calibration.md](../brass-sim/docs/economy-calibration.md)。

## 运行

在仓库根目录执行，要求 Node.js 18 或更新版本，无需安装 npm 依赖：

```sh
npm --prefix brass-sim start -- --port 8086
npm --prefix brass-sim test
```

- 游戏：`http://127.0.0.1:8086/`
- 经济规则校准版：`http://127.0.0.1:8086/?rules=economy-v2`
- 实验室：`http://127.0.0.1:8086/arena.html?rules=economy-v2`

Python 训练还需安装 `brass-sim/world_model/requirements.txt`。
在 `brass-sim/` 中执行 `python -m unittest discover -s tests -p training_test.py` 可验证训练工具。

## 历史与文件完整性

subtree 导入后，`HEAD:brass-sim` 与模拟器来源提交的树完全一致，且两个来源提交均是
合并提交的祖先。迁移后只补充入口文档，并在 `brass-sim/.gitattributes` 中禁用 PNG
继承的 LFS filter，使原仓库直接存储的截图继续作为普通 Git 二进制文件保存。
模型、代码、实验报告和截图的 Git blob 保持不变。

## 验证结果

- `npm --prefix brass-sim test`：43 项通过，0 失败。
- Python 训练工具测试：10 项通过。
- HTTP 检查：游戏、规则校准入口、实验室、脚本、截图，以及历史与校准版 2/3/4 人模型，
  共 18 个资源均返回 200，响应字节与本地文件一致，模型 JSON 可解析。
- 两个来源提交均在合并历史中；除入口 README 和 PNG 属性适配外，
  `brass-sim/` 文件与来源 Git 树保持一致。Rust/Svelte 源码没有修改。

来源仓库在统一仓库推送并核对完成后归档。归档保留原 URL 与提交记录，且可以撤销；
私有模拟器来源仓库保留私有属性。
