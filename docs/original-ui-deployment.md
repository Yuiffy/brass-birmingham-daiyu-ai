# 原版界面部署

主界面是 `ui/` 的 Svelte 客户端：实体棋盘图片、与棋盘区域对应的蓝 / 绿 / 红 /
黄 / 紫地点牌、底部扇形手牌、缩放和城市预览。仓库整合没有覆盖这些组件；
`brass-sim/` 是独立的 JavaScript 模拟器，拥有另一套界面和规则。

## Vercel

项目 `brass-birmingham-daiyu-ai` 从仓库根目录部署。`vercel.json` 将请求交给
`Dockerfile.vercel` 定义的容器服务，构建顺序为 Svelte 静态文件、Rust
`cloud-ui` 二进制。二进制同时提供主界面、资源和游戏 API；无需本机服务。
Vercel 容器服务属于 Beta。

线上构建设置 `VITE_BRASS_BROWSER_SESSIONS=true`，客户端通过
`/api/browser_request` 发送当前浏览器的存档。后端为每次请求重建独立临时
数据库和游戏状态，完成操作后返还更新后的动作日志；临时文件随请求结束清理。
这沿用原引擎的存档回放，避免云端实例重启丢局或不同访客共用一个当前对局。
种子以字符串保存，保留完整的 64 位值。AI 默认使用学习增强 v2 原生适配版：
冻结教师价值头、真人策略、手牌评估与两步原生合法动作前瞻。权重嵌入 Rust
二进制，在 Vercel 容器内以 CPU 推理，无需本机程序或额外推理服务。
见[接入验证](native-trained-ai.md)。

存档保存在 IndexedDB 的 `brass-original-saves`，旧 `localStorage` 存档自动迁移，刷新后可在
**Join Game** 找到。它们属于当前浏览器，不会跨设备同步；清除站点数据会清除存档。
JavaScript 模拟器的旧存档不兼容 Rust 引擎，两个入口使用独立存档键。

## 本地验证

```sh
docker build -f Dockerfile.vercel -t fast-brass-original-cloud .
docker run --rm -p 5178:80 fast-brass-original-cloud
cargo test --lib web::
```

打开 `http://127.0.0.1:5178/` 验证实际线上构建。普通 `cargo run` 和
`npm --prefix ui run dev` 保留本地原有 API 与 SQLite 存档方式。

发布前检查新建对局、地点牌颜色、城市预览、建造 / 取消 / 撤销、AI 回合和
刷新后继续存档。云端回放测试覆盖并发访客隔离、选牌中途恢复、推荐的版本校验、
收入 / 手牌补充及运河到铁路时代切换。
