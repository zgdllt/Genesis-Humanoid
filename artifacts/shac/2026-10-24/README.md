# SHAC 实验状态与原始证据

这是删除原 `limh24` 工作目录前的历史归档，**不是成功训练结果**。正式 SHAC 完成 25 轮，在第 26 轮因物理状态梯度 NaN 退出；正式 PPO 被用户中断，成熟参照尚不存在。完整说明和恢复步骤见 [交接文档](../../../docs/SHAC_HANDOFF.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `remote_runs/shac_direct_seed0_pdcore1/` | 最新正式失败运行：原 config、20 轮前的日志记录、25 轮完成计数、26 轮错误、SwanLab URL |
| `remote_runs/shac_direct_seed0_pdcore1.stdout.log` | 正式运行完整控制台日志 |
| `remote_runs/repair_20261024/gradient_pd_core32.json` | 当前 `20f168d0...` 补丁的 9 项 float32 方向导数检查 |
| `remote_runs/repair_20261024/equivalence_pd_core32.json` | 当前版本的前向、恢复与 VJP 检查 |
| `remote_runs/repair_20261024/fullfootprint_pd10/` | 完整正式评估规模的 10 轮调试记录及 checkpoint |
| `remote_runs/repair_20261024/incident_final.json` | OOM 修复、容量验证及最新 NaN 失败索引 |
| `remote_runs/formal_seed0/` | 已中断 PPO 和已取消串行队列的原始状态 |
| `remote_runs/shac_direct_seed0*/` | 较早的正式尝试及 OOM 错误记录 |
| `workspace_diagnostics/` | 研发阶段的梯度/精度/接触消融、失败诊断及早期调试记录，包含旧版本证据 |
| `engine_metadata/` | 原运行私有引擎补丁标记和实际应用的 patch，匹配根目录 `patches/` |
| `remote_source_manifest.json` | 归档前的服务器源文件 SHA256；本地 314 个原源文件均匹配 |
| `related_checkout/original_workspace_uv_lock.diff` | 原 `/home/limh24/workspace/Genesis-Humanoid` 的独立未提交 lockfile 改动，单独保留，没有合并进 SHAC 的 lockfile |
| `PACKAGING_VALIDATION.json` | 提交前的 31 项 CPU 测试、Ruff 和文件传输校验记录 |
| `MANIFEST.json` | 本目录归档文件的大小和 SHA256 |

部分较早诊断失败、部分使用不同接触设置或旧代码，这是保留的研发记录。仅凭文件名或 `passed=true` 不足以授权新实验，必须检查完整身份。调试曲线和 checkpoint 不能作为高质量/样本效率结论。

## Checkpoint 的限制

当前实现的调试 checkpoint 为：

```text
remote_runs/repair_20261024/fullfootprint_pd10/checkpoint_00010.pt
```

它包含调试第 10 轮的 actor/critic、优化器、target critic。它不是正式第 25 轮的模型，不包含完整环境/RNG 状态，不能无缝续跑原正式实验。本次正式运行在保存点之前退出，**没有正式 checkpoint**。

## 发布与复现边界

- 保存原始 JSON、JSONL、stdout/error、调度记录、调试 checkpoint。原始内容中的服务器路径是历史来源，不是恢复依赖。
- 不纳入 `.venv`、下载/编译缓存、SwanLab SDK 运行目录或认证配置。旧 SDK 目录另外保存在本地忽略的备份中，未发布到此 Git 仓库。
- 大型源码 tarball 没有重复提交；源代码由本分支的 Git tree 和固定上游 Genesis patch 恢复。保留的历史 snapshot JSON 可能引用这些未纳入 Git 的 tarball。
- `remote_source_manifest.json` 的逐文件比对发生在本次添加交接文档之前；发布时对顶层 README 和实现说明增加了状态提示，运行代码保持一致。
- 原始临时诊断脚本归档为 `.py.txt` / `.sh.txt`，保留字节内容，避免被误当作可直接运行的入口。它们可能依赖已删除的旧机器路径。
- SwanLab 链接仍受原账户访问权限约束；本目录中的本地记录可独立阅读。

所有正式质量与样本效率结论仍待后续完成稳定训练和成熟 PPO 对照。
