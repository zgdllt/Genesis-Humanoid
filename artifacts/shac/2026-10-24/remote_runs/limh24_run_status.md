# limh24 实验交付记录

最新状态（2026-10-24）：正式 SHAC 完成 25 轮，在第 26 轮反传时因 `Nan grad in qpos or dofs_vel found at step 91` 退出。此次不是 OOM；第 20 轮快照前仍有约 1.07 GB 空闲显存。具体产生非有限梯度的运算尚未定位，当前没有训练进程，也未自动重跑。

此前容量验证记录：第 2 轮快照 OOM 经按需输出 PD 状态后已跨过；完整配置连续 10 轮和受控数值梯度、状态恢复检查通过。这些检查未覆盖第 26 轮遇到的数值问题，不能视为长跑基线已验证。

- SSH：`limh24`；主机：`limh24-ASUS`；GPU：RTX 5070 Ti 16 GB。
- 实际代码：`/home/limh24/workspace/Genesis-Humanoid-shac`，分支 `codex/genesis-shac`。
- 本地可审查代码：`/Users/limh24/workspace/Genesis-Humanoid-shac`。
- 引擎：Genesis v1.4.5，私有 `.deps/genesis`；patch SHA256：`20f168d0cbc59a761d36096ca1986192714be7fb0f3c49b2e2b653575c8eccac`。

当前版本验证：31 项 CPU 测试和 Ruff 通过；1/4/24 步的 reward、foot velocity、contact force 共 9 项 float32 方向导数检查通过，最大相对差异 5.54%，重复输入的 loss 误差为 0。前向一致性、反传后恢复、contact basis/output VJP 均通过。`--debug --full-evaluation` 的 4096 环境连续 10 轮完成，实际使用 128×1000 的评估和 50000 轮课程计划；第 10 轮快照前空闲约 1.06 GB，原失败时为约 0.11 GB。

这些只用于调试和实现验收，不能证明策略质量或样本效率。有限差分夹具只验证所记录的接触分支，不能证明所有接触切换处都可微。当前 SHAC 热启动后约 25 秒/轮，PPO 约 1 秒/轮，尚无加速结论。

[SHAC 调试记录](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/nt38e21l)

本次正式 SHAC 配置为 4096 环境、50000 轮、seed 0；同卡通过独占锁防止重叠。固定质量门槛和 20 秒独立评估保持不变。当前已退出，完成的 25 轮不足以建立正式质量或样本效率结论。

- 已退出的 SHAC PID：`3514298`。
- 结果目录：`/mnt/big_disk/limh24/genesis-shac-20261023/shac_direct_seed0_pdcore1`。
- 启动记录：`/mnt/big_disk/limh24/genesis-shac-20261023/shac_direct_seed0_pdcore1.launch.json`。
- 标准输出：`/mnt/big_disk/limh24/genesis-shac-20261023/shac_direct_seed0_pdcore1.stdout.log`。
- 冻结源码归档：`/mnt/big_disk/limh24/genesis-shac-20261023/implementation_snapshot.tar.gz`。

[重启后的正式 SHAC SwanLab](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/2f5714lx)

[完整配置 10 轮调试证据](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/int9z1vu)

原失败记录完整保留于 `shac_direct_seed0`。修复证据和新源码归档位于 `repair_20261024`，未改动正式任务、评估窗口或质量标准。私有引擎已逐文件校验后移到大盘并保留原路径链接；缓存和临时文件配置见 `runtime_environment.sh`。
最终修复与证据索引为 `repair_20261024/incident_final.json`，当前源码归档为 `repair_20261024/implementation_snapshot_pd_core.tar.gz`。

[原失败 SHAC SwanLab](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/19qlpf9e)

被中断的 PPO 日志保留于 `formal_seed0/ppo`，原队列记录为 `interrupted_by_user`，不会自动继续启动实验。该 PPO 尚未达到成熟基准，不能用于最终样本效率结论。

[已中断的 PPO SwanLab](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/fi58jbu5)

训练当前已停止，正式高质量策略和最终效率对比尚未产生。
