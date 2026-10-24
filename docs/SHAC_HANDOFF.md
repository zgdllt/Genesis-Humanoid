# G1 walk SHAC 交接与恢复（2026-10-24）

本分支保存当前实现、原始验证证据、失败记录和调试 checkpoint，供删除原服务器工作目录后继续开发。**当前 SHAC 尚未跑稳，没有高质量策略或样本/时间效率提升的结论。**

## 当前状态

| 项目 | 已核实结果 |
| --- | --- |
| 正式 SHAC | `4096` 环境、计划 `50000` 轮、seed `0`；完成 `25` 轮，在第 `26` 轮反传退出 |
| 最新错误 | `Nan grad in qpos or dofs_vel found at step 91`，此次不是 CUDA OOM |
| `step 91` | 反传窗口内引擎报告的物理步，不能解释为训练第 91 轮 |
| 已完成训练控制步 | `2457600 = 4096 × 24 × 25`；失败轮的完整更新未完成 |
| 正式评估 | 独立 `128` 环境 × `1000` 控制步（20 秒），第 0 轮评估已完成；尚未到第 1000 轮评估 |
| 显存 | 第 20 轮快照前约 `1.07 GB` 空闲；曾发生的第 2 轮快照 OOM 经减少冗余 PD 状态输出后跨过 |
| 正式 PPO | 用户要求直接运行 SHAC，故在尝试第 `305` 轮时中断，完成 `304` 轮；不是成熟的 50000 轮基准 |
| 当前进程 | 已退出，无本任务 GPU 训练进程或待续跑的串行队列 |
| 正式 checkpoint | **没有。** 当前 runner 每 1000 轮及结束时保存，正式运行尚未到保存点 |
| 可恢复 checkpoint | 完整评估配置调试第 10 轮；仅供诊断，不是高质量策略，不是正式第 25 轮 |

正式 SHAC：[SwanLab 2f5714lx](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/2f5714lx)。完整配置 10 轮调试：[SwanLab int9z1vu](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/int9z1vu)。中断的 PPO：[SwanLab fi58jbu5](https://swanlab.cn/@zgdllt/g1-walk-shac-vs-ppo/runs/fi58jbu5)。

## 实现边界与版本

- G1 平面 walk 的可微物理 SHAC；没有 AMP discriminator，没有学习到的 world model，也未进行真机验证。
- 保留相同机器人、观测/动作、19 项奖励、PD 时序、随机化、网络结构和初始化的 PPO 对照。PPO 修正了 timeout bootstrap，physical fall 优先。
- SHAC：24 控制步短窗口、重参数化 Gaussian 动作、保留状态导数的冻结 target critic、TD(lambda) critic、target retention 0.4。每控制步 4 个物理步。
- Genesis `v1.4.5`，上游 commit `1b6dfa927f3836d49cf80df7a024116499e68d15`。
- 私有引擎补丁：[genesis-v1.4.5-walk-output-grad.patch](../patches/genesis-v1.4.5-walk-output-grad.patch)，SHA256 `20f168d0cbc59a761d36096ca1986192714be7fb0f3c49b2e2b653575c8eccac`。
- 补丁包括速度/接触力输出 VJP、plane-cylinder 接触几何反传、分块局部 Torch VJP，以及 PD 查询按需省略未使用的 walk 输出。关节位置/速度的 read-node 和梯度保留；观测/奖励读取保持完整输出。
- 完整 solver checkpoint、pending tracked setters 和反传后恢复保留。Torch 缓存分别在快照、反传及恢复前释放。没有部署 CUDA pool trim；该操作只在已归档的诊断脚本中试验，未解决 OOM。
- 原运行：SSH `limh24`，RTX 5070 Ti 16 GB，Python 3.12.14，Torch 2.10.0+cu128，Quadrants 1.3.3，SwanLab 0.10.2。依赖详见 `uv.lock` 和归档的 `config.json`。

原源代码基点是 `bfaed612c25de20944db8fc9265b34525146286b`。归档前比对了服务器的 314 个受版本控制/待提交源文件，本地内容全部一致。只修改了任务私有的 `.deps/genesis`，补丁和 post-apply hash manifest 已归档；无需保留旧 `.venv`、私有引擎副本或编译缓存来恢复源代码。

## 已有验证和仍未解决的问题

- 31 项 CPU 测试通过，相关运行代码 Ruff 检查通过。
- 当前版本 float32 的 `H=1/4/24` × `reward/foot_velocity/contact_force` 共 9 项方向导数检查通过；最大相对误差约 5.54%，重复输入 loss 误差为 0。
- PPO/AD 前向一致性、反传后状态恢复、contact-basis VJP、contact-output VJP 通过。
- `--debug --full-evaluation --num-envs 4096 --iterations 10` 完成；使用正式 128×1000 评估、1000 轮评估间隔和 50000 轮 entropy schedule。
- 这些是受控夹具和有限运行窗口的证据，**没有覆盖所有可能接触状态**。正式第 26 轮的 NaN 根因尚未定位，不能把组件梯度检查当作长期数值稳定性的证明。

错误检测发生在 `RigidSolver.substep_pre_coupling_grad` 的 `kernel_begin_backward_substep` 检查处；该轮已进入物理反传。下一步应保存可重放的失败窗口，并在反传各阶段定位首个非有限量，核查接触/姿态退化、零种子与非法导数的乘积或梯度爆炸等可能性；目前没有证据把其中某个可能性定为根因。不要用 `nan_to_num`、忽略失败样本或降低策略质量标准来宣称修复。

固定验收仍为成熟的 `4096 × 50000` PPO 参照、独立质量评估和连续三次达标确认。短跑仅用于调试；正式比较必须匹配源代码、引擎、运行时及任务指纹，并分别核算新训练控制步、评估步和总耗时。同一张卡不同时运行两个实验。

## 从 GitHub 恢复

在有 CUDA GPU 的 Linux x86_64 主机上，选择有足够空间、当前用户拥有的源码目录：

```bash
git clone --branch codex/genesis-shac https://github.com/zgdllt/Genesis-Humanoid.git
cd Genesis-Humanoid
uv sync --frozen --python 3.12
.venv/bin/python examples/prepare_shac_engine.py
```

`prepare_shac_engine.py` 会重建 `.deps/genesis`、核实上游 commit 后应用补丁。运行入口会验证补丁/文件 hash，并在导入 Genesis 前选择私有引擎、禁用 zero-copy。不要手工修改已安装的 Genesis 或 uv 的共享缓存。

旧路径 `/home/limh24/workspace/Genesis-Humanoid-shac` 和 `/mnt/big_disk/limh24/genesis-shac-20261023` 只是历史运行位置；旧的 `runtime_environment.sh` 不适用于删除目录后的新位置。可为新运行设置实际可写的数据盘目录，例如先把下列第一行替换成自己的路径：

```bash
export GENESIS_ARTIFACT_ROOT=/your/writable/data-disk/genesis-shac-new
mkdir -p "$GENESIS_ARTIFACT_ROOT/cache/quadrants" "$GENESIS_ARTIFACT_ROOT/cache/genesis" "$GENESIS_ARTIFACT_ROOT/tmp" "$GENESIS_ARTIFACT_ROOT/runs"
export QD_OFFLINE_CACHE_FILE_PATH="$GENESIS_ARTIFACT_ROOT/cache/quadrants"
export GS_CACHE_FILE_PATH="$GENESIS_ARTIFACT_ROOT/cache/genesis"
export TMPDIR="$GENESIS_ARTIFACT_ROOT/tmp"
```

源码目录建议使用支持正常 Unix 所有权/文件模式的文件系统。原数据盘使用固定文件模式，旧机器对精确的私有引擎路径配置了 Git trust 和 `core.filemode=false`；这些机器级设置不在 Git 仓库中，也不是物理模型的一部分。

SwanLab 凭据不在仓库中。配置自己的凭据才能上传云端；仅做诊断时可使用已有入口参数 `--swanlab-mode offline`。

## 恢复后先验证、再定位 NaN

```bash
.venv/bin/python -m pytest -q tests/test_shac_returns.py tests/test_ppo_timeout.py tests/test_walk_quality.py tests/test_walk_evidence.py tests/test_walk_comparison.py tests/test_genesis_scene_config.py tests/test_leggedrobot_randomization.py
.venv/bin/python examples/validate_shac_gradients.py --output logs/shac/restored_gradients.json
.venv/bin/python examples/validate_walk_equivalence.py --output logs/shac/restored_equivalence.json

# 仅用于复现/诊断；保留正式评估规模与 entropy schedule。
.venv/bin/python examples/run_walk_benchmark.py --algorithm shac --debug --full-evaluation \
  --num-envs 4096 --iterations 40 --seed 0 \
  --swanlab-mode offline --output logs/shac/restored_nan_probe
```

CUDA 数值行为不保证逐位复现，因此不保证同一轮再次报错。已归档状态不足以准确重放第 26 轮：没有保存该窗口的完整起始状态、随机数和正式第 25 轮模型。先增加失败窗口采集/分阶段诊断，再考虑正式训练。

**不能直接从正式第 25 轮续训。** 调试 checkpoint 也没有完整环境/RNG 状态；当前 benchmark runner 没有可验证的无缝续训 CLI。`SHAC.load()` 可以加载网络/优化器及 target critic，不能据此宣称恢复了原正式轨迹。

归档的 `gradient_gate.json` 只是历史证据，放在归档目录中，不是新机器的有效启动授权。不要直接复制到活动 `logs/shac/gradient_gate.json` 来跳过重新验证。已知 NaN 未解决前，不应把重新启动长跑当作修复完成。

## 证据与 checkpoint 位置

完整索引：[artifacts/shac/2026-10-24/README.md](../artifacts/shac/2026-10-24/README.md)。每个归档文件的大小和 SHA256 记录在同目录的 `MANIFEST.json`。

关键文件：

- 正式失败状态/堆栈：`remote_runs/shac_direct_seed0_pdcore1/{status.json,error.txt}`。
- 当前补丁的梯度/恢复证据：`remote_runs/repair_20261024/{gradient_pd_core32.json,equivalence_pd_core32.json}`。
- 完整配置调试：`remote_runs/repair_20261024/fullfootprint_pd10/`。
- 可恢复的调试模型：`remote_runs/repair_20261024/fullfootprint_pd10/checkpoint_00010.pt`。
- 调度、资源和失败沿革：`remote_runs/repair_20261024/incident_final.json`，以及各 `.launch.json`、stdout、status/config/evaluation/training 文件。

上面的路径均相对于 `artifacts/shac/2026-10-24/`。旧代码/引擎版本的失败和成功诊断一起保留，必须结合各自的 source/engine/common/runtime 指纹解读，不能混用作当前版本的有效证明。

原 `/home/limh24/workspace/Genesis-Humanoid` 也有一份未提交的 `uv.lock` 改动，已单独保存为 `related_checkout/original_workspace_uv_lock.diff`，其基点同为 `bfaed612...`。这份补充材料没有覆盖当前 SHAC 分支的 lockfile。
