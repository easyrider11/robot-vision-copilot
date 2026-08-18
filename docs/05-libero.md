# 05 · LIBERO

**状态：已安装并验证可用**（macOS 15.5 / Apple M3 / arm64，2026-08-13）

```bash
make setup-libero      # 约 2.6 GB，全部装在项目内，无 sudo
make libero-tasks      # 列出任务
```

Stage 0 审计时我把 LIBERO 标为「⚠️ 有条件，可行但脆弱」。**现在可以改成「✅ 已验证」** ——
但这一路踩了 6 个坑，全部记录在下面。

## 实测性能

`libero_spatial` task 0，两路 256×256 离屏相机，MuJoCo 自动选中 `MUJOCO_GL=cgl`：

| 阶段 | 耗时 |
|---|---|
| 建环境 | 2.3 s |
| reset | 0.55 s |
| 步进（稳态） | **24 ms/step** |
| 220 步 rollout | 约 8 s |

**仿真不是瓶颈。** 真实 OpenVLA 单步前向 150–400 ms，比仿真慢一个数量级。

## 磁盘代价

| 项 | 大小 |
|---|---|
| Python wheels（torch, mujoco, robosuite, scipy, opencv, matplotlib, numba…） | ~1.9 GB |
| `external/LIBERO` 仓库浅克隆（含 bddl 与 init_files 资产） | 651 MB |
| **合计** | **约 2.6 GB** |

演示数据集（HDF5，数十 GB）**没有下载**，跑评测不需要它，只有训练才要。

## 安装过程中踩到的 6 个不兼容

全部固化在 [`src/rvc/envs/libero_bootstrap.py`](../src/rvc/envs/libero_bootstrap.py)，
每条都带症状描述。

### 1. `pip install git+…LIBERO` 装出一个空包

```
Successfully installed libero-0.1.0
>>> import libero
ModuleNotFoundError: No module named 'libero'
```

`external/LIBERO/libero/` 没有顶层 `__init__.py`（隐式命名空间包），而他们的 `setup.py`
用的是 `find_packages()` —— 它只认有 `__init__.py` 的目录，于是打包出来只有 dist-info。

**解法**：克隆仓库到 `external/LIBERO`，把它加到 `sys.path`。资产文件本来也必须留在磁盘上。

### 2. 首次导入会在 stdin 上交互提问

```
Do you want to specify a custom path for the dataset folder? (Y/N):
EOFError: EOF when reading a line
```

**解法**：预先写好 `config.yaml`，并把 `LIBERO_CONFIG_PATH` 指到项目内的
`external/.libero-config/`。**你的 `~/.libero` 不会被碰。**

### 3. `libero.libero.benchmark` 在模块层 `import torch`

跑仿真根本用不到 torch，但导入就会失败。**解法**：torch 进 `[libero]` extra。

### 4. `torch.load` 的 `weights_only` 默认值变了

```
UnpicklingError: Weights only load failed …
numpy.core.multiarray._reconstruct was not an allowed global
```

torch ≥2.6 把默认值从 `False` 翻成了 `True`，而 LIBERO 的 `.pruned_init` 是 pickle 的 numpy 数组。

**解法**：`torch_load_compat()` 上下文管理器，**只包住那一次调用**，不做全局 patch ——
全进程静默允许反序列化任意文件正是你不想要的。

### 5. robosuite 1.4.1 vs mujoco ≥3.3

```
AttributeError: 'MjData' object has no attribute 'qM'. Did you mean: 'M'?
```

MuJoCo 3.3.0 把 `MjData.qM` 改名成了 `M`，robosuite 1.4.1（2023 年）还在用旧名。

**解法**：pin `mujoco>=3.2,<3.3`（实测 3.2.7 可用）。`probe()` 会提前检测这个组合并给出明确提示，
而不是让你在 rollout 中途撞上一个莫名其妙的 AttributeError。

### 6. 不要装 LIBERO 的 `requirements.txt`

它 pin 了 `numpy==1.22.4`、`transformers==4.21.1`、`robomimic==0.2.0`、`wandb` ——
**全是给他们的 lifelong learning 训练代码用的**，跑仿真一个都不需要，而且会把 numpy 降级、
破坏项目其余部分。

**解法**：只装 `libero/libero/**` 真正 import 的东西：
`cloudpickle` `gym` `h5py` `huggingface-hub` `matplotlib` `easydict` `bddl` `pyyaml`。

## 一个必须诚实面对的问题：mock 策略跑不了 LIBERO

脚本 mock 策略伺服的是 TabletopSim 的特权状态（物体坐标、盒子坐标）。**LIBERO 不提供这些** ——
它只给 reward 作为成功信号。所以这个组合必然 0% 成功。

处理方式有三层：

1. **`--env auto` 不会选它**。当策略是 mock 时，auto 直接跳过 LIBERO 并说明原因：
   ```
   [SKIP] libero: 跳过：mock 策略驱动不了 LIBERO（要跑请显式 --env libero）
   [OK  ] tabletop: OK
   ```
2. **显式 `--env libero` 仍然被尊重**，但会打出一整块 ⛔ 警告，说明这只是管线验证、
   不是 LIBERO 评测、成功率不代表任何模型能力。见 [`src/rvc/compat.py`](../src/rvc/compat.py)。
3. **mock 策略自己会拒绝**：`can_drive(obs)` 检查它需要的特权键，缺了就返回零动作而不是崩溃。

```bash
make demo-libero ENV=libero STEPS=60     # 管线验证：60 步 2.0 秒，超时失败（符合预期）
```

这次运行验证的是：环境创建、双相机离屏渲染、7-DoF 动作转换、夹爪约定、日志、
帧与 GIF 产出、状态机 —— 全部正常。

## 这次集成暴露的一个真实 bug

LIBERO 跑起来后，动作日志里出现了 `dz = -15.95` —— **安全校验层自己发出了超范围 16 倍的指令**。

根因：`SafetyLimits` 把 TabletopSim 的工作空间（z 上限 0.35 m）写死成了默认值。
LIBERO 的 Panda 末端在 z ≈ 0.91 m，永远在盒子外面，于是每一步都被要求
「回到 0.35」= `(0.35 - 0.91) / 0.035 = -16`。而这个修正值写在幅值夹取之后，没有再次夹取。

修法两条，都有回归测试：

- 工作空间边界改为**由环境声明**（`env.workspace_bounds`），不声明就不检查。
  猜一个边界比不检查更糟 —— 错的盒子会让每一步都变成朝着虚构边界的全速修正。
  LIBERO 声明 `None`，并注明原因：我没有实测过 Panda 的可达空间，
  真正的边界是 OSC 控制器和关节限位。
- 校验函数末尾加了**最终不变量**：无论前面哪一步做了什么，输出都不会超出幅值上限。

```python
def test_workspace_clip_never_exceeds_magnitude_ceiling():
    v = ActionValidator(TABLETOP_LIMITS)
    ee = np.array([0.0, 0.0, 0.91])          # LIBERO 的高度
    a, ok, note = v.validate(Action.zeros(), ee)
    assert np.all(np.abs(a.vector[:6]) <= 1.0), f"validator emitted {a.vector}"
```

这正是「先把管线接到真实环境上」的价值：内置 tabletop 仿真永远碰不到这个 bug。

## 跑一次真正的 LIBERO 评测

需要真实 OpenVLA。检查点和 `unnorm_key` 必须配对：

```bash
# GPU 机器上
RVC_MODEL_ID=openvla/openvla-7b-finetuned-libero-spatial \
RVC_UNNORM_KEY=libero_spatial \
  uvicorn rvc.service.vla_server:app --host 0.0.0.0 --port 8000

# 笔记本上（走 SSH 隧道）
ssh -N -L 8000:localhost:8000 user@gpu-host &
make demo-libero ENV=libero BACKEND=openvla-remote SUITE=libero_spatial TASKIDX=0 STEPS=220
```

用 base 检查点 + `bridge_orig` 跑 LIBERO 是常见的错误对照，`rvc.compat` 会提示。
完整说明见 [04 · 跑真实 OpenVLA](04-real-openvla.md)。

## 任务套件

| 套件 | 任务数 | 侧重 |
|---|---|---|
| `libero_spatial` | 10 | 空间关系（"在盘子和 ramekin 之间的那个碗"） |
| `libero_object` | 10 | 物体识别 |
| `libero_goal` | 10 | 目标条件 |
| `libero_10` | 10 | 长时程任务 |
| `libero_90` | 90 | 大规模预训练集 |

```bash
make libero-tasks SUITE=libero_object
make demo-libero ENV=libero SUITE=libero_goal TASKIDX=3
```

## 已知噪音（无害）

- `[robosuite WARNING] No private macro file found!` —— robosuite 建议建一个私有 macro 文件，
  不影响运行。
- `Gym has been unmaintained since 2022 and does not support NumPy 2.0` —— LIBERO 只用 gym 的
  基础类型，实测无影响。

## 更新 / 卸载

```bash
git -C external/LIBERO pull                    # 更新仓库
rm -rf external/LIBERO external/.libero-config # 卸载（Python 包留在 venv 里）
```

删掉之后 `--env auto` 会自动回落到内置 tabletop 仿真，其余功能不受影响。
