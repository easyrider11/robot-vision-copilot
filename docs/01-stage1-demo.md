# Stage 1 · 最小演示

```bash
make setup && make demo-libero
```

## 这条命令做了什么

```
PREFLIGHT  探测每个后端 → 打印可用性和失败原因
RESOLVE    选出真正能用的策略 + 环境 → 降级时打红字横幅
ROLLOUT    跑完整个 Agent 状态机 → 逐步在终端解释
ARTIFACTS  写出 frames/ · rollout.gif · actions.jsonl · transitions.jsonl · summary.json
```

## 终端输出格式

`--explain full`（默认）每一步输出四行，正好对应「任务 / 观测 / 预测动作 / 执行结果」：

```
── step 007 │ EXECUTE │ 子目标 2/7 · descend ──────────────────────────────
  观测 obs    : 256×256 RGB · red_block conf 0.99 @ (-0.162,-0.120)
  指令 instr  : "descend to the red block"   ← 交给动作模型的全部语言输入
  动作 act    : x+0.00  y+0.00  z-1.00  rpy[+0.00,+0.00,+0.00]  grip=OPEN    ← no clamp · 0.1ms
  结果 result : ee=(-0.162,-0.120,+0.120) hold=✗ r=0.00 │ 进行中 xy误差=0.004m z=0.120m
```

`--explain compact` 每步一行，适合看整条轨迹。

## 常用参数

```bash
make demo-libero INJECT=grasp_slip          # 注入运输滑落
make demo-libero MODE=e2e                   # 不拆子目标，整句直接给动作模型
make demo-libero BACKEND=mock EXPLAIN=compact
make demo-libero BACKEND=openvla-remote     # 配合 RVC_VLA_URL 指向云 GPU
make demo-real                              # 拒绝降级：拿不到真实 VLA 就退出
```

完整参数：`.venv/bin/rvc-demo --help`

## 动作契约

7 维，和 OpenVLA / LIBERO / RLDS 数据集一致：

```
[dx, dy, dz, droll, dpitch, dyaw, gripper]
 └──── 归一化到 [-1,1] 的末端增量 ────┘   └ 夹爪
```

**夹爪约定是最容易踩的坑。** OpenVLA 输出 `[0,1]`，LIBERO 的 OSC 控制器要 `[-1,+1]` 且符号相反。
转换只写在一个地方 —— [`LiberoEnv._to_libero`](../src/rvc/envs/libero_env.py)，对应 openvla 官方
`experiments/robot/robot_utils.py` 里的 `normalize_gripper_action` + `invert_gripper_action`。
搞错的典型症状是「机械臂在动但永远抓不住」。

## 两层边界在哪

`EXECUTE` 是**唯一**调用动作模型的地方：

```python
policy_obs = replace(obs, instruction=sub.text)   # 一张图 + 一句话，仅此而已
raw_action = self.policy.predict(policy_obs)      # ← 模型
action, ok, note = self.validator.validate(raw_action, ee)   # ← Agent 层接管
```

模型不知道有子目标、不知道有重试预算、不知道什么时候该停。这些全在 Agent 层。

## 安全校验

[`ActionValidator`](../src/rvc/agent/validators.py) 有六道检查。`ok=False` 表示**拒绝**（触发
`UNSAFE_ACTION` 重规划）；`ok=True` 但 `note` 非空表示**已夹取**，可以下发但会被记录。

| # | 检查 | 处理 |
|---|---|---|
| 1 | NaN / Inf | 拒绝 |
| 2 | 单维幅值 > 1.0 | 夹取 |
| 3 | `‖dxyz‖` 超速 | 等比缩放 |
| 4 | 相邻两步变化过大（jerk） | 速率限制 |
| 5 | 夹爪抖动（窗口内翻转 ≥4 次） | 拒绝 |
| 6 | 预测位姿越界 | 夹到工作空间边界 |

神经网络会输出 NaN、会输出 40 倍过大的增量、会每步来回开合夹爪。这些都不该到达执行器。

## 失败注入与恢复

| `--inject` | 物理上发生了什么 | 实测 |
|---|---|---|
| `none` | — | 28 步成功 / 0 次恢复 |
| `target_lost` | 遮挡板真的滑过工作区（第 5–12 步），颜色检测器真的返回 `None` | 40 步成功 / **2 次恢复** |
| `grasp_fail` | 第一次闭合夹爪不咬合 | 38 步成功 / 1 次恢复 |
| `grasp_slip` | 抓取后第 6 步、高度 >0.10 m 时脱手，物体落回桌面 | 45 步成功 / 1 次恢复 |

`target_lost` 是真遮挡不是标志位：`TabletopSim.render()` 画了一块不透明挡板，
`ColorDetector` 在像素上真的找不到红色。这一点有测试守着
（`test_occluder_actually_hides_the_block`，同时断言挡板必须会移开，否则恢复不可能成功）。

`RECOVER` 的动作顺序是：**先让机器人安全**（张爪 + 向上退避，记为 `action_source="recovery"`），
再问规划器从哪个子目标重来。恢复次数用尽 → `FAILED`。

## 产物

```
runs/20260812-230032_tabletop_mock_target_lost/
├── frames/step_0001.png …      叠加了检测框、状态、子目标的观测图
├── rollout.gif                 整条轨迹回放
├── actions.jsonl               一行一个动作
├── transitions.jsonl           一行一次状态跳转
└── summary.json                结果 + 降级原因 + 后端尝试记录 + 校验统计
```

`actions.jsonl` 单行示例：

```json
{"step": 7, "state": "EXECUTE", "subgoal": "descend",
 "instruction": "descend to the red block",
 "action": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
 "action_source": "scripted-mock", "validated": true, "validator_note": "",
 "reward": 0.0, "done": false, "failure": "none",
 "ee_xyz": [-0.1617, -0.1195, 0.1203], "holding": false,
 "latency_ms": 0.11, "frame": "frames/step_0007.png"}
```

`action_source` 区分模型动作和 Agent 恢复动作，别把恢复动作当成模型行为来分析。

## 内置 tabletop 仿真是什么

[`TabletopSim`](../src/rvc/envs/tabletop.py) 是一个零依赖（numpy + Pillow）的桌面抓取仿真：
俯视正交相机、7-DoF 增量动作、简化接触模型、真实渲染的 256×256 RGB。

它**不是 LIBERO**，`degraded=True` 永远为真。它存在的理由是：LIBERO 需要 robosuite + MuJoCo +
离屏 GL，OpenVLA 需要 15 GB 权重和 CUDA。在两者都没有的机器上，诚实的做法不是假装跑了一次
OpenVLA rollout，而是提供一个真的能跑、真的渲染图像、真的走完同一条
`Observation → Policy → Action → Env` 回路的小仿真。

装上真 LIBERO（`make setup-libero`）后，`--env auto` 会自动优先用它，Agent 层一行不改。

## 测试

```bash
make test     # 19 个用例
```

守住的性质：动作契约（7 维、有限、夹爪二值）、校验器六道检查、遮挡真的遮挡、
每种注入故障都真的进入 `RECOVER`、恢复预算真的生效、状态机真的走过每个状态、
mock 后端**永远**被标记为降级、`--no-degraded` 真的拒绝静默回退。
