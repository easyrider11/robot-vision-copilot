# Robot Vision Copilot

**一个教学级的机器人操作栈：下层是 OpenVLA 兼容的动作模型接入层，上层是与模型无关的 Agent 运行时 —— 状态机、动作安全校验、失败检测与恢复 —— 在三个仿真器（零依赖桌面仿真、LIBERO、Gazebo + ROS 2）里端到端实跑验证。**

[English README](README.md) · [文档](docs/) · [60 秒上手](#60-秒上手) · [什么是真的、什么是降级](#诚实降级什么在真的跑什么没有)

<p align="center">
  <img src="docs/assets/gazebo-pickplace.gif" width="256" alt="Gazebo 抓取放置">
  &nbsp;&nbsp;
  <img src="docs/assets/tabletop-grasp-slip-recovery.gif" width="256" alt="桌面仿真：注入滑落后恢复">
  &nbsp;&nbsp;
  <img src="docs/assets/libero-bc-success.gif" width="128" alt="LIBERO：行为克隆基线回放">
</p>
<p align="center"><sub>左：Gazebo 里纯像素闭环的 pick-and-place（吸附抓取 + 关节反馈确认 + 像素验收，12.2 秒）。中：桌面仿真中途注入滑落 —— agent 检测到、重规划、完成任务。右：LIBERO：本机训出来的行为克隆基线把碗拿起放到盘子上。</sub></p>

---

## 为什么做这个

OpenVLA 这类视觉-语言-动作模型只回答一个问题：*给定这张相机图和这句话，下一步 7 自由度末端增量是什么？* 真实机器人需要的其他一切 —— 任务拆解、拒绝不安全输出、发现抓取失败、决定怎么补救、知道什么时候停 —— **都不是模型的活**。这个仓库就是这条边界的一个完整样例：

```
                 ┌──────────────────────────────────────────────┐
  动作模型层      │  Policy 协议: (图像, 指令) → Action[7]           │
  （可替换）      │  openvla_local · openvla_remote · visual_servo · mock │
                 └───────────────────────┬──────────────────────┘
                                         │  Action [dx dy dz droll dpitch dyaw grip]
                 ┌───────────────────────▼──────────────────────┐
  Agent 运行时    │  ActionValidator（6 道检查 + 末尾不变量）        │
  （模型无关）    │  PERCEIVE → PLAN → EXECUTE → VERIFY → RECOVER  │
                 │  RuleBasedPlanner / LLMPlanner.replan()        │
                 └───────────────────────┬──────────────────────┘
                                         │  同一套接口
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
        TabletopSim                LIBERO (MuJoCo)           Gazebo + ROS 2
        零依赖教学仿真              OpenVLA 官方评测基准         ros_gz_bridge + agent 节点
```

`EXECUTE` 是全系统唯一调用神经网络的地方，而网络只看到一张图和一句话。安全、重试、终止全在它上面，用普通的、有单元测试的 Python 写成。

## 诚实降级：什么在真的跑、什么没有

项目在 **MacBook Air M3（16 GB 统一内存，无 CUDA）** 上搭建并验证。这台机器跑不动 OpenVLA-7B：bf16 权重就 15.08 GB，4-bit 量化的 `bitsandbytes` 只支持 CUDA，CPU 推理 7B VLA 每步数十秒。所以：

| 层 | 现在真正在跑的 | 已建好、等 GPU 的 |
|---|---|---|
| 动作模型 | `ScriptedMockPolicy`（读仿真状态，仅桌面仿真）、`VisualServoPolicy`（**纯像素**，Gazebo 用）与 **`LiberoBCPolicy`** —— 用 LIBERO 的 50 条人类示教在本机行为克隆出来的 ResNet18×2+MLP，真实基准上的真学习策略 —— 出现在哪里都标 `degraded=true`（"不是 VLA"） | `openvla_local.py`（transformers，预检显存/磁盘）与 `openvla_remote.py` + `vla_server.py`（7B 前向放云 GPU、机器人栈留本地）；prompt 模板与 `unnorm_key` 配对已就位 |
| 感知 | `ColorDetector`（RGB 阈值）与一个**微调过的 YOLO11n**（自动标注的合成数据） | — |
| 规划器 | `RuleBasedPlanner`（确定性恢复表）；`LLMPlanner` 代码路径完整（结构化输出、有边界的重规划），仅在存在 Anthropic key 时激活 | — |
| 仿真器 | 桌面仿真、**LIBERO**（已装，24 ms/step，等一个真策略）、**Gazebo Harmonic + ROS 2 Jazzy**（colima 容器） | — |

每一处降级都有标注 —— 终端横幅、`summary.json`、`/health`、`/infer` —— 后端解析器记录*为什么*跳过了每个更高优先级的后端。`--no-degraded` 会直接拒绝运行而不是降级。这里没有任何东西会在不是模型输出的时候被当成模型输出。

## 结果

以下每个数字都能用一条 `make` 命令复现，来自上面那台机器的真实运行。

| 声明 | 数字 | 复现 |
|---|---|---|
| Agent 运行时在受控故障下的表现 —— seeded、可复现的桌面仿真回合（500 回合中 375 回合注入目标丢失 / 抓取失败 / 运输中滑落） | **500 回合 · 100% 成功 · 故障回合 100% 恢复** | `make eval` |
| 安全校验 | **0 个非法动作到达执行器**（17,478 次策略调用）；8.9% 被夹取进限幅（噪声注入运行） | `make eval` |
| 控制环时延（完整 PERCEIVE→VERIFY 一圈） | **p95 0.44 ms**，p99 0.90 ms | `make eval` |
| Gazebo 抓取放置（像素 + 关节反馈） | INIT→…→VERIFY→**DONE，12.2 秒**，放置偏差 14 px | `make ros-up` |
| Gazebo 视觉伺服到达 | SUCCEEDED，1.8 秒 | `make ros-up` |
| Gazebo 真实故障注入（关节真松开 / 真生成遮挡模型） | 搬运中吸附失效 → 关节反馈 **40 ms** 判定 → 再抓 → DONE；4 秒遮挡 → 目标丢失 → 2 次恢复 → DONE | 容器内 `fault_inject.py` |
| **LIBERO 行为克隆基线**（ResNet18×2+MLP，50 条示教，单任务，MPS 上 19 分钟训完 —— 学习策略，**不是 VLA**） | 50 个官方初始状态上成功率 **50 % (25/50)**，推理 7–15 ms/步 | `make bc-data bc-train bc-eval` |
| LIBERO 在 Apple 芯片上的仿真 | 24 ms/step（双 256² 相机，`MUJOCO_GL=cgl`） | `make setup-libero` |
| 学习型检测器（YOLO11n，合成数据，MPS） | 150 帧 held-out 合成测试集上 P 0.996 / R 0.996（颜色阈值：0.967 / 0.963）；~9 ms/帧；40 epochs ≈ 10 分钟 | `make yolo` |
| 测试 | 72 个（依赖 LIBERO/YOLO/BC 的在缺依赖时自动跳过） | `make test` |

评测报告的 `provenance` 字段明确写着：测的是 **Agent 运行时**，不是任何 VLA。

## 60 秒上手

```bash
git clone https://github.com/easyrider11/robot-vision-copilot && cd robot-vision-copilot
make setup          # uv venv + numpy + pillow —— Stage 1 的全部依赖
make play           # 交互 playground
```

playground 里输入指令、注入故障；ASCII 地图和检测结果都是从策略看到的同一张渲染图算出来的：

```
指令> auto                            # 全自动跑一遍状态机
指令> reset
指令> move above the red block        # 手动一步步指挥
指令> descend to the red block
指令> close the gripper on the red block
指令> inject slip                     # 布防一个运输中滑落
指令> auto                            # 看 RECOVER 介入并完成任务
指令> gif                             # 导出刚才的全过程
```

其他入口：

```bash
make demo-libero                      # 脚本化 rollout，逐步解释 + 产物（GIF、JSONL）
make demo-recover                     # 连续演示三种故障注入
make eval EPISODES=200                # 批量指标
make serve                            # FastAPI + Web 面板 http://127.0.0.1:8080（GET /health, POST /infer, POST /episode）
make bc-data bc-train bc-eval         # LIBERO 行为克隆基线：示教 -> MPS 训练 -> 成功率
```

## 阶段

项目按可验证的阶段搭建，每个阶段都有一份文档记录量了什么、什么坏了。

| 阶段 | 内容 | 状态 | 文档 |
|---|---|---|---|
| 0 | 环境审计 —— CPU/GPU/磁盘，什么能跑什么不能 | ✅ | [00](docs/00-environment-audit.md) |
| 1 | 最小演示：7-DoF 动作契约、状态机、校验器、故障注入、产物 | ✅ | [01](docs/01-stage1-demo.md) |
| 1.5 | 真实 LIBERO：安装（6 处兼容修复，各附症状）、适配器、`unnorm_key` 与夹爪符号约定 | ✅ | [05](docs/05-libero.md) |
| 2 | FastAPI 可观察服务 + 自包含 Web 面板 | ✅ | [02](docs/02-service.md) |
| 3 | 容器里的 ROS 2 Jazzy + Gazebo Harmonic：悬浮夹爪、视觉伺服、DetachableJoint 抓取、九阶段 pick-and-place | ✅ | [03](docs/03-ros2-gazebo.md) |
| 3+ | 学习型感知（YOLO11n 微调于自动标注合成数据）、可选 LLM 规划器 | ✅ | [06](docs/06-perception-yolo.md) |
| 3++ | **VLA 之外的路径**：本机训练并评测的 LIBERO 行为克隆基线；Gazebo 真实故障注入；由此暴露的夹爪符号契约 bug | ✅ | [08](docs/08-bc-baseline.md) · [03](docs/03-ros2-gazebo.md) |
| 4 | 真实 OpenVLA 推理 + LIBERO 评测、LoRA 微调 | 📄 已文档化，需要 GPU | [04](docs/04-real-openvla.md) |

## 值得知道的设计决策

- **7-DoF 动作契约** `[dx, dy, dz, droll, dpitch, dyaw, gripper]` 与 OpenVLA / LIBERO / RLDS 一致。夹爪符号转换（OpenVLA `[0,1]` ↔ LIBERO `[-1,1]` 反号）只写在一个函数里 —— 写错它就是经典的"机械臂永远抓不住"。
- **校验器有末尾不变量。** 六道检查（NaN、幅值、速率、夹爪抖动、工作空间…）*加*一个无论前面代码干了什么都成立的最终限幅。它存在是因为校验器自己曾在 LIBERO 上发出过 16 倍超界的修正 —— 工作空间边界写死成了桌面仿真的。现在边界由各环境自己声明，不声明就不检查。
- **抓取验证在 TRANSPORT 而不是 LIFT。** 俯视相机下，夹爪还悬在方块上方时不管抓没抓住方块都被挡着；只有移开之后"我还能在离夹爪很远的地方看见方块"才有意义。同一家族的洞察：*误差已经很小时目标消失* 是到达，不是失败。
- **每条执行器指令都闭环到传感器反馈。** 第一次 Gazebo 运行拖着一个隐形连接的方块跑，因为 fire-and-forget 的 `detach` 消息在 gz-transport 发现窗口里丢了。INIT/GRASP/RELEASE 现在重发直到 `/gripper/attached_state` 确认 —— 等价于吸盘的真空传感器。
- **序列器和伺服是纯 Python，ROS 节点只做管道。** 单元测试里 30 行的玩具运动学仿真，在 Gazebo 启动之前就抓到了两个设计 bug（过早检查抓取；从遮挡目标的位姿重新感知）。
- **LLM 是可选的、被关在盒子里的。** `LLMPlanner` 拆解任务、选择恢复点，但只能选择*已有*的、在失败点或之前的子目标 id；其他一切 —— 编造的动作、跳步、坏 JSON、拒答、超时 —— 都落回确定性表格并记录原因。结构化输出让 JSON 从构造上就合法。
- **契约错过一次，而且没有任何东西抓住它。** 夹爪符号曾写成"0 = 开"—— 与 OpenVLA dataloader 约定相反。每个组件都和*某个*符号自洽，所以测试全绿；LIBERO 的 hdf5 动作逼着把约定落到纸面，才暴露出来。已改为 OpenVLA 约定（1 = 开），并用 `tests/test_contract.py` 把枚举、LIBERO 映射、仿真、mock、校验器钉在有据可查的外部来源上。教训：一个约定写在两处就会漂移 —— 要钉在外部参考上。
- **恢复预算要覆盖时间，不只是次数。** Gazebo 里 3 秒遮挡在约 1 秒内烧光 3 次恢复（10 Hz 下退开→看→丢→退开）。RECOVER 现在停留 1 秒再重新接近；4 秒遮挡只花 2 次。
- **带免费标签的合成数据。** 仿真器知道自己把每个物体画在哪，所以给自己的帧打标签；`make yolo` 渲染数据集、在 Apple MPS 上约十分钟微调 YOLO11n，并通过 agent 用的*同一个* `Detector` 接口评测，与阈值检测器并排对比。

## 仓库结构

```
src/rvc/
  types.py                  Action / Observation / AgentState / FailureKind / 日志记录
  compat.py                 策略 × 环境兼容性检查（拦住无意义组合）
  policies/                 mock · visual_servo · bc_libero（BC 基线）· openvla_local · openvla_remote · registry
  envs/                     tabletop（零依赖）· libero_env + libero_bootstrap · base 协议
  agent/                    state_machine · validators · planner (+ llm_anthropic) · pickplace · verifier
  perception/               detector（颜色 + YOLO 同一接口）· yolo_train（数据 → 训练 → 评测）
  service/                  app.py（FastAPI + 面板）· vla_server.py（GPU 侧推理服务）
  runners/                  audit · demo_libero · eval · play · bc（data/train/eval）
ros2_ws/                    Dockerfile · compose · rvc_agent（SDF 世界、launch、agent_node、frame_grab、fault_inject）
tests/                      72 个测试
docs/                       阶段文档 00–08 + 资产
scripts/                    setup_libero.sh · setup_ros2.sh · smoke_api.sh
```

## 依赖与约束

- Python 3.10–3.13、[`uv`](https://github.com/astral-sh/uv)。基础安装只有 numpy + pillow；更重的都是可选 extra（`api`、`libero`、`vision`、`bc`、`llm`、`vla`）。
- ROS 2 / Gazebo 跑在容器里；macOS 上 `make ros-up` 安装 colima（用户态、无 sudo）并构建 3.9 GB 无头镜像。
- 项目从不用 `sudo`、不碰全局 Python、不在未确认时下载模型权重。

## 接下来

1. 租 GPU 跑真实 OpenVLA：那边 `make serve-vla`，这边 `make demo-libero BACKEND=openvla-remote ENV=libero` —— 代码路径已在，LIBERO 基线已就绪。
2. 有评测基线之后再做 LoRA 微调。
3. 用 Panda 机械臂 + MoveIt Servo 替换悬浮夹爪（节点已经在发 `TwistStamped`）。
4. 多任务 / 语言条件的 BC 扩展到更多 LIBERO 任务（单任务基线已就位），有 GPU 时再与真实 OpenVLA 检查点对比。

## 许可

MIT
