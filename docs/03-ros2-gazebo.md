# Stage 3 · ROS 2 + Gazebo

> ## ✅ 已于 2026-08-14 在本机实跑验证 —— 含完整 pick-and-place
>
> colima (vz, 无 sudo) + `rvc/ros2-gazebo:jazzy`（3.86 GB）。两个里程碑：
>
> 1. **视觉伺服**：IDLE → PERCEIVE → EXECUTE → SUCCEEDED，1.8 秒（`runs/stage3-final/`）
> 2. **完整 pick-and-place**：INIT → APPROACH → DESCEND → GRASP → LIFT →
>    TRANSPORT → LOWER → RELEASE → RETREAT → VERIFY → **DONE，12.2 秒**，
>    DetachableJoint 吸附抓取 + 关节反馈闭环 + 纯像素验收（方块落在蓝垫上
>    偏差 14px）。产物在 `runs/pickplace/`。
>
> 首跑修正记录见文末 —— 每一处预写错误和它的症状都在。

## 为什么本机不行

| 障碍 | 说明 |
|---|---|
| macOS 无官方 ROS 2 二进制 | Jazzy / Humble 只提供 Linux 与 Windows 包；macOS 需源码编译，代价大且脆弱 |
| Gazebo 需要 GL 上下文 | 无头运行要 `gz sim -s`；GUI 要 XQuartz |
| 无容器引擎 | Docker Desktop 需要管理员密码；colima 不需要但要 4.5–5.5 GB 磁盘，当前水位（约 7 GB）装完会低于安全线 |

## 复现命令

```bash
make ros-up        # colima + 构建 + 冒烟（幂等）
# 或手动：
docker compose -f ros2_ws/docker-compose.yml run --rm ros2 \
  ros2 launch rvc_agent tabletop.launch.py
```

磁盘曾是阻塞项，靠清掉 68 GB 的 Xcode DerivedData（纯可再生编译产物）解决。

## 首跑修正记录（2026-08-14）

预写代码在真实环境暴露的三个问题，与修复：

| # | 症状 | 根因与修复 |
|---|---|---|
| 1 | 构建上下文 2.4 GB 塞满磁盘 | 忘写 `.dockerignore`，`.venv`/`external` 全被打包发给 daemon。补上后 69 kB。 |
| 2 | 非交互 shell 里看不到 `rvc_agent` 包 | `.bashrc` 对非交互 shell 提前 return；改为 patch `/ros_entrypoint.sh` source overlay。 |
| 3 | `axis_map` 符号错 | 预猜 `(0,-1,1,0)`；用真实渲染标定出 `u=128+235x, v=128−235y` ⇒ 正确值 `(1,0,0,−1)`。 |
| 4 | **到达即失败**：伺服完美开到方块正上方，方块被夹爪遮挡 → target_lost → FAILED | 俯视相机的自遮挡问题。加"遮挡即到达"判定：目标消失且上一拍误差 < 26px ⇒ SUCCEEDED。与 Stage 1 的"抓着物体时检测不到是正常的"同源。 |
| 5 | **spawn 附着从未松开**：夹爪拖着隐形连接的方块跑（帧间方块与标记锁步移动是铁证） | DetachableJoint 出生即附着；开环发 3 拍 detach 全部损失在 gz-transport 订阅发现窗口。修复：接入 `/gripper/attached_state` 反馈，INIT/GRASP/RELEASE 全部改为"发指令直到关节确认"（等价真实吸盘的真空传感器闭环）。 |
| 6 | 抓取失败恢复后立即 target_lost | 恢复后夹爪仍悬在方块正上方挡住它。RECOVER 改为先退到检查位再重新感知（由 NeverGrasp 单元测试发现，未浪费 Gazebo 时间）。 |
| 7 | 被携带的方块露边导致 grasp_failed 误报 | 携带的方块可在抓取偏移处露出边缘。改用距离规则：只有距标记 >30px 的可见红方块才算"留在原地"。 |

## 三条可行路径

### 路径 A：colima（macOS 上推荐，无 sudo）

```bash
brew install colima docker docker-compose && colima start --cpu 4 --memory 6
```

或 Docker Desktop（需要你的密码，你自己执行）：

```bash
brew install --cask docker
```

然后：

```bash
make ros-build     # 构建 ros:jazzy-ros-base + ros_gz + 本仓库的 rvc 包
make ros-shell     # 进容器
```

容器内：

```bash
ros2 launch rvc_agent tabletop.launch.py            # 默认降级到 visual-servo
ros2 launch rvc_agent tabletop.launch.py backend:=openvla-remote   # 配合 RVC_VLA_URL
```

> Apple Silicon 上 `ros:jazzy-ros-base` 有 arm64 镜像；Gazebo 在容器内跑无头最稳。
> GUI 需要 XQuartz + `xhost +localhost` + `DISPLAY=host.docker.internal:0`。

### 路径 B：原生 Ubuntu 22.04 / 24.04

```bash
# ROS 2 Jazzy (Ubuntu 24.04) —— 需要 sudo，你自己执行
sudo apt install ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-cv-bridge

git clone <this repo> && cd robot-vision-copilot
python3 -m venv .venv && .venv/bin/pip install -e ".[vision]"

cd ros2_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install
source install/setup.bash
ros2 launch rvc_agent tabletop.launch.py
```

### 路径 C：先不装，只看接口

Agent 逻辑本身**不依赖 ROS 2**。`RobotAgent`、`ActionValidator`、`RuleBasedPlanner`、
verifier 都在 Stage 1 里跑通并测过了。`agent_node.py` 只是把它们接到 topic 上 ——
所以你现在就可以读懂状态机，等有 Linux 机器再接线。

## 设计：悬浮夹爪 + 像素闭环

第一版刻意**不上机械臂**。世界里是一个重力关闭、由 `/model/gripper/cmd_vel`
速度指令驱动的"悬浮夹爪"方块 —— 没有 IK、没有 MoveIt，这一阶段学的是
ROS 2 管线和 Agent 回路，不是机械臂运动学。

```
   Gazebo (gz sim, headless)
     │ /camera/image_raw (gz.msgs.Image)          ▲ /model/gripper/cmd_vel (gz.msgs.Twist)
     ▼                                            │
   ros_gz_bridge ──▶ sensor_msgs/Image     geometry_msgs/Twist ◀── ros_gz_bridge
                        │                                │
                        ▼                                │
               ┌──────────────────┐                      │
  /rvc/task ──▶│    AgentNode     │──▶ /rvc/state        │
  (String)     │ PERCEIVE→EXECUTE │──▶ /rvc/overlay（带检测框，rqt_image_view 可看）
               │ →VERIFY→RECOVER  │──▶ /model/gripper/cmd_vel ────┘
               └──────────────────┘──▶ /servo_node/delta_twist_cmds（给未来机械臂预留）
```

**降级链是 `openvla-local → openvla-remote → visual-servo`**，注意最后一级
不是 tabletop 的 mock —— mock 依赖仿真器特权状态，Gazebo 不提供。取而代之的
[`VisualServoPolicy`](../src/rvc/policies/visual_servo.py) 只用像素闭环：
检测红方块和夹爪顶面的绿色标记，用两者的像素误差做 P 控制。这是本仓库第一个
**不需要任何特权状态**、能跑在任意带相机环境上的策略 —— 但它仍然不是 VLA，
仍然全程标注 degraded。控制律的符号约定（图像轴 → 世界轴）是构造函数显式参数
`axis_map`，且有单元测试钉死 —— 这是"夹爪朝反方向跑"这类经典 bug 的多发地。

VERIFY 不再是占位的步数预算：视觉伺服"在死区内连续稳定 N 拍"是真实可观测的
完成信号。目标连续 3 帧检测不到 → `RECOVER`（先停车再重新感知），超过恢复
预算 → `FAILED`。

容器里设 `RVC_VLA_URL` 即可让同一个节点被云 GPU 上的真 OpenVLA 驱动。

## 已交付的文件

```
ros2_ws/
├── Dockerfile                          ros:jazzy-ros-base + ros_gz + cyclonedds + rvc 包
│                                       （无 rviz / 无 torch，预计解包 2.5–3 GB）
├── docker-compose.yml                  ROS_LOCALHOST_ONLY=1，挂载 src/ 与 runs/
└── src/rvc_agent/
    ├── package.xml  setup.py  setup.cfg
    ├── config/agent.yaml               backend / instruction / max_recoveries
    ├── worlds/tabletop.sdf             桌子 + 红方块 + 蓝垫 + 俯视相机 + 悬浮夹爪
    ├── launch/tabletop.launch.py       gz sim -s + 三路 bridge + agent_node
    └── rvc_agent/agent_node.py         状态机接 topic；视觉伺服闭环
```

`rvc_agent/frame_grab.py` 是容器内的"眼睛"：订阅 `/camera/image_raw`，把帧存到宿主机挂载的
`runs/`，并打印 `ColorDetector` 的检测结果 —— 也就是 agent 的 PERCEIVE 看到的东西。
无 GUI 时用它验收。

配套（在主包里，本机已测）：

- [`src/rvc/policies/visual_servo.py`](../src/rvc/policies/visual_servo.py) —— 像素伺服控制律，8 个单元测试
- `ColorDetector` 新增 `gripper_marker`（绿色）检测规格

## Pick-and-place 设计要点

- **九阶段序列器**是纯 Python（[`rvc/agent/pickplace.py`](../src/rvc/agent/pickplace.py)），
  9 个单元测试用玩具运动学仿真驱动；ROS 节点只做管道。
- **感知契约**：物体/目标位置只来自相机；唯一的非视觉输入是夹爪自身高度
  （odometry = 本体感知，等价真实机器人的编码器）和 DetachableJoint 的
  attached/detached 反馈（等价吸盘的真空传感器）。
- **抓取验证在 TRANSPORT 而非 LIFT**：抬升时夹爪还悬在方块原位上方，
  可见性不携带任何信息；移开后，被携带的方块不可见（或只在夹爪旁露边
  <30px），远处出现红方块 = 留在原地 = 抓取失败。
- **最终验收是纯像素判定**：红方块中心距蓝垫中心 <18px 才算 DONE。

## 真实故障注入（2026-08-19 实跑）

tabletop 仿真一直有故障注入，Gazebo 侧此前没有 —— RECOVER 分支在 Gazebo 里只被
"意外的 bug"触发过。现在补上了 [`fault_inject.py`](../ros2_ws/src/rvc_agent/rvc_agent/fault_inject.py)：
它订阅 agent 节点新增的 `/rvc/phase`（latched QoS，晚启动也能拿到当前相位），在指定相位
对**仿真器本身**动手 —— 不是在 agent 代码里翻一个标志位。

| 注入 | 机制 | 结果 |
|---|---|---|
| `suction_loss`（TRANSPORT +1.0 s） | 外部发一次 `/gripper/detach`，DetachableJoint 真的松开、方块掉到桌上 | **40 ms** 内由关节反馈判定 `grasp_failed` → RECOVER → 退开 → 重新接近 → 再抓 → DONE（偏差 8 px），1 次恢复 |
| `occlude`（APPROACH +0.3 s，持续 4 s） | 通过 `gz service /world/tabletop/create` 在相机与方块之间生成一块无碰撞的深色面板，到时移除 | 11 帧检测不到方块 → `target_lost` → RECOVER（退开、**停留观察 1 s**）→ 面板消失后重新接近 → DONE（偏差 10 px），2 次恢复 |

两段回放：`docs/assets/gazebo-fault-suction-loss.gif`、`docs/assets/gazebo-fault-occlusion.gif`。

**注入暴露的设计问题**：第一次 3 秒遮挡就耗尽了全部 3 次恢复预算 —— 恢复循环
（退开→再看→又丢）在 10 Hz 下 0.4 s 一圈，预算按次数计、实际只覆盖约 1 秒。修法是在
RECOVER 到位后**停留 10 拍再看**（`RECOVER_DWELL_TICKS`），让每次尝试覆盖真实时间；
之后 4 秒遮挡只用 2 次。两个小坑：phase 话题最初不是 latched，注入器晚启动 2 s 就
错过了相位；SDF 塞进 protobuf 文本格式必须去掉换行、转义引号。

复现（容器内）：
```bash
python3 /ws/src/rvc_agent/rvc_agent/fault_inject.py --fault suction_loss --at-phase TRANSPORT &
ros2 launch rvc_agent tabletop.launch.py
```

## 还没做的（按建议顺序）

1. **机械臂** —— Panda 或 UR5e 的 URDF/SDF，配 `ros2_control` + MoveIt Servo。
   `agent_node` 已经在发 `TwistStamped`，正好对接 MoveIt Servo。
4. **YOLO 替换颜色检测** —— [`YoloDetector`](../src/rvc/perception/detector.py) 已写好且接口一致，
   镜像里 `pip install ultralytics` 后把 `ColorDetector()` 换掉即可（会 +2 GB 镜像体积）。
5. ~~失败注入~~ —— 已做（见上节）。可再加：随机扰动力、放置后碰翻方块让 VERIFY 失败。

## macOS 上的坑

- Gazebo GUI 在容器里基本不值得折腾，用 `gz sim -s` 无头 + 订阅 `/rvc/overlay` 看图。
- DDS：compose 已设 `ROS_LOCALHOST_ONLY=1` 并装好 cyclonedds（旧版 compose 声明了
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` 却没在镜像里装它 —— 那会让所有节点起不来，已修）。
- x86 镜像在 M3 上走 Rosetta 会非常慢。确认拉的是 arm64。
