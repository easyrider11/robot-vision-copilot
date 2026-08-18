# Stage 2 · FastAPI 可观察服务

```bash
make setup-api && make serve        # http://127.0.0.1:8080
make smoke PORT=8080                # 12 项接口冒烟
```

> **关于接口契约**：你原始需求里 `POST /infer` 的返回结构说明被截断了（停在「返回结构」）。
> 下面这套是我按「每个响应都必须自带出处」的原则设计的。字段名要改的话告诉我。

自动生成的交互式文档在 `/docs`（Swagger）和 `/redoc`。

## 设计原则

**凡是经过模型的响应，都必须带出处。** 调用方只看 payload、不看日志，就应该能判断这些数字
是 OpenVLA-7B 算出来的，还是脚本兜底算出来的。所以每个响应里都有 `backend.degraded` 和
`warnings`。

服务本身**不含任何机器人逻辑** —— 它复用 Stage 1 的同一批对象。这是刻意的：HTTP 层不可能和
`make demo-libero` 的行为发生漂移。

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 主机能力 + 后端状态 + 每个后端为什么不可用 |
| `POST` | `/infer` | 图像 + 语言任务 → 经安全校验的 7-DoF 动作 |
| `POST` | `/episode` | 跑一整条 Agent rollout，返回摘要和状态机时间线 |
| `GET` | `/runs` | 历史运行列表 |
| `GET` | `/runs/{id}` | 单次运行的摘要 + 逐步日志 + 帧列表 |
| `GET` | `/runs/{id}/frames/{name}` | 单帧 PNG 或 `rollout.gif` |
| `GET` | `/runs/{id}/actions.jsonl` | 原始动作日志 |
| `GET` | `/` | 自包含 Web 面板（无 CDN、无构建步骤） |

### `GET /health`

```json
{
  "status": "degraded",
  "model_loaded": false,
  "backend": {
    "name": "scripted-mock", "kind": "mock", "degraded": true,
    "degraded_reason": "OpenVLA-7B cannot run on this host …",
    "attempts": [
      ["openvla-remote", "no RVC_VLA_URL set - skipped"],
      ["openvla-local",  "torch not installed (ModuleNotFoundError). Run: uv pip install -e '.[vla]'"],
      ["mock", "OK"]
    ]
  },
  "host": {"os":"15.5","cpu":"Apple M3","ram_gb":16.0,"disk_free_gb":28.2,
           "cuda":false,"mps":false,"vram_gb":null},
  "capabilities": {
    "openvla_local": {"available": false, "reason": "torch not installed …"},
    "libero":        {"available": false, "reason": "LIBERO/robosuite not importable …"},
    "tabletop_sim":  {"available": true,  "reason": "built-in, always degraded"},
    "ros2_gazebo":   {"available": false, "reason": "Stage 3; needs ROS 2 natively or Docker"}
  },
  "warnings": ["DEGRADED: 该动作不是 OpenVLA-7B 推理结果，不能作为模型性能证据。", "…"]
}
```

`model_loaded` 是给监控用的单一布尔量：**只有真实 VLA 才为 `true`**。

### `POST /infer`

请求：

```jsonc
{
  "instruction": "move above the red block",   // 必填
  "image_b64": "iVBORw0KG…",                   // 可选；留空则用内置 tabletop 初始观测
  "unnorm_key": "bridge_orig",                 // 可选；覆盖 OpenVLA 反归一化 key
  "validate_action": true                      // 可选；默认走安全校验
}
```

```bash
curl -s -X POST localhost:8080/infer \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"move above the red block"}' | jq
```

响应五个部分：

```jsonc
{
  "request_id": "4b99bc04b2f3",
  "instruction": "move above the red block",

  // ① 动作本体
  "action": {
    "vector": [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    "labels": ["dx","dy","dz","droll","dpitch","dyaw","gripper"],
    "delta_xyz": [-1.0, -1.0, -1.0], "delta_rpy": [0.0, 0.0, 0.0],
    "gripper": 0.0, "gripper_label": "OPEN"
  },

  // ② 安全校验：ok=false 表示这个动作不该下发给机器人
  "validation": { "ok": true, "note": "", "clamped": false,
                  "raw_vector": [-1.0,-1.0,-1.0,0,0,0,0] },   // 校验前的模型原始输出

  // ③ 感知：这一帧上真实的检测结果
  "perception": {
    "detector": "color-threshold", "target_found": true,
    "detections": [
      {"label":"red_block","confidence":0.99,"bbox_px":[46,166,72,192],
       "center_world":[-0.1617,-0.1195]},
      {"label":"blue_box","confidence":0.99,"bbox_px":[168,44,215,91],
       "center_world":[0.1488,0.1418]}
    ]
  },

  // ④ 出处：这个动作到底是谁算的
  "backend": { "name":"scripted-mock", "kind":"mock", "degraded":true,
               "degraded_reason":"…", "attempts":[…] },

  // ⑤ 时延与告警
  "latency_ms": {"policy": 0.12, "total": 6.46},
  "warnings": ["DEGRADED: 该动作不是 OpenVLA-7B 推理结果，不能作为模型性能证据。", "…"],
  "image_size": [256, 256]
}
```

`validation.raw_vector` 保留校验前的模型原始输出，方便对比「模型想做什么」和「实际下发了什么」。

错误：非法 base64 或非图像 → `422`。

### `POST /episode`

```bash
curl -s -X POST localhost:8080/episode -H 'Content-Type: application/json' \
  -d '{"inject":"grasp_slip","backend":"auto","mode":"subgoal","max_steps":200}' | jq
```

返回结果、步数、恢复次数、降级标记、产物链接，以及**完整状态机时间线**：

```json
{
  "run_id": "20260812-230032_tabletop_mock_target_lost",
  "success": true, "steps": 40, "recoveries": 2,
  "final_state": "SUCCEEDED", "failure": "none",
  "degraded": true, "degraded_reason": "…", "injected_fault": "target_lost",
  "state_timeline": [
    {"step": 5, "from": "PERCEIVE", "to": "RECOVER",  "reason": "检测不到 red_block"},
    {"step": 9, "from": "RECOVER",  "to": "PERCEIVE", "reason": "第 1 次恢复：目标丢失 -> 退回第 0 步重新定位"}
  ],
  "artifacts": {"gif": "/runs/…/frames/rollout.gif", "actions_jsonl": "/runs/…/actions.jsonl"}
}
```

**注意**：当前是同步执行。mock 后端一条 rollout ~0.1 s，无所谓；换成真实 OpenVLA
（每步数百毫秒 × 200 步）时应改成后台任务 + 轮询 `run_id`。这一点还没做。

## Web 面板

`GET /` 是一个自包含单页（无 CDN、无构建步骤、跟随系统深浅色）：

- 顶部横幅：降级时黄色警告，真实 VLA 时绿色
- 主机与后端能力卡片
- 一键运行 episode（可选后端 / 注入故障 / 模式 / 步数）
- rollout GIF 回放
- 状态机时间线表
- 动作日志表（step / state / subgoal / dx dy dz / grip / hold / valid / ms / instruction）
- 历史运行列表，可点开任意一次

## 远程 OpenVLA 服务

[`rvc.service.vla_server`](../src/rvc/service/vla_server.py) 是另一个进程，**跑在 GPU 机器上**：

```
MacBook（本仓库）                       云 GPU（A100 / L40S / 4090）
─────────────────                       ──────────────────────────
TabletopSim / LIBERO                    rvc.service.vla_server
Agent 状态机           ──HTTP──▶         OpenVLALocalPolicy
校验 / 恢复 / 日志     ◀─action──        openvla/openvla-7b
```

只有 7B 前向搬走，整个机器人栈留在笔记本。每个控制步一次图像往返，所以延迟是主要瓶颈 ——
客户端每次调用都会记录 `latency_ms`。详见 [04](04-real-openvla.md)。

## 安全

- 路径穿越在两处被挡：路由正则 `[^/]+` 不匹配编码斜杠，以及 `_safe_run_dir` 的显式前缀检查。
  冒烟测试里有对应用例。
- `vla_server` **没有鉴权**，这是刻意的（它是教学产物）。放到不可信网络前请绑
  `127.0.0.1` 并走 SSH 隧道。
