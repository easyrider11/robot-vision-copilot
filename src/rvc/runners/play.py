"""`make play` - interactive tabletop playground.

Type natural-language instructions, watch the agent act, inject faults, and
export a GIF of everything you did. Runs the REAL code paths - the same env,
policy, detector and state machine as `make demo-libero` - just driven by you
instead of the planner.

    指令> auto                          # 全自动跑完整个任务（走规划器+状态机）
    指令> move above the red block      # 手动下达一条子目标指令
    指令> descend to the red block
    指令> close the gripper on the red block
    指令> lift the red block
    指令> move above the blue box
    指令> lower the red block into the blue box
    指令> open the gripper
    指令> inject slip|fail|lost         # 中途注入故障
    指令> look                          # 只看不动：渲染 + 检测器结果
    指令> reset / gif / help / quit

The ASCII map is rendered from the SAME observation image the policy gets -
detections shown are genuinely computed from pixels each time you look.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from rvc.agent.state_machine import AgentConfig, RobotAgent
from rvc.envs.tabletop import X_MAX, X_MIN, Y_MAX, Y_MIN, TabletopSim
from rvc.perception.detector import ColorDetector, draw_overlay
from rvc.policies.mock import ScriptedMockPolicy
from rvc.report import BLU, DIM, GRN, RED, YLW, B, Reporter, rule

REPO_ROOT = Path(__file__).resolve().parents[3]
FAULTS = {"slip": "grasp_slip", "fail": "grasp_fail", "lost": "target_lost"}

HELP = """
  可用指令：
    auto                    全自动跑完任务（规划器拆解 + 状态机 + 恢复）
    <自然语言子目标>         例如下面这些（mock 策略按关键词理解）：
       move above the red block      descend to the red block
       close the gripper on the red block      lift the red block
       move above the blue box       lower the red block into the blue box
       open the gripper
    inject slip|fail|lost   注入故障（滑落 / 抓空 / 目标遮挡）
    look                    只观测不动作：ASCII 地图 + 像素检测结果
    reset [far]             重置（far = 另一个任务布局）
    gif                     把本次会话所有帧导出成 GIF
    help / quit
"""


class Playground:
    def __init__(self, task_id: str = "pick_place_block", seed: int = 0) -> None:
        self.task_id = task_id
        self.seed = seed
        self.env = TabletopSim(task_id=task_id, max_steps=100000, seed=seed)
        self.policy = ScriptedMockPolicy(seed=seed)
        self.detector = ColorDetector()
        self.frames: list[np.ndarray] = []
        self.obs = self.env.reset()
        self._snap("session start")

    # -- rendering -----------------------------------------------------------

    def _snap(self, note: str) -> None:
        dets = self.detector.detect(self.obs.image, ("red_block", "blue_box"))
        self.frames.append(draw_overlay(
            self.obs.image, dets, f"{self.env.t:03d} play", note[:60]
        ))

    def ascii_world(self) -> str:
        """Top-down ASCII map, 46x15, same frame the policy sees."""
        w, h = 46, 15
        grid = [[DIM("·") for _ in range(w)] for _ in range(h)]

        def cell(x: float, y: float) -> tuple[int, int]:
            c = int((x - X_MIN) / (X_MAX - X_MIN) * (w - 1))
            r = int((Y_MAX - y) / (Y_MAX - Y_MIN) * (h - 1))
            return max(0, min(h - 1, r)), max(0, min(w - 1, c))

        p = self.obs.privileged
        # blue box footprint
        bx, by = p["box"]
        half = p["box_half"]
        for dx in (-half, 0, half):
            for dy in (-half, 0, half):
                r, c = cell(bx + dx, by + dy)
                grid[r][c] = BLU("▒")
        # block (unless occluded)
        if not p["occluded"]:
            r, c = cell(*p["block"][:2])
            grid[r][c] = RED(B("■"))
        # occluder band
        if p["occluded"]:
            for c in range(2, w - 2):
                grid[h // 2][c] = DIM("█")
        # gripper
        ee = p["ee"]
        r, c = cell(ee[0], ee[1])
        grid[r][c] = GRN(B("G" if p["grip_closed"] else "U"))

        rows = ["  " + "".join(row) for row in grid]
        return "\n".join(rows)

    def status(self) -> str:
        p = self.obs.privileged
        dets = self.detector.detect(self.obs.image, ("red_block", "blue_box"))
        seen = ", ".join(f"{d.label}@({d.center_world[0]:+.2f},{d.center_world[1]:+.2f})"
                         for d in dets) or RED("什么都没检测到")
        ee = p["ee"]
        return (
            f"  step={self.env.t}  ee=({ee[0]:+.2f},{ee[1]:+.2f},z={ee[2]:.2f})  "
            f"夹爪={'闭' if p['grip_closed'] else '开'}  持有={'✓' if p['holding'] else '✗'}  "
            f"{'★任务完成' if p['success'] else ''}\n"
            f"  {DIM('像素检测:')} {seen}"
        )

    # -- actions -------------------------------------------------------------

    def run_instruction(self, text: str, max_steps: int = 45) -> str:
        if self.env.done:
            return YLW("回合已结束（成功或超时）。输入 reset 重新开始。")
        events, still = [], 0
        for _ in range(max_steps):
            self.obs.instruction = text
            action = self.policy.predict(self.obs)
            self.obs, _reward, done, info = self.env.step(action)
            if info.get("event"):
                events.append(f"step {self.env.t}: {info['event']}")
            self._snap(text)
            if done:
                break
            still = still + 1 if np.all(np.abs(action.vector[:6]) < 1e-6) else 0
            if still >= 2:  # servo settled - instruction finished
                break
        ev = ("  " + YLW(" | ".join(events))) if events else ""
        return f"  执行了 {DIM(text)} ->{ev}"

    def run_auto(self) -> None:
        agent = RobotAgent(
            env=self.env,
            policy=self.policy,
            config=AgentConfig(max_recoveries=3, max_total_steps=self.env.t + 300),
            on_event=Reporter(style="compact"),
            collect_frames=True,
        )
        # RobotAgent resets the env itself; that is the semantics of `auto`.
        agent.run()
        self.frames.extend(f for _, f in agent.trace.frames)
        self.obs = self.env._observe()

    def export_gif(self) -> str:
        if not self.frames:
            return "还没有任何帧。"
        from PIL import Image

        out = REPO_ROOT / "runs" / f"play-{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        ims = [Image.fromarray(f) for f in self.frames]
        path = out / "session.gif"
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0)
        return f"已导出 {len(ims)} 帧 -> {path}"


def main(argv: list[str] | None = None) -> int:
    print(rule("PLAY · 交互式桌面抓取 playground", "═"))
    print(DIM("  mock 策略按关键词理解指令（DEGRADED，不是 VLA）。help 看全部指令。"))
    pg = Playground()
    print()
    print(pg.ascii_world())
    print(pg.status())

    while True:
        try:
            cmd = input(B("\n指令> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd:
            continue
        low = cmd.lower()

        if low in ("quit", "exit", "q"):
            break
        elif low == "help":
            print(HELP)
            continue
        elif low == "auto":
            pg.run_auto()
        elif low.startswith("inject"):
            parts = low.split()
            kind = FAULTS.get(parts[1] if len(parts) > 1 else "")
            if kind is None:
                print(f"  用法: inject {'|'.join(FAULTS)}")
                continue
            print("  " + YLW(pg.env.arm_fault(kind)))
        elif low.startswith("reset"):
            task = "pick_place_block_far" if "far" in low else "pick_place_block"
            pg.env = TabletopSim(task_id=task, max_steps=100000, seed=pg.seed)
            pg.obs = pg.env.reset()
            pg._snap("reset")
            print(f"  已重置（任务: {pg.env.instruction}）")
        elif low == "gif":
            print("  " + pg.export_gif())
            continue
        elif low == "look":
            pg._snap("look")
        else:
            print(pg.run_instruction(cmd))

        print()
        print(pg.ascii_world())
        print(pg.status())

    print("\n" + pg.export_gif())
    print("bye.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
