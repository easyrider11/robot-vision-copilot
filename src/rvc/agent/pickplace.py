"""Pick-and-place sequencing for the Stage 3 floating gripper.

Pure Python, no ROS imports - the ROS node feeds observations in and pipes
commands out. That keeps the interesting logic unit-testable on any machine
(see tests/test_pickplace.py, which drives it with a toy kinematic sim).

PHASES
------
    INIT      detach + rise to travel height (DetachableJoint starts ATTACHED
              at sim spawn, so the very first thing we do is let go)
    APPROACH  visual-servo the gripper marker over the red block (xy)
    DESCEND   proprioceptive z-servo down to grasp height
    GRASP     emit `attach`, hold still a few ticks
    LIFT      z up to travel height. Deliberately NO grasp check here: the
              gripper still hovers over the block's spot, so the block is
              occluded whether we hold it or not - zero information.
    TRANSPORT visual-servo toward the blue pad, and THIS is where the grasp
              is verified: a held block rides invisibly under the gripper, so
              seeing the red block at all means it was left behind -> RECOVER.
              (The pad is wider than the gripper, so it stays detectable as a
              ring around the marker.)
    LOWER     z down to place height
    RELEASE   emit `detach`, hold still a few ticks
    RETREAT   servo away to a home corner + z up, so the camera can see the pad
    VERIFY    PIXELS decide: red block detected within PLACE_TOL of the pad
              center -> done. Anything else -> RECOVER (bounded attempts).

SENSING CONTRACT
----------------
Object and goal positions come from the camera only. The single non-visual
input is the gripper's own height `z` from odometry - that is proprioception,
which every real robot has (encoders), not privileged simulator state.

All the constants that encode world geometry are grouped below with the
derivation written out, because "why 0.835?" is exactly the question the next
reader will have.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rvc.perception.detector import Detection
from rvc.policies.visual_servo import VisualServoPolicy

# --- geometry (metres), derived from worlds/tabletop.sdf ---------------------
# table top z=0.75; block 5cm cube -> resting center 0.775, top 0.800
# gripper body 4cm tall -> bottom touches block top at center z 0.820
# pad top 0.760; block resting on pad -> center 0.785
# attach happens at gripper z~0.825 -> rigid offset to block center ~0.047
Z_TRAVEL = 0.95  # cruise height; carried block bottom clears the pad by ~0.1
Z_GRASP = 0.825  # 5 mm above touch - DetachableJoint needs no contact
Z_PLACE = 0.835  # carried block bottom ~0.763, i.e. just onto the pad (0.760)
Z_TOL = 0.008

# --- pixel-space thresholds (256x256 overhead camera) ------------------------
ARRIVE_PX = 26.0  # target vanished closer than this => occluded by us => arrived
TRANSPORT_TOL_PX = 8.0  # xy error to call the pad reached
PLACE_TOL_PX = 18.0  # |block - pad| after release => success
HOME_PX = (90.0, 200.0)  # retreat corner, far from the pad so the view is clear
RETREAT_TOL_PX = 14.0

KZ = 20.0  # z P-gain (normalized action per metre of error)
MISS_LIMIT = 3  # consecutive frames target may be missing before it is "lost"
HOLD_TICKS = 3  # dwell after attach/detach commands
CONFIRM_TIMEOUT = 20  # ticks (2 s) to wait for joint-state confirmation
LEFT_BEHIND_PX = 30.0  # visible target farther than this from the marker => not carried
TRANSPORT_SETTLE = 4  # consecutive in-tolerance ticks to call transport done
VERIFY_TICKS = 6  # frames to wait for the dropped block to be (re)detected


@dataclass
class PickInput:
    target: Detection | None  # red block (vision)
    pad: Detection | None  # blue pad (vision)
    marker: Detection | None  # gripper marker (vision)
    z: float | None  # gripper height (proprioception / odometry)
    # DetachableJoint feedback: "attached" / "detached" / None (no message yet).
    # This is the suction gripper's own sensor - open-loop commanding proved
    # unreliable on the first live run (the fire-and-forget detach was lost in
    # gz-transport subscription discovery and the spawn attachment survived).
    joint_state: str | None = None


@dataclass
class PickCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    gripper_event: str | None = None  # "attach" | "detach" | None
    phase: str = ""
    note: str = ""
    done: bool = False
    failed: bool = False
    failure: str = ""


def _fake_det(u: float, v: float, label: str = "goal") -> Detection:
    return Detection(
        label=label, confidence=1.0,
        bbox_px=(int(u) - 2, int(v) - 2, int(u) + 2, int(v) + 2),
        center_px=(u, v), center_world=(0.0, 0.0), area_px=16,
    )


@dataclass
class PickPlaceSequencer:
    servo: VisualServoPolicy
    max_recoveries: int = 3

    phase: str = "INIT"
    recoveries: int = 0
    last_err_px: float = field(default=float("inf"))
    _misses: int = 0
    _hold: int = 0
    _settle: int = 0
    _verify_wait: int = 0

    # -- helpers --------------------------------------------------------------

    def _z_cmd(self, z: float | None, z_target: float) -> tuple[float, bool]:
        if z is None:
            return 0.0, False  # no odometry yet - do not guess
        err = z_target - z
        if abs(err) < Z_TOL:
            return 0.0, True
        return float(np.clip(err * KZ, -1.0, 1.0)), False

    def _xy_to(self, det: Detection | None, marker: Detection | None) -> tuple[float, float]:
        a = self.servo.compute(det, marker)
        return float(a.vector[0]), float(a.vector[1])

    def _goto(self, phase: str, cmd: PickCommand, note: str) -> PickCommand:
        cmd.note = f"{self.phase} -> {phase}: {note}"
        cmd.phase = phase  # report the NEW phase on the transition tick
        self.phase = phase
        self._misses = 0
        self._hold = 0
        self._settle = 0
        self._verify_wait = 0
        return cmd

    def _recover(self, cmd: PickCommand, why: str) -> PickCommand:
        cmd.gripper_event = "detach"  # always make-safe first
        if self.recoveries >= self.max_recoveries:
            cmd.failed = True
            cmd.failure = why
            return self._goto("FAILED", cmd, f"恢复预算用尽 ({why})")
        self.recoveries += 1
        self.last_err_px = float("inf")
        return self._goto("RECOVER", cmd, f"第 {self.recoveries} 次恢复: {why}")

    # -- main -----------------------------------------------------------------

    def step(self, obs: PickInput) -> PickCommand:
        cmd = PickCommand(phase=self.phase)

        if self.phase in ("DONE", "FAILED"):
            cmd.done = self.phase == "DONE"
            cmd.failed = self.phase == "FAILED"
            return cmd

        if self.phase == "INIT":
            # DetachableJoint spawns attached. Keep commanding detach EVERY
            # tick until the joint itself confirms - fire-and-forget lost the
            # message during transport discovery on the first live run.
            cmd.gripper_event = "detach"
            cmd.vz, at_z = self._z_cmd(obs.z, Z_TRAVEL)
            self._hold += 1
            confirmed = obs.joint_state == "detached"
            if at_z and (confirmed or self._hold >= CONFIRM_TIMEOUT):
                note = "已确认释放" if confirmed else "无关节反馈，超时继续(查 attached_state 桥)"
                return self._goto("APPROACH", cmd, f"{note}，巡航高度就位")
            return cmd

        if self.phase == "RECOVER":
            # Back off to the home corner, not just upward: after a failed
            # grasp the gripper hovers exactly over the block, occluding it -
            # re-perceiving from the same pose can only report target_lost.
            # (Found via the NeverGrasp unit test, not in Gazebo.)
            cmd.vz, at_z = self._z_cmd(obs.z, Z_TRAVEL)
            cmd.vx, cmd.vy = self._xy_to(_fake_det(*HOME_PX), obs.marker)
            eu, ev = self.servo.status.error_px
            clear = float(np.hypot(eu, ev)) < RETREAT_TOL_PX
            if at_z and clear:
                return self._goto("APPROACH", cmd, "已退到检查位，重新感知")
            return cmd

        if self.phase == "APPROACH":
            cmd.vz, _ = self._z_cmd(obs.z, Z_TRAVEL)
            if obs.target is None:
                if self.last_err_px < ARRIVE_PX:
                    return self._goto("DESCEND", cmd,
                                      f"目标被遮挡即到达 (最后误差 {self.last_err_px:.0f}px)")
                self._misses += 1
                if self._misses >= MISS_LIMIT:
                    return self._recover(cmd, "target_lost")
                return cmd
            self._misses = 0
            cmd.vx, cmd.vy = self._xy_to(obs.target, obs.marker)
            if self.servo.status.marker_found:
                eu, ev = self.servo.status.error_px
                self.last_err_px = float(np.hypot(eu, ev))
            if self.servo.status.settled:
                return self._goto("DESCEND", cmd, "已在目标正上方")
            return cmd

        if self.phase == "DESCEND":
            cmd.vz, at_z = self._z_cmd(obs.z, Z_GRASP)
            if at_z:
                cmd.gripper_event = "attach"
                return self._goto("GRASP", cmd, "到抓取高度，闭合(吸附)")
            return cmd

        if self.phase == "GRASP":
            self._hold += 1
            if obs.joint_state == "attached":
                return self._goto("LIFT", cmd, "关节反馈已吸附，抬升")
            if self._hold >= CONFIRM_TIMEOUT:
                return self._recover(cmd, "grasp_failed (no attach confirmation)")
            cmd.gripper_event = "attach"  # keep commanding until confirmed
            return cmd

        if self.phase == "LIFT":
            # NOTE deliberately NO grasp check here: the gripper is still
            # hovering over the block's original spot, so the block is occluded
            # whether we grasped it or not - visibility carries zero
            # information at this pose. (The toy-sim tests caught this: an
            # earlier version checked here and waved a failed grasp through.)
            cmd.vz, at_z = self._z_cmd(obs.z, Z_TRAVEL)
            if at_z:
                return self._goto("TRANSPORT", cmd, "已抬升，移动中验证抓取")
            return cmd

        if self.phase == "TRANSPORT":
            cmd.vz, _ = self._z_cmd(obs.z, Z_TRAVEL)
            # Grasp verification, two independent signals:
            #  1. the joint itself reports we lost it;
            #  2. the red block is visible FAR from the marker - sitting back
            #     on the table, not merely peeking out from under the gripper
            #     (a carried block can show a sliver at the grasp offset, so
            #     nearby visibility alone must NOT count as failure).
            if obs.joint_state == "detached":
                return self._recover(cmd, "grasp_failed (joint reports detached)")
            if obs.target is not None and obs.marker is not None:
                d = float(np.hypot(
                    obs.target.center_px[0] - obs.marker.center_px[0],
                    obs.target.center_px[1] - obs.marker.center_px[1],
                ))
                if d > LEFT_BEHIND_PX:
                    return self._recover(cmd, f"grasp_failed (block left behind, {d:.0f}px away)")
            if obs.pad is None:
                self._misses += 1
                if self._misses >= MISS_LIMIT:
                    return self._recover(cmd, "pad_lost")
                return cmd
            self._misses = 0
            cmd.vx, cmd.vy = self._xy_to(obs.pad, obs.marker)
            eu, ev = self.servo.status.error_px
            if float(np.hypot(eu, ev)) < TRANSPORT_TOL_PX:
                self._settle += 1
                if self._settle >= TRANSPORT_SETTLE:
                    return self._goto("LOWER", cmd, "已在蓝垫正上方")
            else:
                self._settle = 0
            return cmd

        if self.phase == "LOWER":
            cmd.vz, at_z = self._z_cmd(obs.z, Z_PLACE)
            if at_z:
                cmd.gripper_event = "detach"
                return self._goto("RELEASE", cmd, "到放置高度，松开")
            return cmd

        if self.phase == "RELEASE":
            self._hold += 1
            if obs.joint_state == "detached":
                return self._goto("RETREAT", cmd, "关节反馈已松开，退开让相机验收")
            if self._hold >= CONFIRM_TIMEOUT:
                # proceed anyway - VERIFY will catch a still-carried block
                return self._goto("RETREAT", cmd, "无松开确认，靠 VERIFY 兜底")
            cmd.gripper_event = "detach"
            return cmd

        if self.phase == "RETREAT":
            cmd.vz, _ = self._z_cmd(obs.z, Z_TRAVEL)
            home = _fake_det(*HOME_PX)
            cmd.vx, cmd.vy = self._xy_to(home, obs.marker)
            eu, ev = self.servo.status.error_px
            if float(np.hypot(eu, ev)) < RETREAT_TOL_PX:
                return self._goto("VERIFY", cmd, "已退到检查位")
            return cmd

        if self.phase == "VERIFY":
            # Pixels decide. No simulator state, no wishful thinking.
            self._verify_wait += 1
            if obs.target is not None and obs.pad is not None:
                du = obs.target.center_px[0] - obs.pad.center_px[0]
                dv = obs.target.center_px[1] - obs.pad.center_px[1]
                dist = float(np.hypot(du, dv))
                if dist < PLACE_TOL_PX:
                    cmd.done = True
                    return self._goto("DONE", cmd, f"方块在蓝垫上 (偏差 {dist:.0f}px)")
                return self._recover(cmd, f"place_missed ({dist:.0f}px)")
            if self._verify_wait >= VERIFY_TICKS:
                return self._recover(cmd, "block_not_found_after_release")
            return cmd

        return cmd  # pragma: no cover
