"""Fault injection for the Gazebo pick-and-place - the Stage 3 counterpart of
`TabletopSim(inject=...)`. Runs INSIDE the container next to the launch file.

    python3 fault_inject.py --fault suction_loss --at-phase TRANSPORT
    python3 fault_inject.py --fault occlude      --at-phase APPROACH --hold 3.0

suction_loss  waits for the sequencer to reach --at-phase, then publishes ONE
              /gripper/detach - the DetachableJoint releases, the block drops.
              The sequencer must notice (joint reports "detached" / block
              visible far from the marker) -> RECOVER -> re-grasp -> DONE.
occlude       spawns a collision-free dark panel above the block for --hold s
              (gz EntityFactory service), then removes it. The block vanishes
              from the camera -> target_lost -> RECOVER (retreat, re-perceive)
              -> DONE once the panel is gone.

Both faults are REAL inside the simulator (a joint really releases, a model is
really spawned in front of the camera) - nothing is flagged in agent code.
"""

from __future__ import annotations

import argparse
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

WORLD = "tabletop"
OCCLUDER_SDF = """<?xml version="1.0"?>
<sdf version="1.10"><model name="occluder"><static>true</static>
<pose>{x} {y} {z} 0 0 0</pose>
<link name="l"><visual name="v"><geometry><box><size>0.22 0.22 0.01</size></box></geometry>
<material><ambient>0.2 0.2 0.22 1</ambient><diffuse>0.2 0.2 0.22 1</diffuse></material>
</visual></link></model></sdf>"""


def gz_service(service: str, reqtype: str, req: str, reptype: str = "gz.msgs.Boolean") -> str:
    out = subprocess.run(
        ["gz", "service", "-s", service, "--reqtype", reqtype, "--reptype", reptype,
         "--timeout", "3000", "--req", req],
        capture_output=True, text=True, timeout=10,
    )
    return (out.stdout + out.stderr).strip()


class FaultInjector(Node):
    def __init__(self, args) -> None:
        super().__init__("rvc_fault_injector")
        self.args = args
        self.phase = ""
        self.fired = False
        self.t_armed = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/rvc/phase", self._on_phase, latched)
        self.pub_detach = self.create_publisher(Empty, "/gripper/detach", 10)
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f"armed: {args.fault} at phase {args.at_phase} (+{args.delay}s)")

    def _on_phase(self, msg: String) -> None:
        if msg.data != self.phase:
            self.phase = msg.data
            if self.phase == self.args.at_phase and not self.fired:
                self.t_armed = time.monotonic()

    def _tick(self) -> None:
        if self.fired or self.t_armed is None:
            return
        if time.monotonic() - self.t_armed < self.args.delay:
            return
        self.fired = True
        if self.args.fault == "suction_loss":
            self.pub_detach.publish(Empty())
            self.get_logger().warn("INJECTED suction_loss: published /gripper/detach")
        elif self.args.fault == "occlude":
            x, y, z = self.args.occluder_pose
            # protobuf text format: one line, quotes escaped
            sdf = " ".join(OCCLUDER_SDF.format(x=x, y=y, z=z).split()).replace('"', '\\"')
            r = gz_service(f"/world/{WORLD}/create", "gz.msgs.EntityFactory",
                           f'sdf: "{sdf}" name: "occluder"')
            self.get_logger().warn(f"INJECTED occlude: spawned panel ({r})")
            self._remove_at = time.monotonic() + self.args.hold
            self.create_timer(0.2, self._maybe_remove)

    def _maybe_remove(self) -> None:
        if time.monotonic() < self._remove_at:
            return
        r = gz_service(f"/world/{WORLD}/remove", "gz.msgs.Entity",
                       'name: "occluder" type: MODEL')
        self.get_logger().warn(f"occluder removed ({r})")
        self._remove_at = float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=["suction_loss", "occlude"], required=True)
    ap.add_argument("--at-phase", default="TRANSPORT")
    ap.add_argument("--delay", type=float, default=0.8, help="seconds after entering the phase")
    ap.add_argument("--hold", type=float, default=3.0, help="occluder lifetime (s)")
    ap.add_argument("--occluder-pose", type=float, nargs=3, default=(-0.16, -0.12, 0.86))
    args = ap.parse_args()
    rclpy.init()
    node = FaultInjector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
