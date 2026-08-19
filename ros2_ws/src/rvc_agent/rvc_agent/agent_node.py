"""Stage 3 · the agent as a ROS 2 node - full pick-and-place.

VERIFIED IN GAZEBO (see docs/03 for the run log). The interesting logic lives
in `rvc.agent.pickplace.PickPlaceSequencer` (pure Python, unit-tested); this
file is deliberately just plumbing:

    topic                          dir  type                 purpose
    /camera/image_raw              in   sensor_msgs/Image    overhead camera
    /model/gripper/odometry        in   nav_msgs/Odometry    proprioception (z)
    /gripper/attached_state        in   std_msgs/String      DetachableJoint state
    /rvc/task                      in   std_msgs/String      restart the task
    /rvc/state                     out  std_msgs/String      AgentState, on transitions
    /rvc/phase                     out  std_msgs/String      sequencer phase (fault injector hooks here)
    /rvc/overlay                   out  sensor_msgs/Image    annotated frame
    /model/gripper/cmd_vel         out  geometry_msgs/Twist  gripper velocity
    /gripper/attach, /gripper/detach out std_msgs/Empty      suction grasp

POLICY NOTE. The pick-and-place sequence is driven by the visual-servo
policy - still NOT a VLA, still flagged degraded. When a real OpenVLA backend
is attached (RVC_VLA_URL), set `backend:=openvla-remote` and `mode:=e2e`: the
sequencer is bypassed and the model's 7-DoF actions drive cmd_vel directly.
That path exists but is untested until a GPU server is available.
"""

from __future__ import annotations

import numpy as np

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Empty, String
except ImportError as exc:  # pragma: no cover - only importable inside ROS 2
    raise SystemExit(
        f"ROS 2 (rclpy) not available: {exc}\n"
        "Run this inside the Stage 3 container:\n"
        "  docker compose -f ros2_ws/docker-compose.yml run --rm ros2 bash"
    ) from exc

from rvc.agent.pickplace import PickInput, PickPlaceSequencer
from rvc.perception.detector import ColorDetector, draw_overlay
from rvc.policies.visual_servo import VisualServoPolicy
from rvc.types import AgentState

CONTROL_HZ = 10.0
MAX_SPEED = 0.15  # m/s at |action| = 1.0

# sequencer phase -> AgentState, for the /rvc/state topic
PHASE_STATE = {
    "INIT": AgentState.PLAN,
    "APPROACH": AgentState.EXECUTE,
    "DESCEND": AgentState.EXECUTE,
    "GRASP": AgentState.EXECUTE,
    "LIFT": AgentState.EXECUTE,
    "TRANSPORT": AgentState.EXECUTE,
    "LOWER": AgentState.EXECUTE,
    "RELEASE": AgentState.EXECUTE,
    "RETREAT": AgentState.EXECUTE,
    "VERIFY": AgentState.VERIFY,
    "RECOVER": AgentState.RECOVER,
    "DONE": AgentState.SUCCEEDED,
    "FAILED": AgentState.FAILED,
}


class AgentNode(Node):
    def __init__(self) -> None:
        super().__init__("rvc_agent")

        self.declare_parameter("target_label", "red_block")
        self.declare_parameter("pad_label", "blue_box")
        self.declare_parameter("max_recoveries", 3)

        self.target_label = self.get_parameter("target_label").value
        self.pad_label = self.get_parameter("pad_label").value

        self.detector = ColorDetector()
        self.servo = VisualServoPolicy(self.detector)
        self.seq = PickPlaceSequencer(
            servo=self.servo,
            max_recoveries=int(self.get_parameter("max_recoveries").value),
        )
        self.get_logger().warn(
            "DEGRADED: pick-and-place is driven by a pixel-space visual servo, "
            "NOT a vision-language-action model."
        )

        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self.z: float | None = None
        self.attached_state = "?"
        self.state = AgentState.IDLE
        self._last_phase = ""
        self._reported = False

        self.create_subscription(Image, "/camera/image_raw", self._on_image, 5)
        self.create_subscription(Odometry, "/model/gripper/odometry", self._on_odom, 10)
        self.create_subscription(String, "/gripper/attached_state", self._on_attached, 10)
        self.create_subscription(String, "/rvc/task", self._on_task, 5)

        self.pub_state = self.create_publisher(String, "/rvc/state", 10)
        # Latched (transient-local) so a late subscriber - e.g. the fault
        # injector started after launch - still learns the current phase.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_phase = self.create_publisher(String, "/rvc/phase", latched)
        self.pub_overlay = self.create_publisher(Image, "/rvc/overlay", 5)
        self.pub_cmd_vel = self.create_publisher(Twist, "/model/gripper/cmd_vel", 10)
        self.pub_attach = self.create_publisher(Empty, "/gripper/attach", 10)
        self.pub_detach = self.create_publisher(Empty, "/gripper/detach", 10)

        self.create_timer(1.0 / CONTROL_HZ, self._tick)

    # -- callbacks -----------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _on_odom(self, msg: Odometry) -> None:
        self.z = float(msg.pose.pose.position.z)

    def _on_attached(self, msg: String) -> None:
        self.attached_state = msg.data

    def _on_task(self, msg: String) -> None:
        self.get_logger().info(f"task restart requested: {msg.data}")
        self.seq = PickPlaceSequencer(servo=self.servo, max_recoveries=self.seq.max_recoveries)
        self._last_phase = ""
        self._reported = False

    # -- control loop ---------------------------------------------------------

    def _tick(self) -> None:
        if self.latest_image is None:
            return
        image = self.latest_image

        dets = self.detector.detect(
            image, (self.target_label, self.pad_label, "gripper_marker")
        )
        by_label = {d.label: d for d in dets}

        joint = None
        if self.attached_state in ("attached", "detached"):
            joint = self.attached_state
        cmd = self.seq.step(PickInput(
            target=by_label.get(self.target_label),
            pad=by_label.get(self.pad_label),
            marker=by_label.get("gripper_marker"),
            z=self.z,
            joint_state=joint,
        ))

        # actuate
        tw = Twist()
        tw.linear.x = cmd.vx * MAX_SPEED
        tw.linear.y = cmd.vy * MAX_SPEED
        tw.linear.z = cmd.vz * MAX_SPEED
        self.pub_cmd_vel.publish(tw)
        if cmd.gripper_event == "attach":
            self.pub_attach.publish(Empty())
        elif cmd.gripper_event == "detach":
            self.pub_detach.publish(Empty())

        # observe / narrate
        z_s = "?" if self.z is None else f"{self.z:.3f}"
        self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(
            draw_overlay(image, dets,
                         f"{cmd.phase} | z={z_s} | joint={self.attached_state}",
                         cmd.note or f"recoveries={self.seq.recoveries}"),
            encoding="rgb8",
        ))
        if cmd.phase != self._last_phase:
            self._last_phase = cmd.phase
            self.get_logger().info(f"phase {cmd.phase}  {cmd.note}")
            self.pub_phase.publish(String(data=cmd.phase))
            new_state = PHASE_STATE.get(cmd.phase, AgentState.EXECUTE)
            if new_state is not self.state:
                self.state = new_state
                self.pub_state.publish(String(data=new_state.value))
        if cmd.phase in ("DONE", "FAILED") and not self._reported:
            self._reported = True
            if cmd.done:
                self.get_logger().info("PICK-AND-PLACE SUCCEEDED")
            else:
                self.get_logger().error(f"PICK-AND-PLACE FAILED: {cmd.failure}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
