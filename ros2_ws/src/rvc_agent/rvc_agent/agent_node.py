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
    /rvc/phase                     out  std_msgs/String      sequencer phase (fault injector hooks h
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
    from geometry_msgs.msg import (
        PoseStamped,
        Twist,
        TwistStamped,
    )
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Empty, String
    from tf2_ros import (
        Buffer,
        ConnectivityException,
        ExtrapolationException,
        LookupException,
        TransformListener,
    )
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
DEFAULT_MAX_SPEED = 0.15  # m/s at |action| = 1.0


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

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
        # --- embodiment: "floating" (velocity-controlled box, default) or
        # "panda" (7-DOF arm driven through MoveIt Servo). Same sequencer;
        # what changes is the output channel, the z source and the z targets.
        self.declare_parameter("embodiment", "floating")
        self.declare_parameter("command_frame", "panda_link0")
        self.declare_parameter("ee_frame", "panda_link8")
        self.declare_parameter("base_z_offset", 0.75)  # world z of the arm base
        self.declare_parameter("attach_topic", "/gripper/attach")
        self.declare_parameter("detach_topic", "/gripper/detach")
        self.declare_parameter("z_travel", 0.0)  # 0.0 = sequencer default
        self.declare_parameter("z_grasp", 0.0)
        self.declare_parameter("z_place", 0.0)
        self.declare_parameter("home_u", 0.0)  # 0.0 = sequencer default corner
        self.declare_parameter("home_v", 0.0)
        self.declare_parameter("image_size", 256)
        # calibration probe: non-zero => publish this constant world-frame
        # velocity instead of running the sequencer (used by scripts only)
        self.declare_parameter("debug_twist", [0.0, 0.0, 0.0])
        # Measured (docs/09): this servo generation transforms twists by
        # header.frame_id correctly - plain panda_link0 stamping is right.
        # The EE-frame path remains for the A/B experiment that proved it.
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("srdf_path", "")
        self.declare_parameter("max_speed", DEFAULT_MAX_SPEED)
        self.declare_parameter("z_tol", 0.0)  # 0 = sequencer default

        self.target_label = self.get_parameter("target_label").value
        self.pad_label = self.get_parameter("pad_label").value
        self.embodiment = self.get_parameter("embodiment").value
        self.command_frame = self.get_parameter("command_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.base_z_offset = float(self.get_parameter("base_z_offset").value)

        self.detector = ColorDetector()
        self.servo = VisualServoPolicy(
            self.detector, image_size=int(self.get_parameter("image_size").value))
        seq_kw = {}
        for name in ("z_travel", "z_grasp", "z_place"):
            v = float(self.get_parameter(name).value)
            if v > 0.0:
                seq_kw[name] = v
        hu, hv = (float(self.get_parameter(n).value) for n in ("home_u", "home_v"))
        if hu > 0.0 and hv > 0.0:
            seq_kw["home_px"] = (hu, hv)
        if float(self.get_parameter("z_tol").value) > 0.0:
            seq_kw["z_tol"] = float(self.get_parameter("z_tol").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self._seq_kw = seq_kw
        self.seq = PickPlaceSequencer(
            servo=self.servo,
            max_recoveries=int(self.get_parameter("max_recoveries").value),
            **seq_kw,
        )
        self.get_logger().warn(
            "DEGRADED: pick-and-place is driven by a pixel-space visual servo, "
            "NOT a vision-language-action model."
        )

        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self._ee_rot: np.ndarray | None = None
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
        self.pub_servo_pose = self.create_publisher(
            PoseStamped, "/servo_node/pose_target_cmds", 10)
        self._target_xyz = None  # virtual pose target, seeded from TF
        self._ee_quat = None
        self.pub_attach = self.create_publisher(
            Empty, self.get_parameter("attach_topic").value, 10)
        self.pub_detach = self.create_publisher(
            Empty, self.get_parameter("detach_topic").value, 10)

        # Panda proprioception: EE height via TF (robot_state_publisher's
        # base->link8 chain) instead of the floating gripper's odometry.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        # Panda: deterministic in-node resolved-rate control (docs/09 records
        # why MoveIt Servo was retired from this loop). Jacobian via moveit_py;
        # joint targets go straight to the proven JointGroupPositionController.
        self._servo_ready = True
        self._q: np.ndarray | None = None
        self._arm_joints = [f"panda_joint{i}" for i in range(1, 8)]
        self._robot_state = None
        if self.embodiment == "panda":
            from moveit.core.robot_model import RobotModel
            from moveit.core.robot_state import RobotState
            urdf_path = str(self.get_parameter("urdf_path").value)
            srdf_path = str(self.get_parameter("srdf_path").value)
            model = RobotModel(urdf_xml_path=urdf_path, srdf_xml_path=srdf_path)
            self._robot_state = RobotState(model)
            from std_msgs.msg import Float64MultiArray
            self._FloatArr = Float64MultiArray
            self.pub_joint_cmd = self.create_publisher(
                Float64MultiArray, "/forward_position_controller/commands", 10)
            self.create_subscription(JointState, "/joint_states", self._on_joints, 10)

        self.create_timer(1.0 / CONTROL_HZ, self._tick)

    def _on_joints(self, msg) -> None:
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in self._arm_joints):
            self._q = np.array([pos[j] for j in self._arm_joints])

    # -- callbacks -----------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _on_odom(self, msg: Odometry) -> None:
        self.z = float(msg.pose.pose.position.z)

    def _on_attached(self, msg: String) -> None:
        self.attached_state = msg.data

    def _on_task(self, msg: String) -> None:
        self.get_logger().info(f"task restart requested: {msg.data}")
        self.seq = PickPlaceSequencer(servo=self.servo, max_recoveries=self.seq.max_recoveries,
                                      **self._seq_kw)
        self._last_phase = ""
        self._reported = False

    def _actuate(self, cmd) -> None:
        tw = Twist()
        tw.linear.x = cmd.vx * self.max_speed
        tw.linear.y = cmd.vy * self.max_speed
        tw.linear.z = cmd.vz * self.max_speed
        if self.embodiment != "panda":
            self.pub_cmd_vel.publish(tw)
            return
        # deterministic resolved-rate step (see rvc_agent.arm_control):
        # world axes == panda_link0 axes in this workcell, so the commanded
        # velocity needs no frame gymnastics; orientation is held at the frozen
        # spawn (suction-down) rotation by the correction term.
        if self._q is None or self._ee_rot is None or self._ee_quat is None:
            return
        from rvc_agent.arm_control import quat_to_matrix, resolved_rate_step
        self._robot_state.set_joint_group_positions("panda_arm", self._q)
        self._robot_state.update()
        jac = np.asarray(self._robot_state.get_jacobian(
            joint_model_group_name="panda_arm", reference_point_position=np.zeros(3)))
        r_target = quat_to_matrix(self._ee_quat.x, self._ee_quat.y,
                                  self._ee_quat.z, self._ee_quat.w)
        v = np.array([tw.linear.x, tw.linear.y, tw.linear.z])
        dq = resolved_rate_step(jac, v, r_target, self._ee_rot, dt=1.0 / CONTROL_HZ)
        msg = self._FloatArr()
        msg.data = [float(x) for x in (self._q + dq)]
        self.pub_joint_cmd.publish(msg)

    # -- control loop ---------------------------------------------------------

    def _tick(self) -> None:
        if self.latest_image is None or not self._servo_ready:
            return
        if self.embodiment == "panda":
            try:
                tf = self.tf_buffer.lookup_transform(
                    "panda_link0", self.ee_frame, Time())
                self.z = self.base_z_offset + float(tf.transform.translation.z)
                q = tf.transform.rotation
                self._ee_rot = _quat_to_matrix(q.x, q.y, q.z, q.w)
                if self._target_xyz is None:
                    tr = tf.transform.translation
                    self._target_xyz = np.array([tr.x, tr.y, tr.z])
                    self._ee_quat = q  # freeze the spawn (downward) orientation
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                # TF genuinely not up yet. NOTE deliberately narrow: an earlier
                # version caught Exception and silently ate an AttributeError,
                # leaving z=None forever and the sequencer parked in INIT.
                self.z = None
                self.get_logger().warn(f"EE TF not available yet: {exc}",
                                       throttle_duration_sec=5.0)

        dbg = [float(v) for v in self.get_parameter("debug_twist").value]
        if any(abs(v) > 1e-9 for v in dbg):
            from rvc.agent.pickplace import PickCommand
            probe = PickCommand(vx=dbg[0], vy=dbg[1], vz=dbg[2])
            self._actuate(probe)
            self.get_logger().info(
                f"probe twist={dbg} z={self.z}", throttle_duration_sec=2.0)
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

        self._actuate(cmd)

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
        self.get_logger().info(
            f"phase={cmd.phase} z={z_s} joint={self.attached_state}",
            throttle_duration_sec=5.0)
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
