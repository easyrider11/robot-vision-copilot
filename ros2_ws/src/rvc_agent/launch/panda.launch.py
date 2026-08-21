"""Panda arm pick-and-place - Gazebo Harmonic + ros2_control + MoveIt Servo.

VERIFIED status is tracked in docs/09-panda-moveit.md. Chain:

  agent_node (nine-phase sequencer, unchanged)
      | TwistStamped (panda_link0)
  MoveIt Servo (jacobian IK, singularity scaling, joint-limit stop)
      | Float64MultiArray joint velocities
  forward_velocity_controller (ros2_control)
      | gz_ros2_control
  Gazebo Panda

The URDF comes from moveit_resources_panda_description with three surgical
edits applied at launch time (see _build_urdf): finger joints fixed (suction
story - finger contact physics is deliberately out of scope), a green marker
on the hand for the visual servo, and the gz plugins (ros2_control +
DetachableJoint suction) appended. Doing it as string surgery on the shipped
URDF keeps us byte-compatible with the SRDF that MoveIt Servo loads.
"""

from __future__ import annotations

import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

BASE_XYZ = "-0.55 0 0.75"  # base at the camera view edge, axes aligned with world


def _pkg(name: str) -> str:
    return get_package_share_directory(name)


def _build_urdf() -> str:
    with open(os.path.join(_pkg("moveit_resources_panda_description"),
                           "urdf", "panda.urdf")) as f:
        urdf = f.read()

    # 1. suction story: freeze the fingers (no finger contact physics).
    urdf = re.sub(r'(<joint name="panda_finger_joint[12]" type=)"prismatic"', r'\1"fixed"', urdf)
    urdf = re.sub(r'<mimic joint="panda_finger_joint1"[^/]*/>', "", urdf)
    # fixed joints may not carry axis/limit/dynamics tags in some parsers; strip
    # them inside the finger joints only.
    def _strip(m: re.Match) -> str:
        body = re.sub(r'<(axis|limit|dynamics)[^/]*/>', "", m.group(0))
        return body
    urdf = re.sub(r'<joint name="panda_finger_joint[12]" type="fixed">.*?</joint>',
                  _strip, urdf, flags=re.S)

    # 2. the moveit_resources URDF is a planning/visualisation model: most
    # links carry no <inertial>, which makes urdf2sdf drop them and the whole
    # model fail to spawn ("A model must have at least one link"). Give every
    # massless link a nominal inertial so Gazebo accepts it.
    # A self-consistent FAKE dynamics set (the planning URDF has none): with
    # kp=300 these give omega ~ sqrt(kp/I) ~ 77 rad/s (stable at the 1 kHz
    # physics step) and damping 50 -> overdamped. The first attempt (masses
    # 0.4, I=1e-3, no damping) put omega near the integration limit and the
    # arm literally thrashed itself around the sky.
    INERTIAL = ('<inertial><mass value="{m}"/>'
                '<inertia ixx="5e-2" iyy="5e-2" izz="5e-2" ixy="0" ixz="0" iyz="0"/>'
                "</inertial>")

    def _add_inertial(m: re.Match) -> str:
        block, name = m.group(0), m.group(1)
        if "<inertial" in block or name == "world":
            return block
        mass = "3.0" if name.endswith("link0") else "1.0"
        if block.rstrip().endswith("/>"):  # self-closing frame link e.g. panda_link8
            return f'<link name="{name}">{INERTIAL.format(m=0.05)}</link>'
        return block.replace(">", ">" + INERTIAL.format(m=mass), 1)

    urdf = re.sub(r'<link name="([^"]+)"\s*(?:/>|>.*?</link>)', _add_inertial, urdf, flags=re.S)

    # 3. damping: the P-only position interface (kp=60) on light links is
    # undamped - the first gain test sent the arm thrashing across the sky.
    def _add_damping(m: re.Match) -> str:
        block = m.group(0)
        if "<dynamics" in block:
            return block
        return block.replace("</joint>", '<dynamics damping="15.0" friction="0.5"/></joint>', 1)

    urdf = re.sub(r'<joint name="panda_joint[1-7]" type="revolute">.*?</joint>',
                  _add_damping, urdf, flags=re.S)

    extra = """
  <!-- weld the arm to the world: link0 is otherwise a FREE rigid body, and
       with gravity off every reaction/contact impulse sends the whole robot
       drifting and tumbling (observed: model pose (0.37,-0.57,1.11) with wild
       RPY after one spawn-contact). The mount pose lives HERE, not in spawn
       CLI args, which also dodges ros_gz create's negative-number parsing. -->
  <link name="world"/>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="panda_link0"/>
    <origin xyz="BASE_XYZ_PLACEHOLDER" rpy="0 0 0"/>
  </joint>
  <link name="rvc_marker">
    <inertial><mass value="0.02"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual>
      <!-- thin disc WIDER than the wrist (r 0.075 vs wrist ~0.045): from the
           overhead camera it shows as a green ring around the hand silhouette,
           and roll/pitch stay servo-controlled so it keeps facing up -->
      <geometry><cylinder radius="0.09" length="0.004"/></geometry>
      <material name="rvc_green"><color rgba="0.08 0.85 0.10 1"/></material>
    </visual>
  </link>
  <joint name="rvc_marker_joint" type="fixed">
    <parent link="panda_hand"/>
    <child link="rvc_marker"/>
    <origin xyz="0 0 -0.10" rpy="0 0 0"/>
  </joint>
GRAVITY_OFF
  <gazebo reference="panda_hand_joint"><preserveFixedJoint>true</preserveFixedJoint></gazebo>
  <gazebo reference="panda_joint8"><preserveFixedJoint>true</preserveFixedJoint></gazebo>
  <gazebo reference="rvc_marker">
    <visual><material><ambient>0.08 0.85 0.10 1</ambient>
      <diffuse>0.08 0.85 0.10 1</diffuse></material></visual>
  </gazebo>
  <ros2_control name="panda_gz" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    JOINTS
  </ros2_control>
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>CONTROLLERS_YAML</parameters>
    </plugin>
    <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
      <parent_link>panda_hand</parent_link>
      <child_model>red_block</child_model>
      <child_link>link</child_link>
      <attach_topic>/gripper/attach</attach_topic>
      <detach_topic>/gripper/detach</detach_topic>
      <output_topic>/gripper/attached_state</output_topic>
    </plugin>
  </gazebo>
</robot>"""
    # spawn in the standard Panda "ready" pose: the all-zero configuration is
    # a fully-stretched singularity (arm vertical, hand at the camera) from
    # which MoveIt Servo will not move.
    ready = {1: 0.0, 2: -0.785, 3: 0.0, 4: -2.356, 5: 0.0, 6: 1.571, 7: 0.785}
    joints = "\n".join(
        f'    <joint name="panda_joint{i}">'
        f'<command_interface name="position"/>'
        f'<state_interface name="position">'
        f'<param name="initial_value">{ready[i]}</param></state_interface>'
        f'<state_interface name="velocity"/>'
        f"</joint>" for i in range(1, 8))
    extra = extra.replace("JOINTS", joints)
    extra = extra.replace("BASE_XYZ_PLACEHOLDER", BASE_XYZ)
    # Kinematic-puppet arm: the planning URDF has no real dynamics and the gz
    # position interface is a bare P controller - with invented inertias the
    # combination is either saggy or oscillatory (both observed live). Gravity
    # off + damping turns tracking into pure geometry, the same simplification
    # the floating gripper used. The BLOCK keeps gravity - placement still drops.
    links = [f"panda_link{i}" for i in range(9)] + [
        "panda_hand", "panda_leftfinger", "panda_rightfinger", "rvc_marker"]
    extra = extra.replace("GRAVITY_OFF", "\n".join(
        f'  <gazebo reference="{ln}"><gravity>false</gravity></gazebo>' for ln in links))
    extra = extra.replace("CONTROLLERS_YAML", os.path.join(
        _pkg("rvc_agent"), "config", "panda_controllers.yaml"))
    return urdf.replace("</robot>", extra)


def generate_launch_description() -> LaunchDescription:
    urdf = _build_urdf()
    with open(os.path.join(_pkg("moveit_resources_panda_moveit_config"),
                           "config", "panda.srdf")) as f:
        srdf = f.read()
    # our URDF surgery fixed the finger joints (suction story); the shipped
    # SRDF's hand group_states still assign them values, which aborts the
    # MoveIt robot model ("requires 0 variable values"). Strip those states.
    srdf = re.sub(r'<group_state[^>]*group="hand".*?</group_state>', "", srdf, flags=re.S)
    # no one publishes the virtual_joint's state in this stack; dropping it
    # stops the planning-scene monitor waiting on "Missing virtual_joint".
    srdf = re.sub(r'<virtual_joint[^/]*/>', "", srdf)
    world = os.path.join(_pkg("rvc_agent"), "worlds", "panda_tabletop.sdf")

    # the agent's moveit_py RobotModel wants file paths
    urdf_path, srdf_path = "/tmp/rvc_panda.urdf", "/tmp/rvc_panda.srdf"
    with open(urdf_path, "w") as f:
        f.write(urdf)
    with open(srdf_path, "w") as f:
        f.write(srdf)

    gz = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world], output="screen",
        additional_env={
            # so gz finds libgz_ros2_control-system.so ...
            "GZ_SIM_SYSTEM_PLUGIN_PATH": "/opt/ros/jazzy/lib",
            # ... and resolves model://moveit_resources_panda_description/meshes/*
            "GZ_SIM_RESOURCE_PATH": "/opt/ros/jazzy/share",
        })

    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               output="screen",
               parameters=[{"robot_description": urdf, "use_sim_time": True}])

    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", "panda"])

    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  arguments=[
                      "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
                      "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                      "/gripper/attached_state@std_msgs/msg/String[gz.msgs.StringMsg",
                      "/gripper/attach@std_msgs/msg/Empty]gz.msgs.Empty",
                      "/gripper/detach@std_msgs/msg/Empty]gz.msgs.Empty",
                  ])

    jsb = Node(package="controller_manager", executable="spawner",
               arguments=["joint_state_broadcaster"], output="screen")
    fvc = Node(package="controller_manager", executable="spawner",
               arguments=["forward_position_controller"], output="screen")

    # Between gz spawn and controller activation the joints are unclaimed and
    # the arm sags under gravity (found live: j4 drifted from -2.356 to -2.95,
    # inside MoveIt Servo's 0.12 rad joint-bound margin -> permanent halt).
    # Re-command the ready pose for ~3 s once the position controller is up.
    homing = ExecuteProcess(
        cmd=["ros2", "topic", "pub", "-t", "80", "-r", "10",
             "/forward_position_controller/commands",
             "std_msgs/msg/Float64MultiArray",
             "{data: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]}"],
        output="screen")

    # Geometry (see docs/09): block top 0.800, hand TCP ~0.103 below link8 with
    # fingers fixed at zero; suction face ≈ hand + 0.06. Grasp when link8 is at
    # world 0.800 + 0.107 + 0.005 ≈ 0.912; travel well above; place a hair higher
    # than grasp so the carried block (rigid offset) lands on the 0.760 pad.
    agent = Node(package="rvc_agent", executable="agent_node", output="screen",
                 parameters=[{
                     "embodiment": "panda",
                     "use_sim_time": True,
                     "base_z_offset": float(BASE_XYZ.split()[2]),
                     "z_travel": 1.06, "z_grasp": 0.912, "z_place": 0.922,
                     "image_size": 512,
                     "max_speed": 0.08, "z_tol": 0.02,
                     "urdf_path": urdf_path, "srdf_path": srdf_path,
                     "home_u": 266.0, "home_v": 326.0,
                     "max_recoveries": 3,
                 }])

    # order: sim + description first, controllers once the model is in, then IO
    return LaunchDescription([
        gz, rsp, spawn, bridge,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[fvc])),
        RegisterEventHandler(OnProcessExit(target_action=fvc, on_exit=[homing])),
        RegisterEventHandler(OnProcessExit(target_action=homing, on_exit=[agent])),
    ])
