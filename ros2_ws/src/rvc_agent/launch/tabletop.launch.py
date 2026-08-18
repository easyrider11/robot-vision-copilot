"""Stage 3 launch — SCAFFOLD, never executed on the authoring machine.

Brings up, headless:
    gz sim (server only)  ->  worlds/tabletop.sdf
    ros_gz_bridge         ->  camera in, gripper cmd_vel out, clock
    rvc_agent agent_node  ->  the state machine

    ros2 launch rvc_agent tabletop.launch.py
    ros2 launch rvc_agent tabletop.launch.py backend:=openvla-remote

Watch it work:
    ros2 topic echo /rvc/state
    ros2 run rqt_image_view rqt_image_view /rvc/overlay     # needs X/GUI
    ros2 topic pub -1 /rvc/task std_msgs/String "{data: 'move over the red block'}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    world = PathJoinSubstitution(
        [FindPackageShare("rvc_agent"), "worlds", "tabletop.sdf"]
    )

    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="auto"),

        # -s server-only (headless), -r start running immediately
        ExecuteProcess(cmd=["gz", "sim", "-s", "-r", world], output="screen"),

        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            output="screen",
            arguments=[
                # gz -> ros: overhead camera + sim clock
                "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                # gz -> ros: gripper proprioception (z for DESCEND/LIFT phases)
                "/model/gripper/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                # gz -> ros: DetachableJoint state ("attached"/"detached")
                "/gripper/attached_state@std_msgs/msg/String[gz.msgs.StringMsg",
                # ros -> gz: velocity commands for the floating gripper
                "/model/gripper/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                # ros -> gz: suction grasp commands
                "/gripper/attach@std_msgs/msg/Empty]gz.msgs.Empty",
                "/gripper/detach@std_msgs/msg/Empty]gz.msgs.Empty",
            ],
        ),

        Node(
            package="rvc_agent",
            executable="agent_node",
            output="screen",
            parameters=[{
                "backend": LaunchConfiguration("backend"),
                "use_sim_time": True,
            }],
        ),
    ])
