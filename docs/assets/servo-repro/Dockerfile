# Minimal, stock-only image for reproducing MoveIt Servo POSE-mode behaviour.
# No Gazebo, no custom URDF: ros2_control FakeSystem + moveit_resources panda.
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-moveit-servo \
      ros-jazzy-moveit-resources-panda-moveit-config \
      ros-jazzy-moveit-kinematics \
      ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager \
      ros-jazzy-joint-state-broadcaster \
      ros-jazzy-tf2-ros ros-jazzy-tf2-tools \
      ros-jazzy-rmw-cyclonedds-cpp \
      python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc
WORKDIR /repro
CMD ["bash"]
