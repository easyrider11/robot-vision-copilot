# Stage 3 · ROS 2 Jazzy + Gazebo Harmonic (headless)
#
# NOT VERIFIED ON THE AUTHORING MACHINE: no Docker engine was available, so
# this image has never been built here. It is written against documented
# package names; expect to fix a line or two on first build, then update this.
#
#   docker compose -f ros2_ws/docker-compose.yml build
#   docker compose -f ros2_ws/docker-compose.yml run --rm ros2 bash
#   # inside:
#   ros2 launch rvc_agent tabletop.launch.py
#
# Image budget matters on this host (~7 GB free disk): this stays headless -
# no rviz2, no desktop metapackage, no ultralytics/torch. Expected unpacked
# size ~2.5-3 GB. Add heavier tools only when you actually need them.

FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-colcon-common-extensions \
      ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
      ros-jazzy-cv-bridge ros-jazzy-image-transport \
      ros-jazzy-rmw-cyclonedds-cpp \
      git \
    && rm -rf /var/lib/apt/lists/*

# The agent layer (state machine, planner, validators, visual servo) comes from
# the same `rvc` package Stages 1/2 use - no fork, no copy-paste. Base install
# only: numpy + pillow. No [vision] extra here - ultralytics drags torch
# (~2 GB) into the image; add it deliberately when you reach the YOLO step.
WORKDIR /opt/rvc
COPY pyproject.toml /opt/rvc/
COPY src /opt/rvc/src
RUN pip install --break-system-packages -e .

WORKDIR /ws
COPY ros2_ws/src /ws/src
RUN source /opt/ros/jazzy/setup.bash && colcon build --symlink-install

# .bashrc only helps interactive shells (it early-returns otherwise). The ros
# base image's entrypoint sources /opt/ros but not our overlay, so non-
# interactive `docker run ... bash -lc` could not see rvc_agent. Patch the
# entrypoint so EVERY container invocation gets the workspace.
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc \
 && echo 'source /ws/install/setup.bash'   >> /root/.bashrc \
 && sed -i 's|^source .*|&\nsource /ws/install/setup.bash|' /ros_entrypoint.sh \
 && grep -n "setup.bash" /ros_entrypoint.sh

CMD ["bash"]
