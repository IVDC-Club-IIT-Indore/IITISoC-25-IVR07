# IITISoC-25-IVR007 Dynamic Obstacle Detection and Tracking for Autonomous Vehicles

**Team Members**

_**Shubh Raghuwanshi**:  [@Shubh](https://github.com/Shubhragh)_

_**Manish Kumar**:  [@Manish](https://github.com/Manish-git-tech)_

_**Neh Modi**:  [@Neh](https://github.com/Nehmodi2005)_

_**Chandrashekhar**:  [@Chandrasekhar](https://github.com/ChandrashekarRVN)_

**Mentors**

_**Nambiar Anand Sreenivasan**:  [@NambiarAnand](https://github.com/NambiarAnand)_

_**Mentor 2**:  [@mentor2](https://github.com/mentor2)_

# INSTRUCTIONS FOR RUNNING THE yolov8_realsense 

# YOLOv8 3D Tracking - Installation & Running Guide

## Installation

### 1. Install Dependencies
```bash
# ROS2 packages
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs ros-humble-image-transport
sudo apt install ros-humble-realsense2-*

# Python packages
pip install ultralytics opencv-python numpy torch scipy deep-sort-realtime
```

**📺 Watch Installation Video**: [YOLOv8 RealSense Setup Tutorial](https://youtu.be/9p9fBWrVkEk?si=C9yJdp62l3zSlSTT)

### 2. Build Package
```bash
# Navigate to workspace
cd ~/ros2_ws/src
# Place yolov8_realsense folder here

# Build
cd ~/ros2_ws
colcon build --packages-select yolov8_realsense
source install/setup.bash
```

## Running the System

### Option 1: Live Intel RealSense D455

**📺 Live Camera Demo**: [Real-time 3D Tracking with RealSense](https://youtu.be/g3eS8J1P-QY?si=uusEopx49uOvkrpR)

```bash
# Terminal 1: Start camera
ros2 launch realsense2_camera rs_launch.py

# Terminal 2: Start tracking system
cd ~/ros2_ws
source install/setup.bash
ros2 launch yolov8_realsense yolov8_tracking.launch.py
```

### Option 2: Rosbag Playback

**📺 Rosbag Processing Demo**: [3D Tracking from Recorded Data](https://youtu.be/g3eS8J1P-QY?si=uusEopx49uOvkrpR)

```bash
# Terminal 1: Play rosbag
ros2 bag play your_bag_name --loop

# Terminal 2: Start tracking
cd ~/ros2_ws
source install/setup.bash
ros2 launch yolov8_realsense yolov8_tracking.launch.py
```

## Visualization

### 2D Image View
```bash
ros2 run rqt_image_view rqt_image_view
# Select: /yolov8/tracking_image_3d
```

### 3D RViz Visualization
```bash
rviz2
# Set Fixed Frame: camera_color_optical_frame
# Add MarkerArray: /yolov8/tracking_markers
```

**📺 Visualization Tutorial**: [RViz 3D Tracking Output](https://www.youtube.com/watch?v=x__hBLPtahE)

## Quick Verification
```bash
# Check nodes are running
ros2 node list
# Expected: /yolov8_detector_3d, /yolov8_tracker_3d

# Check tracking output
ros2 topic list | grep yolov8
```

**📺 Complete System Demo**: [Multi-Object Tracking](https://youtu.be/9p9fBWrVkEk?si=C9yJdp62l3zSlSTT)
