# YOLOv8 ROS 2 Object Detection Node

This package provides a ROS 2 (Humble) compatible object detection node using **Ultralytics YOLOv8**. It also includes a standalone test script for running YOLOv8 inference without ROS, useful for quick verification.

---

## 📦 Package: `yolo_detector`

### ✅ Features

* YOLOv8 integration with ROS 2 (Humble)
* Subscribes to image data (`sensor_msgs/Image`)
* Uses `cv_bridge` to convert ROS → OpenCV
* Logs detections (extendable to publish bounding boxes or visual output)
* Includes `test_yolo_local.py` for local non-ROS inference

---

## 📁 Folder Structure

```
yolo_detector/
├── yolo_detector/            # Python module
│   └── yolo_node.py          # Main ROS 2 node with YOLOv8
├── test_yolo_local.py        # Non-ROS test script for YOLOv8
├── package.xml               # ROS 2 metadata
├── setup.py                  # Python packaging setup
├── pyproject.toml            # Ultralytics requirement support
└── resource/
    └── yolo_detector         # ROS 2 resource marker
```

---

## 🚀 How to Use

### 🧠 Prerequisites

* ROS 2 Humble installed and sourced
* Python 3.10
* Required Python packages:

```bash
pip install ultralytics opencv-python
```

### ⚙️ Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select yolo_detector
source install/setup.bash
```

### ▶️ Run the ROS 2 Node

```bash
ros2 run yolo_detector yolo_node
```

You should see:

```
[INFO] [timestamp] [yolo_detector_node]: YOLOv8 detector node initialized.
```

> Note: If no camera or rosbag is running, it will not receive any image data.

---

## 🧪 Test YOLOv8 Locally (without ROS)

```bash
python3 test_yolo_local.py
```

This script uses Ultralytics' default test images (e.g., `bus.jpg`, `zidane.jpg`) to verify inference works.

---

## 🔧 Development Notes

* The ROS node currently logs detections; you can extend it to publish bounding boxes or annotated images.
* Ensure all ROS dependencies like `cv_bridge`, `sensor_msgs`, and `rclpy` are available in your Python environment.

---

## 👨‍💻 Contributor

* **Name:** \[Your Name]
* **Role:** AI/ML Module – IVR07, IITISoC 2025
* **Contribution:** YOLOv8 + ROS 2 object detection integration

---

## 📌 Status

✅ YOLOv8 integrated
✅ ROS 2 node functional
✅ Local test script verified

---
