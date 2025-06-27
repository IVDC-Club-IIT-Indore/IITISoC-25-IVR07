import sys
sys.path.insert(0, '/opt/ros/humble/lib/python3.10/site-packages')

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')

        # Use simulation time for rosbag
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.model = YOLO("yolov8m.pt")  # Use yolov8m or yolov8n.pt
        self.bridge = CvBridge()
        self.window_name = "YOLOv8 + Depth Viewer"
        
        # Video recording state
        self.recording = False
        self.video_writer = None

        # QoS for rosbag topics
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Subscribers
        rgb_sub = Subscriber(self, Image, '/d455_1_rgb_image', qos_profile=qos)
        depth_sub = Subscriber(self, Image, '/d455_1_depth_image', qos_profile=qos)

        # Sync
        self.ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=100, slop=1.0)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info("YOLOv8 + Depth node initialized")
        self.get_logger().info("Waiting for synchronized RGB and Depth frames...")
        self.get_logger().info("Press SPACE to start/stop recording video")
        self.get_logger().info("Press 'q' to quit")

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert RGB
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Convert Depth
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_msg.encoding == '16UC1':
                depth_frame = depth_frame.astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_frame = depth_frame.astype(np.float32)
            else:
                self.get_logger().error(f"Unsupported depth encoding: {depth_msg.encoding}")
                return

            h, w = depth_frame.shape

            # YOLOv8 Inference
            results = self.model(bgr_frame)[0]
            annotated = results.plot()  # Keep YOLO's nice boxes

            for box in results.boxes:
                conf = float(box.conf)
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if 0 <= cx < w and 0 <= cy < h:
                    depth_value = float(depth_frame[cy, cx])
                    if depth_value <= 0.1 or np.isnan(depth_value):
                        continue
                else:
                    continue

                class_id = int(box.cls)
                name = results.names[class_id]

                # Custom green label with only depth info
                label = f"{name} Depth: {depth_value:.2f} m"
                y_pos = max(y2 + 20, y1 + 20)  # Put it below the box
                cv2.putText(
                    annotated, label, (x1, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                self.get_logger().info(label)

            # Handle video recording
            key = cv2.waitKey(1) & 0xFF
            
            # Press SPACE to start/stop recording
            if key == 32:  # Spacebar
                self.recording = not self.recording
                if self.recording:
                    # Initialize VideoWriter
                    frame_height, frame_width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        'output.avi', fourcc, 20.0, (frame_width, frame_height)
                    )
                    self.get_logger().info("Started recording video to 'output.avi'")
                else:
                    if self.video_writer is not None:
                        self.video_writer.release()
                        self.video_writer = None
                        self.get_logger().info("Stopped recording video")
            
            # Press 'q' to quit
            if key == ord('q'):
                if self.video_writer is not None:
                    self.video_writer.release()
                cv2.destroyAllWindows()
                self.get_logger().info("Shutting down...")
                raise KeyboardInterrupt  # Trigger node shutdown
            
            # Write frame if recording
            if self.recording and self.video_writer is not None:
                self.video_writer.write(annotated)
            
            # Show window
            cv2.imshow(self.window_name, annotated)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = YOLODepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        if node.video_writer is not None:
            node.video_writer.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
