import sys
sys.path.insert(0, '/opt/ros/humble/lib/python3.10/site-packages')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')
        
        # Enable simulation time for bag playback
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()
        self.window_name = "YOLOv8 Detection"
        
        # QoS setup for both topics
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # Create subscribers for synchronized topics
        rgb_sub = Subscriber(self, Image, '/d455_1_rgb_image', qos_profile=qos)
        depth_sub = Subscriber(self, Image, '/d455_1_depth_image', qos_profile=qos)
        
        # Synchronize topics with larger tolerance
        self.ts = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=100,
            slop=1.0
        )
        self.ts.registerCallback(self.image_callback)
        
        self.get_logger().info("YOLOv8 + Depth node initialized")
        self.get_logger().info("Waiting for synchronized images...")

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert RGB image
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Convert Depth image and handle units
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_msg.encoding == '16UC1':
                depth_frame = depth_frame.astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_frame = depth_frame.astype(np.float32)
            else:
                self.get_logger().error(f"Unsupported depth encoding: {depth_msg.encoding}")
                return
            
            # Run YOLOv8 inference
            results = self.model(bgr_frame)[0]
            annotated = results.plot()  # Base image with detections
            
            # Prepare and add non-overlapping labels
            boxes = []
            for idx, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                depth_roi = depth_frame[y1:y2, x1:x2]
                valid_depths = depth_roi[depth_roi != 0]
                if valid_depths.size > 0:
                    median_depth = np.median(valid_depths)
                else:
                    median_depth = 0.0
                class_id = int(box.cls)
                conf = float(box.conf)
                boxes.append({
                    'coords': (x1, y1, x2, y2),
                    'depth': median_depth,
                    'class_id': class_id,
                    'conf': conf,
                    'name': results.names[class_id]
                })
                self.get_logger().info(
                    f"Detected {results.names[class_id]} "
                    f"at {median_depth:.2f}m "
                    f"with confidence {conf:.2f}"
                )
            self.add_labels_with_offset(annotated, boxes)
            
            # Show live window (comment these lines if no GUI wanted)
            cv2.imshow(self.window_name, annotated)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def add_labels_with_offset(self, image, boxes, start_y_offset=30, line_height=25):
        """Add labels to image with vertical offset to prevent overlap"""
        boxes_sorted = sorted(boxes, key=lambda b: b['coords'][1])
        for idx, box in enumerate(boxes_sorted):
            x1, y1, x2, y2 = box['coords']
            y_pos = y1 - start_y_offset - idx * line_height
            if y_pos < 20:
                y_pos = y2 + 20 + idx * line_height
            label = f"{box['name']} {box['conf']:.2f} {box['depth']:.2f}m"
            cv2.putText(
                image, label, (x1, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

def main(args=None):
    rclpy.init(args=args)
    node = YOLODepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
