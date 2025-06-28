import sys
sys.path.insert(0, '/opt/ros/humble/lib/python3.10/site-packages')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLORGBNode(Node):
    def __init__(self):
        super().__init__('yolo_rgb_node')
        
        # Enable simulation time for bag playback
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()
        self.window_name = "YOLOv8 Detection"
        
        # QoS setup for RGB topic
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # Create subscriber for RGB images only
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos)
        
        self.get_logger().info("YOLOv8 RGB node initialized")
        self.get_logger().info("Waiting for RGB images...")

    def image_callback(self, rgb_msg):
        try:
            # Convert RGB image
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Run YOLOv8 inference
            results = self.model(bgr_frame)[0]
            annotated = results.plot()  # Base image with detections
            
            # Prepare and add non-overlapping labels
            boxes = []
            for idx, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                conf = float(box.conf)
                boxes.append({
                    'coords': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'conf': conf,
                    'name': results.names[class_id]
                })
                self.get_logger().info(
                    f"Detected {results.names[class_id]} "
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
            label = f"{box['name']} {box['conf']:.2f}"
            cv2.putText(
                image, label, (x1, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

def main(args=None):
    rclpy.init(args=args)
    node = YOLORGBNode()
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
