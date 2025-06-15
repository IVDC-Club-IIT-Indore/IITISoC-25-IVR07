# yolo_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.model = YOLO("yolov8n.pt")  # Adjust path if needed
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info("YOLOv8 detector node initialized.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)[0]  # Single frame inference
        annotated = results.plot()      # Annotated image with boxes
        cv2.imshow("YOLOv8 Detection", annotated)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
