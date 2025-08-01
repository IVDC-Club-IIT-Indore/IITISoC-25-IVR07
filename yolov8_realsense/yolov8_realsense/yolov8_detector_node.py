#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class YOLOv8DetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('publish_image', True)  # New parameter
        self.declare_parameter('image_scale', 0.5)     # New parameter for performance
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.publish_image = self.get_parameter('publish_image').get_parameter_value().bool_value
        self.image_scale = self.get_parameter('image_scale').get_parameter_value().double_value
        
        # Initialize YOLO model
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.get_logger().info(f'YOLOv8 model loaded: {model_path}')
            self.get_logger().info(f'Using device: {self.device}')
            self.get_logger().info(f'Image publishing: {"ON" if self.publish_image else "OFF"}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            return
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera info
        self.camera_info = None
        
        # Frame counter for performance monitoring
        self.frame_count = 0
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            1  # Reduced queue size for better performance
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            1
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/yolov8/detections',
            1  # Reduced queue size
        )
        
        # Only create image publisher if needed
        if self.publish_image:
            self.annotated_image_pub = self.create_publisher(
                Image,
                '/yolov8/detection_image',  # Different topic name
                1
            )
        
        self.get_logger().info('YOLOv8 Detector Node initialized successfully')

    def camera_info_callback(self, msg):
        """Store camera calibration info"""
        if self.camera_info is None:  # Only store once
            self.camera_info = msg

    def image_callback(self, msg):
        """Process incoming images with YOLOv8"""
        try:
            self.frame_count += 1
            
            # Skip frames for performance if needed
            if self.frame_count % 2 != 0:  # Process every 2nd frame
                return
                
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Resize image for faster processing
            if self.image_scale != 1.0:
                height, width = cv_image.shape[:2]
                new_height = int(height * self.image_scale)
                new_width = int(width * self.image_scale)
                cv_image_resized = cv2.resize(cv_image, (new_width, new_height))
            else:
                cv_image_resized = cv_image
            
            # Run YOLOv8 inference
            results = self.model(
                cv_image_resized,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Create detection array message
            detection_array = Detection2DArray()
            detection_array.header = msg.header
            
            # Process results
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Scale factor to convert back to original image coordinates
                    scale_factor = 1.0 / self.image_scale
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Create detection message
                        detection = Detection2D()
                        detection.header = msg.header
                        
                        # Object hypothesis
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = str(class_id)
                        hypothesis.hypothesis.score = float(conf)
                        detection.results = [hypothesis]
                        
                        # Bounding box (scale back to original coordinates)
                        bbox = BoundingBox2D()
                        bbox.center.position.x = float((box[0] + box[2]) / 2 * scale_factor)
                        bbox.center.position.y = float((box[1] + box[3]) / 2 * scale_factor)
                        bbox.center.theta = 0.0
                        bbox.size_x = float((box[2] - box[0]) * scale_factor)
                        bbox.size_y = float((box[3] - box[1]) * scale_factor)
                        detection.bbox = bbox
                        
                        # Store class name in detection id
                        detection.id = self.model.names[class_id]
                        
                        detection_array.detections.append(detection)
                
                # Only publish annotated image if enabled
                if self.publish_image:
                    annotated_frame = result.plot()
                    
                    # Resize back to original size if needed
                    if self.image_scale != 1.0:
                        height, width = cv_image.shape[:2]
                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                    
                    # Add detection count info
                    cv2.putText(annotated_frame, f'Detections: {len(detection_array.detections)}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, 'DETECTION MODE', 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Publish annotated image
                    annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                    annotated_msg.header = msg.header
                    self.annotated_image_pub.publish(annotated_msg)
            
            # Always publish detections
            self.detection_pub.publish(detection_array)
            
        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8DetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

