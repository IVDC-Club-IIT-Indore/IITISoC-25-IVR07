#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Point, PoseArray, Pose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import message_filters
import threading

class YOLOv8Detector3DNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector_3d')
        
        # Parameters - OPTIMIZED FOR PERFORMANCE
        self.declare_parameter('model_path', 'yolov8n.pt')  # Use yolov8n for speed
        self.declare_parameter('confidence_threshold', 0.6)  # Higher threshold = fewer detections = faster
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_detections', 50)  # Reduced from 100
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('depth_camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('process_every_n_frames', 3)  # Process every 3rd frame
        self.declare_parameter('resize_factor', 0.6)  # Resize images for faster processing
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        depth_camera_info_topic = self.get_parameter('depth_camera_info_topic').get_parameter_value().string_value
        self.publish_image = self.get_parameter('publish_image').get_parameter_value().bool_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').get_parameter_value().integer_value
        self.resize_factor = self.get_parameter('resize_factor').get_parameter_value().double_value
        
        # Initialize YOLO model with optimizations
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            
            # YOLO optimizations
            if torch.cuda.is_available():
                self.model.model.half()  # Use half precision for speed
            
            self.get_logger().info(f'YOLOv8 model loaded: {model_path}')
            self.get_logger().info(f'Using device: {self.device}')
            self.get_logger().info(f'Processing every {self.process_every_n_frames}th frame')
            self.get_logger().info(f'Resize factor: {self.resize_factor}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            return
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera info
        self.rgb_camera_info = None
        self.depth_camera_info = None
        
        # Frame counter for performance
        self.frame_count = 0
        self.last_detection_result = None
        
        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        
        # Synchronized subscribers with larger buffer
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=1)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=1)
        
        # Looser synchronization for performance
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=5,  # Reduced queue size
            slop=0.2       # Increased slop for better sync
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # Camera info subscribers
        self.rgb_camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.rgb_camera_info_callback, 1)
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo, depth_camera_info_topic, self.depth_camera_info_callback, 1)
        
        # Publishers with smaller queues
        self.detection_3d_pub = self.create_publisher(PoseArray, '/yolov8/detections_3d', 1)
        self.detection_2d_pub = self.create_publisher(Detection2DArray, '/yolov8/detections_2d', 1)
        
        if self.publish_image:
            self.annotated_image_pub = self.create_publisher(Image, '/yolov8/detection_image_3d', 1)
        
        self.get_logger().info('YOLOv8 3D Detector Node initialized - PERFORMANCE OPTIMIZED')

    def rgb_camera_info_callback(self, msg):
        if self.rgb_camera_info is None:
            self.rgb_camera_info = msg

    def depth_camera_info_callback(self, msg):
        if self.depth_camera_info is None:
            self.depth_camera_info = msg

    def pixel_to_3d_point(self, u, v, depth, camera_info):
        if depth <= 0 or np.isnan(depth) or depth > 8000:  # Max 8m range
            return None
            
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        
        z = depth * self.depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return (x, y, z)

    def get_object_3d_position(self, bbox, depth_image, camera_info):
        # Get bounding box coordinates with resize factor correction
        scale = 1.0 / self.resize_factor
        x1 = int((bbox.center.position.x - bbox.size_x / 2) * scale)
        y1 = int((bbox.center.position.y - bbox.size_y / 2) * scale)
        x2 = int((bbox.center.position.x + bbox.size_x / 2) * scale)
        y2 = int((bbox.center.position.y + bbox.size_y / 2) * scale)
        
        h, w = depth_image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Sample depth from multiple points for robustness
        center_u = int(bbox.center.position.x * scale)
        center_v = int(bbox.center.position.y * scale)
        
        # Sample 5 points: center + 4 corners of inner region
        sample_points = [
            (center_u, center_v),
            (center_u - int(bbox.size_x * scale * 0.2), center_v - int(bbox.size_y * scale * 0.2)),
            (center_u + int(bbox.size_x * scale * 0.2), center_v - int(bbox.size_y * scale * 0.2)),
            (center_u - int(bbox.size_x * scale * 0.2), center_v + int(bbox.size_y * scale * 0.2)),
            (center_u + int(bbox.size_x * scale * 0.2), center_v + int(bbox.size_y * scale * 0.2))
        ]
        
        valid_depths = []
        for u, v in sample_points:
            if 0 <= u < w and 0 <= v < h:
                depth_val = depth_image[v, u]
                if 100 < depth_val < 8000:  # Valid range for RealSense
                    valid_depths.append(depth_val)
        
        if len(valid_depths) < 2:
            return None
        
        # Use median for robustness
        median_depth = np.median(valid_depths)
        return self.pixel_to_3d_point(center_u, center_v, median_depth, camera_info)

    def synchronized_callback(self, rgb_msg, depth_msg):
        try:
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % self.process_every_n_frames != 0:
                # Republish last detection if available
                if self.last_detection_result is not None:
                    detection_2d_array, pose_array = self.last_detection_result
                    detection_2d_array.header = rgb_msg.header
                    pose_array.header = rgb_msg.header
                    self.detection_2d_pub.publish(detection_2d_array)
                    self.detection_3d_pub.publish(pose_array)
                return
            
            if self.rgb_camera_info is None or self.depth_camera_info is None:
                return
            
            with self.processing_lock:
                # Convert images
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
                
                # Resize RGB for faster processing
                if self.resize_factor != 1.0:
                    h, w = rgb_image.shape[:2]
                    new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
                    rgb_resized = cv2.resize(rgb_image, (new_w, new_h))
                else:
                    rgb_resized = rgb_image
                
                # YOLO inference with optimizations
                results = self.model(
                    rgb_resized,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    verbose=False,
                    half=torch.cuda.is_available()  # Use half precision if CUDA
                )
                
                # Process results
                detection_2d_array = Detection2DArray()
                detection_2d_array.header = rgb_msg.header
                
                pose_array = PoseArray()
                pose_array.header = rgb_msg.header
                pose_array.header.frame_id = "camera_depth_optical_frame"
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        # Create 2D detection
                        detection_2d = Detection2D()
                        detection_2d.header = rgb_msg.header
                        
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = str(class_id)
                        hypothesis.hypothesis.score = float(conf)
                        detection_2d.results = [hypothesis]
                        
                        bbox_2d = BoundingBox2D()
                        bbox_2d.center.position.x = float((box[0] + box[2]) / 2)
                        bbox_2d.center.position.y = float((box[1] + box[3]) / 2)
                        bbox_2d.center.theta = 0.0
                        bbox_2d.size_x = float(box[2] - box[0])
                        bbox_2d.size_y = float(box[3] - box[1])
                        detection_2d.bbox = bbox_2d
                        detection_2d.id = self.model.names[class_id]
                        
                        detection_2d_array.detections.append(detection_2d)
                        
                        # Calculate 3D position
                        point_3d = self.get_object_3d_position(bbox_2d, depth_image, self.depth_camera_info)
                        
                        if point_3d is not None:
                            pose = Pose()
                            pose.position.x = point_3d[0]
                            pose.position.y = point_3d[1]
                            pose.position.z = point_3d[2]
                            pose.orientation.w = 1.0
                            pose_array.poses.append(pose)
                
                # Store last result for frame skipping
                self.last_detection_result = (detection_2d_array, pose_array)
                
                # Publish results
                self.detection_2d_pub.publish(detection_2d_array)
                self.detection_3d_pub.publish(pose_array)
                
                # Optional image visualization
                if self.publish_image and len(results) > 0:
                    annotated_frame = results[0].plot()
                    if self.resize_factor != 1.0:
                        h, w = rgb_image.shape[:2]
                        annotated_frame = cv2.resize(annotated_frame, (w, h))
                    
                    # Add performance info
                    cv2.putText(annotated_frame, f'FPS Boost: Process every {self.process_every_n_frames}th frame', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'3D Detections: {len(pose_array.poses)}', 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                    annotated_msg.header = rgb_msg.header
                    self.annotated_image_pub.publish(annotated_msg)
                    
        except Exception as e:
            self.get_logger().error(f'Error in processing: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Detector3DNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

