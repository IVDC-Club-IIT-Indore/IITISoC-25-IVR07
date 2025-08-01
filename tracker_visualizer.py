#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import message_filters

class YOLOv8Detector3DNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector_3d')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('rgb_camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('depth_camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('image_scale', 0.5)
        self.declare_parameter('depth_scale', 0.001)  # RealSense depth scale
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        rgb_camera_info_topic = self.get_parameter('rgb_camera_info_topic').get_parameter_value().string_value
        depth_camera_info_topic = self.get_parameter('depth_camera_info_topic').get_parameter_value().string_value
        self.publish_image = self.get_parameter('publish_image').get_parameter_value().bool_value
        self.image_scale = self.get_parameter('image_scale').get_parameter_value().double_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        
        # Initialize YOLO model
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.get_logger().info(f'YOLOv8 3D model loaded: {model_path} on {self.device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            return
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera info storage
        self.rgb_camera_info = None
        self.depth_camera_info = None
        
        # Frame counter
        self.frame_count = 0
        
        # Synchronized subscribers for RGB + Depth
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        # Time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)
        
        # Camera info subscribers
        self.rgb_camera_info_sub = self.create_subscription(
            CameraInfo, rgb_camera_info_topic, self.rgb_camera_info_callback, 1)
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo, depth_camera_info_topic, self.depth_camera_info_callback, 1)
        
        # Publishers
        self.detection_2d_pub = self.create_publisher(Detection2DArray, '/yolov8/detections_2d', 1)
        self.detection_3d_pub = self.create_publisher(PoseArray, '/yolov8/detections_3d', 1)
        self.detection_markers_pub = self.create_publisher(MarkerArray, '/yolov8/detection_markers', 1)
        
        if self.publish_image:
            self.annotated_image_pub = self.create_publisher(Image, '/yolov8/detection_image_3d', 1)
        
        self.get_logger().info('YOLOv8 3D Detector with RViz markers initialized')

    def rgb_camera_info_callback(self, msg):
        if self.rgb_camera_info is None:
            self.rgb_camera_info = msg
            self.get_logger().info('RGB Camera info received')

    def depth_camera_info_callback(self, msg):
        if self.depth_camera_info is None:
            self.depth_camera_info = msg
            self.get_logger().info('Depth Camera info received')

    def pixel_to_3d(self, u, v, depth, camera_info):
        """Convert pixel coordinates and depth to 3D world coordinates"""
        if depth <= 0 or depth > 10000 or camera_info is None:
            return None
        
        # Camera intrinsics
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        
        # Convert depth to meters
        z = depth * self.depth_scale
        if z < 0.1 or z > 10.0:
            return None
        
        # Project to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return (x, y, z)

    def get_object_depth(self, depth_image, bbox):
        """Get robust depth measurement from bounding box region"""
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(depth_image.shape[1], int(bbox[2]))
        y2 = min(depth_image.shape[0], int(bbox[3]))
        
        # Extract depth region
        depth_roi = depth_image[y1:y2, x1:x2]
        
        # Filter valid depths
        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < 10000)]
        
        if len(valid_depths) > 0:
            return np.median(valid_depths)  # Use median for robustness
        else:
            return None

    def create_3d_bounding_box_marker(self, center_3d, size_3d, header, marker_id, color, class_name, confidence):
        """Create 3D bounding box marker for RViz"""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "camera_color_optical_frame"
        marker.ns = "3d_detection_boxes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        marker.pose.position.x = center_3d[0]
        marker.pose.position.y = center_3d[1]
        marker.pose.position.z = center_3d[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.02  # Line width
        marker.color = color
        marker.lifetime.sec = 1
        
        # Create 3D bounding box vertices
        sx, sy, sz = size_3d[0]/2, size_3d[1]/2, size_3d[2]/2
        
        # 8 vertices of a cube
        vertices = [
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],  # Bottom face
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]       # Top face
        ]
        
        # 12 edges of the cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Create line list for edges
        for edge in edges:
            start_point = Point()
            start_point.x = vertices[edge[0]][0]
            start_point.y = vertices[edge[0]][1]
            start_point.z = vertices[edge[0]][2]
            marker.points.append(start_point)
            
            end_point = Point()
            end_point.x = vertices[edge[1]][0]
            end_point.y = vertices[edge[1]][1]
            end_point.z = vertices[edge[1]][2]
            marker.points.append(end_point)
        
        return marker

    def create_detection_markers(self, detections_3d, detections_2d, header):
        """Create 3D markers for RViz visualization"""
        marker_array = MarkerArray()
        marker_id = 0
        
        for i, (pose_3d, det_2d) in enumerate(zip(detections_3d, detections_2d)):
            # Get object information
            distance = pose_3d.position.z
            bbox_2d = det_2d.bbox
            class_name = det_2d.id
            confidence = det_2d.results[0].hypothesis.score if det_2d.results else 0.0
            
            # Estimate 3D size from 2D bbox and distance
            if self.rgb_camera_info:
                fx = self.rgb_camera_info.k[0]
                fy = self.rgb_camera_info.k[4]
                
                world_width = (bbox_2d.size_x * distance) / fx
                world_height = (bbox_2d.size_y * distance) / fy
                world_depth = min(world_width, world_height) * 0.6  # Estimated depth
            else:
                # Default sizes based on object class
                if 'person' in class_name.lower():
                    world_width, world_height, world_depth = 0.5, 1.7, 0.3
                elif 'car' in class_name.lower():
                    world_width, world_height, world_depth = 4.5, 1.8, 1.8
                else:
                    world_width, world_height, world_depth = 0.4, 0.6, 0.3
            
            # Object center sphere
            sphere_marker = Marker()
            sphere_marker.header = header
            sphere_marker.header.frame_id = "camera_color_optical_frame"
            sphere_marker.ns = "detection_centers"
            sphere_marker.id = marker_id
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            
            sphere_marker.pose.position.x = pose_3d.position.x
            sphere_marker.pose.position.y = pose_3d.position.y
            sphere_marker.pose.position.z = pose_3d.position.z
            sphere_marker.pose.orientation.w = 1.0
            
            size = max(0.05, min(0.15, 0.3 / distance))
            sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = size
            sphere_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            sphere_marker.lifetime.sec = 1
            
            marker_array.markers.append(sphere_marker)
            marker_id += 1
            
            # 3D Bounding Box
            center_3d = [pose_3d.position.x, pose_3d.position.y, pose_3d.position.z]
            size_3d = [world_width, world_height, world_depth]
            bbox_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            
            bbox_marker = self.create_3d_bounding_box_marker(
                center_3d, size_3d, header, marker_id, bbox_color, class_name, confidence)
            marker_array.markers.append(bbox_marker)
            marker_id += 1
            
            # Text marker with information
            text_marker = Marker()
            text_marker.header = header
            text_marker.header.frame_id = "camera_color_optical_frame"
            text_marker.ns = "detection_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = pose_3d.position.x
            text_marker.pose.position.y = pose_3d.position.y - 0.1
            text_marker.pose.position.z = pose_3d.position.z + world_height/2 + 0.2
            text_marker.pose.orientation.w = 1.0
            
            text_marker.text = f"{class_name} ({confidence:.2f})\n({pose_3d.position.x:.2f}, {pose_3d.position.y:.2f}, {pose_3d.position.z:.2f})\nSize: {world_width:.2f}x{world_height:.2f}x{world_depth:.2f}"
            text_marker.scale.z = 0.1
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.lifetime.sec = 1
            
            marker_array.markers.append(text_marker)
            marker_id += 1
        
        return marker_array

    def synchronized_callback(self, rgb_msg, depth_msg):
        """Process synchronized RGB and depth images"""
        try:
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % 2 != 0:
                return
            
            if self.rgb_camera_info is None or self.depth_camera_info is None:
                self.get_logger().warn('Camera info not available yet')
                return
            
            # Convert images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            
            # Resize RGB for faster processing
            if self.image_scale != 1.0:
                height, width = rgb_image.shape[:2]
                new_height = int(height * self.image_scale)
                new_width = int(width * self.image_scale)
                rgb_resized = cv2.resize(rgb_image, (new_width, new_height))
            else:
                rgb_resized = rgb_image
            
            # YOLO detection
            results = self.model(
                rgb_resized,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Prepare messages
            detection_2d_array = Detection2DArray()
            detection_2d_array.header = rgb_msg.header
            
            pose_array = PoseArray()
            pose_array.header = rgb_msg.header
            pose_array.header.frame_id = "camera_color_optical_frame"
            
            # Visualization image
            annotated_image = rgb_image.copy()
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                scale_factor = 1.0 / self.image_scale
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    # Scale back to original coordinates
                    scaled_box = box * scale_factor
                    
                    # 2D Detection
                    detection_2d = Detection2D()
                    detection_2d.header = rgb_msg.header
                    
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = float(conf)
                    detection_2d.results = [hypothesis]
                    
                    bbox_2d = BoundingBox2D()
                    center_x = (scaled_box[0] + scaled_box[2]) / 2
                    center_y = (scaled_box[1] + scaled_box[3]) / 2
                    width = scaled_box[2] - scaled_box[0]
                    height = scaled_box[3] - scaled_box[1]
                    
                    bbox_2d.center.position.x = float(center_x)
                    bbox_2d.center.position.y = float(center_y)
                    bbox_2d.size_x = float(width)
                    bbox_2d.size_y = float(height)
                    detection_2d.bbox = bbox_2d
                    detection_2d.id = self.model.names[class_id]
                    
                    detection_2d_array.detections.append(detection_2d)
                    
                    # 3D Position calculation
                    object_depth = self.get_object_depth(depth_image, scaled_box)
                    
                    if object_depth is not None:
                        coords_3d = self.pixel_to_3d(center_x, center_y, object_depth, self.depth_camera_info)
                        
                        if coords_3d is not None:
                            x, y, z = coords_3d
                            
                            # Create 3D pose
                            pose = Pose()
                            pose.position.x = x
                            pose.position.y = y
                            pose.position.z = z
                            pose.orientation.w = 1.0
                            
                            pose_array.poses.append(pose)
                            
                            # Enhanced image annotation
                            cv2.rectangle(annotated_image, 
                                        (int(scaled_box[0]), int(scaled_box[1])), 
                                        (int(scaled_box[2]), int(scaled_box[3])), 
                                        (0, 255, 0), 3)
                            
                            # Multi-line labels
                            labels = [
                                f"{self.model.names[class_id]}: {conf:.2f}",
                                f"3D: ({x:.2f}, {y:.2f}, {z:.2f})",
                                f"Depth: {z:.2f}m"
                            ]
                            
                            for i, label in enumerate(labels):
                                y_offset = int(scaled_box[1]) - 50 + (i * 20)
                                color = (0, 255, 0) if i == 0 else (255, 255, 0) if i == 1 else (0, 255, 255)
                                cv2.putText(annotated_image, label, 
                                          (int(scaled_box[0]), y_offset), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add status info
            cv2.rectangle(annotated_image, (10, 10), (500, 100), (0, 0, 0), -1)
            cv2.rectangle(annotated_image, (10, 10), (500, 100), (255, 255, 255), 2)
            
            status_lines = [
                'YOLOv8 3D DETECTION WITH RVIZ MARKERS',
                f'Objects Detected: {len(pose_array.poses)}',
                'RViz: Add MarkerArray -> /yolov8/detection_markers'
            ]
            
            for i, line in enumerate(status_lines):
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                cv2.putText(annotated_image, line, (15, 35 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Publish all results
            self.detection_2d_pub.publish(detection_2d_array)
            self.detection_3d_pub.publish(pose_array)
            
            # Create and publish 3D markers
            if len(pose_array.poses) > 0:
                markers = self.create_detection_markers(
                    pose_array.poses, detection_2d_array.detections, rgb_msg.header)
                self.detection_markers_pub.publish(markers)
            
            # Publish annotated image
            if self.publish_image:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
                annotated_msg.header = rgb_msg.header
                self.annotated_image_pub.publish(annotated_msg)
            
            self.get_logger().info(f'Published {len(pose_array.poses)} 3D detections with RViz markers')
            
        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')

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

