#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.optimize import linear_sum_assignment
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import random

class SimpleObjectTracker:
    """Enhanced object tracker with visualization"""
    
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.class_name = detection.id
        self.class_id = detection.results[0].hypothesis.class_id if detection.results else "unknown"
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        
        # Store current position and velocity
        bbox = detection.bbox
        self.center_x = bbox.center.position.x
        self.center_y = bbox.center.position.y
        self.width = bbox.size_x
        self.height = bbox.size_y
        
        # Velocity tracking
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # History for trajectory visualization
        self.track_history = [(self.center_x, self.center_y)]
        self.max_history = 30  # Keep last 30 positions
        
        # Assign unique color for this track
        random.seed(track_id)  # Ensure consistent color for same ID
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        
        # History for velocity calculation
        self.prev_center_x = self.center_x
        self.prev_center_y = self.center_y

    def update(self, detection):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        bbox = detection.bbox
        new_center_x = bbox.center.position.x
        new_center_y = bbox.center.position.y
        
        # Update velocity
        self.vel_x = new_center_x - self.center_x
        self.vel_y = new_center_y - self.center_y
        
        # Update position
        self.prev_center_x = self.center_x
        self.prev_center_y = self.center_y
        self.center_x = new_center_x
        self.center_y = new_center_y
        self.width = bbox.size_x
        self.height = bbox.size_y
        
        # Add to track history
        self.track_history.append((self.center_x, self.center_y))
        if len(self.track_history) > self.max_history:
            self.track_history.pop(0)

    def predict(self):
        """Predict next state using simple linear motion"""
        if self.time_since_update > 0:
            # Apply velocity prediction with damping
            self.center_x += self.vel_x * 0.3
            self.center_y += self.vel_y * 0.3
            
            # Add predicted position to history
            self.track_history.append((self.center_x, self.center_y))
            if len(self.track_history) > self.max_history:
                self.track_history.pop(0)
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_state()

    def get_state(self):
        """Get current state as detection"""
        detection = Detection2D()
        detection.id = f"{self.class_name}_ID{self.track_id}"
        
        # Object hypothesis
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = self.class_id
        hypothesis.hypothesis.score = max(0.3, 0.9 - (self.time_since_update * 0.1))  # Decay confidence
        detection.results = [hypothesis]
        
        # Bounding box
        bbox = BoundingBox2D()
        bbox.center.position.x = float(self.center_x)
        bbox.center.position.y = float(self.center_y)
        bbox.center.theta = 0.0
        bbox.size_x = float(self.width)
        bbox.size_y = float(self.height)
        detection.bbox = bbox
        
        return detection

class YOLOv8TrackerNode(Node):
    def __init__(self):
        super().__init__('yolov8_tracker')
        
        # Parameters
        self.declare_parameter('max_disappeared', 10)
        self.declare_parameter('max_distance', 100.0)
        self.declare_parameter('input_image_topic', '/camera/camera/color/image_raw')
        
        self.max_disappeared = self.get_parameter('max_disappeared').get_parameter_value().integer_value
        self.max_distance = self.get_parameter('max_distance').get_parameter_value().double_value
        input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        
        # Tracker state
        self.trackers = []
        self.next_id = 0
        
        # CV Bridge for image processing
        self.bridge = CvBridge()
        self.current_image = None
        
        # Subscriptions
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/yolov8/detections',
            self.detection_callback,
            1
        )
        
        self.image_sub = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            1
        )
        
        # Publishers
        self.tracked_objects_pub = self.create_publisher(
            Detection2DArray,
            '/yolov8/tracked_objects',
            1
        )
        
        self.tracking_image_pub = self.create_publisher(
            Image,
            '/yolov8/tracking_image',
            1
        )
        
        self.get_logger().info('YOLOv8 Tracker Node initialized')

    def image_callback(self, msg):
        """Store current image for visualization"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

    def calculate_distance(self, det1, det2):
        """Calculate Euclidean distance between two detections"""
        center1_x = det1.bbox.center.position.x
        center1_y = det1.bbox.center.position.y
        center2_x = det2.bbox.center.position.x
        center2_y = det2.bbox.center.position.y
        
        dx = center1_x - center2_x
        dy = center1_y - center2_y
        return np.sqrt(dx*dx + dy*dy)

    def draw_tracking_visualization(self, image):
        """Draw enhanced tracking visualization"""
        if image is None:
            return None
            
        vis_image = image.copy()
        
        for tracker in self.trackers:
            if tracker.time_since_update < 3:  # Show recent tracks
                # Draw bounding box
                x = int(tracker.center_x - tracker.width / 2)
                y = int(tracker.center_y - tracker.height / 2)
                w = int(tracker.width)
                h = int(tracker.height)
                
                # Different colors for different track states
                if tracker.time_since_update == 0:
                    # Active track - bright color
                    color = tracker.color
                    thickness = 3
                else:
                    # Predicted track - dimmer color
                    color = tuple(int(c * 0.5) for c in tracker.color)
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
                
                # Draw track ID and class
                label = f"ID{tracker.track_id}: {tracker.class_name}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for text
                cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), color, -1)
                
                # Text
                cv2.putText(vis_image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw trajectory
                if len(tracker.track_history) > 1:
                    points = np.array([(int(x), int(y)) for x, y in tracker.track_history], 
                                    dtype=np.int32)
                    for i in range(1, len(points)):
                        # Fade the trail
                        alpha = i / len(points)
                        trail_color = tuple(int(c * alpha) for c in tracker.color)
                        cv2.line(vis_image, tuple(points[i-1]), tuple(points[i]), 
                                trail_color, max(1, int(3 * alpha)))
                
                # Draw center point
                cv2.circle(vis_image, (int(tracker.center_x), int(tracker.center_y)), 
                          5, color, -1)
                
                # Draw velocity vector
                if abs(tracker.vel_x) > 1 or abs(tracker.vel_y) > 1:
                    end_x = int(tracker.center_x + tracker.vel_x * 10)
                    end_y = int(tracker.center_y + tracker.vel_y * 10)
                    cv2.arrowedLine(vis_image, 
                                   (int(tracker.center_x), int(tracker.center_y)),
                                   (end_x, end_y), color, 2)
        
        # Add tracking statistics
        active_tracks = len([t for t in self.trackers if t.time_since_update == 0])
        total_tracks = len(self.trackers)
        
        cv2.putText(vis_image, f'Active Tracks: {active_tracks}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis_image, f'Total Tracks: {total_tracks}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis_image, 'TRACKING MODE', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return vis_image

    def detection_callback(self, msg):
        """Process detections and update tracks"""
        detections = msg.detections
        
        # Predict all existing trackers
        predicted_trackers = []
        for tracker in self.trackers:
            pred_det = tracker.predict()
            predicted_trackers.append((tracker, pred_det))
        
        # Calculate cost matrix for assignment
        if len(detections) > 0 and len(predicted_trackers) > 0:
            cost_matrix = np.zeros((len(detections), len(predicted_trackers)))
            
            for i, detection in enumerate(detections):
                for j, (tracker, pred_det) in enumerate(predicted_trackers):
                    # Only match same class
                    if detection.id == tracker.class_name:
                        distance = self.calculate_distance(detection, pred_det)
                        cost_matrix[i, j] = distance
                    else:
                        cost_matrix[i, j] = 1e6
            
            # Hungarian algorithm for optimal assignment
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
            
            # Update matched trackers
            matched_detections = set()
            matched_trackers = set()
            
            for det_idx, track_idx in zip(det_indices, track_indices):
                if cost_matrix[det_idx, track_idx] < self.max_distance:
                    tracker = predicted_trackers[track_idx][0]
                    tracker.update(detections[det_idx])
                    matched_detections.add(det_idx)
                    matched_trackers.add(track_idx)
            
            # Create new trackers for unmatched detections
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    new_tracker = SimpleObjectTracker(detection, self.next_id)
                    self.trackers.append(new_tracker)
                    self.next_id += 1
                    self.get_logger().info(f'New track created: ID{self.next_id-1} ({detection.id})')
            
            # Remove old trackers
            old_count = len(self.trackers)
            self.trackers = [
                tracker for j, (tracker, _) in enumerate(predicted_trackers)
                if j in matched_trackers or tracker.time_since_update < self.max_disappeared
            ] + [tracker for tracker in self.trackers if tracker not in [t[0] for t in predicted_trackers]]
            
            if len(self.trackers) < old_count:
                self.get_logger().info(f'Removed {old_count - len(self.trackers)} old tracks')
        
        elif len(detections) > 0:
            # No existing trackers, create new ones
            for detection in detections:
                new_tracker = SimpleObjectTracker(detection, self.next_id)
                self.trackers.append(new_tracker)
                self.get_logger().info(f'Initial track created: ID{self.next_id} ({detection.id})')
                self.next_id += 1
        
        # Publish tracked objects
        tracked_array = Detection2DArray()
        tracked_array.header = msg.header
        
        for tracker in self.trackers:
            if tracker.time_since_update < 2:  # Publish recent tracks
                detection = tracker.get_state()
                detection.header = msg.header
                tracked_array.detections.append(detection)
        
        self.tracked_objects_pub.publish(tracked_array)
        
        # Create and publish tracking visualization
        if self.current_image is not None:
            tracking_image = self.draw_tracking_visualization(self.current_image)
            if tracking_image is not None:
                tracking_msg = self.bridge.cv2_to_imgmsg(tracking_image, encoding='bgr8')
                tracking_msg.header = msg.header
                self.tracking_image_pub.publish(tracking_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8TrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

