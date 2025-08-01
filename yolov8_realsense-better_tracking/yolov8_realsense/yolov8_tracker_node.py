#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.optimize import linear_sum_assignment
from geometry_msgs.msg import PoseArray, Pose, Point, Vector3
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import random
import math
from collections import deque
import sensor_msgs_py.point_cloud2 as pc2

# --- Helper: project 3D world point to image pixel coordinates ---
def project_3d_to_2d(x, y, z, cam_info):
    # Only works if cam_info.K is valid and z > 0
    if z == 0 or cam_info is None:
        return None
    fx = cam_info.k[0]
    fy = cam_info.k[4]
    cx = cam_info.k[2]
    cy = cam_info.k[5]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return int(round(u)), int(round(v))

class EnhancedKalmanFilter3D:
    """Enhanced Kalman Filter with direction change handling"""

    def __init__(self, initial_pose, track_id):
        self.track_id = track_id
        self.state_dim = 6
        self.obs_dim = 3
        # State: [x, y, z, vx, vy, vz]
        self.x = np.array([
            initial_pose.position.x, initial_pose.position.y, initial_pose.position.z,
            0.0, 0.0, 0.0
        ]).reshape(6, 1)
        self.P = np.eye(6)
        self.P[:3, :3] *= 0.1
        self.P[3:, 3:] *= 0.5
        self.Q_base = np.diag([0.05, 0.05, 0.1, 0.2, 0.2, 0.3])
        self.Q = self.Q_base.copy()
        self.R = np.diag([0.1, 0.1, 0.15])
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.dt = 0.1
        self.update_transition_matrix()
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.velocity_history = []
        self.max_velocity_history = 10
        self.direction_change_threshold = 0.7
        self.predicted_trajectory = []

    def update_transition_matrix(self):
        dt = self.dt
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

    def detect_direction_change(self, new_velocity):
        if len(self.velocity_history) < 2:
            return False
        recent_vel = self.velocity_history[-1]
        dot_product = np.dot(recent_vel, new_velocity)
        norm_product = np.linalg.norm(recent_vel) * np.linalg.norm(new_velocity)
        if norm_product < 1e-6:
            return False
        cosine_sim = dot_product / norm_product
        return cosine_sim < self.direction_change_threshold

    def predict(self):
        current_vel = np.array([self.x[3, 0], self.x[4, 0], self.x[5, 0]])
        self.x = self.F @ self.x
        velocity_magnitude = np.linalg.norm(current_vel)
        self.Q = self.Q_base * 2.0 if velocity_magnitude > 0.5 else self.Q_base
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.predict_trajectory()
        return self.get_pose()

    def update(self, measurement_pose):
        z = np.array([measurement_pose.position.x, measurement_pose.position.y, measurement_pose.position.z]).reshape(3, 1)
        if self.time_since_update <= 1:
            dt_actual = max(0.05, self.time_since_update * self.dt)
            new_velocity = (z - self.x[:3]) / dt_actual
            new_velocity = new_velocity.flatten()
            direction_changed = self.detect_direction_change(new_velocity)
            if direction_changed:
                self.R = np.diag([0.05, 0.05, 0.08])
            else:
                self.R = np.diag([0.1, 0.1, 0.15])
            self.velocity_history.append(new_velocity)
            if len(self.velocity_history) > self.max_velocity_history:
                self.velocity_history.pop(0)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict_trajectory(self, steps=15):
        self.predicted_trajectory = []
        x_pred = self.x.copy()
        for i in range(steps):
            x_pred = self.F @ x_pred
            self.predicted_trajectory.append({
                'x': float(x_pred[0, 0]),
                'y': float(x_pred[1, 0]),
                'z': float(x_pred[2, 0]),
                'step': i + 1
            })

    def get_pose(self):
        pose = Pose()
        pose.position.x = float(self.x[0, 0])
        pose.position.y = float(self.x[1, 0])
        pose.position.z = float(self.x[2, 0])
        pose.orientation.w = 1.0
        return pose

    def get_velocity_3d(self):
        return float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0])

class ObjectTracker3D:
    def __init__(self, initial_pose, track_id, class_name, rgb_info=None):
        self.track_id = track_id
        self.class_name = class_name
        self.kalman_filter = EnhancedKalmanFilter3D(initial_pose, track_id)
        random.seed(track_id)
        self.color = ColorRGBA(
            r=random.uniform(0.3,1), g=random.uniform(0.3,1), b=random.uniform(0.3,1), a=0.8)
        self.track_history = []
        self.max_history = 50
        self.confidence_score = 1.0
        self.stability_counter = 0
        # Store rgb_info for projection to 2d in each frame
        self.rgb_camera_info = rgb_info
        self.current_pos_x = None
        self.current_pos_y = None
        self.width = 60  # Default (can be updated)
        self.height = 60
        self.velocity_x = 0
        self.velocity_y = 0

    def predict(self):
        pose = self.kalman_filter.predict()
        self.track_history.append({'pose': pose, 'timestamp': self.kalman_filter.age, 'predicted': True})
        if len(self.track_history) > self.max_history:
            self.track_history.pop(0)
        return pose

    def update(self, measurement_pose, cam_info=None):
        self.kalman_filter.update(measurement_pose)
        self.track_history.append({'pose': measurement_pose, 'timestamp': self.kalman_filter.age, 'predicted': False})
        if len(self.track_history) > self.max_history:
            self.track_history.pop(0)
        if cam_info is not None:
            self.rgb_camera_info = cam_info
        # Save the most recent projected 2D position
        if self.rgb_camera_info is not None:
            proj = project_3d_to_2d(
                measurement_pose.position.x, measurement_pose.position.y, measurement_pose.position.z,
                self.rgb_camera_info
            )
            if proj:
                self.current_pos_x, self.current_pos_y = proj
        # Estimate box size and velocity for drawing
        if len(self.track_history) >= 2:
            prev_p = self.track_history[-2]['pose'].position
            now_p = measurement_pose.position
            fx = (now_p.x - prev_p.x) if prev_p is not None else 0
            fy = (now_p.y - prev_p.y) if prev_p is not None else 0
            fz = (now_p.z - prev_p.z) if prev_p is not None else 0
            if self.rgb_camera_info is not None and measurement_pose.position.z > 0:
                prev_2d = project_3d_to_2d(prev_p.x, prev_p.y, prev_p.z, self.rgb_camera_info)
                now_2d = project_3d_to_2d(now_p.x, now_p.y, now_p.z, self.rgb_camera_info)
                if prev_2d and now_2d:
                    self.velocity_x = now_2d[0] - prev_2d[0]
                    self.velocity_y = now_2d[1] - prev_2d[1]
            else:
                self.velocity_x = fx
                self.velocity_y = fy

    @property
    def time_since_update(self):
        return self.kalman_filter.time_since_update

class YOLOv8Tracker3DNode(Node):
    def __init__(self):
        super().__init__('yolov8_tracker_3d')

        self.declare_parameter('max_disappeared', 25)
        self.declare_parameter('max_distance', 1.2)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('publish_point_cloud', True)
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        self.max_disappeared = self.get_parameter('max_disappeared').get_parameter_value().integer_value
        self.max_distance = self.get_parameter('max_distance').get_parameter_value().double_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.publish_point_cloud = self.get_parameter('publish_point_cloud').get_parameter_value().bool_value

        self.trackers = []
        self.next_id = 0
        self.bridge = CvBridge()
        self.current_image = None
        self.rgb_camera_info = None

        self.detection_3d_sub = self.create_subscription(
            PoseArray, '/yolov8/detections_3d', self.detection_3d_callback, 1)
        self.image_sub = self.create_subscription(
            Image, rgb_topic, self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        self.tracked_objects_pub = self.create_publisher(PoseArray, '/yolov8/tracked_objects_3d', 1)
        self.tracking_markers_pub = self.create_publisher(MarkerArray, '/yolov8/tracking_markers', 1)
        self.tracking_image_pub = self.create_publisher(Image, '/yolov8/tracking_image_3d', 1)
        if self.publish_point_cloud:
            self.point_cloud_pub = self.create_publisher(PointCloud2, '/yolov8/tracking_point_cloud', 1)

        self.visualizer_helper = TrackerVisualizationHelper()
        self.get_logger().info('Enhanced YOLOv8 3D Tracker with image and 3D path overlays')

    def camera_info_callback(self, msg):
        self.rgb_camera_info = msg

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {str(e)}')

    def calculate_3d_distance(self, pose1, pose2):
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        dz = pose1.position.z - pose2.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def detection_3d_callback(self, msg):
        detections = msg.poses
        predicted_trackers = []
        for tracker in self.trackers:
            pred_pose = tracker.predict()
            predicted_trackers.append((tracker, pred_pose))

        if len(detections) > 0 and len(predicted_trackers) > 0:
            cost_matrix = np.zeros((len(detections), len(predicted_trackers)))
            for i, detection in enumerate(detections):
                for j, (tracker, pred_pose) in enumerate(predicted_trackers):
                    distance = self.calculate_3d_distance(detection, pred_pose)
                    cost_matrix[i, j] = distance
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
            matched_detections = set()
            matched_trackers = set()
            for det_idx, track_idx in zip(det_indices, track_indices):
                if cost_matrix[det_idx, track_idx] < self.max_distance:
                    tracker = predicted_trackers[track_idx][0]
                    tracker.update(detections[det_idx], cam_info=self.rgb_camera_info)
                    matched_detections.add(det_idx)
                    matched_trackers.add(track_idx)
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    new_tracker = ObjectTracker3D(detection, self.next_id, "person", self.rgb_camera_info)
                    new_tracker.update(detection, cam_info=self.rgb_camera_info)
                    self.trackers.append(new_tracker)
                    self.next_id += 1
            self.trackers = [
                tracker for j, (tracker, _) in enumerate(predicted_trackers)
                if j in matched_trackers or tracker.time_since_update < self.max_disappeared
            ] + [tracker for tracker in self.trackers if tracker not in [t[0] for t in predicted_trackers]]
        elif len(detections) > 0:
            for detection in detections:
                new_tracker = ObjectTracker3D(detection, self.next_id, "person", self.rgb_camera_info)
                new_tracker.update(detection, cam_info=self.rgb_camera_info)
                self.trackers.append(new_tracker)
                self.next_id += 1

        # publish standard 3D pose
        tracked_array = PoseArray()
        tracked_array.header = msg.header
        tracked_array.header.frame_id = "camera_depth_optical_frame"
        for tracker in self.trackers:
            if tracker.time_since_update < 5:
                pose = tracker.kalman_filter.get_pose()
                tracked_array.poses.append(pose)
        self.tracked_objects_pub.publish(tracked_array)

        # publish rviz MarkerArray
        marker_array = self.create_enhanced_markers(msg.header)
        self.tracking_markers_pub.publish(marker_array)

        if self.publish_point_cloud:
            point_cloud = self.create_point_cloud_from_tracks(msg.header)
            if point_cloud:
                self.point_cloud_pub.publish(point_cloud)

        if self.current_image is not None:
            self.create_tracking_image(msg.header)

    def create_point_cloud_from_tracks(self, header):
        points = []
        for tracker in self.trackers:
            if tracker.time_since_update < 3:
                pose = tracker.kalman_filter.get_pose()
                points.append([pose.position.x, pose.position.y, pose.position.z,
                               float(tracker.track_id), tracker.confidence_score])
                for i, pred_point in enumerate(tracker.kalman_filter.predicted_trajectory[:10]):
                    alpha = 1.0 - (i * 0.1)
                    points.append([pred_point['x'], pred_point['y'], pred_point['z'],
                                   float(tracker.track_id), alpha * 0.5])
        if not points:
            return None
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='track_id', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='confidence', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        pc = pc2.create_cloud(header, fields, points)
        pc.header.frame_id = "camera_depth_optical_frame"
        return pc

    def create_enhanced_markers(self, header):
        marker_array = MarkerArray()
        marker_id = 0
        for tracker in self.trackers:
            if tracker.time_since_update < 8:
                current_pose = tracker.kalman_filter.get_pose()
                velocity = tracker.kalman_filter.get_velocity_3d()
                marker = Marker()
                marker.header = header
                marker.header.frame_id = "camera_depth_optical_frame"
                marker.ns = "tracked_objects"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose = current_pose
                base_size = 0.15 + (tracker.confidence_score * 0.1)
                marker.scale.x = marker.scale.y = marker.scale.z = base_size
                color = tracker.color
                color.a = 0.4 + (tracker.confidence_score * 0.6)
                marker.color = color
                marker_array.markers.append(marker)
                marker_id += 1
                text_marker = Marker()
                text_marker.header = header
                text_marker.header.frame_id = "camera_depth_optical_frame"
                text_marker.ns = "track_ids"
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose = current_pose
                text_marker.pose.position.z += 0.3
                text_marker.scale.z = 0.15
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                confidence_text = f"C:{tracker.confidence_score:.1f}"
                if tracker.time_since_update > 0:
                    confidence_text += f" P:{tracker.time_since_update}"
                text_marker.text = f"ID{tracker.track_id} {confidence_text}"
                marker_array.markers.append(text_marker)
                marker_id += 1
                vel_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
                if vel_magnitude > 0.05:
                    vel_marker = Marker()
                    vel_marker.header = header
                    vel_marker.header.frame_id = "camera_depth_optical_frame"
                    vel_marker.ns = "velocity_vectors"
                    vel_marker.id = marker_id
                    vel_marker.type = Marker.ARROW
                    vel_marker.action = Marker.ADD
                    sp = current_pose.position
                    ep = Point(
                        x=sp.x + velocity[0]*3.0,
                        y=sp.y + velocity[1]*3.0,
                        z=sp.z + velocity[2]*3.0
                    )
                    vel_marker.points = [sp, ep]
                    vel_marker.scale.x = 0.03
                    vel_marker.scale.y = 0.06
                    vel_marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.9)
                    marker_array.markers.append(vel_marker)
                    marker_id += 1
                if tracker.kalman_filter.predicted_trajectory:
                    traj_marker = Marker()
                    traj_marker.header = header
                    traj_marker.header.frame_id = "camera_depth_optical_frame"
                    traj_marker.ns = "predicted_trajectory"
                    traj_marker.id = marker_id
                    traj_marker.type = Marker.LINE_STRIP
                    traj_marker.action = Marker.ADD
                    traj_marker.scale.x = 0.02
                    traj_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.7 * tracker.confidence_score)
                    traj_marker.points.append(current_pose.position)
                    for i, pred_point in enumerate(tracker.kalman_filter.predicted_trajectory[:12]):
                        point = Point(x=pred_point['x'], y=pred_point['y'], z=pred_point['z'])
                        traj_marker.points.append(point)
                    marker_array.markers.append(traj_marker)
                    marker_id += 1
        return marker_array

    def create_tracking_image(self, header):
        if self.current_image is None or self.rgb_camera_info is None:
            return
        vis_image = self.current_image.copy()
        # For each tracker, overlay 2D projected box and trajectory
        for tracker in self.trackers:
            if tracker.current_pos_x is not None and tracker.current_pos_y is not None:
                # Draw bounding box (fixed size, can be adjusted dynamically)
                color = (
                    int(tracker.color.b * 255),
                    int(tracker.color.g * 255),
                    int(tracker.color.r * 255)
                )
                w = tracker.width
                h = tracker.height
                x = int(tracker.current_pos_x - w//2)
                y = int(tracker.current_pos_y - h//2)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
                # Draw ID
                label = f"ID{tracker.track_id}"
                cv2.putText(vis_image, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Draw velocity arrow
                if hasattr(tracker, 'velocity_x') and hasattr(tracker, 'velocity_y'):
                    pt1 = (int(tracker.current_pos_x), int(tracker.current_pos_y))
                    pt2 = (int(tracker.current_pos_x + tracker.velocity_x * 15),
                           int(tracker.current_pos_y + tracker.velocity_y * 15))
                    cv2.arrowedLine(vis_image, pt1, pt2, color, 2)
                # Draw trajectory line for history (2D projected)
                if hasattr(tracker, 'track_history') and len(tracker.track_history) > 1:
                    pts = []
                    for th in tracker.track_history:
                        pose = th['pose'].position
                        proj = project_3d_to_2d(pose.x, pose.y, pose.z, self.rgb_camera_info)
                        if proj: pts.append(proj)
                    for i in range(1, len(pts)):
                        cv2.line(vis_image, pts[i-1], pts[i], color, 2)
        # Status overlays same as before
        active_tracks = len([t for t in self.trackers if t.time_since_update == 0])
        cv2.putText(vis_image, f'Active Tracks: {active_tracks}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(vis_image, '3D TRACKING WITH IMAGE OVERLAY',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        try:
            tracking_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
            tracking_msg.header = header
            self.tracking_image_pub.publish(tracking_msg)
        except Exception as e:
            self.get_logger().error(f'Tracking image error: {str(e)}')

class TrackerVisualizationHelper:
    def __init__(self):
        self.track_histories = {}

    def update_history(self, tracker_id, xy, maxlen=30):
        if tracker_id not in self.track_histories:
            self.track_histories[tracker_id] = deque(maxlen=maxlen)
        self.track_histories[tracker_id].append(xy)

    def draw(self, img, trackers):
        for tracker in trackers:
            x = int(tracker.current_pos_x - tracker.width / 2)
            y = int(tracker.current_pos_y - tracker.height / 2)
            w = int(tracker.width)
            h = int(tracker.height)
            color = (int(tracker.color.b*255), int(tracker.color.g*255), int(tracker.color.r*255)) if hasattr(tracker.color, 'b') else (0,255,0)
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Update history and draw trajectory
            center = (int(tracker.current_pos_x), int(tracker.current_pos_y))
            self.update_history(tracker.track_id, center)
            history = self.track_histories[tracker.track_id]
            for i in range(1, len(history)):
                cv2.line(img, history[i-1], history[i], color, 2)
            # Velocity arrow
            if hasattr(tracker, 'velocity_x') and hasattr(tracker, 'velocity_y'):
                pt1 = center
                pt2 = (int(center[0] + tracker.velocity_x * 15), int(center[1] + tracker.velocity_y * 15))
                cv2.arrowedLine(img, pt1, pt2, color, 2)
        return img

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Tracker3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

