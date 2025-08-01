#!/usr/bin/env python3
"""
FairMOT Obstacle Tracker Node with Path Tracing
Complete ROS2 implementation for dynamic obstacle detection and tracking
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose, PoseArray, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from collections import defaultdict, deque
import sys
import os

# ============================================================================
# KALMAN FILTER IMPLEMENTATION
# ============================================================================

class KalmanFilter:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor, np.dot(self._update_mat, covariance.T)).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

# ============================================================================
# TRACKING UTILITIES
# ============================================================================

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost_matrix_copy = cost_matrix.copy()
    for _ in range(min(cost_matrix.shape)):
        if cost_matrix_copy.size == 0:
            break
        row_ind, col_ind = np.unravel_index(np.argmin(cost_matrix_copy), cost_matrix_copy.shape)
        if cost_matrix_copy[row_ind, col_ind] < thresh:
            matches.append([row_ind, col_ind])
            cost_matrix_copy[row_ind, :] = np.inf
            cost_matrix_copy[:, col_ind] = np.inf
        else:
            break
    
    matched_rows = set([m[0] for m in matches])
    matched_cols = set([m[1] for m in matches])
    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matched_cols]
    
    return np.array(matches), unmatched_a, unmatched_b

def bbox_ious(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / np.maximum(union, 1e-6)

def iou_distance(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )
    return 1 - ious

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

# ============================================================================
# TRACK CLASS
# ============================================================================

class STrack:
    shared_kalman = KalmanFilter()
    count = 0

    def __init__(self, tlwh, score, temp_feat=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.smooth_feat = None
        self.curr_feat = None
        self.features = []
        self.alpha = 0.9
        self._state = 0
        
        # Path tracing
        self.path_3d = deque(maxlen=100)  # Store 3D positions
        self.path_2d = deque(maxlen=100)  # Store 2D positions
        self.path_timestamps = deque(maxlen=100)

        if temp_feat is not None:
            self.update_features(temp_feat)

    def update_features(self, feat):
        if feat is None:
            return
        feat = feat / np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat = self.smooth_feat / np.linalg.norm(self.smooth_feat)

    def add_path_point(self, pos_3d, pos_2d, timestamp):
        """Add a point to the path history."""
        self.path_3d.append(pos_3d)
        self.path_2d.append(pos_2d)
        self.path_timestamps.append(timestamp)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != 1:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != 1:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = 1
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = 1
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = 1
        self.is_activated = True
        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    def mark_lost(self):
        self.state = 2

    def mark_removed(self):
        self.state = 3

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def next_id():
        STrack.count += 1
        return STrack.count

    @property
    def end_frame(self):
        return self.frame_id

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

# ============================================================================
# DETECTOR CLASSES
# ============================================================================

class SimpleDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        self.min_contour_area = 500
        self.max_contour_area = 50000
        
    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                x1, y1, x2, y2 = x, y, x + w, y + h
                confidence = min(area / self.max_contour_area, 1.0)
                detections.append([x1, y1, x2, y2, confidence])
        
        return np.array(detections) if detections else np.empty((0, 5))

class YOLODetector:
    def __init__(self):
        self.use_yolo = False
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()
            self.use_yolo = True
            print("✅ YOLOv5 loaded successfully")
        except Exception as e:
            self.simple_detector = SimpleDetector()
            self.use_yolo = False
            print(f"⚠️ YOLOv5 not available, using simple motion detection: {e}")
    
    def detect(self, frame):
        if self.use_yolo:
            try:
                results = self.model(frame)
                detections = results.pandas().xyxy[0].values
                person_vehicle_classes = [0, 2, 3, 5, 7]
                filtered_detections = []
                for det in detections:
                    if int(det[5]) in person_vehicle_classes and det[4] > 0.3:
                        filtered_detections.append([det[0], det[1], det[2], det[3], det[4]])
                return np.array(filtered_detections) if filtered_detections else np.empty((0, 5))
            except Exception as e:
                print(f"YOLO detection failed: {e}")
                return np.empty((0, 5))
        else:
            return self.simple_detector.detect(frame)

# ============================================================================
# FAIRMOT TRACKER
# ============================================================================

class FairMOTTracker:
    def __init__(self, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results) == 0:
            for track in self.tracked_stracks:
                if not track.is_activated:
                    track.mark_removed()
                    removed_stracks.append(track)
                else:
                    track.mark_lost()
                    lost_stracks.append(track)
            
            self.tracked_stracks = []
            self.lost_stracks.extend(lost_stracks)
            self.removed_stracks.extend(removed_stracks)
            return []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / scale

        remain_inds = scores > 0.1
        inds_low = scores > 0.1
        inds_high = scores < 0.7
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.8)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == 1:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, False)
                refind_stracks.append(track)

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != 2:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < 0.6:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks

# ============================================================================
# ROS2 TRACKER NODE
# ============================================================================

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('color_image_topic', '/camera/camera/color/image_raw'),
                ('depth_image_topic', '/camera/camera/depth/image_rect_raw'),
                ('camera_info_topic', '/camera/camera/color/camera_info'),
                ('frame_rate', 30.0),
                ('confidence_threshold', 0.3),
                ('max_path_length', 100),
                ('path_update_interval', 5),
                ('show_path_trails', True),
                ('trail_length', 20),
                ('publish_debug_image', True),
                ('publish_path_markers', True),
                ('marker_lifetime', 2.0),
            ]
        )
        
        # Get parameters
        self.color_topic = self.get_parameter('color_image_topic').value
        self.depth_topic = self.get_parameter('depth_image_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.max_path_length = self.get_parameter('max_path_length').value
        self.path_update_interval = self.get_parameter('path_update_interval').value
        self.show_trails = self.get_parameter('show_path_trails').value
        self.trail_length = self.get_parameter('trail_length').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        self.publish_markers = self.get_parameter('publish_path_markers').value
        self.marker_lifetime = self.get_parameter('marker_lifetime').value
        
        # Initialize components
        self.bridge = CvBridge()
        self.tracker = FairMOTTracker(frame_rate=self.frame_rate)
        self.detector = YOLODetector()
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, self.color_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_callback, 10)
        
        # Publishers
        self.poses_pub = self.create_publisher(PoseArray, '/tracked_obstacles', 10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/fairmot_tracker/debug_image', 10)
        if self.publish_markers:
            self.marker_pub = self.create_publisher(MarkerArray, '/fairmot_tracker/path_markers', 10)
        
        # Internal state
        self.camera_info = None
        self.depth_image = None
        self.frame_count = 0
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        
        # Path tracing
        self.track_paths = defaultdict(lambda: {'3d': deque(maxlen=self.max_path_length), 
                                               '2d': deque(maxlen=self.max_path_length),
                                               'timestamps': deque(maxlen=self.max_path_length)})
        
        self.get_logger().info("🚀 FairMOT Tracker Node with Path Tracing initialized!")
        self.get_logger().info(f"📸 Color topic: {self.color_topic}")
        self.get_logger().info(f"📊 Depth topic: {self.depth_topic}")
        self.get_logger().info(f"ℹ️ Info topic: {self.info_topic}")
        self.get_logger().info(f"🛤️ Path tracing: {self.max_path_length} points, update every {self.path_update_interval} frames")

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_count += 1
            current_time = self.get_clock().now()
            
            # Detect objects
            detections = self.detector.detect(cv_image)
            
            if len(detections) > 0:
                img_info = (cv_image.shape[0], cv_image.shape[1])
                img_size = (640, 480)
                
                # Update tracker
                online_targets = self.tracker.update(detections, img_info, img_size)
                
                # Update path tracking
                self.update_paths(online_targets, current_time)
                
                # Publish results
                self.publish_poses(online_targets)
                if self.publish_markers:
                    self.publish_path_markers()
                
                # Create and publish debug image
                if self.publish_debug:
                    debug_image = self.draw_tracking_visualization(cv_image, online_targets)
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                    debug_msg.header = msg.header
                    self.debug_pub.publish(debug_msg)
                
                # Log periodically
                if self.frame_count % 60 == 0:
                    self.get_logger().info(f"Frame {self.frame_count}: {len(online_targets)} active tracks")
                
            else:
                # No detections
                if self.publish_debug:
                    debug_image = cv_image.copy()
                    cv2.putText(debug_image, "No detections", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                    debug_msg.header = msg.header
                    self.debug_pub.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def update_paths(self, tracks, timestamp):
        """Update path history for tracks."""
        active_track_ids = set()
        
        for track in tracks:
            track_id = track.track_id
            active_track_ids.add(track_id)
            
            # Update path every N frames
            if self.frame_count % self.path_update_interval == 0:
                bbox = track.tlbr
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # 2D position
                pos_2d = (center_x, center_y)
                
                # 3D position
                pos_3d = self.get_3d_position(center_x, center_y)
                
                # Store in track's path history
                track.add_path_point(pos_3d, pos_2d, timestamp)
                
                # Also store in global path dictionary
                self.track_paths[track_id]['3d'].append(pos_3d)
                self.track_paths[track_id]['2d'].append(pos_2d)
                self.track_paths[track_id]['timestamps'].append(timestamp)
        
        # Clean up old tracks
        tracks_to_remove = []
        for track_id in self.track_paths:
            if track_id not in active_track_ids:
                # Keep paths for a while after track disappears
                if len(self.track_paths[track_id]['timestamps']) > 0:
                    last_time = self.track_paths[track_id]['timestamps'][-1]
                    time_since_last = (timestamp.nanoseconds - last_time.nanoseconds) / 1e9
                    if time_since_last > 5.0:  # Remove after 5 seconds
                        tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_paths[track_id]

    def get_3d_position(self, x, y):
        """Convert 2D pixel to 3D position using depth."""
        if self.depth_image is None or self.camera_info is None:
            return (float(x), float(y), 0.0)
        
        try:
            h, w = self.depth_image.shape
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            depth_value = self.depth_image[y, x]
            if depth_value > 0 and depth_value < 10000:
                depth = depth_value / 1000.0  # Convert mm to meters
                
                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]
                
                x_3d = (x - cx) * depth / fx
                y_3d = (y - cy) * depth / fy
                z_3d = depth
                
                return (z_3d, -x_3d, -y_3d)  # Camera frame coordinates
        except Exception as e:
            self.get_logger().debug(f"Error getting 3D position: {e}")
        
        return (float(x), float(y), 0.0)

    def draw_tracking_visualization(self, image, tracks):
        """Draw comprehensive tracking visualization with paths."""
        debug_image = image.copy()
        
        # Color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0)
        ]
        
        # Draw path trails first (so they appear behind bounding boxes)
        if self.show_trails:
            for track in tracks:
                if len(track.path_2d) > 1:
                    color = colors[track.track_id % len(colors)]
                    path_points = list(track.path_2d)
                    
                    # Draw path as connected lines with fading effect
                    for i in range(1, len(path_points)):
                        if i < len(path_points):
                            pt1 = (int(path_points[i-1][0]), int(path_points[i-1][1]))
                            pt2 = (int(path_points[i][0]), int(path_points[i][1]))
                            
                            # Fade older points
                            alpha = i / len(path_points)
                            faded_color = tuple(int(c * alpha) for c in color)
                            
                            cv2.line(debug_image, pt1, pt2, faded_color, 2)
                            cv2.circle(debug_image, pt2, 3, faded_color, -1)
        
        # Draw current tracking boxes and info
        for track in tracks:
            bbox = track.tlbr
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            color = colors[track.track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            
            # Track information
            track_id = track.track_id
            score = track.score
            path_length = len(track.path_2d)
            
            # Multi-line label
            labels = [
                f"ID:{track_id}",
                f"Score:{score:.2f}",
                f"Path:{path_length} pts"
            ]
            
            # Draw label background
            label_height = 20 * len(labels) + 5
            cv2.rectangle(debug_image, (x1, y1 - label_height), 
                         (x1 + 120, y1), color, -1)
            
            # Draw labels
            for i, label in enumerate(labels):
                cv2.putText(debug_image, label, (x1 + 2, y1 - label_height + 15 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(debug_image, (center_x, center_y), 5, color, -1)
            
            # Distance if available
            if self.depth_image is not None and self.camera_info is not None:
                pos_3d = self.get_3d_position(center_x, center_y)
                if pos_3d[2] != 0:
                    distance = math.sqrt(pos_3d[0]**2 + pos_3d[1]**2 + pos_3d[2]**2)
                    dist_label = f"{distance:.2f}m"
                    cv2.putText(debug_image, dist_label, (x1, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Frame information
        current_time = self.get_clock().now().nanoseconds / 1e9
        fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time
        
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Active Tracks: {len(tracks)}",
            f"FPS: {fps:.1f}",
            f"Total Paths: {len(self.track_paths)}"
        ]
        
        # Draw info background
        info_height = 25 * len(info_lines)
        cv2.rectangle(debug_image, (10, 10), (250, info_height + 10), (0, 0, 0), -1)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            cv2.putText(debug_image, line, (15, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image

    def publish_poses(self, tracks):
        """Publish tracked obstacle poses."""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "camera_color_optical_frame"
        
        for track in tracks:
            bbox = track.tlbr
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            pose = Pose()
            pos_3d = self.get_3d_position(center_x, center_y)
            
            pose.position.x = pos_3d[0]
            pose.position.y = pos_3d[1]
            pose.position.z = pos_3d[2]
            pose.orientation.w = 1.0
            
            pose_array.poses.append(pose)
        
        self.poses_pub.publish(pose_array)

    def publish_path_markers(self):
        """Publish path visualization markers."""
        marker_array = MarkerArray()
        marker_id = 0
        
        colors_rgb = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.5, 0.0, 0.5), (1.0, 0.65, 0.0),
            (0.0, 0.5, 0.5), (0.5, 0.5, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0)
        ]
        
        current_time = self.get_clock().now()
        
        for track_id, path_data in self.track_paths.items():
            if len(path_data['3d']) < 2:
                continue
            
            # Create line strip marker for path
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = current_time.to_msg()
            marker.ns = "track_paths"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set color
            color_idx = track_id % len(colors_rgb)
            marker.color.r = colors_rgb[color_idx][0]
            marker.color.g = colors_rgb[color_idx][1]
            marker.color.b = colors_rgb[color_idx][2]
            marker.color.a = 0.8
            
            # Set scale
            marker.scale.x = 0.05  # Line width
            
            # Set lifetime
            marker.lifetime.sec = int(self.marker_lifetime)
            marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            # Add points
            for pos_3d in path_data['3d']:
                point = Point()
                point.x = pos_3d[0]
                point.y = pos_3d[1]
                point.z = pos_3d[2]
                marker.points.append(point)
            
            marker_array.markers.append(marker)
            marker_id += 1
            
            # Create sphere marker for current position
            if len(path_data['3d']) > 0:
                sphere_marker = Marker()
                sphere_marker.header.frame_id = "camera_color_optical_frame"
                sphere_marker.header.stamp = current_time.to_msg()
                sphere_marker.ns = "current_positions"
                sphere_marker.id = marker_id
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                
                # Position at latest point
                latest_pos = path_data['3d'][-1]
                sphere_marker.pose.position.x = latest_pos[0]
                sphere_marker.pose.position.y = latest_pos[1]
                sphere_marker.pose.position.z = latest_pos[2]
                sphere_marker.pose.orientation.w = 1.0
                
                # Set color (brighter than path)
                sphere_marker.color.r = colors_rgb[color_idx][0]
                sphere_marker.color.g = colors_rgb[color_idx][1]
                sphere_marker.color.b = colors_rgb[color_idx][2]
                sphere_marker.color.a = 1.0
                
                # Set scale
                sphere_marker.scale.x = 0.1
                sphere_marker.scale.y = 0.1
                sphere_marker.scale.z = 0.1
                
                # Set lifetime
                sphere_marker.lifetime.sec = int(self.marker_lifetime)
                sphere_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
                
                marker_array.markers.append(sphere_marker)
                marker_id += 1
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    print("🚀 Starting FairMOT Tracker Node with Path Tracing")
    print("=" * 60)
    
    rclpy.init(args=args)
    node = TrackerNode()
    
    try:
        print("✅ Node running... Press Ctrl+C to stop")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Stopping tracker...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Tracker stopped successfully")

if __name__ == '__main__':
    main()

