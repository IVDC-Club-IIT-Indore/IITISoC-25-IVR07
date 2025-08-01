#!/usr/bin/env python3
"""
FairMOT Visualizer Node
Separate visualization node for enhanced debugging and monitoring
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.debug_image_sub = self.create_subscription(
            Image, '/fairmot_tracker/debug_image', self.debug_image_callback, 10)
        self.poses_sub = self.create_subscription(
            PoseArray, '/tracked_obstacles', self.poses_callback, 10)
        self.markers_sub = self.create_subscription(
            MarkerArray, '/fairmot_tracker/path_markers', self.markers_callback, 10)
        
        self.tracked_poses = []
        self.get_logger().info("🎨 FairMOT Visualizer Node started")

    def debug_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow('FairMOT Tracking with Path Tracing', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error displaying image: {e}")

    def poses_callback(self, msg):
        self.tracked_poses = msg.poses
        if len(self.tracked_poses) > 0:
            self.get_logger().info(f"📍 Tracking {len(self.tracked_poses)} obstacles")

    def markers_callback(self, msg):
        path_markers = len([m for m in msg.markers if m.ns == "track_paths"])
        pos_markers = len([m for m in msg.markers if m.ns == "current_positions"])
        if path_markers > 0:
            self.get_logger().info(f"🛤️ Visualizing {path_markers} paths, {pos_markers} positions")

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    
    cv2.namedWindow('FairMOT Tracking with Path Tracing', cv2.WINDOW_AUTOSIZE)
    
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

