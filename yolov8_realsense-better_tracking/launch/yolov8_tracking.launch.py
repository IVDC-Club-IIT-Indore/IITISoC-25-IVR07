from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='yolov8n.pt',  # Using nano for speed
            description='Path to YOLOv8 model file (yolov8n.pt for speed, yolov8s.pt for accuracy)'
        ),
        
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.6',  # Higher threshold = less noise
            description='Confidence threshold for detections'
        ),
        
        # Performance-optimized 3D detector
        Node(
            package='yolov8_realsense',
            executable='yolov8_detector_3d',
            name='yolov8_detector_3d',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'rgb_topic': '/camera/camera/color/image_raw',
                'depth_topic': '/camera/camera/depth/image_rect_raw',
                'camera_info_topic': '/camera/camera/color/camera_info',
                'depth_camera_info_topic': '/camera/camera/depth/camera_info',
                'publish_image': True,
                'depth_scale': 0.001,
                'process_every_n_frames': 3,  # Process every 3rd frame for speed
                'resize_factor': 0.6,         # Resize for faster processing
                'max_detections': 50          # Limit detections for speed
            }]
        ),
        
        # Enhanced 3D tracker with ID stability
        Node(
            package='yolov8_realsense',
            executable='yolov8_tracker_3d',
            name='yolov8_tracker_3d',
            output='screen',
            parameters=[{
                'max_disappeared': 25,     # Longer tracking for stability
                'max_distance': 1.2,       # Stricter distance threshold
                'rgb_topic': '/camera/camera/color/image_raw',
                'publish_point_cloud': True  # Enable point cloud visualization
            }]
        )
    ])

