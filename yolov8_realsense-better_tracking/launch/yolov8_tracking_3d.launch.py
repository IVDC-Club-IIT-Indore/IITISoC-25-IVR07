from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='yolov8n.pt',
            description='Path to YOLOv8 model file'
        ),
        
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.5',
            description='Confidence threshold for detections'
        ),
        
        # 3D Detector with depth integration
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
                'depth_scale': 0.001
            }]
        ),
        
        # 3D Kalman Filter Tracker
        Node(
            package='yolov8_realsense',
            executable='yolov8_tracker_3d',
            name='yolov8_tracker_3d',
            output='screen',
            parameters=[{
                'max_disappeared': 20,  # Allow longer prediction periods
                'max_distance': 1.5,    # 3D distance threshold in meters
                'rgb_topic': '/camera/camera/color/image_raw'
            }]
        )
    ])

