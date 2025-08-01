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
        
        # Lightweight detector for tracking mode
        Node(
            package='yolov8_realsense',
            executable='yolov8_detector',
            name='yolov8_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'publish_image': False,  # Disable image publishing to reduce load
                'image_scale': 0.6,      # Reduce resolution for faster processing
                'input_topic': '/camera/camera/color/image_raw',
                'camera_info_topic': '/camera/camera/color/camera_info'
            }]
        ),
        
        # Enhanced tracker with visualization
        Node(
            package='yolov8_realsense',
            executable='yolov8_tracker',
            name='yolov8_tracker',
            output='screen',
            parameters=[{
                'max_disappeared': 15,
                'max_distance': 80.0,
                'input_image_topic': '/camera/camera/color/image_raw'
            }]
        )
    ])

