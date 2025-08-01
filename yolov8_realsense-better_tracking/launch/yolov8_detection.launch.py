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
        
        Node(
            package='yolov8_realsense',
            executable='yolov8_detector',
            name='yolov8_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'publish_image': True,  # Enable image publishing for detection-only mode
                'image_scale': 1.0,     # Full resolution for detection-only
                'input_topic': '/camera/camera/color/image_raw',
                'camera_info_topic': '/camera/camera/color/camera_info'
            }]
        )
    ])

