#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('fairmot_tracker'),
            'config',
            'tracker_params.yaml'
        ]),
        description='Path to the tracker configuration file'
    )
    
    use_visualizer_arg = DeclareLaunchArgument(
        'use_visualizer',
        default_value='true',
        description='Whether to launch the visualizer node'
    )

    # FairMOT tracker node (using the actual package)
    tracker_node = Node(
        package='fairmot_tracker',
        executable='tracker_node',
        name='tracker_node',
        parameters=[LaunchConfiguration('config_file')],
        output='screen',
        emulate_tty=True,
    )
    
    # Visualizer node
    visualizer_node = Node(
        package='fairmot_tracker',
        executable='visualizer_node',
        name='visualizer_node',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        config_file_arg,
        use_visualizer_arg,
        tracker_node,
        visualizer_node,
    ])
