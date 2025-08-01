from setuptools import setup
import os
from glob import glob

package_name = 'yolov8_realsense'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), 
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='YOLOv8 object detection and tracking for Intel RealSense cameras',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_detector = yolov8_realsense.yolov8_detector_node:main',
            'yolov8_tracker = yolov8_realsense.yolov8_tracker_node:main',
        ],
    },
)
