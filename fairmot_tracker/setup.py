from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'fairmot_tracker'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='FairMOT-based dynamic obstacle detection and tracking',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = fairmot_tracker.tracker_node:main',
            'visualizer_node = fairmot_tracker.visualizer_node:main',
        ],
    },
)

