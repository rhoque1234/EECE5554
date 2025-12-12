from setuptools import setup
import os
from glob import glob

package_name = 'vn_driver'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@northeastern.edu',
    description='VectorNav VN-100 IMU driver',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vn_driver_node = vn_driver.vn_driver_node:main',
            'imu_emulator = vn_driver.imu_simulator:main',
        ],
    },
)
