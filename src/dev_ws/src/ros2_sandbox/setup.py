from setuptools import setup

package_name = 'ros2_sandbox'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Martin Falang',
    maintainer_email='falang.martin@gmail.com',
    description='Simple package for experimenting with ROS2 functionality',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start_mission = ros2_sandbox.start_mission:main',
            'simple_mission = ros2_sandbox.simple_mission:main',
        ],
    },
)
