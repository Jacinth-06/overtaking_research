from setuptools import setup, find_packages

setup(
    name='jetracer',
    version='0.1.0',
    description='Custom Lidar-enabled JetRacer for Overtaking Research',
    packages=find_packages(),
    install_requires=[
        'pyserial',  # Required for RPLidar A1
        'smbus',     # Required for I2C communication in your new class
    ],
)