import sys
from distutils.core import setup
from os import path

from setuptools import find_namespace_packages  # This should be place at top!


assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages()



install_requires = [
    "matplotlib>=3.2.2",
    "numpy>=1.18.5",
    "opencv-python>=4.1.1",
    "Pillow>=7.1.2",
    "PyYAML>=5.3.1",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    #"torch>=1.7.0,!=1.12.0",
    #"torchvision>=0.8.1,!=0.13.0",
    "tqdm>=4.41.0",
    "protobuf<4.21.3",
    "tensorboard>=2.4.1",
    "pandas>=1.1.4",
    "seaborn>=0.11.0",
    "ipython",
    "psutil",
    "thop"
]


setup(
    name="yolov7",
    version="7.0",
    description="yolov7",
    url="https://github.com/WongKinYiu/yolov7",
    packages=packages,
    install_requires=install_requires,
    license="GPL 3.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
