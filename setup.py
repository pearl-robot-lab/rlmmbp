"""Installation script for the 'learned_robot_placement' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "protobuf==3.20.2",
    "omegaconf==2.1.1",
    "hydra-core==1.1.1",
]

# Installation operation
setup(
    name="learned_robot_placement",
    author="Snehal Jauhri",
    version="2022.2.0", # For isaac-sim 2022.2.0
    description="RL environments for robot learning for the tiago robot in NVIDIA Isaac Sim. Adapted from omniisaacgymenvs (https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7, 3.8"],
    zip_safe=False,
)

# EOF
