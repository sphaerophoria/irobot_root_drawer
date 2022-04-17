#!/usr/bin/env python3

from distutils.core import setup

setup(name='roomba_drawer',
      version='1.0',
      description='',
      packages=['roomba_drawer'],
      install_requires=['opencv-python', 'scikit-image', 'matplotlib', 'numpy', 'aiorobot']
     )
