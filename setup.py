#!/usr/bin/env python

from distutils.core import setup

setup(name='facenet',
      version='0.1',
      description='Face Recognition using Tensorflow',
      author='David Sandberg',
      author_email='david.o.sandberg@gmail.com',
      url='https://github.com/davidsandberg/facenet/',
      packages=['facenet','facenet.align','facenet.models'],
      package_data={'': ['*.npy']}
     )
