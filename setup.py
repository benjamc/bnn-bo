#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='bnn_bo',
      version='0.1',
      packages=find_packages(include=['bnnbo_test_functions']),
      install_requires=[
            'botorch',
            'gpytorch'
      ]
)