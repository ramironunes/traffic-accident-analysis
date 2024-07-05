# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-30 20:54:25
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-30 20:57:49

""" Information in `setup.py`

The `setup.py` file is used to configure Python packages, including defining
project metadata and specifying dependencies.
"""

from setuptools import setup, find_packages

setup(
    name='project-template',
    version='0.1.0',
    description='A template for creating new projects with a basic structure',
    author='Your Name',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/project-template',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'flask',  # example of dependency
        'requests',  # example of dependency
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
