"""
This is the Setup Script for installing the Library
"""

from setuptools import setup

setup(
    # Common Setup
    name="mpctools",
    version="0.1.2",
    packages=['mpctools.extensions', 'mpctools.multiprocessing'],

    # Requirements
    install_requires=['numpy', 'pathos', 'scikit-multilearn', 'scikit-learn', 'matplotlib', 'pandas'],

    # Meta-Data
    author='Michael P. J. Camilleri',
    author_email='michael.p.camilleri@ed.ac.uk',
    description='A set of tools and utilities for extending common libraries and providing multiprocessing capabilities',
    license='GNU GPL',
    keywords='extensions multiprocessing',
    url='https://github.com/michael-camilleri/mpctools'
)