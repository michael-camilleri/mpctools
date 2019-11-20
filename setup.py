"""
This is the Setup Script for installing the Library
"""

from setuptools import setup

setup(
    # Common Setup
    name="mpctools",
    version="0.3.21",
    packages=['mpctools.extensions', 'mpctools.parallel'], # Eventually, do neural

    # Requirements
    install_requires=['numpy', 'scipy', 'pathos', 'scikit-learn', 'matplotlib', 'seaborn', 'pandas', 'scikit-multilearn', 'numba'],

    # Meta-Data
    author='Michael P. J. Camilleri',
    author_email='michael.p.camilleri@ed.ac.uk',
    description='A set of tools and utilities for extending common libraries and providing parallel capabilities',
    license='GNU GPL',
    keywords='extensions parallel',
    url='https://github.com/michael-camilleri/mpctools'
)