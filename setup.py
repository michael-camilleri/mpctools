"""
This is the Setup Script for installing the Library
"""

from setuptools import setup
import mpctools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    # Common Setup
    name="mpctools",
    version=mpctools.__version__,
    packages=["mpctools", "mpctools.extensions", "mpctools.parallel"],
    # Requirements
    install_requires=[
        "scikit-learn",
        "matplotlib",
        "joblib",
        "lapsolver",
        "hotelling",
        "seaborn",
        "pandas",
        "pathos",
        "scipy",
        "numpy",
        "numba",
    ],
    extras_require = {
        'torch': ["torch"],
        'cv':["opencv-python"],
        'all': ["torch", "opencv-python"]
    },
    # Meta-Data
    author="Michael P. J. Camilleri",
    author_email="michael.p.camilleri@ed.ac.uk",
    description="A set of tools and utilities for extending common libraries and providing parallel"
                " capabilities.",
    license="GNU GPL",
    keywords=["extensions", "parallel", "utilities"],
    url="https://github.com/michael-camilleri/mpctools",
    download_url="https://github.com/michael-camilleri/mpctools/archive/v_01330.tar.gz",
    long_description=long_description,
    long_description_content_type="text/markdown",
    data_files=[("", ["LICENSE"])],
    include_package_data=True,
    # PyPi Data
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
