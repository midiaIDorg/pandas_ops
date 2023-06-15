# This Python file uses the following encoding: utf-8
import glob

from setuptools import find_packages, setup

setup(
    name="pandas_ops",
    packages=find_packages(),
    version="0.0.1",
    description="Description.",
    long_description="Common operations on pandas data frames.",
    author="MatteoLacki",
    author_email="matteo.lacki@gmail.com",
    url="https://github.com/MatteoLacki/pandas_ops.git",
    keywords=["Great module", "Devel Inside"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "numpy",
        "numba",
        "pandas",
        "tqdm",
        # "mmapped_df",
    ],
    scripts=glob.glob("tools/*.py"),
)
