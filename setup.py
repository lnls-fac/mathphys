#!/usr/bin/env python-sirius
"""Setup module."""

from setuptools import setup

with open("README.md", "r") as _f:
    _long_description = _f.read().strip()

with open("VERSION", "r") as _f:
    __version__ = _f.read().strip()

with open("requirements.txt", "r") as _f:
    _requirements = _f.read().strip().split("\n")

setup(
    name="mathphys",
    version=__version__,
    author="lnls-fac",
    author_email="xresende@gmail.com",
    description="LNLS Math and Physics Utilities",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lnls-fac/mathphys",
    download_url="https://github.com/lnls-fac/mathphys",
    license="MIT License",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    packages=["mathphys"],
    install_requires=_requirements,
    package_data={"mathphys": ["VERSION", "data/d_touschek.npz"]},
    test_suite="tests",
    python_requires=">=3.4",
    zip_safe=False,
)
