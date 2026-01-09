#!/usr/bin/env python-sirius
"""Setup module."""

import pathlib
from setuptools import setup


def get_abs_path(relative):
    return str(pathlib.Path(__file__).parent / relative)


with open(get_abs_path("README.md"), "r") as _f:
    _long_description = _f.read().strip()

with open(get_abs_path("mathphys/VERSION"), "r") as _f:
    __version__ = _f.read().strip()

with open(get_abs_path("requirements.txt"), "r") as _f:
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
    include_package_data=True,
    install_requires=_requirements,
    test_suite="tests",
    python_requires=">=3.4",
    zip_safe=False,
)
