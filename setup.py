#!/usr/bin/env python-sirius
"""Setup module."""

from setuptools import setup

with open('VERSION', 'r') as _f:
    __version__ = _f.read().strip()

with open('requirements.txt', 'r') as _f:
    __requirements__ = _f.read().strip().split('\n')

setup(
    name='mathphys',
    version=__version__,
    author='lnls-fac',
    author_email='xresende@gmail.com',
    description='LNLS Math and Physics Utilities',
    url='https://github.com/lnls-fac/mathphys',
    download_url='https://github.com/lnls-fac/mathphys',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=['mathphys'],
    install_requires=__requirements__,
    package_data={'mathphys': ['VERSION', 'data/d_touschek.npz']},
    zip_safe=False,
)
