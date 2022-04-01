#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]
    
setup(
    name="Mini-Project",
    version="v0.0.2",
    author="CA, MC",
    description="Mini-Project for the Master in IA to Idiap",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/M05-project-group5/mini-project",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)
