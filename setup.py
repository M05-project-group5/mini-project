#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="Mini-Project",
    version="2.0.1",
    description="Basic example of a Reproducible Research Project in Python",
    url="https://github.com/M05-project-group5/mini-project",
    license="BSD",
    author="Adrien Chassignet, Cédric Mariéthoz",
    author_email="",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["mini-project-main = src.main:main","mini-project-download = src.download_datasets:main"]},
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
