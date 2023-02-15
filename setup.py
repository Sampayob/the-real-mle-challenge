from setuptools import setup, find_packages

setup(
    name='the-real-mle-challenge',
    packages=find_packages(include=find_packages())
)