"""
setup.py — Tython Python SDK Package
======================================
Install locally:
    pip install -e ./src/sdk

Or build a distributable:
    python setup.py sdist bdist_wheel
    pip install dist/tython-*.whl
"""

from setuptools import setup, find_packages

setup(
    name="tython",
    version="1.0.0",
    author="Tython",
    description="Official Python SDK for the Tython AI Platform",
    long_description=open("README_SDK.md").read() if __import__("os").path.exists("README_SDK.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
