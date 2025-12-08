from setuptools import setup, find_packages
import os

def read_requirements(file:str="requirements.txt"):
    if not os.path.exists(file):
        return []
    with open(file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lane_detection",
    version="0.1.0",
    author="Shane Teel",
    description="Classic lane line detection system using traditional CV models and Kalman filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShaneTeel/lane-detection-classic",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0"
        ]
    }
)