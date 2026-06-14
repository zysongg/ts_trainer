from setuptools import setup, find_packages

setup(
    name="ts_trainer",
    version="0.1.0",
    description="Standardized training framework for time series tasks",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "lightning>=2.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "numpy>=1.21",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
