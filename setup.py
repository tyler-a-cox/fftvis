"""
This file is used to install the package using pip.
"""

from setuptools import setup, find_packages

setup(
    name="fftvis",
    version="0.0.7",
    description="An FFT-based visibility simulator",
    author="Tyler Cox",
    author_email="tyler.a.cox@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "matvis", "finufft", "pyuvdata", "psutil"],
    extras_require={
        "dev": [
            "mpi4py",
            "pyuvsim[sim]",
            "pyradiosky",
            "pytest",
            "pre-commit",
            "pytest-cov",
            "hera_sim",
            "pytest-xdist",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
