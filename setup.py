from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyNNsMD",
    version="1.0.3",
    author="Patrick Reiser",
    author_email="patrick.reiser@kit.edu",
    description="Neural Network for learning potential energy surface for molecular dynamics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=['numpy', "scikit-learn", "matplotlib"],
    extras_require={
        "tf": ["tensorflow>=2.3.0"],
        "tf_gpu": ["tensorflow-gpu>=2.3.0"],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"pyNNsMD": ["*.json", "*.yaml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["materials", "science", "machine", "learning", "deep", "dynamics", "molecular", "potential"]
)
