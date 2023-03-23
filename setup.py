from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ga_vqc",
    version="0.0.2",
    description="Genetic Algorithm for VQC ansatz search.",
    packages=find_packages(include=["ga_vqc"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib >= 3.7.1" "numpy >= 1.20.3",
        "pandas >= 1.5.3",
        "scikit_learn >= 1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            "check-manifest >= 0.47",
        ],
    },
    url="https://github.com/tcoulvert/GA_Ansatz_Search",
    author="Thomas Sievert",
    author_email="63161166+tcoulvert@users.noreply.github.com",
)
