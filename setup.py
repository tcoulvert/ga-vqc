from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ga_vqc",
    version="1.0.3",
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
        "matplotlib >= 2.2.5",
        "numpy >= 1.23",
        "pandas >= 1.5",
        "scikit_learn >= 1.2",
        "pennylane >= 0.29",
        "qulacs-gpu >= 0.3", # Must install qulacs thru qulacs-gpu for qulacs plugin to work
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
