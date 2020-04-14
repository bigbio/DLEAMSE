from __future__ import print_function
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dleamse",
    version="0.3.2",
    author="BigBio Team",
    author_email="qinchunyuan1995@gmail.com",
    description=" dleamse's encoding and embedding methods, and dleamse's faiss index (IndexIDMap type) write.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="'Apache 2.0",
    url="https://github.com/bigbio/DLEAMSE",
    packages=["dleamse"],
    install_requires=['numba>=0.45',
                      'numpy>=1.13.3',
                      'pyteomics>=3.5.1',
                      'torch==1.0.0',
                      'faiss>=1.5.3',
                      'more_itertools == 7.1.0'],
    platforms=['any'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    python_requires='>=3.5',

)
