# setup.py
from Cython.Build import cythonize
import numpy
import os
from setuptools import setup, Extension
import sys
sys.path.insert(1, './cbuilds')

# List of all extensions
extensions = [
    Extension(
        "cmember",
        sources=["cbuilds/cmember.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "cpopulation",
        sources=["cbuilds/cpopulation.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],  # Optional: Additional compilation options
        language="c"
    ),
    Extension(
        "coptimization",
        sources=["cbuilds/coptimization.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],  # Optional: Additional compilation options
        language="c"
    )
]

setup(
    name="cython_extensions",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)