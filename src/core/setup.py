from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        name="cmember",
        sources=["cmember.c"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="cmember",
    ext_modules=ext_modules
)