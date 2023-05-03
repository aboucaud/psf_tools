#!/usr/bin/env python

import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    use_scm_version=True,
    ext_modules=cythonize("psftools/utils/_cytutils.pyx"),
    include_dirs=[numpy.get_include()],
)
