#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# automatic resource extraction
# https://docs.python.org/2/distutils/apiref.html
import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import sys
import distutils.sysconfig as dsc
import sysconfig

pyincdir  = dsc.get_python_inc(plat_specific=1)
plibdir = dsc.get_python_lib(plat_specific=1)
plib = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)
os.environ['BASECFLAGS'] = '-fPIC'
os.environ['OPT'] = '-O3'

try:
    import numpy as np
    npyincdir = np.get_include()
except ImportError:
    npyincdir = os.path.join(plibdir.replace('lib/python',
                                             'local/lib/python'),
                             'numpy', 'core', 'include')
    print("Unable to import numpy, trying header %s".format(npyincdir_alt))

iw_ext = Extension('imgworker._ImgWorker',
                   sources=['imgworker.cpp'],
                   include_dirs=['/usr/include', '/usr/local/include',
                                 pyincdir, npyincdir],
                   library_dirs=['/usr/local/lib'],
                   libraries=['boost_thread', 'boost_system', 'jpeg', plib]
                   )
# install_requires = ['numpy', ]
install_requires = []
test_requires = ['nose', ]

setup(name="imgworker",
      version="0.1.0",
      description="Provides a set of functions for fast jpeg decoding "
                  "and accumulation for image statistics",
      ext_modules = [iw_ext],
      packages=['imgworker'],
      author="Nervanasys",
      author_email="info@nervanasys.com",
      url="http://nervanasys.com",
      install_requires=install_requires,
      tests_require=test_requires,
      cmdclass={'build_ext': build_ext},
)
