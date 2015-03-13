#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# automatic resource extraction
# https://docs.python.org/2/distutils/apiref.html
import distutils.sysconfig as dsc
import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import sys

# Libraries needed for extension
libs = ["python{}.{}".format(sys.version_info.major, sys.version_info.minor),
        'boost_thread-mt' if sys.platform == "darwin" else 'boost_thread',
        'boost_system-mt' if sys.platform == "darwin" else 'boost_system',
        'jpeg']

# Library directories to find the above
pylibdir = dsc.get_python_lib(plat_specific=1)
libdirs = [pylibdir, '/usr/local/lib']

# Hack to make sure it finds the right libpython with homebrew
if pylibdir.find('Cellar') > 0:
    brewpylibdir = os.path.join('/', *pylibdir.split('/')[:-2])
    libdirs.append(brewpylibdir)

# Include directories needed by .cpp files in extension
pyincdir  = dsc.get_python_inc(plat_specific=1)
try:
    import numpy as np
    npyincdir = np.get_include()
except ImportError:
    npyincdir = os.path.join(
                    pylibdir.replace('lib/python', 'local/lib/python'),
                    'numpy', 'core', 'include')
    print("Unable to import numpy, trying header %s".format(npyincdir))

incdirs = ['/usr/include', '/usr/local/include', pyincdir, npyincdir]

# Replace some of the python determined cflags and options
os.environ['BASECFLAGS'] = '-fPIC'
os.environ['OPT'] = '-O3'

iw_ext = Extension('_ImgWorker', sources=['imgworker.cpp'],
                   include_dirs=incdirs, library_dirs=libdirs, libraries=libs)

# install_requires = ['numpy', ]
install_requires = [ ]
test_requires = ['nose', ]

with open('README.md') as file:
    long_desc = file.read()

setup(name="imgworker",
      version="0.1.0",
      description="Provides a set of functions for fast jpeg decoding "
                  "and accumulation for image statistics",
      ext_modules = [iw_ext],
      packages=['imgworker'],
      author="Nervanasys",
      author_email="info@nervanasys.com",
      long_description = long_desc,
      url="http://nervanasys.com",
      install_requires=install_requires,
      tests_require=test_requires,
      cmdclass={'build_ext': build_ext},
)
