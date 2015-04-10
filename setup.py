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
from glob import glob

# Need boost thread and system
libext = '.dylib' if sys.platform == 'darwin' else '.so'
lpaths = ['/usr/local/lib', '/usr/lib']
thrlib = 'libboost_thread*' + libext
syslib = 'libboost_system*' + libext
thrlib = glob(os.path.join(lpaths[0], thrlib)) + glob(os.path.join(lpaths[1], thrlib))
syslib = glob(os.path.join(lpaths[0], syslib)) + glob(os.path.join(lpaths[1], syslib))
print thrlib, syslib
thrlib = os.path.splitext(os.path.basename(thrlib[0]))[0][3:]
syslib = os.path.splitext(os.path.basename(syslib[0]))[0][3:]
print thrlib, syslib
# Libraries needed for extension
pylib = "python{}".format(sys.version[:3])
if sys.version[:3] == '3.4':
    pylib += 'm'
libs = [pylib, thrlib, syslib, 'jpeg']

pyincdir  = dsc.get_python_inc(plat_specific=1)

pylibdir = os.path.join('/', *pyincdir.split('/')[:-2] + ['lib'])

# Library directories to find the above
# pylibdir = dsc.get_python_lib()
libdirs = [pylibdir, '/usr/local/lib']

# Hack to make sure it finds the right libpython with homebrew

# if pylibdir.find('Cellar') > 0:
#     brewpylibdir = os.path.join('/', *pylibdir.split('/')[:-2])
#     libdirs.append(brewpylibdir)

# Include directories needed by .cpp files in extension
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
      version="0.2.2",
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
