#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import platform as plat


def _find_include_file(self, include):
    for directory in self.compiler.include_dirs:
        if os.path.isfile(os.path.join(directory, include)):
            return 1
    return 0

def _find_library_file(self, library):
    return self.compiler.find_library_file(self.compiler.library_dirs, library)

def _add_directory(path, dir, where=None):
    if dir is None:
        return
    dir = os.path.realpath(dir)
    if os.path.isdir(dir) and dir not in path:
        if where is None:
            path.append(dir)
        else:
            path.insert(where, dir)

# Need boost thread and system
libext = '.dylib' if sys.platform == 'darwin' else '.so'

class imgworker_build_ext(build_ext):

    def build_extensions(self):
        pyincdir  = dsc.get_python_inc(plat_specific=1)
        pylibdir = os.path.join('/', *pyincdir.split('/')[:-2] + ['lib'])

        # Include directories needed by .cpp files in extension
        try:
            import numpy as np
            npyincdir = np.get_include()
        except ImportError:
            npyincdir = os.path.join(
                            pylibdir.replace('lib/python', 'local/lib/python'),
                            'numpy', 'core', 'include')
            print("Unable to import numpy, trying header %s".format(npyincdir))

        library_dirs = [pylibdir, '/usr/local/lib']
        include_dirs = ['/usr/include', '/usr/local/include', pyincdir,
                        npyincdir]


        for k in ('CFLAGS', 'LDFLAGS'):
            if k in os.environ:
                for match in re.finditer(r'-I([^\s]+)', os.environ[k]):
                    _add_directory(include_dirs, match.group(1))
                for match in re.finditer(r'-L([^\s]+)', os.environ[k]):
                    _add_directory(library_dirs, match.group(1))

        # include, rpath, if set as environment variables:
        for k in ('C_INCLUDE_PATH', 'CPATH', 'INCLUDE'):
            if k in os.environ:
                for d in os.environ[k].split(os.path.pathsep):
                    _add_directory(include_dirs, d)

        for k in ('LD_RUN_PATH', 'LIBRARY_PATH', 'LIB'):
            if k in os.environ:
                for d in os.environ[k].split(os.path.pathsep):
                    _add_directory(library_dirs, d)

        prefix = dsc.get_config_var("prefix")
        if prefix:
            _add_directory(library_dirs, os.path.join(prefix, "lib"))
            _add_directory(include_dirs, os.path.join(prefix, "include"))

        if sys.platform == "darwin":
            # fink installation directories
            _add_directory(library_dirs, "/sw/lib")
            _add_directory(include_dirs, "/sw/include")
            # darwin ports installation directories
            _add_directory(library_dirs, "/opt/local/lib")
            _add_directory(include_dirs, "/opt/local/include")
            # if Homebrew is installed, use its lib and include directories
            import subprocess
            try:
                prefix = subprocess.check_output(
                    ['brew', '--prefix']
                ).strip().decode('latin1')
            except:
                # Homebrew not installed
                prefix = None

            if prefix:
                # add Homebrew's include and lib directories
                _add_directory(library_dirs, os.path.join(prefix, 'lib'))
                _add_directory(include_dirs, os.path.join(prefix, 'include'))

        elif sys.platform.startswith("linux"):
            arch_tp = (plat.processor(), plat.architecture()[0])
            if arch_tp == ("x86_64", "32bit"):
                # 32 bit build on 64 bit machine.
                _add_directory(library_dirs, "/usr/lib/i386-linux-gnu")
            else:
                for platform_ in arch_tp:
                    if not platform_:
                        continue
                    if platform_ in ["x86_64", "64bit"]:
                        _add_directory(library_dirs, "/lib64")
                        _add_directory(library_dirs, "/usr/lib64")
                        _add_directory(
                            library_dirs, "/usr/lib/x86_64-linux-gnu")
                        break
                    elif platform_ in ["i386", "i686", "32bit"]:
                        _add_directory(
                            library_dirs, "/usr/lib/i386-linux-gnu")
                        break
                    elif platform_ in ["aarch64"]:
                        _add_directory(library_dirs, "/usr/lib64")
                        _add_directory(
                            library_dirs, "/usr/lib/aarch64-linux-gnu")
                        break
                else:
                    raise ValueError(
                        "Unable to identify Linux platform: `%s`" % platform_)

        self.compiler.library_dirs = library_dirs + self.compiler.library_dirs
        self.compiler.include_dirs = include_dirs + self.compiler.include_dirs

        pylib = "python{}".format(sys.version[:3])
        if sys.version[:3] == '3.4':
            pylib += 'm'
        libs = [pylib]

        if _find_include_file(self, "jpeglib.h"):
            if _find_library_file(self, "jpeg"):
                libs.append('jpeg')
            else:
                raise ValueError("Unable to find libjpeg")
        else:
            raise ValueError("Unable to find jpeglib.h")

        if _find_include_file(self, "boost/thread.hpp"):
            if _find_library_file(self, "boost_thread"):
                libs.append("boost_thread")
            elif _find_library_file(self, "boost_thread-mt"):
                libs.append("boost_thread-mt")
            else:
                raise ValueError("Unable to find libboost_thread")

            if _find_library_file(self, "boost_system"):
                libs.append("boost_system")
            elif _find_library_file(self, "boost_system-mt"):
                libs.append("boost_system-mt")
            else:
                raise ValueError("Unable to find libboost_system")
        else:
            raise ValueError("Unable to find boost headers")

        print(libs)
        iwt = Extension('_ImgWorker', sources=['imgworker.cpp'],
                                   include_dirs=self.compiler.include_dirs,
                                   library_dirs=self.compiler.library_dirs,
                                   libraries=libs)
        iwt._needs_stub = False
        exts = [iwt]
        self.extensions[:] = exts
        build_ext.build_extensions(self)

install_requires = [ ]
test_requires = ['nose', ]

with open('README.md') as file:
    long_desc = file.read()

setup(name="imgworker",
      version="0.2.5",
      description="Provides a set of functions for fast jpeg decoding "
                  "and accumulation for image statistics",
      ext_modules = [Extension('_ImgWorker', sources=['imgworker.cpp'])],
      packages=['imgworker'],
      author="Nervanasys",
      author_email="info@nervanasys.com",
      long_description = long_desc,
      url="http://nervanasys.com",
      install_requires=install_requires,
      tests_require=test_requires,
      cmdclass={'build_ext': imgworker_build_ext},
)

