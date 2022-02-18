# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Setup mnm package."""
import os
import shutil
import sys
import sysconfig
import platform

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

CURRENT_DIR = os.path.dirname(__file__)


def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(CURRENT_DIR, "./mnm/_lib.py")
    libinfo = {"__file__": libinfo_py}
    with open(libinfo_py, "rb") as f:
        ss = f.read()
    exec(compile(ss, libinfo_py, "exec"), libinfo, libinfo)
    version = "0.0.2.dev"
    if not os.getenv("CONDA_BUILD"):
        lib_path = libinfo["find_lib_path"]()
        libs = [lib_path[0]]
        if libs[0].find("runtime") == -1:
            for name in lib_path[1:]:
                if name.find("runtime") != -1:
                    libs.append(name)
                    break
    else:
        libs = None
    return libs, version


LIB_LIST, __version__ = get_lib_path()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


include_libs = False
wheel_include_libs = False
if not os.getenv("CONDA_BUILD"):
    if "bdist_wheel" in sys.argv:
        wheel_include_libs = True
    else:
        include_libs = True

setup_kwargs = {}

# For bdist_wheel only
if wheel_include_libs:
    with open("MANIFEST.in", "w") as fo:
        for path in LIB_LIST:
            if os.path.normpath(path) != os.path.normpath(
                os.path.join(CURRENT_DIR, "mnm/libmnm.so")
            ):
                shutil.copy(path, os.path.join(CURRENT_DIR, "mnm"))
            _, libname = os.path.split(path)
            fo.write("include mnm/%s\n" % libname)
    setup_kwargs = {"include_package_data": True}

if include_libs:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    for i, path in enumerate(LIB_LIST):
        LIB_LIST[i] = os.path.relpath(path, curr_path)
    setup_kwargs = {"include_package_data": True, "data_files": [("mnm", LIB_LIST)]}

# Local change: Write out version to file and include in package
os.makedirs("../build/private/mnm/version", exist_ok=True)
with open("../build/private/mnm/version/__init__.py", "w") as fd:
    fd.write("__version__ = '" + __version__ + "'")
setup_kwargs["package_dir"] = {"mnm.version": "../build/private/mnm/version"}
# End local change

setup(
    name="mnm",
    version=__version__,
    description="An End to End Compiler for Deep Learning Systems",
    zip_safe=False,
    install_requires=[
        "tvm",
        "numpy",
    ],
    extras_require={
        "test": ["torch==1.6.0", "pytest"],
    },
    packages=find_packages() + ["mnm.version"],
    package_data={"mnm": [os.path.join(CURRENT_DIR, "../build/lib/libmnm.so")]},
    distclass=BinaryDistribution,
    url="https://github.com/meta-project/meta",
    python_requires=">=3.6",
    **setup_kwargs
)

if wheel_include_libs:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in LIB_LIST:
        _, libname = os.path.split(path)
        os.remove("mnm/%s" % libname)
