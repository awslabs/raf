# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup RAF package."""
import os
import shutil
import subprocess
import sys

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

SCRIPT_DIR = os.path.dirname(__file__)


def get_env_flag(name, default=""):
    """Get environment bololean flag by all means."""
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(SCRIPT_DIR, "./raf/_lib.py")
    libinfo = {"__file__": libinfo_py}
    with open(libinfo_py, "rb") as f:
        ss = f.read()
    exec(compile(ss, libinfo_py, "exec"), libinfo, libinfo)
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
    return libs


def get_build_version():
    """Generate the build version."""
    cwd = os.path.abspath(SCRIPT_DIR)
    git_sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
    version = os.getenv("RAF_VERSION", default="0.1")
    if not get_env_flag("RELEASE_VERSION", default="0"):
        version += "+git" + git_sha
    return version


LIB_LIST = get_lib_path()
__version__ = get_build_version()


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
    with open(os.path.join(SCRIPT_DIR, "MANIFEST.in"), "w") as fo:
        for path in LIB_LIST:
            if os.path.normpath(path) != os.path.normpath(
                os.path.join(SCRIPT_DIR, "raf/libraf.so")
            ):
                shutil.copy(path, os.path.join(SCRIPT_DIR, "raf"))
            _, libname = os.path.split(path)
            fo.write("include raf/%s\n" % libname)
    setup_kwargs = {"include_package_data": True}

if include_libs:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    for i, path in enumerate(LIB_LIST):
        LIB_LIST[i] = os.path.relpath(path, curr_path)
    setup_kwargs = {"include_package_data": True, "data_files": [("raf", LIB_LIST)]}

# Local change: Write out version to file and include in package
os.makedirs(os.path.join(SCRIPT_DIR, "../build/private/raf/version"), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, "../build/private/raf/version/__init__.py"), "w") as fd:
    fd.write("__version__ = '" + __version__ + "'")
setup_kwargs["package_dir"] = {
    "raf.version": os.path.join(SCRIPT_DIR, "../build/private/raf/version")
}
# End local change

setup(
    name="raf",
    version=__version__,
    description="RAF Accelerates deep learning Frameworks",
    zip_safe=False,
    install_requires=[
        "tvm",
        "numpy",
    ],
    extras_require={
        "test": ["torch==1.6.0", "pytest"],
    },
    packages=find_packages() + ["raf.version"],
    package_data={"raf": [os.path.join(SCRIPT_DIR, "../build/lib/libraf.so")]},
    distclass=BinaryDistribution,
    url="https://github.com/awslabs/raf",
    python_requires=">=3.6",
    **setup_kwargs
)

if wheel_include_libs:
    # Wheel cleanup
    os.remove(os.path.join(SCRIPT_DIR, "MANIFEST.in"))
    for path in LIB_LIST:
        _, libname = os.path.split(path)
        os.remove("raf/%s" % libname)
