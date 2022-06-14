# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup RAF package."""
import os
import shutil
import subprocess
import sys
import sysconfig

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
FFI_MODE = os.environ.get("TVM_FFI", "auto")
CONDA_BUILD = os.getenv("CONDA_BUILD") is not None


def get_env_flag(name, default=""):
    """Get environment bololean flag by all means."""
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


# FIXME: get_lib_path() does not work as line 36: "exec(compile(ss, libinfo_py, "exec"), libinfo, libinfo)" errors
def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(SCRIPT_DIR, "./raf/python/raf/_lib.py")
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


# FIXME: commented out LIB_LIST because it relies on get_lib_path()
# LIB_LIST = get_lib_path()

# FIXME: commented out _version_ because subprocess.CalledProcessError: Command '['git', 'rev-parse', '--short', 'HEAD']' returned non-zero exit status 128.
# __version__ = get_build_version()

__version__ = "0.1"


def config_cython():
    """Try to configure cython and return cython configuration"""
    if FFI_MODE not in ("cython"):
        if os.name == "nt" and not CONDA_BUILD:
            print("WARNING: Cython is not supported on Windows, will compile without cython module")
            return []
        sys_cflags = sysconfig.get_config_var("CFLAGS")
        if sys_cflags and "i386" in sys_cflags and "x86_64" in sys_cflags:
            print("WARNING: Cython library may not be compiled correctly with both i386 and x64")
            return []
    try:
        from Cython.Build import cythonize

        if sys.version_info >= (3, 0):
            subdir = "_cy3"
        else:
            subdir = "_cy2"
        ret = []
        path = "./raf/3rdparty/tvm/python/tvm/_ffi/_cython"
        extra_compile_args = ["-std=c++14", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
        if os.name == "nt":
            library_dirs = ["tvm", "../build/Release", "../build"]
            libraries = ["tvm"]
            extra_compile_args = None
            # library is available via conda env.
            if CONDA_BUILD:
                library_dirs = [os.environ["LIBRARY_LIB"]]
        else:
            library_dirs = None
            libraries = None

        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "tvm._ffi.%s.%s" % (subdir, fn[:-4]),
                    ["./raf/3rdparty/tvm/python/tvm/_ffi/_cython/%s" % fn],
                    include_dirs=[
                        "../include/",
                        "../3rdparty/dmlc-core/include",
                        "../3rdparty/dlpack/include",
                    ],
                    extra_compile_args=extra_compile_args,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    language="c++",
                )
            )
        return cythonize(ret, compiler_directives={"language_level": 3})
    except ImportError as error:
        if FFI_MODE == "cython":
            raise error
        print("WARNING: Cython is not installed, will compile without cython module")
        return []


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

# FIXME: this if statement does not work as it relies on get_lib_path()
"""
if include_libs:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    for i, path in enumerate(LIB_LIST):
        LIB_LIST[i] = os.path.relpath(path, curr_path)
    setup_kwargs = {"include_package_data": True, "data_files": [("raf", LIB_LIST)]}
"""

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
    license="Apache",
    description="RAF Accelerates Deep Learning Frameworks",
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
    ext_modules=config_cython(),
    **setup_kwargs
)

if wheel_include_libs:
    # Wheel cleanup
    os.remove(os.path.join(SCRIPT_DIR, "MANIFEST.in"))
    for path in LIB_LIST:
        _, libname = os.path.split(path)
        os.remove("raf/%s" % libname)       
