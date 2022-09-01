# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup RAF package."""
import os
import shutil
import subprocess
import sys

from setuptools import find_packages
from setuptools.dist import Distribution

from datetime import date

if "--inplace" in sys.argv:
    from distutils.core import setup
else:
    from setuptools import setup

SCRIPT_DIR = os.path.dirname(__file__)
CONDA_BUILD = os.getenv("CONDA_BUILD") is not None


def get_env_flag(name, default=""):
    """Get environment bololean flag by all means."""
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_lib_path():
    """Get library path, name and version"""
    # We can not import `_lib_utils.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences.
    libinfo_py = os.path.join(SCRIPT_DIR, "./raf/_lib_utils.py")
    libinfo = {"__file__": libinfo_py}
    with open(libinfo_py, "rb") as f:
        ss = f.read()
    exec(compile(ss, libinfo_py, "exec"), libinfo, libinfo)
    if not os.getenv("CONDA_BUILD"):
        # Look for the libs.
        # "libraf" is the core RAF lib.
        # "raf" is the lib required by win32 systems.
        libs = libinfo["find_lib_path"](["libraf", "raf"])
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
    version = "+git" + git_sha
    return version


LIB_LIST = get_lib_path()
git_version = get_build_version()


def get_build_raf_version():
    """Build the version based on PEP-440. Specifically, we have the following formats:
    - Stable release: MAJOR.MINOR+PLATFORM
    - Nightly release: MAJOR.MINOR.devDATE+PLATFORM
    - Dev (local build): MAJOR.MINOR.dev+GIT_SHA
    Note that we do not put the PLATFORM in dev version due to the limit of PEP 440 format.
    """
    raf_build_version = os.getenv("RAF_BUILD_VERSION", default="dev")
    raf_build_platform = os.getenv("RAF_BUILD_PLATFORM", default="cu113")
    with open("./raf/version.txt", "r") as version_file:
        version = version_file.readline()
        raf_version = version
        if raf_build_version == "stable":
            raf_version = version + "+" + raf_build_platform
        elif raf_build_version == "nightly":
            version = inc_minor(version)
            today = date.today().strftime("%Y%m%d")
            raf_version = version + ".dev" + str(today) + "+" + raf_build_platform
        elif raf_build_version == "dev":
            version = inc_minor(version)
            raf_version = version + ".dev+" + git_version
        else:
            raise ValueError("Unsupported RAF build version: " % raf_build_version)
        return raf_version


def inc_minor(version):
    split_version = version.split(".")
    inc_version = int(split_version[-1]) + 1
    next_version = ".".join(split_version[:-1]) + "." + str(inc_version)
    return next_version


__version__ = get_build_raf_version()
__raf_version__ = repr(__version__.split("+")[0])
__full_version__ = repr(__version__)
__gitrev__ = repr(git_version)

with open("./raf/version.py", "w") as version_file:
    version_file.write(
        '"""Auto-generated. Do not touch."""'
        + "\n"
        + "__version__ = "
        + __raf_version__
        + "\n"
        + "__full_version__ = "
        + __full_version__
        + "\n"
        + "__gitrev__ = "
        + __gitrev__
        + "\n"
    )


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


class ExtModules(list):
    """bdist_wheel determines whether the package is pure or not based on ext_modules.
    However, all RAF modules are prebuilt (libraf.so) and packaged as data, so the package
    is not pure, although we have no Extensions to build.

    Ideally we should override has_ext_modules, but https://bugs.python.org/issue32957
    mentions that has_ext_modules is not always called in setuptools <= 57.0.0.

    Thus, we provide a customized empty list. This causes the package to be treated
    as non-pure on all relevant setuptools versions.

    This solution is inspired by https://github.com/microsoft/debugpy/pull/700
    """

    def __bool__(self):
        return True


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
            try:
                shutil.copy(path, os.path.join(SCRIPT_DIR, "raf"))
            except shutil.SameFileError:
                pass
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
setup_kwargs["package_dir"] = {"raf.version": "../build/private/raf/version"}
# End local change

setup(
    name="raf",
    version=__version__,
    license="Apache-2.0",
    author="AWS AI team",
    author_email="",
    description="RAF Accelerates Deep Learning Frameworks",
    zip_safe=False,
    install_requires=[
        "numpy",
    ],
    extras_require={
        "test": ["torch==1.12.0", "pytest"],
    },
    packages=find_packages() + ["raf.version"],
    package_data={"raf": [os.path.join(SCRIPT_DIR, "../build/lib/libraf.so")]},
    distclass=BinaryDistribution,
    url="https://github.com/awslabs/raf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ExtModules(),
    has_ext_modules=lambda: True,
    python_requires=">=3.7",
    **setup_kwargs
)

if wheel_include_libs:
    # Wheel cleanup
    os.remove(os.path.join(SCRIPT_DIR, "MANIFEST.in"))
    for path in LIB_LIST:
        _, libname = os.path.split(path)
        try:
            os.remove("raf/%s" % libname)
        except FileNotFoundError:
            pass
