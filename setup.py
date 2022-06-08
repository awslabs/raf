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




setup(
    name="rafproject",
    version= 0.2,
    license = "Apache",
    description="RAF Accelerates Deep Learning Frameworks",
    zip_safe=False,
    install_requires=[
        "tvm",
        "numpy",
    ],
    extras_require={
        "test": ["torch==1.6.0", "pytest"],
    },
    packages=find_packages(), 
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




