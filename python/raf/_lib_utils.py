# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The utility for library loading."""
import os
import sys


def find_lib_path(name=None, search_path=None):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.
    search_path : str
        Root path to search.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    package_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..")

    dll_path = []

    if os.environ.get("RAF_LIBRARY_PATH", None):
        dll_path.append(os.environ["RAF_LIBRARY_PATH"])

    if sys.platform.startswith("linux") and os.environ.get("LD_LIBRARY_PATH", None):
        dll_path.extend([p.strip() for p in os.environ["LD_LIBRARY_PATH"].split(":")])
    elif sys.platform.startswith("darwin") and os.environ.get("DYLD_LIBRARY_PATH", None):
        dll_path.extend([p.strip() for p in os.environ["DYLD_LIBRARY_PATH"].split(":")])

    # Package data directory when pip installed
    dll_path.append(package_dir)
    # Pip lib directory
    dll_path.append(os.path.join(ffi_dir, ".."))
    # Default cmake build directory
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "lib"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))
    dll_path.append(install_lib_dir)

    dll_path = [os.path.realpath(x) for x in dll_path]

    if search_path is not None:
        if not isinstance(search_path, list):
            search_path = [search_path]
        dll_path.extend(search_path)

    ext = ".so"
    if sys.platform.startswith("win32"):
        ext = ".dll"
    elif sys.platform.startswith("darwin"):
        ext = ".dylib"

    if name is not None:
        if not isinstance(name, list):
            name = [name]
        lib_dll_path = [os.path.join(p, n + ext) for n in name for p in dll_path]
    else:
        lib_dll_path = [os.path.join(p, "libraf" + ext) for p in dll_path]
        if sys.platform.startswith("win32"):
            lib_dll_path += [os.path.join(p, "raf" + ext) for p in dll_path]

    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        message = (
            "Cannot find the files.\n" + "List of candidates:\n" + str("\n".join(lib_dll_path))
        )
        raise RuntimeError(message)

    return lib_found
