"""Library information.

This file is borrowed from TVM with minor modification that allows us to load MNM libraries.
"""
import ctypes
import os
import sys


def find_lib_path(name=None, search_path=None):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..", "..")

    dll_path = []

    if os.environ.get('MNM_LIBRARY_PATH', None):
        dll_path.append(os.environ['MNM_LIBRARY_PATH'])

    if sys.platform.startswith('linux') and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip()
                         for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    elif sys.platform.startswith('darwin') and os.environ.get('DYLD_LIBRARY_PATH', None):
        dll_path.extend([p.strip()
                         for p in os.environ['DYLD_LIBRARY_PATH'].split(":")])

    # Pip lib directory
    dll_path.append(os.path.join(ffi_dir, ".."))
    # Default cmake build directory
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))
    dll_path.append(install_lib_dir)

    dll_path = [os.path.realpath(x) for x in dll_path]
    if search_path is not None:
        if isinstance(search_path, list):
            dll_path.extend(search_path)
        elif isinstance(search_path, str):
            dll_path.append(search_path)
    if name is not None:
        if isinstance(name, list):
            lib_dll_path = []
            for n in name:
                lib_dll_path += [os.path.join(p, n) for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, name) for p in dll_path]
    else:
        if sys.platform.startswith('win32'):
            lib_dll_path = [os.path.join(p, 'libmnm.dll') for p in dll_path] +\
                           [os.path.join(p, 'mnm.dll') for p in dll_path]
        elif sys.platform.startswith('darwin'):
            lib_dll_path = [os.path.join(p, 'libmnm.dylib') for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, 'libmnm.so') for p in dll_path]

    lib_found = [p for p in lib_dll_path if os.path.exists(
        p) and os.path.isfile(p)]

    if not lib_found:
        message = ('Cannot find the files.\n' +
                   'List of candidates:\n' +
                   str('\n'.join(lib_dll_path)))
        raise RuntimeError(message)

    return lib_found


def _load_lib():
    lib_path = find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib, os.path.basename(lib_path[0])


_LIB, _LIB_NAME = _load_lib()
