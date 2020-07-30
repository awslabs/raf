"""Runtime profiler"""
import json
from mnm import build
from mnm._ffi.profiler import EnableProfiler, DisableProfiler
from mnm._ffi.profiler import CollectCudaProfile, GetProfile


def start():
    """Enable the profiler in backend and start to profile the execution from now.
    """
    EnableProfiler()


def stop():
    """Disable the profiler in backend and stop to profile the execution from now.
    """
    DisableProfiler()


def dump(filename="profile.json"):
    """Dump the profiling results to `filename`.

    Parameters
    ----------
    filename : str
        The location to store the profiling results.
        Default lcoation is "profile.json" in the current folder.
    """
    with open(filename, "w") as f: # pylint: disable=invalid-name
        json.dump(get(), f, indent=4)


def get():
    """Dump the profiling results to string.

    Return
    ----------
        The profiling results in json format.
    """
    if build.with_cuda():
        CollectCudaProfile()
    return json.loads(GetProfile())
