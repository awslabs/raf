# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory Profiler."""

from raf._ffi.memory_profiler import EnableMemoryProfiler, DisableMemoryeProfiler
from raf._ffi.memory_profiler import ResetMemoryProfiler, GetMaxMemoryInfo, GetMemoryTrace


def start():
    """Enable the profiler in backend to start profiling."""
    EnableMemoryProfiler()


def stop():
    """Disable the profiler in backend to stop profiling."""
    DisableMemoryeProfiler()


def reset():
    """Reset the profiler stats in backend."""
    ResetMemoryProfiler()


def get_max_memory_info(device):
    """Get the max memory info in backend.

    Parameters
    ----------
    device: Device
        The device to get the max memory info.

    Returns
    -------
    ret: Dict[str, float]
        A map of max memory usage info, including max used, max allocated and the trace ID that
        achieves the mex usage.
    """
    return GetMaxMemoryInfo(device)


def get_memory_trace(device):
    """Get the memory trace.

    Parameters
    ----------
    device: Device
        The device to fetch.

    Returns
    -------
    ret: str
        The complete trace in a string.
    """
    return GetMemoryTrace(device)
