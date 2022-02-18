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

"""Memory Profiler."""

from mnm._ffi.memory_profiler import EnableMemoryProfiler, DisableMemoryeProfiler
from mnm._ffi.memory_profiler import ResetMemoryProfiler, GetMaxMemoryInfo, GetMemoryTrace


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
