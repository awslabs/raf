# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime profiler"""
import json
from raf import build
from raf._ffi.profiler import EnableProfiler, DisableProfiler
from raf._ffi.profiler import CollectBaseProfile, CollectCudaProfile, GetProfile


def start(prof_level=1):
    """Enable the profiler in backend and start to profile the execution from now.

    Parameters
    ----------
    prof_level : int
        Specify the profiling level.
    """
    EnableProfiler(prof_level)


def stop():
    """Disable the profiler in backend and stop to profile the execution from now."""
    DisableProfiler()


def dump(filename="profile.json"):
    """Dump the profiling results to `filename`.

    Parameters
    ----------
    filename : str
        The location to store the profiling results.
        Default lcoation is "profile.json" in the current folder.
    """
    with open(filename, "w") as f:  # pylint: disable=invalid-name
        json.dump(get(), f, indent=4)


def get():
    """Dump the profiling results to string.

    Return
    ----------
        The profiling results in json format.
    """
    CollectBaseProfile()
    if build.with_cuda():
        CollectCudaProfile()
    return json.loads(GetProfile())


def get_duration(data, event, category=None):
    """
    Get the duration of given event on given category in milliseconds.

    Parameters
    ----------
    data : Dict[str, ...]
        The traced data in google trace event format, See
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview.
        It can be get by raf.utils.profiler.get().

    event : str
        The event name.

    category : Optional[str]
        The category name. None matches any category. Default: None. The available categories
        includes:
        - 'VMInstruction': The events of each virtual machine instruction.
        - 'Default Stream': The kernel executed on the default stream.
        - 'Stream 1': The kernels executed on the first computation stream. There are also
          categories such as 'Stream 2', 'Stream 3' and so on.

    Returns
    -------
    ret : float
        The duration of the event in milliseconds.
    """
    start_time_stamp = None
    end_time_stamp = None
    for e in data["traceEvents"]:
        if (not category or e["cat"] == category) and e["name"] == event and e["ph"] == "B":
            assert start_time_stamp is None, "Multiple events with the same event name"
            start_time_stamp = int(e["ts"])
        if (not category or e["cat"] == category) and e["name"] == event and e["ph"] == "E":
            assert end_time_stamp is None, "Multiple events with the same event name"
            end_time_stamp = int(e["ts"])
    if start_time_stamp is None or end_time_stamp is None:
        raise ValueError(f"The start or end time stamp of event {event} does not exist")
    return float((end_time_stamp - start_time_stamp) / 1000.0)
