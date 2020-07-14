from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/profiler/cuda/cuda_profiler.cc
CollectCudaProfile = _APIS.get("mnm.profiler.CollectCudaProfile", None)
# Defined in ./src/profiler/base/profiler.cc
DisableProfiler = _APIS.get("mnm.profiler.DisableProfiler", None)
# Defined in ./src/profiler/base/profiler.cc
EnableProfiler = _APIS.get("mnm.profiler.EnableProfiler", None)
# Defined in ./src/profiler/base/profiler.cc
GetProfile = _APIS.get("mnm.profiler.GetProfile", None)
