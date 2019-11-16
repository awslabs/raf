from mnm._lib import _APIS

# pylint: disable=invalid-name
# Defined in ./src/device_api/cpu/cpu.cc, line 69
cpu = _APIS.get("mnm.device_api._make.cpu", None)
# Defined in ./src/device_api/cuda/cuda.cc, line 74
cuda = _APIS.get("mnm.device_api._make.cuda", None)
