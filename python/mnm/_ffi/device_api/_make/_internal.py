from mnm._lib import _APIS

# Defined in ./src/device_api/cpu/cpu.cc, line 64
cpu = _APIS.get("mnm.device_api._make.cpu", None)
# Defined in ./src/device_api/cuda/cuda.cc, line 70
cuda = _APIS.get("mnm.device_api._make.cuda", None)
