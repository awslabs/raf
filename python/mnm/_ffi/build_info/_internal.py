from mnm._lib import _APIS

# pylint: disable=invalid-name
# Defined in ./src/impl/build_info.cc, line 27
git_version = _APIS.get("mnm.build_info.git_version", None)
# Defined in ./src/impl/build_info.cc, line 28
use_cuda = _APIS.get("mnm.build_info.use_cuda", None)
# Defined in ./src/impl/build_info.cc, line 29
use_cudnn = _APIS.get("mnm.build_info.use_cudnn", None)
# Defined in ./src/impl/build_info.cc, line 30
use_llvm = _APIS.get("mnm.build_info.use_llvm", None)
