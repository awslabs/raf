from mnm._lib import _APIS

# pylint: disable=invalid-name
# Defined in ./src/impl/build_info.cc
git_version = _APIS.get("mnm.build_info.git_version", None)
# Defined in ./src/impl/build_info.cc
use_cuda = _APIS.get("mnm.build_info.use_cuda", None)
# Defined in ./src/impl/build_info.cc
use_cudnn = _APIS.get("mnm.build_info.use_cudnn", None)
# Defined in ./src/impl/build_info.cc
use_llvm = _APIS.get("mnm.build_info.use_llvm", None)
