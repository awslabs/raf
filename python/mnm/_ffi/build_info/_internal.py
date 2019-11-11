from mnm._lib import _APIS

# Defined in ./src/impl/build_info.cc, line 22
git_version = _APIS.get("mnm.build_info.git_version", None)
# Defined in ./src/impl/build_info.cc, line 23
use_cuda = _APIS.get("mnm.build_info.use_cuda", None)
# Defined in ./src/impl/build_info.cc, line 24
use_cudnn = _APIS.get("mnm.build_info.use_cudnn", None)
# Defined in ./src/impl/build_info.cc, line 25
use_llvm = _APIS.get("mnm.build_info.use_llvm", None)
