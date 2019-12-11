from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/memory_pool/no_pool/no_pool.cc
no_pool = _APIS.get("mnm.memory_pool._make.no_pool", None)
