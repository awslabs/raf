from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/impl/memory_pool.cc
InitPool = _APIS.get("mnm.memory_pool.InitPool", None)
# Defined in ./src/impl/memory_pool.cc
RemovePool = _APIS.get("mnm.memory_pool.RemovePool", None)
