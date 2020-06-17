from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/memory_pool/no_pool/no_pool.cc
no_pool = _APIS.get("mnm.memory_pool._make.no_pool", None)
# Defined in ./src/memory_pool/page_unit_pool/page_unit_pool.cc
page_unit_pool = _APIS.get("mnm.memory_pool._make.page_unit_pool", None)
