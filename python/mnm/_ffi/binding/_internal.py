from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/impl/binding.cc
BindNDArray = _APIS.get("mnm.binding.BindNDArray", None)
# Defined in ./src/impl/binding.cc
BindSymbol = _APIS.get("mnm.binding.BindSymbol", None)
# Defined in ./src/impl/binding.cc
LookupBoundValue = _APIS.get("mnm.binding.LookupBoundValue", None)
# Defined in ./src/impl/binding.cc
SetRequiresGrad = _APIS.get("mnm.binding.SetRequiresGrad", None)
