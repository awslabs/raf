from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/impl/binding.cc
BindConstValue = _APIS.get("mnm.binding.BindConstValue", None)
# Defined in ./src/impl/binding.cc
BindExprValue = _APIS.get("mnm.binding.BindExprValue", None)
# Defined in ./src/impl/binding.cc
ExtractLetList = _APIS.get("mnm.binding.ExtractLetList", None)
# Defined in ./src/impl/binding.cc
GetGrad = _APIS.get("mnm.binding.GetGrad", None)
# Defined in ./src/impl/binding.cc
LookupBoundExpr = _APIS.get("mnm.binding.LookupBoundExpr", None)
# Defined in ./src/impl/binding.cc
LookupBoundValue = _APIS.get("mnm.binding.LookupBoundValue", None)
# Defined in ./src/impl/binding.cc
SetRequiresGrad = _APIS.get("mnm.binding.SetRequiresGrad", None)
