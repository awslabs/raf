from mnm._lib import _APIS

# pylint: disable=invalid-name,redefined-builtin
# Defined in ./src/pass/gradient.cc
AutoDiff = _APIS.get("mnm.pass_.AutoDiff", None)
# Defined in ./src/pass/extract_binding.cc
ExtractBinding = _APIS.get("mnm.pass_.ExtractBinding", None)
# Defined in ./src/pass/rename_vars.cc
RenameVars = _APIS.get("mnm.pass_.RenameVars", None)
