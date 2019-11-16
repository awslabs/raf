from mnm._lib import _APIS

# pylint: disable=invalid-name
# Defined in ./src/impl/interpreter.cc, line 350
CreateInterpreter = _APIS.get("mnm.executor.CreateInterpreter", None)
# Defined in ./src/impl/interpreter.cc, line 351
InterpretWithGlobal = _APIS.get("mnm.executor.InterpretWithGlobal", None)
