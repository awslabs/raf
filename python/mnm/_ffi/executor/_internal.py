from mnm._lib import _APIS

# pylint: disable=invalid-name
# Defined in ./src/impl/interpreter.cc, line 347
CreateInterpreter = _APIS.get("mnm.executor.CreateInterpreter", None)
# Defined in ./src/impl/interpreter.cc, line 348
InterpretWithGlobal = _APIS.get("mnm.executor.InterpretWithGlobal", None)
