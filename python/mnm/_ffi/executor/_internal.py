from mnm._lib import _APIS

# Defined in ./src/impl/interpreter.cc, line 347
CreateInterpreter = _APIS.get("mnm.executor.CreateInterpreter", None)
# Defined in ./src/impl/interpreter.cc, line 348
InterpretWithGlobal = _APIS.get("mnm.executor.InterpretWithGlobal", None)
