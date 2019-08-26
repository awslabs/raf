from dataclasses import dataclass
from typing import Any
from collections import OrderedDict

import mnm
import tvm


@dataclass
class Argument:
    type: str = None
    default: Any = None


config = {
    "mnm.op.add": {
        "args": OrderedDict(
            x1=Argument(type="array_like"),
            x2=Argument(type="array_like"),
        ),
        "attrs": {},
    },
    "mnm.op.subtract": {
        "args": OrderedDict(
            x1=Argument(type="array_like"),
            x2=Argument(type="array_like"),
        ),
        "attrs": {},
    },
    "mnm.op.multiply": {
        "args": OrderedDict(
            x1=Argument(type="array_like"),
            x2=Argument(type="array_like"),
        ),
        "attrs": {},
    },
    "mnm.op.divide": {
        "args": OrderedDict(
            x1=Argument(type="array_like"),
            x2=Argument(type="array_like"),
        ),
        "attrs": {},
    },
    "mnm.op.mod": {
        "args": OrderedDict(
            x1=Argument(type="array_like"),
            x2=Argument(type="array_like"),
        ),
        "attrs": {},
    },
    "mnm.op.negative": {
        "args": OrderedDict(
            x=Argument(type="array_like"),
        ),
        "attrs": {},
    },
}


print(mnm._core.op.OP_DICT)
