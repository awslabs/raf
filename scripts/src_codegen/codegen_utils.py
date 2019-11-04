import os
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Arg:
    name: str
    cxx_type: str
    cxx_default: Any = None
    cxx_normalizer: Any = None
    py_type: Any = None
    py_default: Any = None
    py_normalizer: Any = None


@dataclass
class Op:
    name: str
    schema_name: str
    schema: List[Arg] = None


@dataclass
class API:
    name: str
    path: str
    lineno: int


def snake_to_pascal(snake):
    return "".join([x.title() for x in snake.split("_")])


def write_to_file(path, content):
    content = content.strip() + "\n"

    if os.path.exists(path):
        with open(path, "r") as i_f:
            prev_content = i_f.read()
        prev_content = prev_content.strip() + "\n"

        if content == prev_content:
            print("Skip", path)

            return
    print("Writing to", path)
    with open(path, "w") as o_f:
        o_f.write(content)


NORM_MAP = {
    "value::Value": "ToAny",
    "value::TensorValue": "ToTensor",
    "IntTuple": "ToIntTuple",
    "OptionalIntTuple": "ToOptionalIntTuple",
    "int": "ToInt",
    "int64_t": "ToInt",
    "double": "ToDouble",
    "float": "ToDouble",
    "bool": "ToBool",
    "std::string": "ToString",
}
