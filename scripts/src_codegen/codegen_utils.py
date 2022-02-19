# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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


def split_chunks(list_, chunk_size):
    list_ = list(list_)
    for i in range(0, len(list_), chunk_size):
        yield list_[i : i + chunk_size]


NORM_MAP = {
    "value::Value": "ToAny",
    "ir::Optional<value::Value>": "ToAnyOptional",
    "value::BaseTensorValue": "ToTensor",
    "ir::Optional<value::BaseTensorValue>": "ToOptionalTensor",
    "IntTuple": "ToIntTuple",
    "IntArray": "ToIntArray",
    "OptionalIntTuple": "ToOptionalIntTuple",
    "int": "ToInt",
    "int64_t": "ToInt",
    "double": "ToDouble",
    "float": "ToDouble",
    "bool": "ToBool",
    "std::string": "ToString",
    "TensorTuple": "ToTensorTuple",
}

PY_NORM_MAP = {
    "value::Value": "to_any",
    "ir::Optional<value::Value>": "to_any",
    "value::BaseTensorValue": "to_tensor",
    "ir::Optional<value::BaseTensorValue>": "to_tensor",
    "IntTuple": "to_int_tuple",
    "IntArray": "to_int_tuple",
    "OptionalIntTuple": "to_optional_int_tuple",
    "int": "to_int",
    "int64_t": "to_int",
    "double": "to_double",
    "float": "to_double",
    "bool": "to_bool",
    "std::string": "to_string",
    "TensorTuple": "to_tensor_tuple",
}
