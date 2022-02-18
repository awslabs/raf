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
from numbers import Number

from . import def_schema
from .codegen_utils import snake_to_pascal, write_to_file


def gen_file(schemas, filename):
    FILE = """
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Auto generated. Do not touch.
 * \\file {FILENAME}
 * \\brief Operator schema.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/value.h"
namespace mnm {{
namespace op {{
namespace schema {{
{CLASSES}
}}  // namespace schema
}}  // namespace op
}}  // namespace mnm
""".strip()
    result = []
    for name in sorted(schemas.keys()):
        schema = schemas[name]
        result.append(gen_class(name, schema))
    result = "\n\n".join(result)
    if filename.startswith("./"):
        filename = filename[2:]
    return FILE.format(CLASSES=result, FILENAME=filename)


def gen_class(name, schema):
    CLASS = """
class {CLASS_NAME} : public ir::AttrsNode<{CLASS_NAME}> {{
 public:
{ARGS}
  MNM_OP_SCHEMA({CLASS_NAME}, "{SCHEMA_NAME}");
}};
""".strip()
    class_name = snake_to_pascal(name) + "Args"
    schema_name = "mnm.args." + name
    args = "\n".join(gen_arg(entry) for entry in schema)
    return CLASS.format(CLASS_NAME=class_name, SCHEMA_NAME=schema_name, ARGS=args)


def gen_arg(entry):
    ARG = (
        " " * 2
        + """
  {TYPE} {NAME}{DEFAULT};
""".strip()
    )
    typ = entry.cxx_type
    name = entry.name
    default = entry.cxx_default
    if default is None:
        default = None
    elif isinstance(default, bool):
        default = str(default).lower()
    elif isinstance(default, (Number, str)):
        if entry.cxx_normalizer == "IntArray" and len(default) > 2:
            value = default[1:-1].split(",")
            re = "ir::Array<value::IntValue> {"
            for i in value:
                re += "value::IntValue::make(" + i + ")"
            re += "}"
            default = re
        else:
            default = str(default)
    else:
        raise NotImplementedError(entry)
    if default is None:
        default = ""
    elif not (default.startswith("{") and default.endswith("}")):
        default = "{" + default + "}"
    return ARG.format(TYPE=typ, NAME=name, DEFAULT=default)


def main(path_prefix="./src/op/schema/"):
    files = def_schema.by_file()
    for file_name in sorted(files.keys()):
        schemas = files[file_name]
        path = os.path.join(path_prefix, file_name)
        result = gen_file(schemas, path)
        write_to_file(path, result)


if __name__ == "__main__":
    main()
