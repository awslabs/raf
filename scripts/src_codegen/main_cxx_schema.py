import os
from numbers import Number

import def_schema
from codegen_utils import snake_to_pascal, write_to_file


def gen_file(schemas, filename):
    FILE = """
/*!
 * Copyright (c) 2019 by Contributors
 * \\file {FILENAME}
 * \\brief Operator schema. Auto generated. Do not touch.
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
    result = "\n".join(result)
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
    ARG = " " * 2 + """
  {TYPE} {NAME}{DEFAULT};
""".strip()
    typ = entry.cxx_type
    name = entry.name
    default = entry.cxx_default
    if default is None:
        default = None
    elif isinstance(default, bool):
        default = str(default).lower()
    elif isinstance(default, (Number, str)):
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
