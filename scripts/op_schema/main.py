import os
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from typing import Any, List

CXX_SCHEMA_HEADER = """
#pragma once
#include "./utils.h"
namespace mnm {
namespace op {
namespace args {
""".strip()
CXX_SCHEMA_FOOTER = """
}  // namespace args
}  // namespace op
}  // namespace mnm
""".strip()

CXX_REG_HEADER = """
namespace mnm {
namespace op {
namespace args {
namespace {
MNM_REGISTER_NODE_TYPE(ListArgs);
""".strip()
CXX_REG_FOOTER = """
}
}  // namespace args
}  // namespace op
}  // namespace mnm
""".strip()


@dataclass
class Arg:
    name: str
    cxx_type: str
    cxx_default: Any = None
    cxx_normalizer: str = ""
    py_type = None
    py_default = None
    py_normalizer = None


@dataclass
class Op:
    cxx_path: str
    py_path: str


schemas = {
    "nn.h::conv": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="w", cxx_type="value::TensorValue"),
        Arg(name="stride", cxx_type="std::vector<int64_t>",
            cxx_default="{1}", cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="padding", cxx_type="std::vector<int64_t>",
            cxx_default="{0}", cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>",
            cxx_default="{1}", cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="groups", cxx_type="int64_t", cxx_default=1),
    ],
    "nn.h::pool": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="kernel", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="stride", cxx_type="std::vector<int64_t>",
            cxx_default="{}", cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="padding", cxx_type="std::vector<int64_t>",
            cxx_default="{0}", cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>",
            cxx_default="{1}", cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="ceil_mode", cxx_type="bool", cxx_default=False),
        Arg(name="include_pad", cxx_type="bool", cxx_default=True),
    ],
    "nn.h::softmax": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "nn.h::batch_norm": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="running_mean", cxx_type="value::TensorValue"),
        Arg(name="running_var", cxx_type="value::TensorValue"),
        Arg(name="scale", cxx_type="value::TensorValue", cxx_default="nullptr"),
        Arg(name="bias", cxx_type="value::TensorValue", cxx_default="nullptr"),
        Arg(name="eps", cxx_type="double", cxx_default="1e-5"),
        Arg(name="momentum", cxx_type="double", cxx_default="0.1"),
    ],
    "nn.h::local_response_norm": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="size", cxx_type="int64_t"),
        Arg(name="alpha", cxx_type="double", cxx_default="1e-4"),
        Arg(name="beta", cxx_type="double", cxx_default="0.75"),
        Arg(name="k", cxx_type="double", cxx_default="1.0"),
    ],
    "nn.h::conv_dxw": [
        Arg(name="x_or_w", cxx_type="value::TensorValue"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="stride", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="padding", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="groups", cxx_type="int64_t"),
    ],
    "nn.h::pool_dx": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="kernel", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="stride", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="padding", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>",
            cxx_normalizer="NormalizeTupleOrInt"),
        Arg(name="ceil_mode", cxx_type="bool"),
        Arg(name="include_pad", cxx_type="bool"),
    ],
    "nn.h::softmax_dx": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "ufunc.h::unary_ufunc": [
        Arg(name="x", cxx_type="value::Value"),
        Arg(name="out", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="where", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "ufunc.h::binary_ufunc": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="out", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="where", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "ufunc.h::ternary_ufunc": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
        Arg(name="out", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="where", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "ufunc.h::unary": [
        Arg(name="x", cxx_type="value::Value"),
    ],
    "ufunc.h::binary": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
    ],
    "ufunc.h::ternary": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
    ],
    "ufunc.h::unary_dx": [
        Arg(name="x", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
    ],
    "ufunc.h::binary_dx": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
    ],
    "ufunc.h::ternary_dx": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
    ],
}


def snake_to_pascal(snake):
    return "".join([x.title() for x in snake.split("_")])


def print_arg(arg: Arg, o_f):
    cxx_type = arg.cxx_type
    name = arg.name
    default = arg.cxx_default
    if default is None:
        print(f"  {cxx_type} {name};", file=o_f)
        return
    if isinstance(default, (Number, str)):
        if isinstance(default, bool):
            default = str(default).lower()
        else:
            default = str(default)
        if not (default.startswith("{") and default.endswith("}")):
            default = "{" + default + "}"
        print(f"  {cxx_type} {name}{default};", file=o_f)
        return
    raise NotImplementedError(default)


def print_schema(idx: int, arg: Arg, o_f):
    cxx_type = arg.cxx_type
    is_required = arg.cxx_default is None
    if arg.cxx_normalizer:
        cxx_normalizer = arg.cxx_normalizer
    elif cxx_type.startswith("value::"):
        cxx_normalizer = "ir::Downcast<" + cxx_type + ">"
    else:
        cxx_normalizer = "To" + cxx_type.replace("value::", "").replace("std::", "").title().replace("_T", "")
    name = arg.name
    prefix = "MNM_ARG_REQUIRED" if is_required else "MNM_ARG_OPTIONAL"
    prefix = "    " + prefix
    print(f"{prefix}({idx}, {cxx_normalizer}, {name});", file=o_f)


def gen_cxx_schema(path_prefix="./src/op/args/"):
    files = defaultdict(dict)
    for name, schema in schemas.items():
        file_name, schema_name = name.split("::")
        files[file_name][schema_name] = schema
        is_optional = False
        for arg in schema:
            if arg.cxx_default is None and is_optional:
                raise ValueError(
                    f"In {name}, required arguments should precede optional arguments")
            if arg.cxx_default is not None:
                is_optional = True

    for file_name in sorted(files.keys()):
        o_f = open(os.path.join(path_prefix, file_name), "w")
        print(CXX_SCHEMA_HEADER, file=o_f)
        for schema_name in sorted(files[file_name]):
            schema = files[file_name][schema_name]
            class_name = snake_to_pascal(schema_name) + "Args"
            schema_name = "mnm.args." + schema_name
            print("class {0} : public ir::AttrsNode<{0}> {{".format(
                class_name), file=o_f)
            print(" public:", file=o_f)
            for arg in schema:
                print_arg(arg, o_f)
            print(
                f"  MNM_OP_SCHEMA({class_name}, \"{schema_name}\") {{", file=o_f)
            for i, arg in enumerate(schema):
                print_schema(i, arg, o_f)
            print("  }", file=o_f)
            print("};", file=o_f)
        print(CXX_SCHEMA_FOOTER, file=o_f)
        o_f.close()

    with open(os.path.join(path_prefix, "args.cc"), "w") as o_f:
        for header in sorted(list(files.keys()) + ["list_args.h"]):
            print(f"#include \"./{header}\"", file=o_f)
        print(CXX_REG_HEADER, file=o_f)
        for name in sorted(schemas.keys()):
            _, schema_name = name.split("::")
            class_name = snake_to_pascal(schema_name) + "Args"
            print(f"MNM_REGISTER_NODE_TYPE({class_name});", file=o_f)
        print(CXX_REG_FOOTER, file=o_f)


gen_cxx_schema()
