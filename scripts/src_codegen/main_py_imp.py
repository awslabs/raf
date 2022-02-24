# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from numbers import Number

from . import def_op
from .codegen_utils import PY_NORM_MAP as NORM_MAP
from .codegen_utils import split_chunks, write_to_file


def gen_file():
    FILE = """
# pylint: disable=invalid-name,line-too-long,too-many-lines
# pylint: disable=too-many-arguments,redefined-builtin,redefined-outer-name
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=protected-access, too-many-lines
\"\"\"Auto generated. Do not touch.\"\"\"
import raf._ffi.op.imp as ffi
from raf._core.core_utils import set_module
from . import imp_utils

__all__ = [
{OP_NAMES}
]

{METHODS}
""".strip()
    ops = def_op.by_name()
    methods = "\n\n".join(gen_method(ops[name]) for name in sorted(ops.keys()))
    op_names = "\n".join(
        map(
            lambda x: '    "' + '", "'.join([i.replace(".", "_") for i in x]) + '",',
            split_chunks(sorted(ops.keys()), chunk_size=5),
        )
    )
    return FILE.format(METHODS=methods, OP_NAMES=op_names)


def gen_method(op):
    METHOD = """
@set_module("raf")
def {FUNC_NAME}({PARAMS_W_DEFAULT}):
{NORMS}
    return imp_utils.ret(ffi.{OP_NAME}({PARAMS_WO_DEFAULT}))
""".strip()
    norms = "\n".join(map(gen_norm, op.schema))
    param_w = gen_param_w_default(op.schema)
    param_wo = gen_param_wo_default(op.schema)
    return METHOD.format(
        FUNC_NAME=op.name.replace(".", "_"),
        OP_NAME=op.name,
        NORMS=norms,
        PARAMS_W_DEFAULT=param_w,
        PARAMS_WO_DEFAULT=param_wo,
    )


def gen_norm(entry):
    NORM = (
        " " * 4
        + """
    {NAME} = imp_utils.{NORM}({NAME})
""".strip()
    )
    name = entry.name
    norm = NORM_MAP[entry.py_normalizer or (entry.cxx_normalizer or entry.cxx_type)]
    return NORM.format(NAME=name, NORM=norm)


def gen_param_w_default(schema):
    result = []
    for entry in schema:
        name = entry.name
        # case 1: no default value
        if entry.cxx_default is None:
            result.append(name)
            continue
        # case 2: python-specific default value is provided
        if entry.py_default is not None:
            default = entry.py_default
        # case 3: no python-specific default is provided
        elif entry.cxx_default == "nullptr":
            default = None
        elif isinstance(entry.cxx_default, Number):
            default = entry.cxx_default
        else:
            raise NotImplementedError(entry)
        result.append(f"{name}={default}")
    return ", ".join(result)


def gen_param_wo_default(schema):
    return ", ".join(arg.name for arg in schema)


def main(path="./python/raf/_op/imp.py"):
    result = gen_file()
    write_to_file(path, result)


if __name__ == "__main__":
    main()
