import def_op
import def_schema
from codegen_utils import NORM_MAP, snake_to_pascal, write_to_file


def gen_file(filename):
    FILE = """
/*!
 * Copyright (c) 2019 by Contributors
 * \\file {FILENAME}
 * \\brief Auto generated. Do not touch.
 */
#include "./regs_utils.h"
{INCLUDES}

namespace mnm {{
namespace op {{
namespace schema {{
namespace {{
{ARG_REGS}
}}  // namespace
}}  // namespace schema
}}  // namespace op
}}  // namespace mnm

namespace mnm {{
namespace op {{
namespace args {{
using namespace mnm::ir;
using namespace mnm::value;
#define MNM_REQUIRED(i, norm, name) attrs->name = norm(values[i]);
#define MNM_OPTIONAL(i, norm, name) \\
  if (size > i) attrs->name = norm(values[i]);
{ARG_INITS}
#undef MNM_OPTIONAL
#undef MNM_REQUIRED
}}  // namespace args
}}  // namespace op
}}  // namespace mnm

namespace mnm {{
namespace op {{
namespace ffi {{
using namespace mnm::ir;
using namespace mnm::value;
using registry::TVMArgs;
#define MNM_REQUIRED(i, norm, name) result.push_back(norm(values[i]));
#define MNM_OPTIONAL(i, norm, name) \\
  if (size > i) result.push_back(norm(values[i]));
{FFI_INITS}
#undef MNM_OPTIONAL
#undef MNM_REQUIRED
}}  // namespace ffi
}}  // namespace op
}}  // namespace mnm

namespace mnm {{
namespace op {{
namespace args {{
#define MNM_BIND_SCHEMA(op_name, schema_name) \\
  MNM_OP_REGISTER(op_name).set_attr<::mnm::op::FMNMSchema>("FMNMSchema", schema_name);
{OP_SCHEMAS}
#undef MNM_BIND_SCHEMA
}}  // namespace args
}}  // namespace op
}}  // namespace mnm

namespace mnm {{
namespace op {{
namespace ffi {{
using registry::TVMArgs;
using registry::TVMRetValue;
{OP_FFI_SYMS}
}}  // namespace ffi
}}  // namespace op
}}  // namespace mnm

namespace mnm {{
namespace op {{
namespace ffi {{
using registry::TVMArgs;
using registry::TVMRetValue;
using executor::interpreter::Interpret;
{OP_FFI_IMPS}
}}  // namespace ffi
}}  // namespace op
}}  // namespace mnm
""".strip()
    schema_headers = def_schema.by_file()
    ops = def_op.by_name()
    schemas = dict()
    for sub_schemas in schema_headers.values():
        schemas.update(sub_schemas)
    includes = "\n".join(map(gen_include, sorted(schema_headers.keys())))
    arg_regs = "\n".join(map(gen_arg_reg, sorted(schemas.keys())))
    arg_inits = "\n".join(gen_arg_init(
        name, schemas[name]) for name in sorted(schemas.keys()))
    ffi_inits = "\n".join(gen_ffi_init(
        name, schemas[name]) for name in sorted(schemas.keys()))
    op_schemas = "\n".join(gen_op_schema(
        name, ops[name].schema_name) for name in sorted(ops.keys()))
    op_ffi_syms = "\n".join(gen_op_ffi_sym(
        ops[name]) for name in sorted(ops.keys()))
    op_ffi_imps = "\n".join(gen_op_ffi_imp(
        ops[name]) for name in sorted(ops.keys()))
    if filename.startswith("./"):
        filename = filename[2:]
    return FILE.format(INCLUDES=includes,
                       ARG_REGS=arg_regs,
                       ARG_INITS=arg_inits,
                       FFI_INITS=ffi_inits,
                       OP_SCHEMAS=op_schemas,
                       OP_FFI_SYMS=op_ffi_syms,
                       OP_FFI_IMPS=op_ffi_imps,
                       FILENAME=filename)


def gen_include(filename):
    INCLUDE = """
#include "../schema/{FILE}"
""".strip()
    return INCLUDE.format(FILE=filename)


def gen_arg_reg(name):
    ARG_REG = """
MNM_REGISTER_OBJECT_REFLECT({CLASS_NAME});
""".strip()
    class_name = snake_to_pascal(name) + "Args"
    return ARG_REG.format(CLASS_NAME=class_name)


def gen_arg_init(name, schema):
    ARG_INIT = """
Attrs {NAME}(const Array<Value> &values) {{
  const int size = values.size();
  CHECK({N_ARG_LB} <= size && size <= {N_ARG_UB});
  auto attrs = make_object<schema::{NAME}Args>();
{ARG_ENTRIES}
  return Attrs(attrs);
}}
""".strip()
    ARG_ENTRY = " " * 2 + """
MNM_{OPTION}({I}, args::{NORM}, {ENTRY});
""".strip()
    name = snake_to_pascal(name)
    n_arg_lb = sum(int(entry.cxx_default is None) for entry in schema)
    n_arg_ub = len(schema)
    arg_entries = []
    for i, entry in enumerate(schema):
        option = "REQUIRED" if entry.cxx_default is None else "OPTIONAL"
        norm = NORM_MAP[entry.cxx_normalizer or entry.cxx_type]
        entry = entry.name
        arg_entries.append(ARG_ENTRY.format(
            I=i, NORM=norm, ENTRY=entry, OPTION=option))
    arg_entries = "\n".join(arg_entries)
    return ARG_INIT.format(NAME=name, N_ARG_LB=n_arg_lb, N_ARG_UB=n_arg_ub, ARG_ENTRIES=arg_entries)


def gen_ffi_init(name, schema):
    ARG_INIT = """
Array<Expr> {NAME}(const TVMArgs &values) {{
  const int size = values.size();
  CHECK({N_ARG_LB} <= size && size <= {N_ARG_UB});
  std::vector<Expr> result;
{ARG_ENTRIES}
  return Array<Expr>(result);
}}
""".strip()
    ARG_ENTRY = " " * 2 + """
MNM_{OPTION}({I}, ffi::{NORM}, {ENTRY});
""".strip()
    name = snake_to_pascal(name)
    n_arg_lb = sum(int(entry.cxx_default is None) for entry in schema)
    n_arg_ub = len(schema)
    arg_entries = []
    for i, entry in enumerate(schema):
        option = "REQUIRED" if entry.cxx_default is None else "OPTIONAL"
        norm = NORM_MAP[entry.cxx_normalizer or entry.cxx_type]
        entry = entry.name
        arg_entries.append(ARG_ENTRY.format(
            I=i, NORM=norm, ENTRY=entry, OPTION=option))
    arg_entries = "\n".join(arg_entries)
    return ARG_INIT.format(NAME=name, N_ARG_LB=n_arg_lb, N_ARG_UB=n_arg_ub, ARG_ENTRIES=arg_entries)


def gen_op_schema(op_name, schema_name):
    OP_SCHEMA = """
MNM_BIND_SCHEMA("mnm.op.{OP_NAME}", args::{SCHEMA_NAME});
""".strip()
    schema_name = snake_to_pascal(schema_name)
    return OP_SCHEMA.format(OP_NAME=op_name, SCHEMA_NAME=schema_name)


def gen_op_ffi_sym(op):
    OP_FFI_SYM = """
MNM_REGISTER_GLOBAL("mnm.op.sym.{OP_NAME}")
.set_body([](TVMArgs args, TVMRetValue *ret) {{
  static Op op = Op::Get("mnm.op.{OP_NAME}");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::{SCHEMA_NAME}(args)),
                                               NullValue<Value>());
}});
""".strip()
    op_name = op.name
    schema_name = snake_to_pascal(op.schema_name)
    return OP_FFI_SYM.format(OP_NAME=op_name, SCHEMA_NAME=schema_name)


def gen_op_ffi_imp(op):
    OP_FFI_IMP = """
MNM_REGISTER_GLOBAL("mnm.op.imp.{OP_NAME}")
.set_body([](TVMArgs args, TVMRetValue *ret) {{
  static Op op = Op::Get("mnm.op.{OP_NAME}");
  *ret = Interpret(CallNode::make(op, ffi::{SCHEMA_NAME}(args)));
}});
""".strip()
    op_name = op.name
    schema_name = snake_to_pascal(op.schema_name)
    return OP_FFI_IMP.format(OP_NAME=op_name, SCHEMA_NAME=schema_name)


def main(path="./src/op/regs/regs.cc"):
    result = gen_file(path)
    write_to_file(path, result)


if __name__ == "__main__":
    main()
