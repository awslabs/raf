/*!
 * Copyright (c) 2020 by Contributors
 * \file ./src/op/dialect/tvm/annotation.cc
 * \brief annotation operators bridged from TVM.
 */
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/annotation.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using schema::CompilerArgs;

std::vector<Value> CompilerSchema2Args(const CompilerArgs* args) {
  return {args->x};
}

std::vector<std::string> CompilerSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs CompilerSchema2Attrs(const CompilerArgs* args) {
  auto attrs = make_object<CompilerAttrs>();
  attrs->compiler = args->compiler;
  return Attrs(attrs);
}

HashKey CompilerHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const CompilerArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  return key;
}

MNM_TVM(compiler_begin, CompilerBegin, CompilerArgs, CompilerSchema2Args, CompilerSchemaArgNames,
        CompilerSchema2Attrs, CompilerHasher, kOpaque);
MNM_TVM(compiler_end, CompilerEnd, CompilerArgs, CompilerSchema2Args, CompilerSchemaArgNames,
        CompilerSchema2Attrs, CompilerHasher, kOpaque);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
