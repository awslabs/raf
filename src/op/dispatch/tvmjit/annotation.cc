/*!
 * Copyright (c) 2020 by Contributors
 * \file ./src/op/dispatch/tvmjit/annotation.cc
 * \brief annotation operators bridged from TVM.
 */
#include <tvm/relay/attrs/annotation.h>
#include "./tvmjit_utils.h"
#include "../../schema/annotation.h"

namespace mnm {
namespace op {
namespace tvmjit {

using schema::CompilerArgs;
using namespace tvm;
using namespace tvm::relay;

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

MNM_TVMJIT(CompilerBegin, "mnm.op.compiler_begin", CompilerArgs, CompilerSchema2Args,
           CompilerSchemaArgNames, CompilerSchema2Attrs, CompilerHasher);
MNM_TVMJIT(CompilerEnd, "mnm.op.compiler_end", CompilerArgs, CompilerSchema2Args,
           CompilerSchemaArgNames, CompilerSchema2Attrs, CompilerHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
