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

Attrs CompilerNormalizer(TVMOpEnv* env, const CompilerArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->x);
  auto attrs = make_object<CompilerAttrs>();
  attrs->compiler = args->compiler;
  return Attrs(attrs);
}

void CompilerTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  y_type[0] = GetTensorType(env->outputs[0]);
  *param_types = {GetTensorType(env->inputs[0])};
}

HashKey CompilerHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const CompilerArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  return key;
}

MNM_TVMJIT(CompilerBegin, "mnm.op.compiler_begin", CompilerArgs, CompilerNormalizer, CompilerTyper,
           CompilerHasher);
MNM_TVMJIT(CompilerEnd, "mnm.op.compiler_end", CompilerArgs, CompilerNormalizer, CompilerTyper,
           CompilerHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
