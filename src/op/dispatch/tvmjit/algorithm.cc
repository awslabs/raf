/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/vision.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/algorithm.h>
#include <mnm/value.h>
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/nn.h"
#include "../../schema/algorithm.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace ::tvm::relay;

Attrs ArgsortNormalizer(TVMOpEnv* env, const ArgsortArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->data);
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = args->axis;
  attrs->is_ascend = args->is_ascend;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

void ArgsortTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTupleType(env->outputs);
  *param_types = {GetTensorType(env->inputs[0])};
}

HashKey ArgsortHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ArgsortArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->is_ascend;
  key << ir::String2DLDataType(args->dtype);
  return key;
}

MNM_TVMJIT(Argsort, "mnm.op.argsort", ArgsortArgs, ArgsortNormalizer, ArgsortTyper, ArgsortHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
