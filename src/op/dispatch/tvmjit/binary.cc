/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/likes.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::op::schema;
using common::shape_utils::GetNumel;

Attrs BinaryNormalizer(TVMOpEnv* env, const BinaryUfuncArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x1),
      GetDLTensor(args->x2),
  };
  return Attrs();
}

void BinaryTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

MNM_TVMJIT(Add, "mnm.op.add", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper, GenericHasher);
MNM_TVMJIT(Subtract, "mnm.op.subtract", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper,
           GenericHasher);
MNM_TVMJIT(Divide, "mnm.op.divide", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper, GenericHasher);
MNM_TVMJIT(Multiply, "mnm.op.multiply", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper,
           GenericHasher);
MNM_TVMJIT(Greater, "mnm.op.greater", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper,
           GenericHasher);
MNM_TVMJIT(Maximum, "mnm.op.maximum", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper,
           GenericHasher);
MNM_TVMJIT(Minimum, "mnm.op.minimum", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper,
           GenericHasher);

struct SumAttrs : public tvm::AttrsNode<SumAttrs> {
  Array<Integer> axis;
  Array<Integer> keepdims;
  TVM_DECLARE_ATTRS(SumAttrs, "attrs.SumAttrs") {
    TVM_ATTR_FIELD(axis);
    TVM_ATTR_FIELD(keepdims);
  }
};
TVM_REGISTER_NODE_TYPE(SumAttrs);

Attrs SumNormalizer(TVMOpEnv* env, const SumArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
  };
  auto attrs = make_object<SumAttrs>();
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    attrs->axis.push_back(args->axis[i]);
  }
  for (int i = 0, n = args->keepdims.size(); i < n; ++i) {
    attrs->keepdims.push_back(args->keepdims[i]);
  }
  return Attrs(attrs);
}

void SumTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
  };
}

HashKey SumHasher(const std::vector<Type>& param_types, const Type& ret_type, const SumArgs* args) {
  HashKey key = GenericHasher<std::nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
    key << args->keepdims[i];
  }
  return key;
}

MNM_TVMJIT(Sum, "mnm.op.sum", SumArgs, SumNormalizer, SumTyper, SumHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
