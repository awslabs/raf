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
using common::shape_utils::GetNumel;
using schema::BinaryUfuncArgs;
using schema::SumArgs;

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

MNM_TVMJIT(Add, "mnm.op.add", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper);
MNM_TVMJIT(Subtract, "mnm.op.subtract", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper);
MNM_TVMJIT(Multiply, "mnm.op.multiply", BinaryUfuncArgs, BinaryNormalizer, BinaryTyper);


struct SumAttrs : public tvm::AttrsNode<SumAttrs> {
  Array<Integer> axis;
  Array<Integer> keep;
  // declare attribute fields in header file
  TVM_DECLARE_ATTRS(SumAttrs, "attrs.SgdAttrs") {
    TVM_ATTR_FIELD(axis);
    TVM_ATTR_FIELD(keep);
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
    attrs->keep.push_back(args->keep[i]);
  }
  return Attrs(attrs);
}

void SumTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
  };
}

MNM_TVMJIT(Sum, "mnm.op.sum", SumArgs, SumNormalizer, SumTyper);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
