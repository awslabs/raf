/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/nn.h>
#include <array>
#include "./tvmjit_utils.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using schema::BiasAddArgs;
using schema::BinaryArgs;
using schema::SoftmaxArgs;
using schema::SoftmaxDxArgs;

Attrs GEMMNormalizer(TVMOpEnv* env, const BinaryArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x1),
      GetDLTensor(args->x2),
  };
  return Attrs();
}

void GEMMTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

Attrs DenseNormalizer(TVMOpEnv* env, const BinaryArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x1),
      GetDLTensor(args->x2),
  };
  auto attrs = make_object<tvm::relay::DenseAttrs>();
  return Attrs(attrs);
}

MNM_TVMJIT(BatchMatmul, "mnm.op.batch_matmul", BinaryArgs, GEMMNormalizer, GEMMTyper,
           GenericHasher);
MNM_TVMJIT(Dense, "mnm.op.dense", BinaryArgs, DenseNormalizer, GEMMTyper, GenericHasher);

Attrs SoftmaxNormalizer(TVMOpEnv* env, const SoftmaxArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
  };
  auto attrs = make_object<tvm::relay::SoftmaxAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

void SoftmaxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
  };
}

HashKey SoftmaxHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const SoftmaxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

Attrs SoftmaxDxNormalizer(TVMOpEnv* env, const SoftmaxDxArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
      GetDLTensor(args->y),
      GetDLTensor(args->dy),
  };
  auto attrs = make_object<tvm::relay::SoftmaxAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

void SoftmaxDxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
  };
}

HashKey SoftmaxDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const SoftmaxDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Softmax, "mnm.op.softmax", SoftmaxArgs, SoftmaxNormalizer, SoftmaxTyper, SoftmaxHasher);
MNM_TVMJIT(SoftmaxDx, "mnm.op.softmax_dx", SoftmaxDxArgs, SoftmaxDxNormalizer, SoftmaxDxTyper,
           SoftmaxDxHasher);

Attrs BiasAddNormalizer(TVMOpEnv* env, const BiasAddArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
      GetDLTensor(args->bias),
  };
  auto attrs = make_object<tvm::relay::BiasAddAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

void BiasAddTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

HashKey BiasAddHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const BiasAddArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(BiasAdd, "mnm.op.bias_add", BiasAddArgs, BiasAddNormalizer, BiasAddTyper, BiasAddHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
