/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/nn.h>
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../op_utils.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace tvm_attrs;
using schema::BiasAddArgs;
using schema::BinaryArgs;
using schema::ConvArgs;
using schema::ConvDxwArgs;
using schema::LayerNormArgs;
using schema::LayerNormDxArgs;
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
MNM_TVMJIT(Matmul, "mnm.op.matmul", BinaryArgs, DenseNormalizer, GEMMTyper, GenericHasher);
MNM_TVMJIT(MatmulTN, "mnm.op.matmul_tn", BinaryArgs, DenseNormalizer, GEMMTyper, GenericHasher);
MNM_TVMJIT(MatmulNT, "mnm.op.matmul_nt", BinaryArgs, DenseNormalizer, GEMMTyper, GenericHasher);
MNM_TVMJIT(MatmulTT, "mnm.op.matmul_tt", BinaryArgs, DenseNormalizer, GEMMTyper, GenericHasher);
MNM_TVMJIT(Dense, "mnm.op.dense", BinaryArgs, DenseNormalizer, GEMMTyper, GenericHasher);

Attrs Conv2dNormalizer(TVMOpEnv* env, const ConvArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
      GetDLTensor(args->w),
  };
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  auto attrs = make_object<Conv2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  attrs->groups = args->groups;
  attrs->channels = NullValue<tvm::relay::IndexExpr>();
  attrs->kernel_size = NullValue<Array<tvm::relay::IndexExpr> >();
  attrs->data_layout = "NCHW";
  attrs->kernel_layout = "OIHW";
  attrs->out_layout = "NCHW";

  return Attrs(attrs);
}

void Conv2dTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
  };
}

HashKey Conv2dHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const ConvArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->dilation;
  key << args->groups;
  return key;
}

MNM_TVMJIT(Conv2d, "mnm.op.conv2d", ConvArgs, Conv2dNormalizer, Conv2dTyper, Conv2dHasher);

Attrs Conv2dDxwNormalizer(TVMOpEnv* env, const ConvDxwArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x_or_w),
      GetDLTensor(args->y),
      GetDLTensor(args->dy),
  };
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  auto attrs = make_object<Conv2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  // FIXME: (workaround) we use kernel size to store the shape of X (for dx) or W (for dw)
  for (int i = 0; i < args->shape.size(); ++i) {
    attrs->kernel_size.push_back(IntImm(tvm::runtime::DataType::Int(32), args->shape[i]));
  }
  attrs->groups = args->groups;
  attrs->channels = NullValue<tvm::relay::IndexExpr>();
  attrs->data_layout = "NCHW";
  attrs->kernel_layout = "OIHW";
  attrs->out_layout = "NCHW";

  return Attrs(attrs);
}

void Conv2dDxwTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
  };
}

HashKey Conv2dDxwHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const ConvDxwArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->dilation;
  key << args->groups;
  return key;
}

MNM_TVMJIT(Conv2dDx, "mnm.op.conv2d_dx", ConvDxwArgs, Conv2dDxwNormalizer, Conv2dDxwTyper,
           Conv2dDxwHasher);
MNM_TVMJIT(Conv2dDw, "mnm.op.conv2d_dw", ConvDxwArgs, Conv2dDxwNormalizer, Conv2dDxwTyper,
           Conv2dDxwHasher);

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

Attrs LayerNormNormalizer(TVMOpEnv* env, const LayerNormArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->x);
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<tvm::relay::LayerNormAttrs>();
  attrs->axis = args->axis;
  attrs->epsilon = args->eps;
  return Attrs(attrs);
}

void LayerNormTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
  };
}

HashKey LayerNormHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const LayerNormArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->eps;
  return key;
}

MNM_TVMJIT(LayerNorm, "mnm.op.layer_norm", LayerNormArgs, LayerNormNormalizer, LayerNormTyper,
           LayerNormHasher);

Attrs LayerNormDxNormalizer(TVMOpEnv* env, const LayerNormDxArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs = {
      GetDLTensor(args->x),
      GetDLTensor(args->y),
      GetDLTensor(args->dy),
  };
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<tvm::relay::LayerNormAttrs>();
  attrs->axis = args->axis;
  attrs->epsilon = args->eps;
  return Attrs(attrs);
}

void LayerNormDxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
  };
}

HashKey LayerNormDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                          const LayerNormDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->eps;
  return key;
}

MNM_TVMJIT(LayerNormDx, "mnm.op.layer_norm_dx", LayerNormDxArgs, LayerNormDxNormalizer,
           LayerNormDxTyper, LayerNormDxHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
