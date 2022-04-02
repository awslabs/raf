/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/lans.cc
 * \brief layer_norm cuda backend
 */
#include "raf/op.h"
#include "raf/device_api.h"
#include "../../schema/nn.h"
#include "./kernels/layer_norm.cuh"

namespace raf {
namespace op {
namespace cuda {

using namespace raf::value;
using device_api::DeviceAPI;

using namespace c10;

class LayerNormTrainImpl : public raf::op::OpEnv {
 public:
  explicit LayerNormTrainImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto layer_norm_op = ir::Op::Get("raf.op.layer_norm_train");
    auto args = cv->args.as<op::schema::LayerNormArgs>();
    this->arg_indices = {
        fschema_index[layer_norm_op]("x"),
        fschema_index[layer_norm_op]("scale"),
        fschema_index[layer_norm_op]("bias"),
    };
    eps_ = args->eps;
    axis_ = args->axis;
    DLTensor *x, *scale, *bias;
    x = ir::Downcast<TensorValue>(args->x);

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, cv->device.device_id()));
    maxGridY_ = deviceProp.maxGridSize[1];

    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());
    compute_stream_ = cuda_device_api->GetStream();
    if (args->scale.defined()) {
      scale = ir::Downcast<TensorValue>(args->scale.value());
    } else {
      scale = nullptr;
    }
    if (args->bias.defined()) {
      bias = ir::Downcast<TensorValue>(args->bias.value());
    } else {
      bias = nullptr;
    }
    int64_t* normalized_shape;
    int64_t idiff;
    int normalized_ndim;
    if (scale) {
      normalized_shape = scale->shape;
      idiff = x->ndim - scale->ndim;
      normalized_ndim = scale->ndim;
    } else {
      normalized_shape = &x->shape[x->ndim - 1];
      idiff = x->ndim - 1;
      normalized_ndim = 1;
    }
    n2_ = 1;
    for (int i = 0; i < normalized_ndim; ++i) {
      n2_ *= normalized_shape[i];
    }
    n1_ = 1;
    for (int i = 0; i < idiff; ++i) {
      n1_ *= x->shape[i];
    }
    auto datatype = x->dtype;
    CHECK(datatype.code == kDLFloat);
    CHECK((datatype.bits == 32) || (datatype.bits == 16));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::LayerNormArgs>();
    Execute(std::vector<Value>{args->x, args->scale.value(), args->bias.value()}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    DLTensor* x = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* scale = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* bias = ir::Downcast<TensorValue>(inputs[2]);

    TupleValue out_tuple = ir::Downcast<TupleValue>(output);
    DLTensor* out = ir::Downcast<TensorValue>(out_tuple->fields[0]);
    DLTensor* mean = ir::Downcast<TensorValue>(out_tuple->fields[1]);
    float* mean_p = static_cast<float*>(mean->data);
    DLTensor* invvar = ir::Downcast<TensorValue>(out_tuple->fields[2]);
    float* invvar_p = static_cast<float*>(invvar->data);
    switch (x->dtype.bits) {
      case 16: {
        HostApplyLayerNorm<Half, float, Half>(
            static_cast<Half*>(out->data), mean_p, invvar_p, static_cast<Half*>(x->data), n1_, n2_,
            static_cast<Half*>(scale->data), static_cast<Half*>(bias->data), eps_, compute_stream_,
            maxGridY_);

        break;
      }
      case 32: {
        HostApplyLayerNorm<float, float, float>(
            static_cast<float*>(out->data), mean_p, invvar_p, static_cast<float*>(x->data), n1_,
            n2_, static_cast<float*>(scale->data), static_cast<float*>(bias->data), eps_,
            compute_stream_, maxGridY_);

        break;
      }
      default: {
        LOG(FATAL) << "Unsupported dtype: " << DType(x->dtype).c_str();
        throw;
      }
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.layer_norm_train"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new LayerNormTrainImpl(cv);
  }

 private:
  int axis_;
  double eps_;
  int n1_, n2_;
  uint64_t maxGridY_;
  void* compute_stream_;
};

RAF_REGISTER_DIALECT_OP(cuda, layer_norm_train, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.layer_norm_train", LayerNormTrainImpl::make);

class LayerNormTrainDxImpl : public raf::op::OpEnv {
 public:
  explicit LayerNormTrainDxImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto dx_op = ir::Op::Get("raf.op.layer_norm_train_dx");
    auto args = cv->args.as<op::schema::LayerNormTrainDxArgs>();
    this->arg_indices = {
        fschema_index[dx_op]("x"),    fschema_index[dx_op]("scale"),  fschema_index[dx_op]("dy"),
        fschema_index[dx_op]("mean"), fschema_index[dx_op]("invvar"),
    };
    eps_ = args->eps;
    axis_ = args->axis;
    DLTensor *x, *scale;
    x = ir::Downcast<TensorValue>(args->x);
    if (args->scale.defined()) {
      scale = ir::Downcast<TensorValue>(args->scale.value());
    } else {
      scale = nullptr;
    }
    int64_t* normalized_shape;
    int64_t idiff;
    int normalized_ndim;
    if (scale) {
      normalized_shape = scale->shape;
      idiff = x->ndim - scale->ndim;
      normalized_ndim = scale->ndim;
    } else {
      normalized_shape = &x->shape[x->ndim - 1];
      idiff = x->ndim - 1;
      normalized_ndim = 1;
    }
    n2_ = 1;
    for (int i = 0; i < normalized_ndim; ++i) {
      n2_ *= normalized_shape[i];
    }
    n1_ = 1;
    for (int i = 0; i < idiff; ++i) {
      n1_ *= x->shape[i];
    }
    if (scale) {
      const int part_size = 16;
      RequestWorkspace(&part_grad_gamma_, x->device, 4 * part_size * n2_);
      RequestWorkspace(&part_grad_beta_, x->device, 4 * part_size * n2_);
    }

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, cv->device.device_id()));
    maxGridY_ = deviceProp.maxGridSize[1];

    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());
    compute_stream_ = cuda_device_api->GetStream();

    auto datatype = x->dtype;
    CHECK(datatype.code == kDLFloat);
    CHECK((datatype.bits == 32) || (datatype.bits == 16));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::LayerNormTrainDxArgs>();
    Execute(std::vector<Value>{args->x, args->scale.value(), args->dy, args->mean, args->invvar},
            cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    DLTensor* x = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* scale = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = ir::Downcast<TensorValue>(inputs[2]);
    DLTensor* mean = ir::Downcast<TensorValue>(inputs[3]);
    DLTensor* invvar = ir::Downcast<TensorValue>(inputs[4]);
    float* mean_p = static_cast<float*>(mean->data);
    float* invvar_p = static_cast<float*>(invvar->data);

    TupleValue out_tuple = ir::Downcast<TupleValue>(output);
    DLTensor* dx = ir::Downcast<TensorValue>(out_tuple->fields[0]);
    DLTensor* dw = ir::Downcast<TensorValue>(out_tuple->fields[1]);
    DLTensor* db = ir::Downcast<TensorValue>(out_tuple->fields[2]);
    CHECK(x->dtype.code == kDLFloat);
    CHECK((x->dtype.bits == 32) || (x->dtype.bits == 16));

    switch (x->dtype.bits) {
      case 16: {
        HostLayerNormGradient<Half, Half>(
            static_cast<Half*>(dy->data), mean_p, invvar_p, static_cast<Half*>(x->data), n1_, n2_,
            static_cast<Half*>(scale->data), eps_, static_cast<Half*>(dx->data),
            static_cast<Half*>(dw->data), static_cast<Half*>(db->data),
            part_grad_gamma_ != NULL ? static_cast<float*>(part_grad_gamma_) : NULL,
            part_grad_beta_ != NULL ? static_cast<float*>(part_grad_beta_) : NULL, compute_stream_,
            maxGridY_);
        break;
      }
      case 32: {
        HostLayerNormGradient<float, float>(
            static_cast<float*>(dy->data), mean_p, invvar_p, static_cast<float*>(x->data), n1_, n2_,
            static_cast<float*>(scale->data), eps_, static_cast<float*>(dx->data),
            static_cast<float*>(dw->data), static_cast<float*>(db->data),
            part_grad_gamma_ != NULL ? static_cast<float*>(part_grad_gamma_) : NULL,
            part_grad_beta_ != NULL ? static_cast<float*>(part_grad_beta_) : NULL, compute_stream_,
            maxGridY_);

        break;
      }
      default: {
        LOG(FATAL) << "Unsupported dtype: " << DType(x->dtype).c_str();
        throw;
      }
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.layer_norm_train_dx"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new LayerNormTrainDxImpl(cv);
  }

 private:
  int axis_;
  double eps_;
  int n1_, n2_;
  void* part_grad_gamma_ = nullptr;
  void* part_grad_beta_ = nullptr;
  uint64_t maxGridY_;
  void* compute_stream_;
};

RAF_REGISTER_DIALECT_OP(cuda, layer_norm_train_dx, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.layer_norm_train_dx", LayerNormTrainDxImpl::make);

}  // namespace cuda
}  // namespace op
}  // namespace raf
