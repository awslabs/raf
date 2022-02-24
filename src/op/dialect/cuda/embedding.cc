/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/embedding.cc
 * \brief embedding_dx cuda backend
 */
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/device_api.h"
#include "../../schema/nn.h"
#include "./kernels/kernel_util.cuh"

namespace raf {
namespace op {
namespace cuda {

using namespace raf::value;
using device_api::DeviceAPI;

class EmbeddingDxImpl : public raf::op::OpEnv {
 public:
  explicit EmbeddingDxImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto op = ir::Op::Get("raf.op.embedding_dx");
    auto args = cv->args.as<op::schema::EmbeddingDxArgs>();
    n_out_elements_ = 1;
    std::vector<int64_t> num_weight = GetShapeVecFromValue(args->num_weight);
    for (int i = 0; i < num_weight.size(); ++i) {
      n_out_elements_ *= num_weight[i];
    }
    this->arg_indices = {
        fschema_index[op]("dy"),
        fschema_index[op]("indices"),
    };
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::EmbeddingDxArgs>();
    Execute(std::vector<value::Value>{args->dy, args->indices}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());
    DLTensor* dy = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* indices = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* out = ir::Downcast<TensorValue>(output);
    int stride = dy->shape[dy->ndim - 1];
    int index_range = out->shape[0];
    int n_indices = 1;
    for (int i = 0; i < indices->ndim; ++i) {
      n_indices *= indices->shape[i];
    }

    CHECK(out->dtype.code == kDLFloat);
    CHECK((out->dtype.bits == 32) || (out->dtype.bits == 16));
    switch (out->dtype.bits) {
      case 32:
        embedding_dense_backward_cuda<float, float>(
            static_cast<const float*>(dy->data), static_cast<float*>(out->data),
            static_cast<const int64_t*>(indices->data), n_indices, index_range, stride,
            cuda_device_api->GetStream(), n_out_elements_);
        return;
      case 16:
        embedding_dense_backward_cuda<__half, __half>(
            static_cast<const __half*>(dy->data), static_cast<__half*>(out->data),
            static_cast<const int64_t*>(indices->data), n_indices, index_range, stride,
            cuda_device_api->GetStream(), n_out_elements_);
        return;
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.embedding_dx"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new EmbeddingDxImpl(cv);
  }

 private:
  int64_t n_out_elements_;
};

RAF_REGISTER_DIALECT_OP(cuda, embedding_dx, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.embedding_dx", EmbeddingDxImpl::make);

}  // namespace cuda
}  // namespace op
}  // namespace raf
