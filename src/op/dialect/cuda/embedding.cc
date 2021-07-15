/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dialect/cuda/kernels/transform.cc
 * \brief embedding_dx cuda backend
 */
#include "mnm/op.h"
#include "../../schema/nn.h"
#include "./kernels/kernel_util.cuh"

namespace mnm {
namespace op {
namespace cuda {

using namespace mnm::value;

class EmbeddingDxImpl : public mnm::op::OpEnv {
 public:
  explicit EmbeddingDxImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    static auto op = ir::Op::Get("mnm.op.embedding_dx");
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
    DLTensor* dy = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* indices = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* out = ir::Downcast<TensorValue>(output);
    int stride = dy->shape[dy->ndim - 1];
    int range = out->shape[0];
    int n = 1;
    for (int i = 0; i < indices->ndim; ++i) {
      n *= indices->shape[i];
    }

    CHECK(out->dtype.code == kDLFloat);
    CHECK((out->dtype.bits == 32) || (out->dtype.bits == 16));
    switch (out->dtype.bits) {
      case 32:
        embedding_dense_backward_cuda<float, float>(
            static_cast<const float*>(dy->data), static_cast<float*>(out->data),
            static_cast<const int64_t*>(indices->data), n, range, stride);
        return;
      case 16:
        embedding_dense_backward_cuda<__half, __half>(
            static_cast<const __half*>(dy->data), static_cast<__half*>(out->data),
            static_cast<const int64_t*>(indices->data), n, range, stride);
        return;
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cuda.embedding_dx"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new EmbeddingDxImpl(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cuda, embedding_dx);
MNM_OP_ENV_MAKER("mnm.op.cuda.embedding_dx", EmbeddingDxImpl::make);
MNM_OP_DISPATCH_DIALECT_PLEVEL(embedding_dx, cuda, DevType::kCUDA(), 20);

}  // namespace cuda
}  // namespace op
}  // namespace mnm
