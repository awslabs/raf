/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cublas/l2_norm.cc
 * \brief cuBLAS L2 Norm operator
 */
#include <cublas.h>
#include "raf/op.h"

#include "./cublas_utils.h"
#include "../../schema/reduce.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "../../../profiler/cuda/cuda_profiler.h"

namespace raf {
namespace op {
namespace cublas {
namespace manual {

using namespace raf::value;

class L2NormImpl : public raf::op::OpEnv {
 public:
  explicit L2NormImpl(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.l2norm");
    static auto fschema_index = op::GetOpAttr<op::FRAFSchemaFieldIndex>(op, "FRAFSchemaFieldIndex");
    auto args = cv->args.as<op::schema::L2NormArgs>();
    CHECK(args != nullptr);
    DLTensor* x = ir::Downcast<TensorValue>(args->x);
    n_elements_ = 1;
    for (int i = 0; i < x->ndim; ++i) {
      n_elements_ *= x->shape[i];
    }
    this->arg_indices = {
        fschema_index("x"),
    };
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cublas.l2norm"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::L2NormArgs>();
    Execute(std::vector<value::Value>{args->x}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    DLTensor* x = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* out = ir::Downcast<TensorValue>(output);
    auto handle = CUBlasThreadEntry::ThreadLocal()->handle;
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    if (x->dtype.code == kDLFloat) {
      switch (x->dtype.bits) {
        case 32: {
          CUBLAS_CALL(cublasSnrm2(handle, n_elements_, static_cast<const float*>(x->data), 1,
                                  static_cast<float*>(out->data)));
          CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
          return;
        }
        case 64: {
          CUBLAS_CALL(cublasDnrm2(handle, n_elements_, static_cast<const double*>(x->data), 1,
                                  static_cast<double*>(out->data)));
          CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
          return;
        }
      }
    }
    LOG(FATAL) << "Unsupported output dtype: " << DType(x->dtype).c_str();
    throw;
  }

  static OpEnv* make(const CallValues& cv) {
    return new L2NormImpl(cv);
  }

 private:
  int n_elements_;
};

RAF_REGISTER_DIALECT_OP(cublas, l2norm, 15);
RAF_OP_ENV_MAKER("raf.op.cublas.l2norm", L2NormImpl::make);

}  // namespace manual
}  // namespace cublas
}  // namespace op
}  // namespace raf
