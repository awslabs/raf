/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/lans.cc
 * \brief LANS cuda backend
 */
#include "raf/op.h"
#include "raf/device_api.h"
#include "../../schema/optimizer.h"
//#include "./kernels/multi_tensor_apply.cuh"
#include "./kernels/kernel_util.cuh"

namespace raf {
namespace op {
namespace cuda {

using namespace raf::value;
using device_api::DeviceAPI;
#define CHUNK_SIZE 65536
#define FLOAT_BYTES 4
#define HALF_BYTES 2

class LansImpl : public raf::op::OpEnv {
 public:
  explicit LansImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    static auto lans_op = ir::Op::Get("raf.op.lans");
    auto args = cv->args.as<op::schema::LansArgs>();
    this->arg_indices = {
        fschema_index[lans_op]("tensor_list"),
        fschema_index[lans_op]("step"),
    };
    learning_rate_ = args->learning_rate;
    DLTensor* t0 = ir::Downcast<TensorValue>(args->tensor_list[0]);
    auto datatype = t0->dtype;
    CHECK(datatype.code == kDLFloat);
    CHECK((datatype.bits == 32) || (datatype.bits == 16));

    beta1_ = args->beta1;
    beta2_ = args->beta2;
    eps_ = args->eps;
    bias_correction_ = args->bias_correction;
    weight_decay_ = args->weight_decay;
    grad_averaging_ = args->grad_averaging;
    mode_ = args->mode;
    normalize_grad_ = args->normalize_grad;
    int64_t tensor_elements = 0;

    int n = args->tensor_list.size();
    CHECK(n % 4 == 0);
    param_group_n_ = n / 4;
    for (int i = 0; i < param_group_n_; ++i) {
      DLTensor* t = ir::Downcast<TensorValue>(args->tensor_list[i]);
      int numel = 1;
      for (int j = 0; j < t->ndim; ++j) {
        numel *= t->shape[j];
      }
      void* q_tensor;
      numels_.push_back(numel);
      tensor_elements += numel;
    }
    max_chunks_per_tensor_ = -1;
    if (datatype.bits == 32) {
      RequestWorkspace(&q_tensor_buf_, cv->device, FLOAT_BYTES * tensor_elements);
    } else {
      RequestWorkspace(&q_tensor_buf_, cv->device, HALF_BYTES * tensor_elements);
    }
    for (int t = 0; t < param_group_n_; t++) {
      int max_chunks_this_tensor = (numels_[t] + CHUNK_SIZE - 1) / CHUNK_SIZE;
      if (max_chunks_this_tensor > max_chunks_per_tensor_) {
        max_chunks_per_tensor_ = max_chunks_this_tensor;
      }
    }
    RequestWorkspace(&output_per_tensor_, cv->device, 4 * param_group_n_ * max_chunks_per_tensor_);
    RequestWorkspace(&grad_norm_tensor_, cv->device, 4 * param_group_n_);
    RequestWorkspace(&param_norm_tensor_, cv->device, 4 * param_group_n_);
    RequestWorkspace(&update_m_norm_, cv->device, 4 * param_group_n_);
    RequestWorkspace(&q_norm_tensor_, cv->device, 4 * param_group_n_);
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::LansArgs>();
    Array<Value> tvalue = {args->tensor_list.begin(), args->tensor_list.end()};
    Value tuple = TupleValue::make(tvalue);
    Execute(std::vector<Value>{tuple, args->step}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());
    void* compute_stream = cuda_device_api->GetStream();
    TupleValue tuple = ir::Downcast<TupleValue>(inputs[0]);
    DLTensor* t0 = ir::Downcast<TensorValue>(tuple->fields[0]);
    CHECK(t0->dtype.code == kDLFloat);
    CHECK((t0->dtype.bits == 32) || (t0->dtype.bits == 16));
    DLDevice cpu_ctx;
    cpu_ctx.device_type = kDLCPU;
    cpu_ctx.device_id = 0;
    auto* tstep = inputs[1].as<value::TensorValueObj>();
    tensor::Tensor step_tensor = tstep->tensor;
    CHECK(step_tensor->ndim == 0);
    tvm::runtime::NDArray step_array = step_tensor.CopyTo(cpu_ctx);
    float fstep = reinterpret_cast<float*>(step_array->data)[0];
    int step = (int)fstep;
    float bias_correction1 = 1.0f;
    float bias_correction2 = 1.0f;
    if (bias_correction_ == 1) {
      bias_correction1 = 1 - std::pow(beta1_, step);
      bias_correction2 = 1 - std::pow(beta2_, step);
    }
    float beta3 = 1.0f;
    if (grad_averaging_ == 1) {
      beta3 = 1 - beta1_;
    }

    switch (t0->dtype.bits) {
      case 32: {
        std::vector<float*> tlist;
        for (int i = 0; i < param_group_n_; ++i) {
          DLTensor* tensor = ir::Downcast<TensorValue>(tuple->fields[i]);
          tlist.push_back(static_cast<float*>(tensor->data));
        }
        tlist.push_back(static_cast<float*>(q_tensor_buf_));
        for (int i = 1; i < numels_.size(); ++i) {
          tlist.push_back(static_cast<float*>(q_tensor_buf_) + numels_[i - 1]);
        }
        for (int i = param_group_n_; i < tuple->fields.size(); ++i) {
          DLTensor* tensor = ir::Downcast<TensorValue>(tuple->fields[i]);
          tlist.push_back(static_cast<float*>(tensor->data));
        }
        multi_tensor_lans_cuda<float>(
            CHUNK_SIZE, tlist, learning_rate_, beta1_, beta2_, eps_, bias_correction_,
            bias_correction1, bias_correction2, beta3, weight_decay_, grad_averaging_, mode_,
            normalize_grad_, numels_, compute_stream, static_cast<float*>(output_per_tensor_),
            static_cast<float*>(grad_norm_tensor_), static_cast<float*>(param_norm_tensor_),
            static_cast<float*>(update_m_norm_), static_cast<float*>(q_norm_tensor_),
            max_chunks_per_tensor_);
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported dtype: " << DType(t0->dtype).c_str();
        throw;
      }
    }
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.lans"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new LansImpl(cv);
  }

 private:
  float learning_rate_;
  float beta1_;
  float beta2_;
  float eps_;
  int bias_correction_;
  float weight_decay_;
  int grad_averaging_;
  int mode_;
  bool normalize_grad_;
  std::vector<int> numels_;
  int param_group_n_;
  void* output_per_tensor_;
  void* grad_norm_tensor_;
  void* param_norm_tensor_;
  void* update_m_norm_;
  void* q_norm_tensor_;
  int max_chunks_per_tensor_;
  void* q_tensor_buf_;
};

RAF_REGISTER_DIALECT_OP(cuda, lans, 20);
RAF_OP_ENV_MAKER("raf.op.cuda.lans", LansImpl::make);

}  // namespace cuda
}  // namespace op
}  // namespace raf
