/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dispatch/cudnn/softmax.cc
 * \brief CUDNN softmax operators.
 */
#include "../../schema/nn.h"
#include "./cudnn_utils.h"
#include "mnm/ir.h"
#include "mnm/op_utils.h"

namespace mnm {
namespace op {
namespace cudnn {

using namespace mnm::value;
using namespace mnm::ir;

static auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");

class LogSoftmaxImplementedByCUDNNSoftmaxForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;

  explicit LogSoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.log_softmax");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.log_softmax"));
  }

 public:
  ~LogSoftmaxImplementedByCUDNNSoftmaxForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG, mode,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG, mode,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new LogSoftmaxImplementedByCUDNNSoftmaxForward(cv);
  }
};

MNM_OP_DISPATCH_PLEVEL("mnm.op.log_softmax", LogSoftmaxImplementedByCUDNNSoftmaxForward::make,
                       DevType::kCUDA(), "cudnn", 10);

class LogSoftmaxDxImplementedByCUDNNSoftmaxBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;

  explicit LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.log_softmax_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, axis, axis + 1, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, axis, axis + 1, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.log_softmax_dx"));
  }

 public:
  ~LogSoftmaxDxImplementedByCUDNNSoftmaxBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(cv);
  }
};

MNM_OP_DISPATCH_PLEVEL("mnm.op.log_softmax_dx", LogSoftmaxDxImplementedByCUDNNSoftmaxBackward::make,
                       DevType::kCUDA(), "cudnn", 7);

class SoftmaxImplementedByCUDNNSoftmaxForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;

  explicit SoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.softmax");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.softmax"));
  }

 public:
  ~SoftmaxImplementedByCUDNNSoftmaxForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                   mode, CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                   mode, CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new SoftmaxImplementedByCUDNNSoftmaxForward(cv);
  }
};

MNM_OP_DISPATCH_PLEVEL("mnm.op.softmax", SoftmaxImplementedByCUDNNSoftmaxForward::make,
                       DevType::kCUDA(), "cudnn", 10);

class SoftmaxDxImplementedByCUDNNSoftmaxBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;

  explicit SoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.softmax_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, axis, axis + 1, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, axis, axis + 1, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.softmax_dx"));
  }

 public:
  ~SoftmaxDxImplementedByCUDNNSoftmaxBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new SoftmaxDxImplementedByCUDNNSoftmaxBackward(cv);
  }
};

MNM_OP_DISPATCH_PLEVEL("mnm.op.softmax_dx", SoftmaxDxImplementedByCUDNNSoftmaxBackward::make,
                       DevType::kCUDA(), "cudnn", 7);

}  // namespace cudnn
}  // namespace op
}  // namespace mnm
