/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/softmax.cc
 * \brief CUDNN softmax operators.
 */
#include "../../schema/nn.h"
#include "./cudnn_utils.h"
#include "raf/ir.h"
#include "raf/op_utils.h"

namespace raf {
namespace op {
namespace cudnn {

using namespace raf::value;
using namespace raf::ir;

static auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");

class LogSoftmaxImplementedByCUDNNSoftmaxForward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;

  explicit LogSoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
    auto op = Op::Get("raf.op.log_softmax");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<raf::op::schema::SoftmaxArgs>();
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
  }

 public:
  ~LogSoftmaxImplementedByCUDNNSoftmaxForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.log_softmax"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::SoftmaxArgs>();
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

RAF_REGISTER_DIALECT_OP(cudnn, log_softmax, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.log_softmax", LogSoftmaxImplementedByCUDNNSoftmaxForward::make);

class LogSoftmaxDxImplementedByCUDNNSoftmaxBackward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;

  explicit LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
    auto op = Op::Get("raf.op.log_softmax_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::SoftmaxDxArgs>();
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
  }

 public:
  ~LogSoftmaxDxImplementedByCUDNNSoftmaxBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.log_softmax_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::SoftmaxDxArgs>();
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

RAF_REGISTER_DIALECT_OP(cudnn, log_softmax_dx, 7);
RAF_OP_ENV_MAKER("raf.op.cudnn.log_softmax_dx",
                 LogSoftmaxDxImplementedByCUDNNSoftmaxBackward::make);

class SoftmaxImplementedByCUDNNSoftmaxForward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;

  explicit SoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
    auto op = Op::Get("raf.op.softmax");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<raf::op::schema::SoftmaxArgs>();
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
  }

 public:
  ~SoftmaxImplementedByCUDNNSoftmaxForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.softmax"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::SoftmaxArgs>();
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

RAF_REGISTER_DIALECT_OP(cudnn, softmax, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.softmax", SoftmaxImplementedByCUDNNSoftmaxForward::make);

class SoftmaxDxImplementedByCUDNNSoftmaxBackward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;

  explicit SoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
    auto op = Op::Get("raf.op.softmax_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::SoftmaxDxArgs>();
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
  }

 public:
  ~SoftmaxDxImplementedByCUDNNSoftmaxBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.softmax_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::SoftmaxDxArgs>();
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

RAF_REGISTER_DIALECT_OP(cudnn, softmax_dx, 7);
RAF_OP_ENV_MAKER("raf.op.cudnn.softmax_dx", SoftmaxDxImplementedByCUDNNSoftmaxBackward::make);

}  // namespace cudnn
}  // namespace op
}  // namespace raf
