/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/pool.cc
 * \brief CUDNN pooling operators.
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
using dmlc::BeginPtr;

static auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");

class AvgPool2DImplementedByCUDNNPoolingForward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolingDesc;

  explicit AvgPool2DImplementedByCUDNNPoolingForward(const CallValues& cv) {
    auto op = Op::Get("raf.op.avg_pool2d");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<raf::op::schema::PoolArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        args->include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                          : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN, 2, BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
  }

 public:
  ~AvgPool2DImplementedByCUDNNPoolingForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.avg_pool2d"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::PoolArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new AvgPool2DImplementedByCUDNNPoolingForward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, avg_pool2d, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.avg_pool2d", AvgPool2DImplementedByCUDNNPoolingForward::make);

class AvgPool2DDxImplementedByCUDNNPoolingBackward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit AvgPool2DDxImplementedByCUDNNPoolingBackward(const CallValues& cv) {
    auto op = Op::Get("raf.op.avg_pool2d_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::PoolDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        args->include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                          : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN, 2, BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
  }

 public:
  ~AvgPool2DDxImplementedByCUDNNPoolingBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.avg_pool2d_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::PoolDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new AvgPool2DDxImplementedByCUDNNPoolingBackward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, avg_pool2d_dx, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.avg_pool2d_dx", AvgPool2DDxImplementedByCUDNNPoolingBackward::make);

class MaxPool2DImplementedByCUDNNPoolingForward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolingDesc;

  explicit MaxPool2DImplementedByCUDNNPoolingForward(const CallValues& cv) {
    auto op = Op::Get("raf.op.max_pool2d");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<raf::op::schema::PoolArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2,
                                           BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
  }

 public:
  ~MaxPool2DImplementedByCUDNNPoolingForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.max_pool2d"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::PoolArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MaxPool2DImplementedByCUDNNPoolingForward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, max_pool2d, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.max_pool2d", MaxPool2DImplementedByCUDNNPoolingForward::make);

class MaxPool2DDxImplementedByCUDNNPoolingBackward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnPoolingDescriptor_t poolingDesc;

  explicit MaxPool2DDxImplementedByCUDNNPoolingBackward(const CallValues& cv) {
    auto op = Op::Get("raf.op.max_pool2d_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::PoolDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2,
                                           BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
  }

 public:
  ~MaxPool2DDxImplementedByCUDNNPoolingBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.max_pool2d_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::PoolDxArgs>();
    DLTensor* x = args->x;
    DLTensor* y = args->y;
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new MaxPool2DDxImplementedByCUDNNPoolingBackward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, max_pool2d_dx, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.max_pool2d_dx", MaxPool2DDxImplementedByCUDNNPoolingBackward::make);

}  // namespace cudnn
}  // namespace op
}  // namespace raf
