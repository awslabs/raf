/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/op/dialect/cudnn/activation.cc
 * \brief CUDNN activation operators.
 */
#include "../../schema/ufunc.h"
#include "./cudnn_utils.h"
#include "mnm/ir.h"
#include "mnm/op_utils.h"

namespace mnm {
namespace op {
namespace cudnn {

using namespace mnm::value;
using namespace mnm::ir;

static auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");

class ReluImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;

  explicit ReluImplementedByCUDNNActivationForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.relu");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }

 public:
  ~ReluImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cudnn.relu"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new ReluImplementedByCUDNNActivationForward(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cudnn, relu, 15);
MNM_OP_ENV_MAKER("mnm.op.cudnn.relu", ReluImplementedByCUDNNActivationForward::make);

class ReluDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;

  explicit ReluDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.relu_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }

 public:
  ~ReluDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cudnn.relu_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new ReluDxImplementedByCUDNNActivationBackward(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cudnn, relu_dx, 15);
MNM_OP_ENV_MAKER("mnm.op.cudnn.relu_dx", ReluDxImplementedByCUDNNActivationBackward::make);

class SigmoidImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;

  explicit SigmoidImplementedByCUDNNActivationForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.sigmoid");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }

 public:
  ~SigmoidImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cudnn.sigmoid"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new SigmoidImplementedByCUDNNActivationForward(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cudnn, sigmoid, 15);
MNM_OP_ENV_MAKER("mnm.op.cudnn.sigmoid", SigmoidImplementedByCUDNNActivationForward::make);

class SigmoidDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;

  explicit SigmoidDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.sigmoid_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }

 public:
  ~SigmoidDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cudnn.sigmoid_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new SigmoidDxImplementedByCUDNNActivationBackward(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cudnn, sigmoid_dx, 15);
MNM_OP_ENV_MAKER("mnm.op.cudnn.sigmoid_dx", SigmoidDxImplementedByCUDNNActivationBackward::make);

class TanhImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;

  explicit TanhImplementedByCUDNNActivationForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.tanh");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }

 public:
  ~TanhImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cudnn.tanh"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    DLTensor* x = args->x;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }

  static OpEnv* make(const CallValues& cv) {
    return new TanhImplementedByCUDNNActivationForward(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cudnn, tanh, 15);
MNM_OP_ENV_MAKER("mnm.op.cudnn.tanh", TanhImplementedByCUDNNActivationForward::make);

class TanhDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;

  explicit TanhDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.tanh_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }

 public:
  ~TanhDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    DLTensor* dy = args->dy;
    DLTensor* out = cv->out;
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cudnn.tanh_dx"));
  }

  static OpEnv* make(const CallValues& cv) {
    return new TanhDxImplementedByCUDNNActivationBackward(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cudnn, tanh_dx, 15);
MNM_OP_ENV_MAKER("mnm.op.cudnn.tanh_dx", TanhDxImplementedByCUDNNActivationBackward::make);

}  // namespace cudnn
}  // namespace op
}  // namespace mnm
