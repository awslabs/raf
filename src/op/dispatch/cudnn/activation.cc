/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dispatch/cudnn/activation.cc
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
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.relu"));
  }

 public:
  ~ReluImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
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

MNM_OP_DISPATCH_PLEVEL("mnm.op.relu", ReluImplementedByCUDNNActivationForward::make,
                       DevType::kCUDA(), "cudnn", 10);

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
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.relu_dx"));
  }

 public:
  ~ReluDxImplementedByCUDNNActivationBackward() {
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

  static OpEnv* make(const CallValues& cv) {
    return new ReluDxImplementedByCUDNNActivationBackward(cv);
  }
};

MNM_OP_DISPATCH_PLEVEL("mnm.op.relu_dx", ReluDxImplementedByCUDNNActivationBackward::make,
                       DevType::kCUDA(), "cudnn", 10);

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
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.sigmoid"));
  }

 public:
  ~SigmoidImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
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

MNM_OP_DISPATCH_PLEVEL("mnm.op.sigmoid", SigmoidImplementedByCUDNNActivationForward::make,
                       DevType::kCUDA(), "cudnn", 10);

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
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.sigmoid_dx"));
  }

 public:
  ~SigmoidDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
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

MNM_OP_DISPATCH_PLEVEL("mnm.op.sigmoid_dx", SigmoidDxImplementedByCUDNNActivationBackward::make,
                       DevType::kCUDA(), "cudnn", 10);

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
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.tanh"));
  }

 public:
  ~TanhImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
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

MNM_OP_DISPATCH_PLEVEL("mnm.op.tanh", TanhImplementedByCUDNNActivationForward::make,
                       DevType::kCUDA(), "cudnn", 10);

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
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.tanh_dx"));
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

  static OpEnv* make(const CallValues& cv) {
    return new TanhDxImplementedByCUDNNActivationBackward(cv);
  }
};

MNM_OP_DISPATCH_PLEVEL("mnm.op.tanh_dx", TanhDxImplementedByCUDNNActivationBackward::make,
                       DevType::kCUDA(), "cudnn", 10);

}  // namespace cudnn
}  // namespace op
}  // namespace mnm
