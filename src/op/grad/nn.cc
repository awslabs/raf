/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/nn.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Expr Shape(const Expr& expr) {
  static auto op_shape = Op::Get("mnm.op.shape");
  return Call(op_shape, {expr});
}

Array<Expr> BiasAddGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.bias_add", BiasAddGrad);

template <const char* GradOp>
Array<Expr> PoolGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& kernel = call->args[1];
  const Expr& stride = call->args[2];
  const Expr& padding = call->args[3];
  const Expr& dilation = call->args[4];
  const Expr& ceil_mode = call->args[5];
  const Expr& include_pad = call->args[6];
  return {Call(op_dx, {x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad})};
}

const char MAX_POOL2D_DX[] = "mnm.op.max_pool2d_dx";
auto MaxPool2dGrad = PoolGrad<MAX_POOL2D_DX>;
MNM_OP_GRAD("mnm.op.max_pool2d", MaxPool2dGrad);

const char AVG_POOL2D_DX[] = "mnm.op.avg_pool2d_dx";
auto AvgPool2dGrad = PoolGrad<AVG_POOL2D_DX>;
MNM_OP_GRAD("mnm.op.avg_pool2d", AvgPool2dGrad);

Array<Expr> Conv2dGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  // schema for conv2d is:
  //    x, w, stride, padding, dilation, groups
  // schema for conv2d_grad is:
  //    x_or_w, y, dy, shape, stride, padding, dilation, groups
  static auto op_dx = Op::Get("mnm.op.conv2d_dx");
  static auto op_dw = Op::Get("mnm.op.conv2d_dw");
  const CallNode* call = orig_call.as<CallNode>();
  // TODO(@junrushao1994): this piece of code is particularly suitable for auto-gen
  CHECK_GE(call->args.size(), 6);
  const Expr& x = call->args[0];
  const Expr& w = call->args[1];
  const Expr& stride = call->args[2];
  const Expr& padding = call->args[3];
  const Expr& dilation = call->args[4];
  const Expr& groups = call->args[5];
  // dx: w, y, dy, shape(x), stride, padding, dilation, groups
  // dw: x, y, dy, shape(w), stride, padding, dilation, groups
  return {Call(op_dx, {w, y, dy, Shape(x), stride, padding, dilation, groups}),
          Call(op_dw, {x, y, dy, Shape(w), stride, padding, dilation, groups})};
}

MNM_OP_GRAD("mnm.op.conv2d", Conv2dGrad);

template <const char* GradOp>
Array<Expr> UnaryGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  // schema for relu is:
  //    x
  // schema for relu_dx is:
  //    x, y, dy
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  return {Call(op_dx, {x, y, dy})};
}

const char RELU_DX[] = "mnm.op.relu_dx";
auto ReluGrad = UnaryGrad<RELU_DX>;
MNM_OP_GRAD("mnm.op.relu", ReluGrad);

const char TANH_DX[] = "mnm.op.tanh_dx";
auto TanhGrad = UnaryGrad<TANH_DX>;
MNM_OP_GRAD("mnm.op.tanh", TanhGrad);

const char SIGMOID_DX[] = "mnm.op.sigmoid_dx";
auto SigmoidGrad = UnaryGrad<SIGMOID_DX>;
MNM_OP_GRAD("mnm.op.sigmoid", SigmoidGrad);

const char ERF_DX[] = "mnm.op.erf_dx";
auto ErfGrad = UnaryGrad<ERF_DX>;
MNM_OP_GRAD("mnm.op.erf", ErfGrad);

const char SQRT_DX[] = "mnm.op.sqrt_dx";
auto SqrtGrad = UnaryGrad<SQRT_DX>;
MNM_OP_GRAD("mnm.op.sqrt", SqrtGrad);

Array<Expr> BatchNormTrainGrad(const Expr& orig_call, const Var& y, const Expr& dymv,
                               const Array<Expr>& igrads) {
  // schema for batch_norm_train is:
  //    x, running_mean,running_var, w, b, momentum, eps
  // schema for batch_norm_train_dxwb is:
  //    dy, x, w, b, eps
  static auto op_dxwb = Op::Get("mnm.op.batch_norm_train_dxwb");
  const Expr& dy = AsTupleExpr(dymv, 3)[0];
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& w = call->args[3];
  const Expr& b = call->args[4];
  const Expr& eps = call->args[6];
  const Expr& ret = Call(op_dxwb, {dy, x, w, b, eps});
  return {
      TupleGetItem(ret, 0), NullValue<Expr>(),    NullValue<Expr>(),
      TupleGetItem(ret, 1), TupleGetItem(ret, 2),
  };
}

MNM_OP_FUSED_GRAD("mnm.op.batch_norm_train", BatchNormTrainGrad);

template <const char* GradOp>
Array<Expr> SoftmaxGradImpl(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  return {Call(op_dx, {x, y, dy, axis})};
}

const char SOFTMAX_DX[] = "mnm.op.softmax_dx";
auto SoftmaxGrad = SoftmaxGradImpl<SOFTMAX_DX>;
MNM_OP_GRAD("mnm.op.softmax", SoftmaxGrad);

const char LOG_SOFTMAX_DX[] = "mnm.op.log_softmax_dx";
auto LogSoftmaxGrad = SoftmaxGradImpl<LOG_SOFTMAX_DX>;
MNM_OP_GRAD("mnm.op.log_softmax", LogSoftmaxGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
