/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/nn.cc
 * \brief Declaration of gradients
 */
#include <raf/value.h>
#include <raf/op_utils.h>
#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;
using namespace raf::value;

Array<Expr> BiasAddGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                        const Expr& dy) {
  using namespace raf::value;
  static auto sum = Op::Get("raf.op.sum");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& bias = call->args[1];
  const Expr& axis = call->args[2];
  Expr keep_dims = MakeConstant(ScalarValue::make((int64_t)0));
  Expr exclude = MakeConstant(ScalarValue::make(true));
  return {GetReshapeLike(dy, x), Call(sum, {dy, axis, keep_dims, exclude})};
}

RAF_OP_GRAD("raf.op.bias_add", BiasAddGrad);

Array<Expr> ContribDropoutGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                               const Expr& dout) {
  const static auto dropout_dx = Op::Get("raf.op._contrib_dropout_dx");
  const Expr& dy = AsTupleExpr(dout, 2)[0];
  const Expr& mask = TupleGetItem(y, 1);
  const Expr& reserve_space = TupleGetItem(y, 3);
  const Expr& p = orig_args[1];
  return {Call(dropout_dx, {dy, mask, reserve_space, p})};
}

RAF_OP_GRAD("raf.op._contrib_dropout", ContribDropoutGrad);

template <const char* GradOp>
Array<Expr> PoolGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& kernel = call->args[1];
  const Expr& stride = call->args[2];
  const Expr& padding = call->args[3];
  const Expr& dilation = call->args[4];
  const Expr& ceil_mode = call->args[5];
  const Expr& include_pad = call->args[6];
  const Expr& layout = orig_args[7];
  const auto* layout_const = layout.as<ConstantNode>();
  if (layout_const) {
    const auto* layout_str = layout_const->value.as<value::StringValueObj>();
    CHECK(layout_str && layout_str->value == "NCHW")
        << "PoolGrad support NCHW layout only. Layout = " << layout_str->value;
  }
  return {Call(op_dx, {x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad})};
}

const char MAX_POOL2D_DX[] = "raf.op.max_pool2d_dx";
auto MaxPool2dGrad = PoolGrad<MAX_POOL2D_DX>;
RAF_OP_GRAD("raf.op.max_pool2d", MaxPool2dGrad);

const char AVG_POOL2D_DX[] = "raf.op.avg_pool2d_dx";
auto AvgPool2dGrad = PoolGrad<AVG_POOL2D_DX>;
RAF_OP_GRAD("raf.op.avg_pool2d", AvgPool2dGrad);

template <const char* GradOp>
Array<Expr> AdaptivePoolGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                             const Expr& dy) {
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& shape = call->args[1];
  const Expr& layout = call->args[2];
  const auto* layout_const = layout.as<ConstantNode>();
  if (layout_const) {
    const auto* layout_str = layout_const->value.as<value::StringValueObj>();
    CHECK(layout_str && layout_str->value == "NCHW")
        << "AdaptivePoolGrad support NCHW layout only. Layout = " << layout_str->value;
  }
  return {Call(op_dx, {x, y, dy, shape})};
}

const char ADAPTIVE_MAX_POOL2D_DX[] = "raf.op.adaptive_max_pool2d_dx";
auto AdaptiveMaxPool2dGrad = AdaptivePoolGrad<ADAPTIVE_MAX_POOL2D_DX>;
RAF_OP_GRAD("raf.op.adaptive_max_pool2d", AdaptiveMaxPool2dGrad);

const char ADAPTIVE_AVG_POOL2D_DX[] = "raf.op.adaptive_avg_pool2d_dx";
auto AdaptiveAvgPool2dGrad = AdaptivePoolGrad<ADAPTIVE_AVG_POOL2D_DX>;
RAF_OP_GRAD("raf.op.adaptive_avg_pool2d", AdaptiveAvgPool2dGrad);

Array<Expr> Conv2dGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                       const Expr& dy) {
  // schema for conv2d is:
  //    x, w, stride, padding, dilation, groups
  // schema for conv2d_grad is:
  //    x_or_w, y, dy, shape, stride, padding, dilation, groups
  static auto op_dx = Op::Get("raf.op.conv2d_dx");
  static auto op_dw = Op::Get("raf.op.conv2d_dw");
  const CallNode* call = orig_call.as<CallNode>();
  // TODO(@junrushao1994): this piece of code is particularly suitable for auto-gen
  CHECK_GE(call->args.size(), 6);
  const Expr& x = call->args[0];
  const Expr& w = call->args[1];
  const Expr& stride = call->args[2];
  const Expr& padding = call->args[3];
  const Expr& dilation = call->args[4];
  const Expr& groups = call->args[5];
  const Expr& layout = orig_args[6];
  const auto* layout_const = layout.as<ConstantNode>();
  if (layout_const) {
    const auto* layout_str = layout_const->value.as<value::StringValueObj>();
    CHECK(layout_str && layout_str->value == "NCHW")
        << "PoolGrad support NCHW layout only. Layout = " << layout_str->value;
  }
  // dx: w, y, dy, shape(x), stride, padding, dilation, groups
  // dw: x, y, dy, shape(w), stride, padding, dilation, groups
  return {Call(op_dx, {w, y, dy, GetShape(x), stride, padding, dilation, groups}),
          Call(op_dw, {x, y, dy, GetShape(w), stride, padding, dilation, groups})};
}

RAF_OP_GRAD("raf.op.conv2d", Conv2dGrad);

Array<Expr> Conv2dTransGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.conv2d_transpose_dx");
  static auto op_dw = Op::Get("raf.op.conv2d_transpose_dw");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 7);
  const Expr& x = call->args[0];
  const Expr& w = call->args[1];
  const Expr& stride = call->args[2];
  const Expr& padding = call->args[3];
  const Expr& output_padding = call->args[4];
  const Expr& dilation = call->args[5];
  const Expr& groups = call->args[6];
  const Expr& layout = orig_args[7];
  const auto* layout_const = layout.as<ConstantNode>();
  if (layout_const) {
    const auto* layout_str = layout_const->value.as<value::StringValueObj>();
    CHECK(layout_str && layout_str->value == "NCHW")
        << "PoolGrad support NCHW layout only. Layout = " << layout_str->value;
  }
  // dx: w, y, dy, shape(x), stride, padding, dilation, groups
  // dw: x, y, dy, shape(w), stride, padding, dilation, groups
  return {Call(op_dx, {w, y, dy, GetShape(x), stride, padding, output_padding, dilation, groups}),
          Call(op_dw, {x, y, dy, GetShape(w), stride, padding, output_padding, dilation, groups})};
}

RAF_OP_GRAD("raf.op.conv2d_transpose", Conv2dTransGrad);

template <const char* GradOp>
Array<Expr> UnaryGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  // schema for unary_op is:
  //    x
  // schema for unary_dx_op is:
  //    x, y, dy
  // Note that unary_dx schema allows y to be optional. If the unary_dx op supports
  // optional y, we need to specify it in GradientInputSelection pass to remove the
  // y dependency.
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  return {Call(op_dx, {x, y, dy})};
}

Array<Expr> RsqrtGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  static auto op_sqrt = Op::Get("raf.op.sqrt");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_divide = Op::Get("raf.op.divide");
  static auto op_negative = Op::Get("raf.op.negative");
  static auto op_add = Op::Get("raf.op.add");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call one = Call(op_divide, {x, x});
  Call two = Call(op_add, {one, one, MakeNull(), MakeNull()});
  Call half = Call(op_divide, {one, two});
  Call neg_half = Call(op_negative, {half});
  Call x_pow_2 = Call(op_multiply, {x, x});
  Call x_pow_3 = Call(op_multiply, {x_pow_2, x});
  Call sqrt_x_pow_3 = Call(op_sqrt, {x_pow_3});
  Call rsqrt_x_pow_3 = Call(op_divide, {one, sqrt_x_pow_3});
  Call dx = Call(op_multiply, {neg_half, rsqrt_x_pow_3});
  return {Call(op_multiply, {dy, dx})};
}
RAF_OP_GRAD("raf.op.rsqrt", RsqrtGrad);

Array<Expr> TruncGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];

  Call grads;
  if (x->checked_type_.defined()) {
    static auto op_zeros = Op::Get("raf.op.zeros");
    auto ttype = x->checked_type().as<TensorTypeNode>();
    auto shape = MakeConstant(ArrayToIntTuple(ttype->shape));
    auto dtype = MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(ttype->dtype)));
    grads = Call(op_zeros, {shape, dtype});
  } else {
    static auto op_zeros_like = Op::Get("raf.op.zeros_like");
    grads = Call(op_zeros_like, {x});
  }
  return {grads};
}
RAF_OP_GRAD("raf.op.trunc", TruncGrad);

Array<Expr> LogGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  // give zero gradient for any gradient
  static auto op_div = Op::Get("raf.op.divide");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_ones = Op::Get("raf.op.ones_like");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call ones = Call(op_ones, {x});
  Call one_by_x = Call(op_div, {ones, x});
  Call result = Call(op_multiply, {dy, one_by_x});
  return {result};
}
RAF_OP_GRAD("raf.op.log", LogGrad);

Array<Expr> Log2Grad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  // give zero gradient for any gradient
  static auto op_div = Op::Get("raf.op.divide");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_ones = Op::Get("raf.op.ones_like");
  static auto op_add = Op::Get("raf.op.add");
  static auto op_log = Op::Get("raf.op.log");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call ones = Call(op_ones, {x});
  Call two = Call(op_add, {ones, ones, MakeNull(), MakeNull()});
  Call log_two = Call(op_log, {two});
  Call log_two_x = Call(op_multiply, {log_two, x});
  Call dx = Call(op_div, {ones, log_two_x});
  return {Call(op_multiply, {dy, dx})};
}
RAF_OP_GRAD("raf.op.log2", Log2Grad);

Array<Expr> CosGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_sin = Op::Get("raf.op.sin");
  static auto op_negative = Op::Get("raf.op.negative");
  static auto op_multiply = Op::Get("raf.op.multiply");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call sin_x = Call(op_sin, {x});
  Call dx = Call(op_negative, {sin_x});
  return {Call(op_multiply, {dy, dx})};
}
RAF_OP_GRAD("raf.op.cos", CosGrad);

Array<Expr> SinGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_cos = Op::Get("raf.op.cos");
  static auto op_multiply = Op::Get("raf.op.multiply");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call dx = Call(op_cos, {x});
  return {Call(op_multiply, {dy, dx})};
}
RAF_OP_GRAD("raf.op.sin", SinGrad);

Array<Expr> ExpGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_exp = Op::Get("raf.op.exp");
  static auto op_multiply = Op::Get("raf.op.multiply");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call dx = Call(op_exp, {x});
  return {Call(op_multiply, {dy, dx})};
}
RAF_OP_GRAD("raf.op.exp", ExpGrad);

Array<Expr> AtanGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_divide = Op::Get("raf.op.divide");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_add = Op::Get("raf.op.add");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 1);
  const Expr& x = call->args[0];
  Call one = Call(op_divide, {x, x});
  Call x_square = Call(op_multiply, {x, x});
  Call denominator = Call(op_add, {x_square, one, MakeNull(), MakeNull()});
  Call dx = Call(op_divide, {one, denominator});
  return {Call(op_multiply, {dy, dx})};
}
RAF_OP_GRAD("raf.op.atan", AtanGrad);

const char RELU_DX[] = "raf.op.relu_dx";
auto ReluGrad = UnaryGrad<RELU_DX>;
RAF_OP_GRAD("raf.op.relu", ReluGrad);

const char GELU_DX[] = "raf.op.gelu_dx";
auto GeluGrad = UnaryGrad<GELU_DX>;
RAF_OP_GRAD("raf.op.gelu", GeluGrad);

const char TANH_DX[] = "raf.op.tanh_dx";
auto TanhGrad = UnaryGrad<TANH_DX>;
RAF_OP_GRAD("raf.op.tanh", TanhGrad);

const char SIGMOID_DX[] = "raf.op.sigmoid_dx";
auto SigmoidGrad = UnaryGrad<SIGMOID_DX>;
RAF_OP_GRAD("raf.op.sigmoid", SigmoidGrad);

const char ERF_DX[] = "raf.op.erf_dx";
auto ErfGrad = UnaryGrad<ERF_DX>;
RAF_OP_GRAD("raf.op.erf", ErfGrad);

const char SQRT_DX[] = "raf.op.sqrt_dx";
auto SqrtGrad = UnaryGrad<SQRT_DX>;
RAF_OP_GRAD("raf.op.sqrt", SqrtGrad);

Array<Expr> BatchNormTrainGrad(const Expr& orig_call, const Var& y, const Expr& dymv,
                               const Array<Expr>& igrads) {
  // schema for batch_norm_train is:
  //    x, running_mean,running_var, w, b, momentum, eps
  // schema for batch_norm_train_dxwb is:
  //    dy, x, w, b, eps
  static auto op_dxwb = Op::Get("raf.op.batch_norm_train_dxwb");
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

RAF_OP_FUSED_GRAD("raf.op.batch_norm_train", BatchNormTrainGrad);

template <const char* GradOp>
Array<Expr> SoftmaxGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  return {Call(op_dx, {x, y, dy, axis})};
}

const char SOFTMAX_DX[] = "raf.op.softmax_dx";
auto SoftmaxGrad = SoftmaxGradImpl<SOFTMAX_DX>;
RAF_OP_GRAD("raf.op.softmax", SoftmaxGrad);

Array<Expr> LogSoftmaxGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  using namespace raf::value;
  static auto op_softmax = Op::Get("raf.op.softmax");
  static auto op_sum = Op::Get("raf.op.sum");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_subtract = Op::Get("raf.op.subtract");
  static auto op_divide = Op::Get("raf.op.divide");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr softmax = Call(op_softmax, {x, axis});
  Expr keep_dims = MakeConstant(ScalarValue::make((int64_t)1));
  Expr e_1 = Call(op_sum, {dy, axis, keep_dims, MakeConstant(BoolValue::make(false))});
  Expr e_2 = Call(op_multiply, {e_1, softmax});
  Expr e_3 = Call(op_subtract, {dy, e_2, MakeNull(), MakeNull()});
  return {e_3};
}

RAF_OP_GRAD("raf.op.log_softmax", LogSoftmaxGrad);

Array<Expr> LayerNormGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.layer_norm_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& scale = call->args[1];
  const Expr& axis = call->args[3];
  const Expr& eps = call->args[4];
  const Expr& ret = Call(op_dx, {x, scale, dy, axis, eps});
  const auto* kscale = scale.as<tvm::relay::ConstantNode>();
  if (kscale && !static_cast<const ConstantNode*>(kscale)->value.defined()) {
    // scale and bias are not learnable parameters.
    return {ret};
  }
  return {
      TupleGetItem(ret, 0),
      TupleGetItem(ret, 1),
      TupleGetItem(ret, 2),
  };
}

RAF_OP_GRAD("raf.op.layer_norm", LayerNormGrad);

Array<Expr> ThresholdGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.threshold_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& threshold = call->args[1];
  return {Call(op_dx, {x, dy, threshold})};
}

RAF_OP_GRAD("raf.op.threshold", ThresholdGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
