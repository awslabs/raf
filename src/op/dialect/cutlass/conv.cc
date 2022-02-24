/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/cutlass/conv.cc
 * \brief Implementation of cutlass convolution dispatch
 */
#include "./conv.h"

namespace raf {
namespace op {
namespace cutlass {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
using raf::op::regs::value2schema::TupleInt;
using raf::registry::PackedFunc;
using raf::registry::TypedPackedFunc;

bool CutlassConv2dOpEnv::Pattern(const CallValues& cv) {
  Expr expr = Downcast<ClosureValue>(cv->callee)->func->body;
  const static std::vector<std::string> conv_ops = {"raf.op.cutlass.conv2d"};
  const static std::vector<std::string> epilogue_ops = {"raf.op.cutlass.relu"};
  auto conv2d = IsOps(conv_ops);
  auto epilogue = IsOps(epilogue_ops);
  auto x = IsVar("");
  auto w = IsVar("");
  auto bias = IsVar("");
  auto stride = IsVar("");
  auto padding = IsVar("");
  auto dilation = IsVar("");
  auto groups = IsVar("");
  auto layout = IsVar("");
  auto kernel_layout = IsVar("");
  auto out_layout = IsVar("");
  DFPattern pat =
      conv2d({x, w, stride, padding, dilation, groups, layout, kernel_layout, out_layout});
  DFPattern with_bias = Add()(pat, bias);
  pat = with_bias || pat;
  DFPattern with_epilogue = epilogue({pat});
  pat = with_epilogue || pat;

  if (!RAFMatchPattern(pat, expr)) {
    return false;
  }

  // RAFRewritePatterns serves as a visitor here: it does not rewrite, instead information
  // is recorded for later process.
  TypedPackedFunc<Expr(const Expr&, const Expr&, const Map<DFPattern, Array<Expr>>&)> func(
      [&](const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
        x_ = GetPattern<Var>(node_map, x);
        w_ = GetPattern<Var>(node_map, w);
        stride_ = Pad<2>(TupleInt(GetValue<TupleValue>(cv, GetPattern<Var>(node_map, stride))));
        padding_ = Pad<2>(TupleInt(GetValue<TupleValue>(cv, GetPattern<Var>(node_map, padding))));
        dilation_ = Pad<2>(TupleInt(GetValue<TupleValue>(cv, GetPattern<Var>(node_map, dilation))));
        layout_ = GetValue<StringValue>(cv, GetPattern<Var>(node_map, layout))->value;
        out_layout_ = GetValue<StringValue>(cv, GetPattern<Var>(node_map, out_layout))->value;
        kernel_layout_ = GetValue<StringValue>(cv, GetPattern<Var>(node_map, kernel_layout))->value;
        with_bias_ = GetPattern<Var>(node_map, bias).defined();
        epilogue_op_ = GetEpilogueKind(GetPattern<Op>(node_map, epilogue));
        if (with_bias_) {
          bias_ = GetPattern<Var>(node_map, bias);
        }
        return post;
      });
  DFPatternCallback cb(pat, func.operator PackedFunc(), false);
  RAFRewritePatterns({cb}, expr);
  return true;
}

bool CutlassConv2dOpEnv::IsValid(const CallValues& cv) {
  return x_.defined() && w_.defined() && (!with_bias_ || bias_.defined()) && layout_ == "NHWC" &&
         kernel_layout_ == "OHWI" && out_layout_ == "NHWC";
}

void CutlassConv2dOpEnv::Init(const CallValues& cv) {
  DLTensor* x = GetValue<TensorValue>(cv, x_);
  DLTensor* w = GetValue<TensorValue>(cv, w_);
  DLTensor* out = cv->out;
  DLTensor* bias = with_bias_ ? GetValue<TensorValue>(cv, bias_) : out;
  int N = x->shape[0], H = x->shape[1], W = x->shape[2], C = x->shape[3];
  int K = w->shape[0], R = w->shape[1], S = w->shape[2];
  InitConvOperation(SplitKMode::kSerial, N, H, W, C, K, R, S, padding_[0], padding_[1], stride_[0],
                    stride_[1], dilation_[0], dilation_[1], GetNumericTypeID(out->dtype),
                    NumericTypeID::kF32, const_addr<1>(cudaDataType_t(DType(out->dtype))),
                    GetNumericTypeID(x->dtype), LayoutTypeID::kTensorNHWC, x->data,
                    GetNumericTypeID(w->dtype), LayoutTypeID::kTensorNHWC, w->data,
                    with_bias_ ? const_addr<1>(cudaDataType_t(DType(out->dtype)))
                               : const_addr<0>(cudaDataType_t(DType(out->dtype))),
                    GetNumericTypeID(out->dtype), bias->data, out->data, epilogue_op_);
  arg_indices = GetArgIndices(
      cv, with_bias_ ? std::vector<Var>({x_, w_, bias_}) : std::vector<Var>({x_, w_}));
}

OpEnv* CutlassConv2dOpEnv::make(const CallValues& cv) {
  std::unique_ptr<CutlassConv2dOpEnv> op_env(std::make_unique<CutlassConv2dOpEnv>(cv));
  if (!op_env->Pattern(cv) || !op_env->IsValid(cv)) {
    return nullptr;
  }
  op_env->Init(cv);
  return op_env.release();
}

void CutlassConv2dOpEnv::Execute(const std::vector<Value>& inputs, Value output) {
  DLTensor* x1 = inputs[0];
  DLTensor* x2 = inputs[1];
  DLTensor* out = output;
  DLTensor* bias = with_bias_ ? inputs[2] : out;
  arguments_.A = x1->data;
  arguments_.B = x2->data;
  arguments_.C = bias->data;
  arguments_.D = out->data;
  CUTLASS_CALL(operation_->run(&arguments_, host_workspace_, workspace_, GetStream()));
}

// TODO(@hzfan): Using plevel 0 due to lack of OpEnvMaker
RAF_REGISTER_DIALECT_OP(cutlass, conv2d, 0);

}  // namespace cutlass
}  // namespace op
}  // namespace raf
