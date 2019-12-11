/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/regs/tvmjit_regs.cc
 * \brief Auto generated. Do not touch.
 */
#include "mnm/ir.h"
#include "mnm/op.h"
namespace mnm {
namespace op {
namespace {
using ir::Array;
using ir::Attrs;
using ir::Op;
using ir::Type;
using tvm::Schedule;
using tvm::Target;
using tvm::Tensor;
using tvm::relay::FTVMCompute;
using tvm::relay::FTVMSchedule;
using tvm::relay::OpPatternKind;
using tvm::relay::TOpPattern;
#define MNM_TVM_OP(MNM_OP, OP, PATTERN)                                                         \
  MNM_OP_REGISTER(MNM_OP)                                                                       \
      .set_attr<FTVMCompute>("FTVMCompute",                                                     \
                             [](const Attrs& attrs, const Array<Tensor>& inputs,                \
                                const Type& out_type, const Target& target) -> Array<Tensor> {  \
                               auto fcompute =                                                  \
                                   Op::GetAttr<FTVMCompute>("FTVMCompute")[Op::Get(OP)];        \
                               return fcompute(attrs, inputs, out_type, target);                \
                             })                                                                 \
      .set_attr<FTVMSchedule>(                                                                  \
          "FTVMSchedule",                                                                       \
          [](const Attrs& attrs, const Array<Tensor>& outs, const Target& target) -> Schedule { \
            auto fschedule = Op::GetAttr<FTVMSchedule>("FTVMSchedule")[Op::Get(OP)];            \
            return fschedule(attrs, outs, target);                                              \
          })                                                                                    \
      .set_attr<TOpPattern>("TOpPattern", tvm::relay::PATTERN);
MNM_TVM_OP("mnm.op.abs", "abs", kElemWise);
MNM_TVM_OP("mnm.op.add", "add", kBroadcast);
MNM_TVM_OP("mnm.op.all", "all", kCommReduce);
MNM_TVM_OP("mnm.op.any", "any", kCommReduce);
MNM_TVM_OP("mnm.op.arange", "arange", kOpaque);
MNM_TVM_OP("mnm.op.argmax", "argmax", kCommReduce);
MNM_TVM_OP("mnm.op.argmin", "argmin", kCommReduce);
MNM_TVM_OP("mnm.op.argsort", "argsort", kOpaque);
MNM_TVM_OP("mnm.op.argwhere", "argwhere", kOpaque);
MNM_TVM_OP("mnm.op.atan", "atan", kElemWise);
MNM_TVM_OP("mnm.op.broadcast_to", "broadcast_to", kBroadcast);
MNM_TVM_OP("mnm.op.broadcast_to_like", "broadcast_to_like", kBroadcast);
MNM_TVM_OP("mnm.op.cast", "cast", kElemWise);
MNM_TVM_OP("mnm.op.cast_like", "cast_like", kElemWise);
MNM_TVM_OP("mnm.op.ceil", "ceil", kElemWise);
MNM_TVM_OP("mnm.op.clip", "clip", kElemWise);
MNM_TVM_OP("mnm.op.collapse_sum_like", "collapse_sum_like", kCommReduce);
MNM_TVM_OP("mnm.op.concatenate", "concatenate", kInjective);
MNM_TVM_OP("mnm.op.copy", "copy", kElemWise);
MNM_TVM_OP("mnm.op.cos", "cos", kElemWise);
MNM_TVM_OP("mnm.op.divide", "divide", kBroadcast);
MNM_TVM_OP("mnm.op.equal", "equal", kBroadcast);
MNM_TVM_OP("mnm.op.erf", "erf", kElemWise);
MNM_TVM_OP("mnm.op.exp", "exp", kElemWise);
MNM_TVM_OP("mnm.op.expand_dims", "expand_dims", kBroadcast);
MNM_TVM_OP("mnm.op.floor", "floor", kElemWise);
MNM_TVM_OP("mnm.op.floor_divide", "floor_divide", kBroadcast);
MNM_TVM_OP("mnm.op.floor_mod", "floor_mod", kBroadcast);
MNM_TVM_OP("mnm.op.full", "full", kElemWise);
MNM_TVM_OP("mnm.op.full_like", "full_like", kElemWise);
MNM_TVM_OP("mnm.op.gather_nd", "gather_nd", kInjective);
MNM_TVM_OP("mnm.op.greater", "greater", kBroadcast);
MNM_TVM_OP("mnm.op.greater_equal", "greater_equal", kBroadcast);
MNM_TVM_OP("mnm.op.image.resize", "image.resize", kInjective);
MNM_TVM_OP("mnm.op.layout_transform", "layout_transform", kInjective);
MNM_TVM_OP("mnm.op.left_shift", "left_shift", kBroadcast);
MNM_TVM_OP("mnm.op.less", "less", kBroadcast);
MNM_TVM_OP("mnm.op.less_equal", "less_equal", kBroadcast);
MNM_TVM_OP("mnm.op.log", "log", kElemWise);
MNM_TVM_OP("mnm.op.logical_and", "logical_and", kBroadcast);
MNM_TVM_OP("mnm.op.logical_not", "logical_not", kElemWise);
MNM_TVM_OP("mnm.op.logical_or", "logical_or", kBroadcast);
MNM_TVM_OP("mnm.op.max", "max", kCommReduce);
MNM_TVM_OP("mnm.op.maximum", "maximum", kBroadcast);
MNM_TVM_OP("mnm.op.mean", "mean", kCommReduce);
MNM_TVM_OP("mnm.op.min", "min", kCommReduce);
MNM_TVM_OP("mnm.op.minimum", "minimum", kBroadcast);
MNM_TVM_OP("mnm.op.mod", "mod", kBroadcast);
MNM_TVM_OP("mnm.op.multiply", "multiply", kBroadcast);
MNM_TVM_OP("mnm.op.negative", "negative", kElemWise);
MNM_TVM_OP("mnm.op.not_equal", "not_equal", kBroadcast);
MNM_TVM_OP("mnm.op.one_hot", "one_hot", kOutEWiseFusable);
MNM_TVM_OP("mnm.op.ones", "ones", kElemWise);
MNM_TVM_OP("mnm.op.ones_like", "ones_like", kElemWise);
MNM_TVM_OP("mnm.op.power", "power", kBroadcast);
MNM_TVM_OP("mnm.op.prod", "prod", kCommReduce);
MNM_TVM_OP("mnm.op.reinterpret", "reinterpret", kElemWise);
MNM_TVM_OP("mnm.op.repeat", "repeat", kBroadcast);
MNM_TVM_OP("mnm.op.reshape", "reshape", kInjective);
MNM_TVM_OP("mnm.op.reshape_like", "reshape_like", kInjective);
MNM_TVM_OP("mnm.op.reverse", "reverse", kInjective);
MNM_TVM_OP("mnm.op.right_shift", "right_shift", kBroadcast);
MNM_TVM_OP("mnm.op.round", "round", kElemWise);
MNM_TVM_OP("mnm.op.rsqrt", "rsqrt", kElemWise);
MNM_TVM_OP("mnm.op.sequence_mask", "sequence_mask", kInjective);
MNM_TVM_OP("mnm.op.shape_of", "shape_of", kOpaque);
MNM_TVM_OP("mnm.op.sigmoid", "sigmoid", kElemWise);
MNM_TVM_OP("mnm.op.sign", "sign", kElemWise);
MNM_TVM_OP("mnm.op.sin", "sin", kElemWise);
MNM_TVM_OP("mnm.op.slice_like", "slice_like", kInjective);
MNM_TVM_OP("mnm.op.split", "split", kInjective);
MNM_TVM_OP("mnm.op.sqrt", "sqrt", kElemWise);
MNM_TVM_OP("mnm.op.squeeze", "squeeze", kInjective);
MNM_TVM_OP("mnm.op.stack", "stack", kInjective);
MNM_TVM_OP("mnm.op.strided_slice", "strided_slice", kInjective);
MNM_TVM_OP("mnm.op.subtract", "subtract", kBroadcast);
MNM_TVM_OP("mnm.op.sum", "sum", kCommReduce);
MNM_TVM_OP("mnm.op.take", "take", kInjective);
MNM_TVM_OP("mnm.op.tanh", "tanh", kElemWise);
MNM_TVM_OP("mnm.op.tile", "tile", kBroadcast);
MNM_TVM_OP("mnm.op.topk", "topk", kOpaque);
MNM_TVM_OP("mnm.op.transpose", "transpose", kInjective);
MNM_TVM_OP("mnm.op.trunc", "trunc", kElemWise);
MNM_TVM_OP("mnm.op.variance", "variance", kCommReduce);
MNM_TVM_OP("mnm.op.where", "where", kBroadcast);
MNM_TVM_OP("mnm.op.zeros", "zeros", kElemWise);
MNM_TVM_OP("mnm.op.zeros_like", "zeros_like", kElemWise);
}  // namespace
}  // namespace op
}  // namespace mnm
