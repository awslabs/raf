/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file ./src/op/regs/tvmjit_regs.cc
 * \brief Register TVM ops.
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
using tvm::Target;
using tvm::relay::FTVMCompute;
using tvm::relay::FTVMSchedule;
using tvm::te::Schedule;
using tvm::te::Tensor;
#define MNM_TVM_OP(MNM_OP, OP)                                                                  \
  MNM_OP_REGISTER(MNM_OP)                                                                       \
      .set_attr<FTVMCompute>("FTVMCompute",                                                     \
                             [](const Attrs& attrs, const Array<Tensor>& inputs,                \
                                const Type& out_type) -> Array<Tensor> {                        \
                               auto fcompute =                                                  \
                                   Op::GetAttrMap<FTVMCompute>("FTVMCompute")[Op::Get(OP)];     \
                               return fcompute(attrs, inputs, out_type);                        \
                             })                                                                 \
      .set_attr<FTVMSchedule>(                                                                  \
          "FTVMSchedule",                                                                       \
          [](const Attrs& attrs, const Array<Tensor>& outs, const Target& target) -> Schedule { \
            auto fschedule = Op::GetAttrMap<FTVMSchedule>("FTVMSchedule")[Op::Get(OP)];         \
            return fschedule(attrs, outs, target);                                              \
          })

MNM_TVM_OP("mnm.op.abs", "abs");
MNM_TVM_OP("mnm.op.add", "add");
MNM_TVM_OP("mnm.op.all", "all");
MNM_TVM_OP("mnm.op.any", "any");
MNM_TVM_OP("mnm.op.arange", "arange");
MNM_TVM_OP("mnm.op.argmax", "argmax");
MNM_TVM_OP("mnm.op.argmin", "argmin");
MNM_TVM_OP("mnm.op.argsort", "argsort");
MNM_TVM_OP("mnm.op.argwhere", "argwhere");
MNM_TVM_OP("mnm.op.atan", "atan");
MNM_TVM_OP("mnm.op.avg_pool2d", "nn.avg_pool2d");
MNM_TVM_OP("mnm.op.batch_matmul", "nn.batch_matmul");
MNM_TVM_OP("mnm.op.bias_add", "nn.bias_add");
MNM_TVM_OP("mnm.op.broadcast_to", "broadcast_to");
MNM_TVM_OP("mnm.op.broadcast_to_like", "broadcast_to_like");
MNM_TVM_OP("mnm.op.cast", "cast");
MNM_TVM_OP("mnm.op.cast_like", "cast_like");
MNM_TVM_OP("mnm.op.ceil", "ceil");
MNM_TVM_OP("mnm.op.clip", "clip");
MNM_TVM_OP("mnm.op.collapse_sum_like", "collapse_sum_like");
MNM_TVM_OP("mnm.op.concatenate", "concatenate");
MNM_TVM_OP("mnm.op.copy", "copy");
MNM_TVM_OP("mnm.op.cos", "cos");
MNM_TVM_OP("mnm.op.dense", "nn.dense");
MNM_TVM_OP("mnm.op.divide", "divide");
MNM_TVM_OP("mnm.op.equal", "equal");
MNM_TVM_OP("mnm.op.erf", "erf");
MNM_TVM_OP("mnm.op.exp", "exp");
MNM_TVM_OP("mnm.op.expand_dims", "expand_dims");
MNM_TVM_OP("mnm.op.floor", "floor");
MNM_TVM_OP("mnm.op.floor_divide", "floor_divide");
MNM_TVM_OP("mnm.op.floor_mod", "floor_mod");
MNM_TVM_OP("mnm.op.full", "full");
MNM_TVM_OP("mnm.op.full_like", "full_like");
MNM_TVM_OP("mnm.op.gather_nd", "gather_nd");
MNM_TVM_OP("mnm.op.get_valid_counts", "get_valid_counts");
MNM_TVM_OP("mnm.op.greater", "greater");
MNM_TVM_OP("mnm.op.greater_equal", "greater_equal");
MNM_TVM_OP("mnm.op.image.resize", "image.resize");
MNM_TVM_OP("mnm.op.layout_transform", "layout_transform");
MNM_TVM_OP("mnm.op.left_shift", "left_shift");
MNM_TVM_OP("mnm.op.less", "less");
MNM_TVM_OP("mnm.op.less_equal", "less_equal");
MNM_TVM_OP("mnm.op.log", "log");
MNM_TVM_OP("mnm.op.logical_and", "logical_and");
MNM_TVM_OP("mnm.op.logical_not", "logical_not");
MNM_TVM_OP("mnm.op.logical_or", "logical_or");
MNM_TVM_OP("mnm.op.max", "max");
MNM_TVM_OP("mnm.op.max_pool2d", "nn.max_pool2d");
MNM_TVM_OP("mnm.op.maximum", "maximum");
MNM_TVM_OP("mnm.op.mean", "mean");
MNM_TVM_OP("mnm.op.min", "min");
MNM_TVM_OP("mnm.op.minimum", "minimum");
MNM_TVM_OP("mnm.op.mod", "mod");
MNM_TVM_OP("mnm.op.multiply", "multiply");
MNM_TVM_OP("mnm.op.negative", "negative");
MNM_TVM_OP("mnm.op.non_max_suppression", "non_max_suppression");
MNM_TVM_OP("mnm.op.not_equal", "not_equal");
MNM_TVM_OP("mnm.op.one_hot", "one_hot");
MNM_TVM_OP("mnm.op.ones", "ones");
MNM_TVM_OP("mnm.op.ones_like", "ones_like");
MNM_TVM_OP("mnm.op.power", "power");
MNM_TVM_OP("mnm.op.prod", "prod");
MNM_TVM_OP("mnm.op.reinterpret", "reinterpret");
MNM_TVM_OP("mnm.op.relu", "nn.relu");
MNM_TVM_OP("mnm.op.repeat", "repeat");
MNM_TVM_OP("mnm.op.reverse", "reverse");
MNM_TVM_OP("mnm.op.reverse_sequence", "reverse_sequence");
MNM_TVM_OP("mnm.op.right_shift", "right_shift");
MNM_TVM_OP("mnm.op.round", "round");
MNM_TVM_OP("mnm.op.rsqrt", "rsqrt");
MNM_TVM_OP("mnm.op.sequence_mask", "sequence_mask");
MNM_TVM_OP("mnm.op.sigmoid", "sigmoid");
MNM_TVM_OP("mnm.op.sign", "sign");
MNM_TVM_OP("mnm.op.sin", "sin");
MNM_TVM_OP("mnm.op.slice_like", "slice_like");
MNM_TVM_OP("mnm.op.softmax", "nn.softmax");
MNM_TVM_OP("mnm.op.split", "split");
MNM_TVM_OP("mnm.op.sqrt", "sqrt");
MNM_TVM_OP("mnm.op.squeeze", "squeeze");
MNM_TVM_OP("mnm.op.stack", "stack");
MNM_TVM_OP("mnm.op.strided_slice", "strided_slice");
MNM_TVM_OP("mnm.op.subtract", "subtract");
MNM_TVM_OP("mnm.op.take", "take");
MNM_TVM_OP("mnm.op.tanh", "tanh");
MNM_TVM_OP("mnm.op.tile", "tile");
MNM_TVM_OP("mnm.op.topk", "topk");
MNM_TVM_OP("mnm.op.transpose", "transpose");
MNM_TVM_OP("mnm.op.trunc", "trunc");
MNM_TVM_OP("mnm.op.variance", "variance");
MNM_TVM_OP("mnm.op.where", "where");
MNM_TVM_OP("mnm.op.zeros", "zeros");
MNM_TVM_OP("mnm.op.zeros_like", "zeros_like");
}  // namespace
}  // namespace op
}  // namespace mnm
