/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file ./src/op/regs/tvm_op_regs.cc
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
  MNM_REGISTER_OP(MNM_OP)                                                                       \
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

MNM_TVM_OP("mnm.op.tvm.abs", "abs");
MNM_TVM_OP("mnm.op.tvm.adaptive_avg_pool2d", "nn.adaptive_avg_pool2d");
MNM_TVM_OP("mnm.op.tvm.adaptive_avg_pool2d_dx", "nn.avg_pool2d_grad");
MNM_TVM_OP("mnm.op.tvm.adaptive_max_pool2d", "nn.adaptive_max_pool2d");
MNM_TVM_OP("mnm.op.tvm.adaptive_max_pool2d_dx", "nn.max_pool2d_grad");
MNM_TVM_OP("mnm.op.tvm.add", "add");
MNM_TVM_OP("mnm.op.tvm.all", "all");
MNM_TVM_OP("mnm.op.tvm.any", "any");
MNM_TVM_OP("mnm.op.tvm.arange", "arange");
MNM_TVM_OP("mnm.op.tvm.argmax", "argmax");
MNM_TVM_OP("mnm.op.tvm.argmin", "argmin");
MNM_TVM_OP("mnm.op.tvm.argsort", "argsort");
MNM_TVM_OP("mnm.op.tvm.atan", "atan");
MNM_TVM_OP("mnm.op.tvm.avg_pool2d", "nn.avg_pool2d");
MNM_TVM_OP("mnm.op.tvm.avg_pool2d_dx", "nn.avg_pool2d_grad");
MNM_TVM_OP("mnm.op.tvm.batch_flatten", "nn.batch_flatten");
MNM_TVM_OP("mnm.op.tvm.batch_matmul_nt", "nn.batch_matmul");
MNM_TVM_OP("mnm.op.tvm.bias_add", "nn.bias_add");
MNM_TVM_OP("mnm.op.tvm.broadcast_to", "broadcast_to");
MNM_TVM_OP("mnm.op.tvm.broadcast_to_like", "broadcast_to_like");
MNM_TVM_OP("mnm.op.tvm.cast", "cast");
MNM_TVM_OP("mnm.op.tvm.cast_like", "cast_like");
MNM_TVM_OP("mnm.op.tvm.ceil", "ceil");
MNM_TVM_OP("mnm.op.tvm.clip", "clip");
MNM_TVM_OP("mnm.op.tvm.collapse_sum_like", "collapse_sum_like");
MNM_TVM_OP("mnm.op.tvm.compiler_begin", "annotation.compiler_begin");
MNM_TVM_OP("mnm.op.tvm.compiler_end", "annotation.compiler_end");
MNM_TVM_OP("mnm.op.tvm.concatenate", "concatenate");
MNM_TVM_OP("mnm.op.tvm.copy", "copy");
MNM_TVM_OP("mnm.op.tvm.cos", "cos");
MNM_TVM_OP("mnm.op.tvm.dense", "nn.dense");
MNM_TVM_OP("mnm.op.tvm.device_copy", "device_copy");
MNM_TVM_OP("mnm.op.tvm.divide", "divide");
MNM_TVM_OP("mnm.op.tvm.equal", "equal");
MNM_TVM_OP("mnm.op.tvm.erf", "erf");
MNM_TVM_OP("mnm.op.tvm.exp", "exp");
MNM_TVM_OP("mnm.op.tvm.expand_dims", "expand_dims");
MNM_TVM_OP("mnm.op.tvm.floor", "floor");
MNM_TVM_OP("mnm.op.tvm.floor_divide", "floor_divide");
MNM_TVM_OP("mnm.op.tvm.floor_mod", "floor_mod");
MNM_TVM_OP("mnm.op.tvm.gather", "gather");
MNM_TVM_OP("mnm.op.tvm.gather_nd", "gather_nd");
MNM_TVM_OP("mnm.op.tvm.get_valid_counts", "get_valid_counts");
MNM_TVM_OP("mnm.op.tvm.greater", "greater");
MNM_TVM_OP("mnm.op.tvm.greater_equal", "greater_equal");
MNM_TVM_OP("mnm.op.tvm.layout_transform", "layout_transform");
MNM_TVM_OP("mnm.op.tvm.left_shift", "left_shift");
MNM_TVM_OP("mnm.op.tvm.less", "less");
MNM_TVM_OP("mnm.op.tvm.less_equal", "less_equal");
MNM_TVM_OP("mnm.op.tvm.log", "log");
MNM_TVM_OP("mnm.op.tvm.log2", "log2");
MNM_TVM_OP("mnm.op.tvm.log_softmax", "nn.log_softmax");
MNM_TVM_OP("mnm.op.tvm.logical_and", "logical_and");
MNM_TVM_OP("mnm.op.tvm.logical_not", "logical_not");
MNM_TVM_OP("mnm.op.tvm.logical_or", "logical_or");
MNM_TVM_OP("mnm.op.tvm.max", "max");
MNM_TVM_OP("mnm.op.tvm.max_pool2d", "nn.max_pool2d");
MNM_TVM_OP("mnm.op.tvm.max_pool2d_dx", "nn.max_pool2d_grad");
MNM_TVM_OP("mnm.op.tvm.maximum", "maximum");
MNM_TVM_OP("mnm.op.tvm.mean", "mean");
MNM_TVM_OP("mnm.op.tvm.min", "min");
MNM_TVM_OP("mnm.op.tvm.minimum", "minimum");
MNM_TVM_OP("mnm.op.tvm.mod", "mod");
MNM_TVM_OP("mnm.op.tvm.multiply", "multiply");
MNM_TVM_OP("mnm.op.tvm.negative", "negative");
MNM_TVM_OP("mnm.op.tvm.non_max_suppression", "non_max_suppression");
MNM_TVM_OP("mnm.op.tvm.not_equal", "not_equal");
MNM_TVM_OP("mnm.op.tvm.one_hot", "one_hot");
MNM_TVM_OP("mnm.op.tvm.ones", "ones");
MNM_TVM_OP("mnm.op.tvm.ones_like", "ones_like");
MNM_TVM_OP("mnm.op.tvm.power", "power");
MNM_TVM_OP("mnm.op.tvm.prod", "prod");
MNM_TVM_OP("mnm.op.tvm.reinterpret", "reinterpret");
MNM_TVM_OP("mnm.op.tvm.relu", "nn.relu");
MNM_TVM_OP("mnm.op.tvm.repeat", "repeat");
MNM_TVM_OP("mnm.op.tvm.reshape", "reshape");
MNM_TVM_OP("mnm.op.tvm.reverse", "reverse");
MNM_TVM_OP("mnm.op.tvm.reverse_sequence", "reverse_sequence");
MNM_TVM_OP("mnm.op.tvm.right_shift", "right_shift");
MNM_TVM_OP("mnm.op.tvm.roi_align", "vision.roi_align");
MNM_TVM_OP("mnm.op.tvm.round", "round");
MNM_TVM_OP("mnm.op.tvm.rsqrt", "rsqrt");
MNM_TVM_OP("mnm.op.tvm.scatter", "scatter");
MNM_TVM_OP("mnm.op.tvm.sequence_mask", "sequence_mask");
MNM_TVM_OP("mnm.op.tvm.sigmoid", "sigmoid");
MNM_TVM_OP("mnm.op.tvm.sign", "sign");
MNM_TVM_OP("mnm.op.tvm.sin", "sin");
MNM_TVM_OP("mnm.op.tvm.slice_like", "slice_like");
MNM_TVM_OP("mnm.op.tvm.softmax", "nn.softmax");
MNM_TVM_OP("mnm.op.tvm.split", "split");
MNM_TVM_OP("mnm.op.tvm.sqrt", "sqrt");
MNM_TVM_OP("mnm.op.tvm.squeeze", "squeeze");
MNM_TVM_OP("mnm.op.tvm.stack", "stack");
MNM_TVM_OP("mnm.op.tvm.strided_slice", "strided_slice");
MNM_TVM_OP("mnm.op.tvm.subtract", "subtract");
MNM_TVM_OP("mnm.op.tvm.take", "take");
MNM_TVM_OP("mnm.op.tvm.tanh", "tanh");
MNM_TVM_OP("mnm.op.tvm.threefry_generate", "random.threefry_generate");
MNM_TVM_OP("mnm.op.tvm.threefry_split", "random.threefry_split");
MNM_TVM_OP("mnm.op.tvm.tile", "tile");
MNM_TVM_OP("mnm.op.tvm.topk", "topk");
MNM_TVM_OP("mnm.op.tvm.transpose", "transpose");
MNM_TVM_OP("mnm.op.tvm.trunc", "trunc");
MNM_TVM_OP("mnm.op.tvm.variance", "variance");
MNM_TVM_OP("mnm.op.tvm.where", "where");
MNM_TVM_OP("mnm.op.tvm.zeros", "zeros");
MNM_TVM_OP("mnm.op.tvm.zeros_like", "zeros_like");
}  // namespace
}  // namespace op
}  // namespace mnm
