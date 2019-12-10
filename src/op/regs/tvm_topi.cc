/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/regs/tvm_topi.cc
 * \brief Auto generated. Do not touch.
 */
#include "mnm/ir.h"
#include "mnm/op.h"
namespace mnm {
namespace op {
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
#define SET_COMPUTE_SCHEDULE(OP)                                                                  \
  set_attr<FTVMCompute>("FTVMCompute",                                                            \
                        [](const Attrs& attrs, const Array<Tensor>& inputs, const Type& out_type, \
                           const Target& target) -> Array<Tensor> {                               \
                          auto fcompute = Op::GetAttr<FTVMCompute>("FTVMCompute")[Op::Get(OP)];   \
                          return fcompute(attrs, inputs, out_type, target);                       \
                        })                                                                        \
      .set_attr<FTVMSchedule>(                                                                    \
          "FTVMSchedule",                                                                         \
          [](const Attrs& attrs, const Array<Tensor>& outs, const Target& target) -> Schedule {   \
            auto fschedule = Op::GetAttr<FTVMSchedule>("FTVMSchedule")[Op::Get(OP)];              \
            return fschedule(attrs, outs, target);                                                \
          })
MNM_OP_REGISTER("mnm.op.abs")
.SET_COMPUTE_SCHEDULE("abs")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.add")
.SET_COMPUTE_SCHEDULE("add")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.all")
.SET_COMPUTE_SCHEDULE("all")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.any")
.SET_COMPUTE_SCHEDULE("any")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.arange")
.SET_COMPUTE_SCHEDULE("arange")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kOpaque);
MNM_OP_REGISTER("mnm.op.argmax")
.SET_COMPUTE_SCHEDULE("argmax")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.argmin")
.SET_COMPUTE_SCHEDULE("argmin")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.argsort")
.SET_COMPUTE_SCHEDULE("argsort")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kOpaque);
MNM_OP_REGISTER("mnm.op.argwhere")
.SET_COMPUTE_SCHEDULE("argwhere")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kOpaque);
MNM_OP_REGISTER("mnm.op.atan")
.SET_COMPUTE_SCHEDULE("atan")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.broadcast_to")
.SET_COMPUTE_SCHEDULE("broadcast_to")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.broadcast_to_like")
.SET_COMPUTE_SCHEDULE("broadcast_to_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.cast")
.SET_COMPUTE_SCHEDULE("cast")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.cast_like")
.SET_COMPUTE_SCHEDULE("cast_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.ceil")
.SET_COMPUTE_SCHEDULE("ceil")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.clip")
.SET_COMPUTE_SCHEDULE("clip")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.collapse_sum_like")
.SET_COMPUTE_SCHEDULE("collapse_sum_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.concatenate")
.SET_COMPUTE_SCHEDULE("concatenate")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.copy")
.SET_COMPUTE_SCHEDULE("copy")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.cos")
.SET_COMPUTE_SCHEDULE("cos")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.divide")
.SET_COMPUTE_SCHEDULE("divide")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.equal")
.SET_COMPUTE_SCHEDULE("equal")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.erf")
.SET_COMPUTE_SCHEDULE("erf")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.exp")
.SET_COMPUTE_SCHEDULE("exp")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.expand_dims")
.SET_COMPUTE_SCHEDULE("expand_dims")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.floor")
.SET_COMPUTE_SCHEDULE("floor")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.floor_divide")
.SET_COMPUTE_SCHEDULE("floor_divide")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.floor_mod")
.SET_COMPUTE_SCHEDULE("floor_mod")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.full")
.SET_COMPUTE_SCHEDULE("full")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.full_like")
.SET_COMPUTE_SCHEDULE("full_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.gather_nd")
.SET_COMPUTE_SCHEDULE("gather_nd")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.greater")
.SET_COMPUTE_SCHEDULE("greater")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.greater_equal")
.SET_COMPUTE_SCHEDULE("greater_equal")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.image.resize")
.SET_COMPUTE_SCHEDULE("image.resize")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.layout_transform")
.SET_COMPUTE_SCHEDULE("layout_transform")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.left_shift")
.SET_COMPUTE_SCHEDULE("left_shift")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.less")
.SET_COMPUTE_SCHEDULE("less")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.less_equal")
.SET_COMPUTE_SCHEDULE("less_equal")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.log")
.SET_COMPUTE_SCHEDULE("log")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.logical_and")
.SET_COMPUTE_SCHEDULE("logical_and")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.logical_not")
.SET_COMPUTE_SCHEDULE("logical_not")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.logical_or")
.SET_COMPUTE_SCHEDULE("logical_or")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.max")
.SET_COMPUTE_SCHEDULE("max")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.maximum")
.SET_COMPUTE_SCHEDULE("maximum")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.mean")
.SET_COMPUTE_SCHEDULE("mean")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.min")
.SET_COMPUTE_SCHEDULE("min")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.minimum")
.SET_COMPUTE_SCHEDULE("minimum")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.mod")
.SET_COMPUTE_SCHEDULE("mod")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.multiply")
.SET_COMPUTE_SCHEDULE("multiply")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.negative")
.SET_COMPUTE_SCHEDULE("negative")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.not_equal")
.SET_COMPUTE_SCHEDULE("not_equal")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.one_hot")
.SET_COMPUTE_SCHEDULE("one_hot")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kOutEWiseFusable);
MNM_OP_REGISTER("mnm.op.ones")
.SET_COMPUTE_SCHEDULE("ones")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.ones_like")
.SET_COMPUTE_SCHEDULE("ones_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.power")
.SET_COMPUTE_SCHEDULE("power")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.prod")
.SET_COMPUTE_SCHEDULE("prod")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.reinterpret")
.SET_COMPUTE_SCHEDULE("reinterpret")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.repeat")
.SET_COMPUTE_SCHEDULE("repeat")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.reshape")
.SET_COMPUTE_SCHEDULE("reshape")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.reshape_like")
.SET_COMPUTE_SCHEDULE("reshape_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.reverse")
.SET_COMPUTE_SCHEDULE("reverse")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.right_shift")
.SET_COMPUTE_SCHEDULE("right_shift")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.round")
.SET_COMPUTE_SCHEDULE("round")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.rsqrt")
.SET_COMPUTE_SCHEDULE("rsqrt")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.sequence_mask")
.SET_COMPUTE_SCHEDULE("sequence_mask")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.shape_of")
.SET_COMPUTE_SCHEDULE("shape_of")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kOpaque);
MNM_OP_REGISTER("mnm.op.sigmoid")
.SET_COMPUTE_SCHEDULE("sigmoid")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.sign")
.SET_COMPUTE_SCHEDULE("sign")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.sin")
.SET_COMPUTE_SCHEDULE("sin")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.slice_like")
.SET_COMPUTE_SCHEDULE("slice_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.split")
.SET_COMPUTE_SCHEDULE("split")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.sqrt")
.SET_COMPUTE_SCHEDULE("sqrt")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.squeeze")
.SET_COMPUTE_SCHEDULE("squeeze")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.stack")
.SET_COMPUTE_SCHEDULE("stack")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.strided_slice")
.SET_COMPUTE_SCHEDULE("strided_slice")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.subtract")
.SET_COMPUTE_SCHEDULE("subtract")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.sum")
.SET_COMPUTE_SCHEDULE("sum")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.take")
.SET_COMPUTE_SCHEDULE("take")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.tanh")
.SET_COMPUTE_SCHEDULE("tanh")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.tile")
.SET_COMPUTE_SCHEDULE("tile")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.topk")
.SET_COMPUTE_SCHEDULE("topk")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kOpaque);
MNM_OP_REGISTER("mnm.op.transpose")
.SET_COMPUTE_SCHEDULE("transpose")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kInjective);
MNM_OP_REGISTER("mnm.op.trunc")
.SET_COMPUTE_SCHEDULE("trunc")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.variance")
.SET_COMPUTE_SCHEDULE("variance")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kCommReduce);
MNM_OP_REGISTER("mnm.op.where")
.SET_COMPUTE_SCHEDULE("where")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kBroadcast);
MNM_OP_REGISTER("mnm.op.zeros")
.SET_COMPUTE_SCHEDULE("zeros")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
MNM_OP_REGISTER("mnm.op.zeros_like")
.SET_COMPUTE_SCHEDULE("zeros_like")
.set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise);
}  // namespace op
}  // namespace mnm
