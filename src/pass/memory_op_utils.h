/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_op_utils.cc
 * \brief Utilities for memory copy and collective ops.
 */
#pragma once
#include <unordered_map>
#include <algorithm>
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/dist_config.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "./common.h"
#include "raf/stream_pool.h"
#include "../op/ty/utils.h"
#include "../op/schema/memory.h"
#include "../op/schema/communication.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {
namespace memory_op_utils {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::op::schema;
using namespace raf::analysis;
using namespace raf::common::shape_utils;
using op::IsMemcpyOp;
using raf::distributed::DistConfig;

enum class FuseOp { kFuse, kFuseReorder, kNone };
enum class DefuseOp { kDefuse, kNone };
using FuseAndDefuseOp = std::pair<FuseOp, DefuseOp>;
using FuseAndDefuseOpToAddMap =
    std::unordered_map<Op, FuseAndDefuseOp, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * Use collective ops to decide tensor fusion and defusion ops to add.
 *
 * For allreduce, we add fuse tensor op before
 * and defuse tensor op after it.
 *
 * For reduce scatter, we should add fuse reorder op before it and
 * defuse tensor op after it.
 *
 * Other collective ops do not need any memory copy ops.
 */
static FuseAndDefuseOpToAddMap fuse_and_defuse_op_to_add_map = {
    {Op::Get("raf.op._allreduce"), {FuseOp::kFuse, DefuseOp::kDefuse}},
    {Op::Get("raf.op._allgather"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("raf.op._reduce_scatter"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("raf.op._broadcast"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("raf.op._reduce"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("raf.op._send"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("raf.op._recv"), {FuseOp::kNone, DefuseOp::kNone}},
};

/*!
 * If multiple tensors are in the tuple, then should split the collective op.
 */
inline bool ShouldAddFuseAndDefuseOps(const CallNode* call) {
  TupleType tt = Downcast<TupleType>(call->args[0]->checked_type_);
  return tt->fields.size() > 1;
}

inline FuseAndDefuseOp GetFuseAndDefuseOpToAdd(const CallNode* call) {
  Op op = Downcast<Op>(call->op);
  Op op_n = IsDialectOp(op) ? GetBaseOp(op) : op;
  if (!fuse_and_defuse_op_to_add_map.count(op_n) || !ShouldAddFuseAndDefuseOps(call)) {
    return std::make_pair(FuseOp::kNone, DefuseOp::kNone);
  }
  return fuse_and_defuse_op_to_add_map.at(op_n);
}

using DefuseTensorArgs =
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>;

DefuseTensorArgs GetDefuseTensorArgsFromTupleType(TupleType tt) {
  std::vector<int64_t> shapes = {};
  std::vector<int64_t> shape_indices = {};
  std::vector<int64_t> sizes = {};
  int64_t shape_index = 0;
  for (auto ty : tt->fields) {
    TensorType ty_ = Downcast<TensorType>(ty);
    int64_t size = 1;
    for (auto axis : ty_->shape) {
      auto node = axis.as<IntImmNode>();
      CHECK(node != nullptr) << "Axis " << axis << " is not IntImmNode";
      int64_t axis_ = node->value;
      shapes.push_back(axis_);
      size *= axis_;
    }
    shape_index += ty_->shape.size();
    shape_indices.push_back(shape_index);
    sizes.push_back(size);
  }
  DefuseTensorArgs args = std::make_tuple(sizes, shapes, shape_indices);
  return args;
}

DefuseTensorArgs GetDefuseTensorArgs(const CallNode* call) {
  TupleType tt = Downcast<TupleType>(call->args[0]->checked_type_);
  return GetDefuseTensorArgsFromTupleType(tt);
}

}  // namespace memory_op_utils
}  // namespace pass
}  // namespace raf
