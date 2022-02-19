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
 * \file memory_op_utils.cc
 * \brief Utilities for memory copy and collective ops.
 */
#pragma once
#include <unordered_map>
#include <algorithm>
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/dist_context.h"
#include "mnm/pass.h"
#include "mnm/analysis.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "./common.h"
#include "mnm/stream_pool.h"
#include "../op/ty/utils.h"
#include "../op/schema/memory.h"
#include "../op/schema/communication.h"
#include "../common/shape_utils.h"

namespace mnm {
namespace pass {
namespace memory_op_utils {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace mnm::analysis;
using namespace mnm::common::shape_utils;
using mnm::distributed::DistContext;
using op::IsMemcpyOp;

enum class FuseOp { kFuse, kFuseReorder, kNone };
enum class DefuseOp { kDefuse, kNone };
using FuseAndDefuseOp = std::pair<FuseOp, DefuseOp>;
using FuseAndDefuseOpToAddMap =
    std::unordered_map<Op, FuseAndDefuseOp, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * Use collective ops to decide tensor fusion and defusion ops to add.
 *
 * For allreduce, broadcast and reduce, we add fuse tensor op before
 * and defuse tensor op after it.
 *
 * For reduce scatter, we should add fuse reorder op before it and
 * defuse tensor op after it.
 *
 * Other collective ops do not need any memory copy ops.
 */
static FuseAndDefuseOpToAddMap fuse_and_defuse_op_to_add_map = {
    {Op::Get("mnm.op._allreduce"), {FuseOp::kFuse, DefuseOp::kDefuse}},
    {Op::Get("mnm.op._allgather"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("mnm.op._reduce_scatter"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("mnm.op._broadcast"), {FuseOp::kFuse, DefuseOp::kDefuse}},
    {Op::Get("mnm.op._reduce"), {FuseOp::kFuse, DefuseOp::kDefuse}},
    {Op::Get("mnm.op._send"), {FuseOp::kNone, DefuseOp::kNone}},
    {Op::Get("mnm.op._recv"), {FuseOp::kNone, DefuseOp::kNone}},
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
}  // namespace mnm
