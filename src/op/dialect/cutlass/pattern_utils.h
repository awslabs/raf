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
 * \file src/op/dialect/cutlass/pattern_utils.h
 * \brief dataflow pattern helpers
 */
#pragma once

#include <cutlass/library/library.h>
#include <cutlass/library/handle.h>
#include <cutlass/library/singleton.h>
#include <cutlass/library/operation_table.h>
#include <cutlass_ext/library/library_ext.h>

#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/memory_pool.h"

#include "../../../common/cuda_utils.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace ::cutlass;
using namespace ::cutlass::library;

static const char add[] = "mnm.op.cutlass.add";
static const char subtract[] = "mnm.op.cutlass.subtract";
static const char multiply[] = "mnm.op.cutlass.multiply";
static const char divide[] = "mnm.op.cutlass.divide";

template <const char* Binary>
struct BinaryOp {
  ir::DFPattern operator()(ir::DFPattern x1, ir::DFPattern x2) {
    using namespace mnm::ir;
    auto op = IsOp(Binary);
    return op({x1, x2});
  }
};

template <const char* Binary>
struct BinaryUfuncOp {
  ir::DFPattern operator()(ir::DFPattern x1, ir::DFPattern x2) {
    using namespace mnm::ir;
    auto op = IsOp(Binary);
    return op({x1, x2, IsWildcard(), IsWildcard()});
  }
};

using Add = BinaryUfuncOp<add>;
using Subtract = BinaryUfuncOp<subtract>;
using Multiply = BinaryOp<multiply>;
using Divide = BinaryOp<divide>;

ir::DFPattern IsOps(std::vector<std::string> ops);

EpilogueKindExt GetEpilogueKind(const ir::Op& op);

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
