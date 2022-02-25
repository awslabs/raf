/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "raf/ir.h"
#include "raf/value.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/memory_pool.h"

#include "../../../common/cuda_utils.h"

namespace raf {
namespace op {
namespace cutlass {

using namespace ::cutlass;
using namespace ::cutlass::library;

static const char add[] = "raf.op.cutlass.add";
static const char subtract[] = "raf.op.cutlass.subtract";
static const char multiply[] = "raf.op.cutlass.multiply";
static const char divide[] = "raf.op.cutlass.divide";

template <const char* Binary>
struct BinaryOp {
  ir::DFPattern operator()(ir::DFPattern x1, ir::DFPattern x2) {
    using namespace raf::ir;
    auto op = IsOp(Binary);
    return op({x1, x2});
  }
};

template <const char* Binary>
struct BinaryUfuncOp {
  ir::DFPattern operator()(ir::DFPattern x1, ir::DFPattern x2) {
    using namespace raf::ir;
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
}  // namespace raf
