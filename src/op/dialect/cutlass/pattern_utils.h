/*!
 * Copyright (c) 2021 by Contributors
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

static const char add[] = "mnm.op.add";
static const char subtract[] = "mnm.op.subtract";
static const char multiply[] = "mnm.op.multiply";
static const char divide[] = "mnm.op.divide";

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
