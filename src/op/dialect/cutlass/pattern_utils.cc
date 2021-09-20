/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dialect/cutlass/pattern_utils.cc
 * \brief dataflow pattern helpers
 */
#include "./pattern_utils.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::ir;

DFPattern IsOps(std::vector<std::string> ops) {
  CHECK_GE(ops.size(), 1U);
  auto op = IsOp(ops[0]);
  for (const auto& name : ops) {
    op = op || IsOp(name);
  }
  return op;
}

EpilogueKindExt GetEpilogueKind(const Op& op) {
  if (!op.defined()) {
    return EpilogueKindExt::kLinearCombination;
  }
  const static std::unordered_map<Op, EpilogueKindExt, ObjectPtrHash, ObjectPtrEqual> epilogue_map =
      {{Op::Get("mnm.op.cutlass.relu"), EpilogueKindExt::kLinearCombinationRelu},
       {Op::Get("mnm.op.cutlass.gelu"), EpilogueKindExt::kLinearCombinationGelu}};
  auto it = epilogue_map.find(op);
  if (it == epilogue_map.end()) {
    LOG(FATAL) << "Unknown epilogue op: " << op->name;
  }
  return it->second;
}

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
