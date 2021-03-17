/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/dead_code.cc
 * \brief  Remove code that does not effect the program result.
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
 */
#include <unordered_map>
#include <unordered_set>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace tvm {
namespace relay {
extern Expr DeadCodeElimination(const Expr& e, bool inline_once);
}  // namespace relay
}  // namespace tvm

namespace mnm {
namespace pass {

ir::Expr DeadCodeElimination(const ir::Expr& expr) {
  // Don't inline let because Meta uses ANF
  return tvm::relay::DeadCodeElimination(expr, false);
}

// TODO - Cleanup when pass manager is introduced.
ir::IRModule DeadCodeElimination(const ir::IRModule mod) {
  ir::IRModule updated_mod = ir::IRModule(mod->functions);
  std::vector<std::pair<ir::GlobalVar, ir::Function>> updated_funcs;
  for (auto kv : updated_mod->functions) {
    if (kv.second.as<ir::FunctionNode>()) {
      auto expr = tvm::relay::DeadCodeElimination(kv.second, false);
      auto func = tvm::runtime::Downcast<ir::Function>(expr);
      updated_funcs.emplace_back(kv.first, func);
    }
  }

  for (const auto& it : updated_funcs) {
    updated_mod->Add(it.first, it.second, true);
  }
  return updated_mod;
}

MNM_REGISTER_GLOBAL("mnm.pass_.DeadCodeElimination").set_body_typed([](ir::IRModule mod) {
  return DeadCodeElimination(mod);
});

}  // namespace pass
}  // namespace mnm
