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

Pass DeadCodeElimination() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(DeadCodeElimination(f));
  };
  return CreateMNMFunctionPass(pass_func, 1, "DeadCodeElimination", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.DeadCodeElimination").set_body_typed([]() {
  return DeadCodeElimination();
});

}  // namespace pass
}  // namespace mnm
