/*!
 * Copyright (c) 2020 by Contributors
 * \file model.cc
 * \brief Helpers for running models.
 */
#include "mnm/binding.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/executor.h"
#include "mnm/pass.h"
#include "mnm/dist_context.h"

namespace mnm {
namespace model {

using namespace mnm::ir;
using namespace mnm::value;
using binding::DeStruct;
using binding::DeTuple;
using binding::GradTape;
using binding::NDArrayBindingObj;
using executor::interpreter::Interpret;
using pass::AutoDataParallel;
using pass::AutoDiff;
using pass::BindParam;
using pass::CanonicalizeOps;
using pass::FoldConstant;

ObjectRef RunModel(ir::IRModule mod, Array<Expr> args) {
  ir::IRModule updated_mod = ir::IRModule(mod->functions);
  std::vector<GradTape> grads;
  ir::Array<Bool> requires_grads;
  grads.reserve(args.size());
  bool requires_grad = false;
  for (const Expr& arg : args) {
    if (const auto* a = arg.as<VarNode>()) {
      if (const auto* bound = binding::LookupBinding(a).as<NDArrayBindingObj>()) {
        if (bound->tape.defined()) {
          requires_grad = true;
        }
        requires_grads.push_back(Bool(bound->tape.defined()));
        grads.push_back(bound->tape);
      }
    }
  }

  // TODO - Revisit which passes require update due to presence of module
  Function func = Downcast<Function>(updated_mod->Lookup("main"));
  func = Downcast<Function>(BindParam(func, args));
  auto gvar = updated_mod->GetGlobalVar("main");
  updated_mod->Add(gvar, func);

  if (!requires_grad) {
    // TODO(haibin): add simplify inference pass - simplify the compute of
    // BN, LN, Dropout, GN, etc.
    mnm::pass::MNMSequential seq({CanonicalizeOps(), FoldConstant()});
    updated_mod = seq(updated_mod);
    func = Downcast<Function>(updated_mod->Lookup("main"));
    auto call_node = Call(func, args);
    return DeTuple(Interpret(call_node, updated_mod));
  }

  // Glob the needed passes.
  Array<tvm::transform::Pass> passes;
  // run canonicalize ops pass (it needs "inter type pass" to work properly.)
  passes.push_back(CanonicalizeOps());
  // run auto diff pass
  passes.push_back(AutoDiff(requires_grads));

  // run auto parallel
  if (distributed::DistContext::Global()->enable_data_parallel) {
    passes.push_back(AutoDataParallel());
  }

  // run const folding pass
  passes.push_back(FoldConstant());
  mnm::pass::MNMSequential seq(passes);
  updated_mod = seq(updated_mod);
  func = Downcast<Function>(updated_mod->Lookup("main"));
  TupleValue result = Downcast<TupleValue>(Interpret(Call(func, args), updated_mod));
  CHECK_EQ(result->fields.size(), 2U);
  return DeStruct(/*value=*/result->fields[0],
                  /*bp=*/Downcast<ClosureValue>(result->fields[1]),
                  /*prev_tapes=*/{grads.begin(), grads.end()});
}

MNM_REGISTER_GLOBAL("mnm.model.RunModel").set_body_typed(RunModel);

}  // namespace model
}  // namespace mnm
