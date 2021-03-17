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
  if (!requires_grad) {
    // TODO(haibin): add simplify inference pass - simplify the compute of
    // BN, LN, Dropout, GN, etc.
    func = Downcast<Function>(CanonicalizeOps(func));
    func = Downcast<Function>(FoldConstant(func, updated_mod));
    auto call_node = Call(func, args);
    return DeTuple(Interpret(call_node, updated_mod));
  }
  // run canonicalize ops pass (it needs "inter type pass" to work properly.)
  func = Downcast<Function>(CanonicalizeOps(func));
  // run auto diff pass
  // TODO (janimesh) - Clean this up when pass manager is introduced
  updated_mod->Add(updated_mod->GetGlobalVar("main"), func, true);
  updated_mod = AutoDiff(updated_mod, requires_grads);
  func = Downcast<Function>(updated_mod->Lookup("main"));

  // run auto parallel
  if (distributed::DistContext::Global()->enable_data_parallel) {
    func = AutoDataParallel(func);
  }

  // run const folding pass
  func = Downcast<Function>(FoldConstant(func, updated_mod));
  TupleValue result = Downcast<TupleValue>(Interpret(Call(func, args), updated_mod));
  CHECK_EQ(result->fields.size(), 2U);
  return DeStruct(/*value=*/result->fields[0],
                  /*bp=*/Downcast<ClosureValue>(result->fields[1]),
                  /*prev_tapes=*/{grads.begin(), grads.end()});
}

MNM_REGISTER_GLOBAL("mnm.model.RunModel").set_body_typed(RunModel);

}  // namespace model
}  // namespace mnm
