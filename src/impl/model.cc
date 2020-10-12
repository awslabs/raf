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

ObjectRef RunModel(Function func, Array<Expr> args) {
  std::vector<GradTape> grads;
  grads.reserve(args.size());
  bool requires_grad = false;
  for (const Expr& arg : args) {
    if (const auto* a = arg.as<VarNode>()) {
      if (const auto* bound = binding::LookupBinding(a).as<NDArrayBindingObj>()) {
        if (bound->tape.defined()) {
          requires_grad = true;
        }
        grads.push_back(bound->tape);
      }
    }
  }

  auto mod = Module::Global();
  func = Downcast<Function>(BindParam(func, args));
  if (!requires_grad) {
    // TODO(haibin): add simplify inference pass - simplify the compute of
    // BN, LN, Dropout, GN, etc.
    func = Downcast<Function>(CanonicalizeOps(func));
    func = Downcast<Function>(FoldConstant(func, mod));
    auto call_node = Call(func, args);
    return DeTuple(Interpret(call_node));
  }
  // run canonicalize ops pass (it needs "inter type pass" to work properly.)
  func = Downcast<Function>(CanonicalizeOps(func));
  // run auto diff pass
  func = AutoDiff(func);

  // run auto parallel
  if (distributed::DistContext::Global()->enable_data_parallel) {
    func = AutoDataParallel(func);
  }

  // run const folding pass
  func = Downcast<Function>(FoldConstant(func, mod));
  TupleValue result = Downcast<TupleValue>(Interpret(Call(func, args)));
  CHECK_EQ(result->fields.size(), 2U);
  return DeStruct(/*value=*/result->fields[0],
                  /*bp=*/Downcast<ClosureValue>(result->fields[1]),
                  /*prev_tapes=*/{grads.begin(), grads.end()});
}

MNM_REGISTER_GLOBAL("mnm.model.RunModel").set_body_typed(RunModel);

}  // namespace model
}  // namespace mnm
