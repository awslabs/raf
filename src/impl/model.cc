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

namespace mnm {
namespace model {

using namespace mnm::ir;
using namespace mnm::value;
using binding::DeStruct;
using binding::DeTuple;
using binding::GradTape;
using binding::NDArrayBindingObj;
using executor::interpreter::Interpret;
using executor::interpreter::Interpret;
using pass::AutoDiff;

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
  if (!requires_grad) {
    return DeTuple(Interpret(Call(func, args)));
  }
  func = AutoDiff(func);
  TupleValue result = Downcast<TupleValue>(Interpret(Call(func, args)));
  CHECK_EQ(result->fields.size(), 2U);
  return DeStruct(/*value=*/result->fields[0],
                  /*bp=*/Downcast<ClosureValue>(result->fields[1]),
                  /*prev_tapes=*/{grads.begin(), grads.end()});
}

MNM_REGISTER_GLOBAL("mnm.model.RunModel").set_body_typed(RunModel);

}  // namespace model
}  // namespace mnm
