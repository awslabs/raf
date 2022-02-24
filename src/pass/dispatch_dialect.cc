/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/dispatch_dialect.cc
 * \brief Dispatch the base ops to device-specific dialect ops based on predefined plevels. Note
 * that some ops such as VM related ops do not have dialect ops, and they will remain the same after
 * this pass.
 */
#include <vector>
#include "raf/device.h"
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"

namespace raf {
namespace pass {
namespace dispatch_dialect {

using namespace raf::ir;
using namespace raf::op;

class DispatchMutator : public MixedModeMutator {
 public:
  DispatchMutator(DevType dev_type) : dev_type_(dev_type) {
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    if (node->HasNonzeroAttr(attr::kPrimitive)) {
      // Don't go into fused functions
      return GetRef<Function>(node);
    }
    return ExprMutator::VisitExpr_(node);
  }

  Expr VisitExpr_(const OpNode* node) final {
    auto op = GetRef<Op>(node);
    if (!IsDialectOp(op)) {
      auto dialect_op = OpDialect::Dispatch(op, dev_type_, {});
      if (dialect_op.defined()) {
        return dialect_op;
      }
    }
    return op;
  }

 private:
  DevType dev_type_;
};

Expr Dispatch(const Expr& expr) {
  auto dev = Device::Current(true);
  if (dev->device_type == DevType::kUnknown() || dev->device_id < 0) {
    LOG(WARNING) << "Device is not specified, skip DispatchDialect pass.";
    return expr;
  }
  DevType dev_type = dev.device_type();
  return DispatchMutator(dev_type).Mutate(expr);
}

}  // namespace dispatch_dialect

Pass DispatchDialect() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(dispatch_dialect::Dispatch(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "DispatchDialect", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.DispatchDialect").set_body_typed(DispatchDialect);

}  // namespace pass
}  // namespace raf
