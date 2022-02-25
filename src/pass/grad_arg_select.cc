/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file  grad_arg_select.cc
 * \brief Gradient operator input selection pass
 */
#include <sstream>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include <string>
#include <vector>

namespace raf {
namespace pass {
namespace arg_select {

using namespace raf::ir;
using namespace raf::op;

// Init a map to set which argument should be skipped (set to Null) for which op
RAF_OP_GRAD_SKIP_INPUTS("raf.op.relu_dx", "x");
RAF_OP_GRAD_SKIP_INPUTS("raf.op.gelu_dx", "y");
RAF_OP_GRAD_SKIP_INPUTS("raf.op.sqrt_dx", "x");
RAF_OP_GRAD_SKIP_INPUTS("raf.op.conv2d_dx", "y");
RAF_OP_GRAD_SKIP_INPUTS("raf.op.conv2d_dw", "y");

class GradientOp : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      const Op& op = Downcast<Op>(node->op);
      auto skip_arg_map = Op::GetAttrMap<std::string>("GradientInputSkip");
      auto fschema_index = Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
      if (skip_arg_map.count(op)) {
        Array<Expr> args;
        std::set<int> skip_index_set{};
        std::stringstream arg_string(skip_arg_map[op]);
        std::string field;
        while (getline(arg_string, field, ',')) {
          int idx = fschema_index[op](field);
          skip_index_set.insert(idx);
        }
        for (int i = 0; i < node->args.size(); ++i) {
          if (skip_index_set.count(i)) {
            args.push_back(MakeConstant(NullValue<value::Value>()));
          } else {
            args.push_back(node->args[i]);
          }
        }
        auto calln = Call(node->op, args);
        return Call(node->op, args);
      }
    }
    return ExprMutator::VisitExpr_(node);
  }
};

}  // namespace arg_select

Pass GradInputSelect() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::GradInputSelect";
        ir::IRModule updated_mod = ir::IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<ir::FunctionNode>()) {
            auto func =
                tvm::runtime::Downcast<ir::Function>(arg_select::GradientOp().VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "GradientInputSelection", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.GradientInputSelection").set_body_typed(GradInputSelect);

}  // namespace pass
}  // namespace raf
