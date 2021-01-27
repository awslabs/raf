/*!
 * Copyright (c) 2021 by Contributors
 * \file  grad_arg_select.cc
 * \brief Gradient operator input selection pass
 */
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include <string>
#include <vector>

namespace mnm {
namespace pass {
namespace arg_select {

using namespace mnm::ir;
using namespace mnm::op;

// Init a map to set which argument should be skipped (set to Null) for which op
MNM_OP_GRAD_SKIP_INPUTS("mnm.op.relu_dx", "x");
MNM_OP_GRAD_SKIP_INPUTS("mnm.op.conv2d_dx", "y");
MNM_OP_GRAD_SKIP_INPUTS("mnm.op.conv2d_dw", "y");

class GradientOp : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      const Op& op = Downcast<Op>(node->op);
      auto skip_arg_map = Op::GetAttrMap<std::string>("GradientInputSkip");
      auto fschema_index = Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
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
      } else {
        return ExprMutator::VisitExpr_(node);
      }
    } else {
      LOG(FATAL) << "NotImplementedError: Calling unsupported type: " << callee->GetTypeKey();
      throw;
    }
  }
};

}  // namespace arg_select

ir::Expr GradInputSelect(ir::Expr func) {
  return arg_select::GradientOp().VisitExpr(func);
}
MNM_REGISTER_GLOBAL("mnm.pass_.GradientInputSelection").set_body_typed(GradInputSelect);

}  // namespace pass
}  // namespace mnm
