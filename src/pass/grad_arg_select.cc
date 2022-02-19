/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file  grad_arg_select.cc
 * \brief Gradient operator input selection pass
 */
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include <string>
#include <vector>

namespace mnm {
namespace pass {
namespace arg_select {

using namespace mnm::ir;
using namespace mnm::op;

// Init a map to set which argument should be skipped (set to Null) for which op
MNM_OP_GRAD_SKIP_INPUTS("mnm.op.relu_dx", "x");
MNM_OP_GRAD_SKIP_INPUTS("mnm.op.gelu_dx", "y");
MNM_OP_GRAD_SKIP_INPUTS("mnm.op.sqrt_dx", "x");
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

MNM_REGISTER_GLOBAL("mnm.pass_.GradientInputSelection").set_body_typed(GradInputSelect);

}  // namespace pass
}  // namespace mnm
