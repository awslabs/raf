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
 * \file canonicalize_ops.cc
 * \brief Canonicalize Ops
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "mnm/executor.h"
#include "mnm/binding.h"
#include "../op/schema/nn.h"
#include "../op/schema/transform.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace canonicalize_ops {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::op::schema;
using namespace mnm::value;

inline Expr ExpandBiasToMatchAxis(Expr bias, int target_ndim, const Array<Integer>& axes,
                                  LetList* ll) {
  static const Op& expand_dims = Op::Get("mnm.op.expand_dims");
  for (int64_t i = axes.size(); i != 0; --i) {
    if (i == axes.size()) {
      int64_t num_pad_axis = target_ndim - axes[i - 1]->value - 1;
      if (num_pad_axis > 0) {
        bias = ll->Push(Call(expand_dims, {bias, MakeConstant(ScalarValue::make(i)),
                                           MakeConstant(ScalarValue::make(num_pad_axis))}));
      }
    } else {
      int64_t diff = axes[i]->value - axes[i - 1]->value;
      CHECK_GE(diff, 0L);
      if (diff > 0) {
        bias = ll->Push(Call(expand_dims, {bias, MakeConstant(ScalarValue::make(i)),
                                           MakeConstant(ScalarValue::make(diff))}));
      }
    }
  }
  return bias;
}

inline Expr Add(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("mnm.op.add");
  return Call(op, {lhs, rhs, MakeNull(), MakeNull()}, Attrs(), {});
}

class BiasAddSimplifier : public ExprMutator {
 public:
  BiasAddSimplifier() : bias_add_op_(Op::Get("mnm.op.bias_add")) {
  }

  Expr VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {};
    auto post_visit = [this](const LetNode* op) {
      memo_[GetRef<Let>(op)] = ExprMutator::VisitExpr_(op);
      if (const auto* cn = op->value.as<CallNode>()) {
        if (cn->op == bias_add_op_) {
          CHECK_EQ(cn->args.size(), 3U);
          Expr x = cn->args[0];
          Expr bias = cn->args[1];
          const auto& axis =
              Downcast<value::IntValue>(ConstantExtractValue(Downcast<Constant>(cn->args[2])));
          int normalized_axis = axis->value;
          if (!x->checked_type_.defined()) return;
          const auto* ttype = x->type_as<TensorTypeNode>();
          CHECK(ttype);
          size_t n_dim = ttype->shape.size();
          if (normalized_axis < 0) {
            normalized_axis += n_dim;
          }
          memo_[GetRef<Let>(op)] = LetList::With([&](LetList* ll) {
            bias = ExpandBiasToMatchAxis(bias, n_dim, {normalized_axis}, ll);
            ll->Push(op->var, Add(x, bias));
            return VisitExpr(op->body);
          });
        }
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

 private:
  // Cache the bias_add for equivalence checking.
  const Op& bias_add_op_;
};

}  // namespace canonicalize_ops

Pass CanonicalizeOps() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto simplifier = canonicalize_ops::BiasAddSimplifier();
    return Downcast<Function>(simplifier(f));
  };
  return CreateMNMFunctionPass(pass_func, 1, "CanonicalizeOps", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.CanonicalizeOps").set_body_typed(CanonicalizeOps);

}  // namespace pass
}  // namespace mnm
