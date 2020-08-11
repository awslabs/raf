/*!
 * Copyright (c) 2020 by Contributors
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

namespace mnm {
namespace pass {
namespace canonicalize_ops {

using namespace mnm::ir;
using namespace mnm::op;
using namespace tvm;
using namespace runtime;
using namespace mnm::op::schema;
using namespace mnm::value;

inline Expr ExpandBiasToMatchAxis(Expr bias, int target_ndim, const Array<Integer>& axes) {
  static const Op& expand_dims = Op::Get("mnm.op.expand_dims");
  for (size_t i = axes.size(); i != 0; --i) {
    if (i == axes.size()) {
      int64_t num_pad_axis = target_ndim - axes[i - 1]->value - 1;
      if (num_pad_axis > 0) {
        bias = Call(
            expand_dims,
            {bias, MakeConstant(IntValue::make(i)), MakeConstant(IntValue::make(num_pad_axis))},
            Attrs(), {});
      }
    } else {
      int64_t diff = axes[i]->value - axes[i - 1]->value;
      CHECK_GE(diff, 0L);
      if (diff > 0) {
        bias = Call(expand_dims,
                    {bias, MakeConstant(IntValue::make(i)), MakeConstant(IntValue::make(diff))},
                    Attrs(), {});
      }
    }
  }
  return bias;
}

inline Expr Add(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("mnm.op.add");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

class BiasAddSimplifier : public ExprRewriter {
 public:
  BiasAddSimplifier() : bias_add_op_(Op::Get("mnm.op.bias_add")) {
  }

  Expr Rewrite_(const LetNode* n, const Expr& post) override {
    const CallNode* node = n->value.as<CallNode>();
    if (node != nullptr && node->op == bias_add_op_) {
      Call call = Downcast<Call>(n->value);
      CHECK_EQ(call->args.size(), 3);
      const ConstantNode* axis_p = call->args[2].as<ConstantNode>();
      int axis = Downcast<value::IntValue>(axis_p->value)->data;
      auto x = call->args[0];
      if (!x->checked_type_.defined()) {
        // skip rewrite when type does not exist
        return post;
      }
      auto ttype = x->type_as<TensorTypeNode>();
      size_t n_dim = ttype->shape.size();
      if (axis < 0) {
        axis += n_dim;
      }
      Expr body = n->body;
      Var tmp("exp_bias_tmp", {});
      body = Let(n->var, Add(call->args[0], tmp), body);
      body = Let(tmp, ExpandBiasToMatchAxis(call->args[1], n_dim, {axis}), body);
      return body;
    }
    return post;
  }

 private:
  // Cache the bias_add for equivalence checking.
  const Op& bias_add_op_;
};

Expr CanonicalizeOps(const Expr& e) {
  auto rewriter = BiasAddSimplifier();
  return PostOrderRewrite(e, &rewriter);
}
}  // namespace canonicalize_ops

ir::Expr CanonicalizeOps(ir::Expr expr) {
  return canonicalize_ops::CanonicalizeOps(expr);
}

MNM_REGISTER_GLOBAL("mnm.pass_.CanonicalizeOps").set_body_typed(CanonicalizeOps);

}  // namespace pass
}  // namespace mnm
