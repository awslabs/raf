/*!
 * Copyright (c) 2020 by Contributors
 * \file common.h
 * \brief common utilities
 */
#pragma once

#include <vector>
#include "mnm/ir.h"

namespace mnm {
namespace pass {

using namespace tvm;
using namespace tvm::relay;

struct ExplicitLetList {
 public:
  std::vector<ir::Var> vars;
  std::vector<ir::Expr> exprs;
  ir::Var ret;

  ir::Expr AsExpr() {
    CHECK_EQ(vars.size(), exprs.size());
    ir::Expr body = ret;
    int n = exprs.size();
    for (int i = n - 1; i >= 0; --i) {
      body = ir::Let(vars[i], exprs[i], body);
    }
    return body;
  }

  static std::unique_ptr<ExplicitLetList> make(const ir::Expr& node) {
    std::unique_ptr<ExplicitLetList> ell = std::make_unique<ExplicitLetList>();
    Maker(ell.get()).VisitExpr(node);
    return ell;
  }

  struct Maker : public ir::ExprVisitor {
    explicit Maker(ExplicitLetList* ell) : ell(ell) {
    }

    void VisitExpr_(const ir::VarNode* node) final {
      ell->ret = ir::GetRef<ir::Var>(node);
    }

    void VisitExpr_(const ir::LetNode* node) final {
      ell->vars.push_back(node->var);
      ell->exprs.push_back(node->value);
      const ir::Expr& expr = node->body;
      if (expr->IsInstance<ir::LetNode>()) {
        VisitExpr(expr);  // tail call
      } else if (expr->IsInstance<ir::VarNode>()) {
        VisitExpr(expr);
      } else {
        LOG(FATAL) << "ValueError: assumes ANF";
        throw;
      }
    }
    ExplicitLetList* ell;
  };
};

/*!
 * \brief Cache the compiler_begin annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_begin op
 */
inline const Op& CompilerBeginOp() {
  static auto op = Op::Get("mnm.op.compiler_begin");
  return op;
}

/*!
 * \brief Cache the compiler_end annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_end op
 */
inline const Op& CompilerEndOp() {
  static auto op = Op::Get("mnm.op.compiler_end");
  return op;
}

/*!
 * \brief Remove the compiler_begin/end annotation of the
 * expression.
 * \param expr The input expression to remove annotations from.
 * \param ann_op The specific annotation to remove.
 * \return The expression after remove annotation.
 */
inline Expr RemoveAnnotation(const Expr& expr, const Op& ann_op) {
  const Op& begin_op = CompilerBeginOp();
  const Op& end_op = CompilerEndOp();

  if (ann_op == begin_op) {
    if (expr.as<CallNode>()) {
      const CallNode* call = expr.as<CallNode>();

      // If the CallNode is annotated by compiler_end, then get
      // the args of the compiler_end.
      if (call->op == CompilerEndOp()) {
        CHECK_EQ(call->args.size(), 1U);
        auto input_expr = call->args[0];

        // Remove the compiler_begin annotation of this input_call,
        // and return the expr after annotate it with compiler_end.
        auto new_expr = RemoveAnnotation(input_expr, begin_op);
        Expr ret_expr = Call(call->op, {new_expr}, call->attrs);
        ret_expr->checked_type_ = expr->checked_type_;

        return ret_expr;
      } else if (call->args[0].as<CallNode>() &&
                 call->args[0].as<CallNode>()->op == CompilerBeginOp()) {
        // Remove compiler_begin if exists.
        Array<Expr> new_args;
        for (auto& arg : call->args) {
          const CallNode* arg_call = arg.as<CallNode>();
          CHECK_EQ(arg_call->op, CompilerBeginOp()) << "ValueError: compiler_begin not found";
          CHECK_EQ(arg_call->args.size(), 1U);
          new_args.push_back(arg_call->args[0]);
        }

        Expr new_expr = {Call(call->op, new_args, call->attrs)};
        new_expr->checked_type_ = call->checked_type_;

        return new_expr;
      } else {
        // This expr is not annotated with compiler_begin, return it directly.
        return expr;
      }
    } else if (expr.as<TupleNode>()) {
      // Remove the annotation for TupleNode.
      const TupleNode* tuple = expr.as<TupleNode>();

      // If the fields of the TupleNode is annotated, then remove
      // the annotation, else return this TupleNode directly.
      if (tuple->fields[0].as<CallNode>()->op == CompilerBeginOp()) {
        Array<Expr> new_fields;
        for (auto field : tuple->fields) {
          auto field_call = field.as<CallNode>();
          CHECK_EQ(field_call->op, CompilerBeginOp()) << "ValueError: compiler_begin not found";
          CHECK_EQ(field_call->args.size(), 1U);
          new_fields.push_back(field_call->args[0]);
        }

        Expr new_tuple = {Tuple(new_fields)};
        new_tuple->checked_type_ = expr->checked_type_;

        return new_tuple;
      } else {
        return expr;
      }
    }
  }
  // Remove the compiler_end annotation inside the CallNode.
  else if (ann_op == end_op) {
    if (expr.as<CallNode>()) {
      const CallNode* call = expr.as<CallNode>();
      if (call->op == CompilerEndOp()) {
        // Remove compiler_begin annotations of the input call's arguments.
        return call->args[0];
      } else {
        // If the compiler_end annotation is already removed, then do nothing.
        return expr;
      }
    } else {
      return expr;
    }
  } else {
    LOG(FATAL) << "ValueError: unknown op";
  }
}

};  // namespace pass
};  // namespace mnm
