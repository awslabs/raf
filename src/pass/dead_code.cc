/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/dead_code.cc
 * \brief  Remove code that does not effect the program result.
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
 */
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"

namespace raf {
namespace pass {
namespace dead_code_elimination {
// acknowledgement: the code in dead_code_elimination namespace is adopted from tvm
template <typename X>
using VarMap = std::unordered_map<Var, X, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using op::TRAFSideEffect;

class CalcDep;

/*! \brief Get the bindings of each var in let expression. */
class FindDef : public ExprVisitor {
  void VisitExpr_(const LetNode* l) final {
    auto pre_visit = [this](const LetNode* op) {
      ICHECK_EQ(expr_map_.count(op->var), 0);
      expr_map_[op->var] = op->value;
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(l, pre_visit, post_visit);
  }

 public:
  /*! \brief The var binding map */
  VarMap<Expr> expr_map_;
};

/*!
 * \brief Detect whether an expression has side effect. An expression has side effect if and only
 * if it contains a call expression and the call's operator has TRAFSideEffect attribute.
 */
class SideEffectDetector : public ExprVisitor {
 public:
  bool Detect(Expr expr) {
    has_side_effect_ = false;
    VisitExpr(expr);
    return has_side_effect_;
  }

  void VisitExpr(const Expr& expr) override {
    if (has_side_effect_) return;
    ExprVisitor::VisitExpr(expr);
  }

  void VisitExpr_(const LetNode* op) override {
    auto pre_visit = [](const LetNode* op) {};
    auto post_visit = [this](const LetNode* op) {
      VisitExpr(op->value);
      VisitExpr(op->body);
      visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const CallNode* op) override {
    static auto fside_effect = Op::GetAttrMap<TRAFSideEffect>("TRAFSideEffect");
    if (auto op_node = op->op.as<OpNode>()) {
      if (fside_effect.get(GetRef<Op>(op_node), false)) {
        has_side_effect_ = true;
        return;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

 private:
  /*! \brief Whether the given expr has side effect. */
  bool has_side_effect_;
};

/*!
 * \brief Eliminate the let expression when its bound var is not used by its body.
 * If inline_once is true, inline the var that is only used once. Note that this may break the ANF.
 */
class Eliminator : public ExprMutator {
 public:
  explicit Eliminator(const VarMap<Expr>& expr_map, const VarMap<size_t>& use_map, bool inline_once)
      : expr_map_(expr_map), use_map_(use_map), inline_once_(inline_once) {
  }

  /*!
   * Check whether we should keep the let expression that binds var v.
   * We first check whether the value expr has side effect. If so, we can not remove this let expr.
   * Otherwise, there are three cases:
   *  case 1: The var is never used in let's body. In this case, we can remove the let safely.
   *  case 2: The var is only used once. In this case, we can inline let or keep it, which is
   *    determined by inline_once flag.
   *  case 3: The var is used twice or more. In this case, we should keep the let expr.
   * \param v The var of the let expression.
   * \return Whether we should keep the let expression.
   */
  bool HasLet(const Var& v) {
    Expr value = expr_map_[v];
    if (SideEffectDetector().Detect(value)) return true;
    switch (use_map_[v]) {
      case 0:
        return false;
      case 1:
        return !inline_once_;
      default:
        return true;
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    return (expr_map_.count(v) == 0 || HasLet(v)) ? v : VisitExpr(expr_map_[v]);
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      if (HasLet(op->var)) {
        Expr value = this->VisitExpr(op->value);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      Var v = op->var;
      if (HasLet(v)) {
        Expr value = this->VisitExpr(op->value);
        this->memo_[expr] = Let(v, value, body);
      } else {
        this->memo_[expr] = body;
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

 private:
  /*! \brief The binding map */
  VarMap<Expr> expr_map_;
  /*! \brief The number of used times for each bound var */
  VarMap<size_t> use_map_;
  /*! \brief Whether inline the var that is only used once */
  bool inline_once_;
};

/*! * \brief Calculate the use count for given expr */
class CalcDep : public MixedModeVisitor {
 public:
  explicit CalcDep(const VarMap<Expr>& expr_map) : MixedModeVisitor(2), expr_map_(expr_map) {
  }
  using MixedModeVisitor::VisitExpr_;

  void VisitLeaf(const Expr& e) final {
    visit_counter_[e.get()]++;
    // The dce code separate variable into three parts:
    // used 0 times (remove)
    // used 1 times (inline)
    // used 2 times (dont do anything).
    if (visit_counter_[e.get()] <= 2) {
      using TParent = ExprFunctor<void(const Expr&)>;
      TParent::VisitExpr(e);
    }
  }

  void VisitExpr_(const LetNode* l) final {
    Expr let_binding = GetRef<Expr>(l);
    const LetNode* let;
    while ((let = let_binding.as<LetNode>())) {
      let_binding = let->body;
      visit_counter_[l] += 1;
    }
    VisitExpr(let_binding);
  }

  void VisitExpr_(const VarNode* v) final {
    Var var = GetRef<Var>(v);
    ++use_map_[var];
    if (use_map_[var] == 1 && expr_map_.count(var) > 0) {
      VisitExpr(expr_map_[var]);
    }
  }

 public:
  /*! \brief The var binding map */
  VarMap<Expr> expr_map_;
  /*! \brief The use count for each bound var */
  VarMap<size_t> use_map_;
};
Expr Eliminate(const Expr& e, bool inline_once) {
  FindDef fd;
  fd(e);
  CalcDep cd(fd.expr_map_);
  cd(e);
  Eliminator el(fd.expr_map_, cd.use_map_, inline_once);
  return el(e);
}
}  // namespace dead_code_elimination

ir::Expr DeadCodeElimination(const ir::Expr& expr) {
  // Don't inline let because RAF uses ANF
  return raf::pass::dead_code_elimination::Eliminate(expr, false);
}

Pass DeadCodeElimination() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(DeadCodeElimination(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "DeadCodeElimination", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.DeadCodeElimination").set_body_typed([]() {
  return DeadCodeElimination();
});

}  // namespace pass
}  // namespace raf
