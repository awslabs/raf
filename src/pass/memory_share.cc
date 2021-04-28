/*!
 * Copyright (c) 2020 by Contributors
 * \file memory_share.cc
 * \brief check the validity of memory sharing
 */
#include <vector>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "tvm/ir/type_functor.h"
#include "./let_list.h"
#include "./common.h"
#include "./liveness_analysis.h"

namespace mnm {
namespace pass {
namespace memory_share {

/*
 * The major challenge is as follows:
 * 1. Two kinds of memory sharing are present in our scenario: 1) mandatory
 *    memory sharing determined by VM / interpreter, like %a = (%x0, %x1, %x2).
 *    where %a shares memory with %x0, %x1, %x2. It exists before the introduction
 *    of effect ir. 2) memory sharing newly introduced by effect ir, like
 *    %b = mnm.add(%a, 1). This pass is to tell whether there is a chance
 *    to make %b and %a in the above example share memory.
 *    Typical liveness analysis does not handle mandatory memory sharing as is
 *    denoted by 1).
 * 2. The memory sharing relation (denoted by ~) is not transitive:
 *    Say %a = (%x0, %x1, %x2), %a ~ %x0, %a ~ %x1, %a ~ %x2. But chances
 *    are that %x0 !~ %x1, %x0 !~ %x2, %x1 !~ %x2
 *
 * Note that for plain liveness analysis [1], neither of 1. and
 * 2. holds. So we transform the IR so that plain liveness analysis can
 * be applied. See LivenessAnalysis for more details.
 *
 * Our algorithm works as follows:
 * 1. run liveness analysis and obtain 1) the set of tensor var contained by each original var,
 *    and 2) the set of live tensor vars at each line.
 * 2. inplace write from ty to tx is invalid if and only if there exists a line l, such
 *    that the following two holds simutaneously:
 *    - live(l, x)
 *    - live(l, y)
 *    That is, the inplace write is valid iff the intersection of live(*, x) and live(*, y)
 *    is empty.
 */

class MemoryShareMutator : public ExprMutator {
 public:
  MemoryShareMutator(const Expr& body, liveness_analysis::LivenessAnalyzer& analyzer)
      : body_(body), ell_(ExplicitLetList::make(body)), analyzer_(analyzer) {
  }

  Expr VisitExpr_(const LetNode* node) override {
    Var var_ = node->var;
    const auto* var = static_cast<const ExtendedVarNode*>(var_.operator->());
    var->may_share = Var();
    Expr value = VisitExpr(node->value);
    Expr body = VisitExpr(node->body);
    return Let(var_, value, body);
  }

  Expr Run() {
    if (!analyzer_.IsSuccess()) {
      return VisitExpr(body_);
    }
    auto& vars = ell_->vars;
    auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();
    // memory sharing introduced by users
    for (int i = 0; i < n; ++i) {
      const auto* var = static_cast<const ExtendedVarNode*>(vars[i].operator->());
      if (var->may_share.defined()) {
        Var fout = *analyzer_.GetTensorVars(vars[i]).begin();
        Var fin = *analyzer_.GetTensorVars(var->may_share).begin();
        CHECK(fout.defined());
        CHECK(fin.defined());
        fout = analyzer_.Find(fout);
        fin = analyzer_.Find(fin);
        if (fout != fin && analyzer_.Intersect(fout, fin)) {
          // invalidate the inplace update
          var->may_share = Var();
        } else {
          // the inplace update is valid
          analyzer_.Unite(fin, fout);
        }
      }
      if (const auto* node = exprs[i].as<IfNode>()) {
        // handle if branches recursively
        Expr true_branch = MemoryShareMutator(node->true_branch, analyzer_).Run();
        Expr false_branch = MemoryShareMutator(node->false_branch, analyzer_).Run();
        exprs[i] = If(node->cond, true_branch, false_branch);
      } else if (const auto* node = exprs[i].as<FunctionNode>()) {
        exprs[i] = VisitExpr(exprs[i]);
      }
    }
    return ell_->AsExpr();
  }

 private:
  /*! \brief the expression to be analyzed */
  const Expr& body_;
  /*! \brief the explicit let list of func_ */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief the analyzer it belongs to */
  liveness_analysis::LivenessAnalyzer& analyzer_;
};

}  // namespace memory_share

Pass MemShare() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto analyzer = liveness_analysis::LivenessAnalyzer(f);
        analyzer.Run();
        auto body = memory_share::MemoryShareMutator(f->body, analyzer).Run();
        return Function(f->params, body, f->ret_type, f->type_params, f->attrs);
      };
  return CreateMNMFunctionPass(pass_func, 1, "MemShare", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.MemShare").set_body_typed(MemShare);

}  // namespace pass
}  // namespace mnm
