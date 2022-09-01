/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dataflow_matcher_impl.h
 * \brief The auxiliary data structure for dataflow matcher.
 */

#include <relay/ir/dataflow_matcher_impl.h>
#include "raf/ir_ext.h"
#include "raf/pass.h"

namespace raf {
namespace pass {

using namespace raf::ir;

class RAFDFPatternMatcher : public tvm::relay::DFPatternMatcher {
 public:
  explicit RAFDFPatternMatcher(const tvm::relay::IndexedGraph<Expr>* expr_graph)
      : DFPatternMatcher(expr_graph) {
  }

 protected:
  // override this to match raf ops and their dialects, not tvm ops
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  // override this to match raf values in raf constant nodes
  bool VisitDFPattern_(const RelayConstantPatternNode* op, const Expr& expr) override;
  // override these three to use raf infertype
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
};

class RAFPatternGrouper : public tvm::relay::PatternGrouper {
 public:
  const std::unordered_map<int, Group>& GroupMatches(const DFPattern& pattern, const Expr& pre);
};

class RAFPatternRewriter : protected tvm::relay::PatternRewriter {
 public:
  RAFPatternRewriter(IRModule mod) : PatternRewriter(mod) {
  }

  // override LetNode visit to non-recursive version to avoid stack-overflow
  Expr VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {
      this->Mutate(op->var);
      this->Mutate(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->Mutate(op->value);
      Expr body = this->Mutate(op->body);

      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        this->memo_[GetRef<Expr>(op)] = GetRef<Expr>(op);
      } else {
        this->memo_[GetRef<Expr>(op)] = Let(var, value, body, op->span);
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) override;
};

}  // namespace pass
}  // namespace raf
