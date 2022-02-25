/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dataflow_matcher_impl.h
 * \brief The auxiliary data structure for dataflow matcher.
 */

#include <relay/ir/dataflow_matcher_impl.h>
#include "raf/ir.h"
#include "raf/pass.h"

namespace raf {
namespace pass {

using namespace raf::ir;

class RAFDFPatternMatcher : public tvm::relay::DFPatternMatcher {
 public:
  explicit RAFDFPatternMatcher(const Expr& root_expr) : DFPatternMatcher(root_expr) {
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
  Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) override;
};

}  // namespace pass
}  // namespace raf
