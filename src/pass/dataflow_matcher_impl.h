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
 * \file dataflow_matcher_impl.h
 * \brief The auxiliary data structure for dataflow matcher.
 */

#include <relay/ir/dataflow_matcher_impl.h>
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {

using namespace mnm::ir;

class MNMDFPatternMatcher : public tvm::relay::DFPatternMatcher {
 public:
  explicit MNMDFPatternMatcher(const Expr& root_expr) : DFPatternMatcher(root_expr) {
  }

 protected:
  // override this to match meta ops and their dialects, not tvm ops
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  // override this to match meta values in meta constant nodes
  bool VisitDFPattern_(const RelayConstantPatternNode* op, const Expr& expr) override;
  // override these three to use meta infertype
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
};

class MNMPatternGrouper : public tvm::relay::PatternGrouper {
 public:
  const std::unordered_map<int, Group>& GroupMatches(const DFPattern& pattern, const Expr& pre);
};

class MNMPatternRewriter : protected tvm::relay::PatternRewriter {
 public:
  MNMPatternRewriter(IRModule mod) : PatternRewriter(mod) {
  }
  Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) override;
};

}  // namespace pass
}  // namespace mnm
