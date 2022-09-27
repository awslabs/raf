/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file liveness_analysis.h
 * \brief A pass for analyzing tensor liveness.
 */
#pragma once
#include <vector>
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "./let_list.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace liveness_analysis {

/*
 * Note that plain liveness analysis [1] is not applicable to non-effect IR nor transitive,
 * so we transform the IR in advance:
 *
 * We analyze against (dummy) tensor vars, instead of the original vars
 * in our function. Each tensor var (%t[0...3] in the following example)
 * is the smallest unit for memory allocation. We first obtain the set
 * of tensor var contained by each original var:
 *
 * let %a1 = batch_norm(%x, %mean, %var, %w, %b)    | %a1 = {%t0, %t1, %t2}
 * let %a2 = %a1.0                                  | %a2 = {%t0,}
 * let %a3 = %a1.1                                  | %a3 = {%t1,}
 * let %a4 = %a1.2                                  | %a4 = {%t2,}
 * let %a5 = add(%a3, %a4)                          | %a5 = {%t3,}
 * let %a6 = (%a2, %5)                              | %a6 = {%t0, %t3}
 * %a6                                              |
 *
 * The memory sharing relations over tensor vars are transitive:
 * %tx ~ %ty, %ty ~ %tz => %tx ~ %tz
 *
 * Our algorithm works as follows:
 * 1. obtain the set of tensor var contained by each original var, in ForwardAnalyzer
 * 2. obtain the set of live tensor vars at each line, in BackwardAnalyzer.
 *    Following liveness analysis for registers described in [1], live(l, t) denotes
 *    tensor var t has been defined at line l, and its value will be used at or after
 *    line l. We have rules:
 *    - use(l, x) => live(l, x)
 *    - live(l + 1, x) && !define(l, x) => live(l, x)
 *    where use(l, x) denotes that the computation of line l uses the value of x,
 *    and define(l, x) denotes that line l defines the value of x. x is a tensor var.
 *
 * References:
 * [1] https://www.cs.cmu.edu/~rjsimmon/15411-f15/lec/04-liveness.pdf
 */

using namespace raf::ir;
using namespace raf::op;
using tvm::TypeFunctor;
using VSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using MapVar = StdMap<Var>;
using MapVSet = StdMap<VSet>;
using MapFunction = StdMap<Function>;

class LivenessAnalyzer {
 public:
  LivenessAnalyzer(const Function& func) : func_(func) {
  }

  MapVSet Run();

  bool IsSuccess() {
    return !failure_;
  }

  /*! \brief Get live in tensors of the given line (var). */
  VSet GetLiveVars(const Var& x) {
    if (live_.count(x) == 0) {
      return VSet();
    }
    return live_.at(x);
  }

  /*! \brief Get the dummy tensor variables of the final outputs. */
  VSet GetOutputTensorVars() {
    return GetLiveVars(dummy_output_);
  }

  /*! \brief Get the dummy tensor variables created by CreateTensor. */
  Array<Var> GetTensorVars(const Var& x) {
    if (vtuple_.count(x) > 0) {
      // Return the dummy tensors in order when x is a tuple
      return vtuple_[x];
    }

    Array<Var> ret;
    if (vset_.count(x) == 0) {
      return ret;
    }
    CHECK(vset_.find(x) != vset_.end());
    for (auto var : vset_.at(x)) {
      ret.push_back(var);
    }
    return ret;
  }

  /*! \brief Check if the variable is alive given the live dummy var set. */
  bool IsAlive(const Var& var, const VSet& live_vars) {
    if (live_vars.count(var)) {
      return true;
    }
    // deal with the live dummy vars.
    for (Var v : GetTensorVars(var)) {
      if (v == var) {
        continue;
      }
      if (live_vars.count(v) || IsAlive(v, live_vars)) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Union-find Forest: Get root in Union-find Forest */
  Var Find(const Var& x) {
    CHECK(union_find_forest_.find(x) != union_find_forest_.end());
    if (x == union_find_forest_.at(x)) {
      return x;
    }
    Var root = Find(union_find_forest_.at(x));
    union_find_forest_[x] = root;
    return root;
  }

  /*! \brief Union-find Forest: Unite two trees in Union-find Forest */
  Var Unite(const Var& x, const Var& y) {
    Var fx = Find(x);
    Var fy = Find(y);
    union_find_forest_[fx] = fy;
    CHECK_GT(inv_live_.count(fx), 0);
    CHECK_GT(inv_live_.count(fy), 0);
    inv_live_[fy].insert(inv_live_.at(fx).begin(), inv_live_.at(fx).end());
    return fy;
  }

  /*! \brief check if inv_live_[x] and inv_live_[y] intersects or not */
  bool Intersect(const Var& x, const Var& y) {
    CHECK_GT(inv_live_.count(x), 0);
    CHECK_GT(inv_live_.count(y), 0);
    const VSet& sx = inv_live_.at(x);
    const VSet& sy = inv_live_.at(y);
    for (const auto& v : sx) {
      if (sy.find(v) != sy.end()) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Debug output: vset[x] */
  std::string DebugDump_(MapVSet vset, Var x) {
    std::ostringstream os;
    os << x << ": ";
    if (vset.find(x) != vset.end()) {
      const VSet& vs = vset.at(x);
      for (const auto& v : vs) {
        os << v << ", ";
      }
      os << "\n";
    } else {
      os << "not exsit"
         << "\n";
    }
    return os.str();
  }

  /*! \brief Debug output: vset_ */
  std::string DebugDump(MapVSet vset, Var x = Var()) {
    std::ostringstream os;
    if (x.defined()) {
      return DebugDump_(vset, x);
    }
    for (const auto& kv : vset) {
      Var v = kv.first;
      os << DebugDump_(vset, v);
    }
    return os.str();
  }

  /*! \brief Debug output: live_ */
  std::string DebugDumpLiveIn() {
    return DebugDump(live_);
  }

  /*! \brief Debug output: vset_ */
  std::string DebugDumpDummyVars() {
    return DebugDump(vset_);
  }

 private:
  /*! \brief Create a dummy variable. */
  Var CreateTensorVar(const std::string& name = "t") {
    if (label_.find(name) == label_.end()) {
      label_[name] = 0;
    }
    int label = label_[name]++;
    std::string fullname = name + "_" + std::to_string(label);
    return MakeVar(fullname, {});
  }

  /*! \brief Create a dummy variable, which contains nothing. */
  Var CreateNull(const std::string& name = "n") {
    Var var = CreateTensorVar(name);
    vset_[var] = {};
    return var;
  }

  /*! \brief Create a dummy tensor variable, which contains itself. */
  Var CreateTensor(const std::string& name = "t") {
    Var var = CreateTensorVar(name);
    vset_[var] = {var};
    return var;
  }

  /*! \brief vset1 - vset2 */
  static VSet Remove(const VSet& vset1, const VSet& vset2) {
    VSet ret(vset1);
    for (const auto& var : vset2) {
      ret.erase(var);
    }
    return ret;
  }

  /*! \brief the union of vset1 and vset2 */
  static VSet Merge(const VSet& vset1, const VSet& vset2) {
    VSet ret(vset1);
    ret.insert(vset2.begin(), vset2.end());
    return ret;
  }

  /*! \brief Remove vset_[v2] from vset_[v1] */
  Var Remove(Var v1, Var v2) {
    bool v1_legel = (v1.defined() && vset_.find(v1) != vset_.end());
    bool v2_legel = (v2.defined() && vset_.find(v2) != vset_.end());
    if (v1_legel && !v2_legel) {
      // v1 - v2 = v1 - null = v1
      return v1;
    } else if (!v1_legel) {
      // null - v2 = null
      return Var();
    }
    const VSet& vset1 = vset_.at(v1);
    const VSet& vset2 = vset_.at(v2);
    Var rs = CreateTensorVar("rs");
    vset_[rs] = Remove(vset1, vset2);
    return rs;
  }

  /*! \brief Merge vset_[v1] and vset_[v2] */
  Var Merge(Var v1, Var v2) {
    bool v1_legel = (v1.defined() && vset_.find(v1) != vset_.end());
    bool v2_legel = (v2.defined() && vset_.find(v2) != vset_.end());
    if (!v1_legel && !v2_legel) {
      return Var();
    } else if (!v1_legel || !v2_legel) {
      return (v1_legel) ? v1 : v2;
    }
    const VSet& vset1 = vset_.at(v1);
    const VSet& vset2 = vset_.at(v2);
    Var ms = CreateTensorVar("ms");
    vset_[ms] = Merge(vset1, vset2);
    return ms;
  }

  /*! \brief Merge vset_[vars[i]] */
  Var Merge(Array<Var> vars) {
    size_t n = vars.size();
    if (n == 0) {
      return CreateNull();
    } else if (n == 1) {
      CHECK(vset_.find(vars[0]) != vset_.end());
      return vars[0];
    } else {
      Var ret = Merge(vars[0], vars[1]);
      for (size_t i = 2; i < n; ++i) {
        ret = Merge(ret, vars[i]);
      }
      return ret;
    }
  }

  /*! \brief Init vtuple_[to] and vset_[to] with from */
  void Init(Var to, Var from) {
    if (vtuple_.count(from) > 0) {
      CHECK_EQ(vtuple_.count(to), 0);
      vtuple_.Set(to, vtuple_.at(from));
    }
    CHECK(vset_.find(to) == vset_.end());
    vset_[to] = vset_[from];
  }

  /*! \brief Get free variables */
  static Array<Var> FreeVars(Expr e) {
    if (e.as<LetNode>()) {
      Function f({}, e, {}, {});
      Array<Var> free_vars = ::tvm::relay::FreeVars(f);
      return free_vars;
    } else if (e.as<VarNode>()) {
      return {Downcast<Var>(e)};
    } else if (e.as<FunctionNode>()) {
      return ::tvm::relay::FreeVars(Downcast<Function>(e));
    } else {
      LOG(FATAL) << "NotImplementedError: FreeVars for: " << e->GetTypeKey();
    }
  }

 private:
  class ForwardAnalyzer;
  class BackwardAnalyzer;
  class FormChecker;
  class VarCreator;

  /*!
   * \brief invoke ForwardAnalyzer for func:
   *        populate vset_ for all variables in e
   *        populate vtuple_ for all variables of TupleType in e
   * \param e the expression to be analyzed
   * \return the value of e
   * \note vset_ and vtuple_ free variables in e should be available already
   */
  Var Forward(const Expr& e);

  /*!
   * \brief invoke BackwardAnalyzer for func:
   *        populate live_ for each line in e
   * \param e the expression to be analyzed
   * \param next_var live_[next_var] is the live-out variables of e
   * \note vset_ should be available already
   */
  void Backward(const Expr& e, const Var& next_var);

  /*! \brief Check if e contains closure invoke */
  void FormCheck(const Expr& e);

  /*! \brief Create a variable of specified type */
  Var CreateTensorVar(const Type& type);

 private:
  /*! \brief the function to be analyzed */
  const Function& func_;
  /*! \brief whether func_ contains closure invoke */
  bool failure_{false};
  /*! \brief maps a var to the set of real or fake variables which share memory with the key */
  MapVSet vset_;
  /*! \brief maps a variable with TupleType to its constituent (fake) variables */
  Map<Var, Array<Var>> vtuple_;
  /*! \brief the live-in variables at a specific line */
  MapVSet live_;
  /*! \brief The dummy value of the final output */
  Var dummy_output_;
  /*! \brief count the occurences of a var name, to avoid name collision */
  std::unordered_map<std::string, int> label_;
  /*! \brief mandatory memory sharing between a pair of vars */
  Array<Var> var_out_, var_in_;
  /*! \brief vars that share memory with one another are merged in the union find forest */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> union_find_forest_;
  /*! \brief the lines where a variable is live.
             Initially it's the inversion of live_: inv_live_[x] = {y | x \in live_[y]} */
  MapVSet inv_live_;
};

class LivenessAnalyzer::FormChecker : public ExprVisitor {
 public:
  FormChecker(const Expr& body, LivenessAnalyzer* analyzer) : body_(body), analyzer_(analyzer) {
  }

  void VisitExpr_(const CallNode* node);
  void VisitExpr_(const IfNode* node) override;
  void VisitExpr_(const FunctionNode* node) override;
  void VisitExpr_(const LetNode* node) override;
  void Run() {
    VisitExpr(body_);
  }

 private:
  /*! \brief the expression to be analyzed */
  const Expr& body_;
  /*! \brief the analyzer it belongs to */
  LivenessAnalyzer* analyzer_;
};

class LivenessAnalyzer::VarCreator : public TypeFunctor<Var(const Type& n)> {
 public:
  VarCreator(LivenessAnalyzer* analyzer) : analyzer_(analyzer) {
  }

  Var VisitType_(const TupleTypeNode* op) override {
    Array<Var> fields;
    for (const auto& field : op->fields) {
      Var var = VisitType(field);
      fields.push_back(var);
    }
    Var tvar = analyzer_->Merge(fields);
    analyzer_->vtuple_.Set(tvar, fields);
    return tvar;
  }

  Var VisitType_(const TensorTypeNode* op) override {
    return analyzer_->CreateTensor();
  }

  Var Run(const Type& type) {
    return VisitType(type);
  }

 private:
  /*! \brief the analyzer it belongs to */
  LivenessAnalyzer* analyzer_;
};

class LivenessAnalyzer::ForwardAnalyzer : public ExprVisitor {
 public:
  ForwardAnalyzer(const Expr& body, LivenessAnalyzer* analyzer)
      : body_(body), ell_(ExplicitLetList::make(body)), analyzer_(analyzer) {
  }

  void VisitExpr_(const VarNode* node) override;
  void VisitExpr_(const FunctionNode* node) override;
  void VisitExpr_(const CallNode* node) override;
  void VisitExpr_(const TupleNode* node) override;
  void VisitExpr_(const TupleGetItemNode* node) override;
  void VisitExpr_(const IfNode* node) override;
  void Match(Var v1, Var v2);
  Var Run();

 private:
  /*! \brief the expression to be analyzed */
  const Expr& body_;
  /*! \brief the explicit let list of func_ */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief a variable that is set for each let expr */
  Var let_var_;
  /*! \brief the analyzer it belongs to */
  LivenessAnalyzer* analyzer_;
};

class LivenessAnalyzer::BackwardAnalyzer : public ExprVisitor {
 public:
  BackwardAnalyzer(const Expr& body, LivenessAnalyzer* analyzer)
      : body_(body), ell_(ExplicitLetList::make(body)), analyzer_(analyzer) {
  }

  void VisitExpr_(const VarNode* node) override;
  void VisitExpr_(const FunctionNode* node) override;
  void VisitExpr_(const CallNode* node) override;
  void VisitExpr_(const TupleNode* node) override;
  void VisitExpr_(const TupleGetItemNode* node) override;
  void VisitExpr_(const IfNode* node) override;
  void VisitBranch(const Expr& branch, const Var& def);
  void Run(Var next_var);

 private:
  /*! \brief returns live_[next_var_] - vset_[def] + vset_[cur]
             it's an instantiation of the following rule:
             live(l + 1, x) && !define(l, x) => live(l, x) */
  Var MergeLive(const Var& cur, const Var& def = Var()) {
    Var next_line_var = analyzer_->CreateTensorVar("ml");
    CHECK(analyzer_->live_.find(next_var_) != analyzer_->live_.end());
    analyzer_->vset_[next_line_var] = analyzer_->live_.at(next_var_);
    Var remain = next_line_var;
    if (def.defined()) {
      remain = analyzer_->Remove(remain, def);
    }
    Var ret = analyzer_->Merge(remain, cur);
    return ret;
  }

 private:
  /*! \brief the expression to be analyzed */
  const Expr& body_;
  /*! \brief the explicit let list of func_ */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief a variable that is set for each let expr */
  Var let_var_;
  /*! \brief the variable next to let_var_ */
  Var next_var_;
  /*! \brief the analyzer it belongs to */
  LivenessAnalyzer* analyzer_;
};

/*! \brief Calculate the byte compact size of the given type. If the type is a tuple,
 * then the size of each tensor in the tuple will be returned. Note that size 0 means
 * a tensor with dynamic shape.
 */
std::vector<int64_t> CalcBytesCompactSizes(const Type& type);

/*! \brief Dump liveness analysis result statistics. */
void DumpLivenessStat(const MapVSet& live_in);

}  // namespace liveness_analysis
}  // namespace pass
}  // namespace raf
