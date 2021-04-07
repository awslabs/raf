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
 * be applied.
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
 * 3. inplace write from ty to tx is invalid if and only if there exists a line l, such
 *    that the following two holds simutaneously:
 *    - live(l, x)
 *    - live(l, y)
 *    That is, the inplace write is valid iff the intersection of live(*, x) and live(*, y)
 *    is empty.
 *
 * References:
 * [1] https://www.cs.cmu.edu/~rjsimmon/15411-f15/lec/04-liveness.pdf
 */

using namespace mnm::ir;
using namespace mnm::op;
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

  Function Run() {
    Expr body;
    FormCheck(func_->body);
    if (failure_) {
      body = Mutate(func_->body);
    } else {
      // TODO(@hzfan): support TupleType params
      for (const auto& var : func_->params) {
        CHECK(var->checked_type().as<TensorTypeNode>());
        Var tvar = CreateTensor("param");
        Init(var, tvar);
      }
      // forward analysis
      Forward(func_->body);
      // backward analysis
      Var dummy = CreateNull();
      live_[dummy] = {};
      Backward(func_->body, dummy);
      // init find
      for (const auto& kv : vset_) {
        const Var& var = kv.first;
        const Var& tensor = GetTensorVar(var);
        if (tensor.defined() && tensor == var) {
          union_find_forest_[var] = var;
        }
      }
      // init inv
      for (const auto& kv : live_) {
        const Var& k = kv.first;
        const VSet& vs = kv.second;
        for (const auto& v : vs) {
          inv_live_[v].insert(k);
        }
      }
      // mandatory memory sharing
      CHECK_EQ(var_out_.size(), var_in_.size());
      int m = var_out_.size();
      for (int i = 0; i < m; ++i) {
        Var fout = GetTensorVar(var_out_[i]);
        Var fin = GetTensorVar(var_in_[i]);
        CHECK(fout.defined());
        CHECK(fin.defined());
        fout = Find(fout);
        fin = Find(fin);
        if (fout != fin && Intersect(fout, fin)) {
          // the mandatory inplace update is invalid
          // something goes wrong here
          LOG(WARNING) << "Mandatory memory sharing between " << fin << " and " << fout
                       << " is invalid. Such cases cannot be handled by "
                       << "the memory_share pass.";
          failure_ = true;
        } else {
          // the mandatory inplace update is valid
          Unite(fin, fout);
        }
      }
      // check the validity of VarNode with may_share
      body = Mutate(func_->body);
    }
    return Function(func_->params, body, func_->ret_type, func_->type_params, func_->attrs);
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
    const VSet& vset1 = vset_.at(v1);
    const VSet& vset2 = vset_.at(v2);
    Var rs = CreateTensorVar("rs");
    vset_[rs] = Remove(vset1, vset2);
    return rs;
  }

  /*! \brief Merge vset_[v1] and vset_[v2] */
  Var Merge(Var v1, Var v2) {
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

  /*! \brief Get the dummy tensor variable created by CreateTensor.
             Undefined if no 1:1 correspondence */
  Var GetTensorVar(const Var& x) {
    const VSet& vset = vset_.at(x);
    if (vset.size() != 1) {
      return Var();
    }
    return *vset.begin();
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
    inv_live_[fy].insert(inv_live_.at(fx).begin(), inv_live_.at(fx).end());
    return fy;
  }

  /*! \brief check if inv_live_[x] and inv_live_[y] intersects or not */
  bool Intersect(const Var& x, const Var& y) {
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

 private:
  class ForwardAnalyzer;
  class BackwardAnalyzer;
  class Mutator;
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

  /*!
   * \brief remove invalid memory sharing in e.
   * \param e the expression to be analyzed
   * \param find parent pointers in Union-find Forest (before any inplace rewrite)
   * \param inv lines where a tensor variable is live (before any inplace rewrite)
   * \return the expression with invalid memory sharing removed
   */
  Expr Mutate(const Expr& e);

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
  Map<Var, Array<Var> > vtuple_;
  /*! \brief the live-in variables at a specific line */
  MapVSet live_;
  /*! \brief count the occurences of a var name, to avoid name collision */
  std::unordered_map<std::string, int> label_;
  /*! \brief mandatory memory sharing between a pair of vars */
  Array<Var> var_out_, var_in_;
  /*! \brief vars that share memory with one another are merged in the union find forest */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> union_find_forest_;
  /*! \brief the lines where a variable is live.
             Initially it's the inversion of live_: inv_live_[x] = {y | x \in live_[y]}*/
  MapVSet inv_live_;
};

class LivenessAnalyzer::FormChecker : public ExprVisitor {
 public:
  FormChecker(const Expr& body, LivenessAnalyzer* analyzer) : body_(body), analyzer_(analyzer) {
  }

  void VisitExpr_(const CallNode* node) {
    if (!node->op.as<OpNode>()) {
      // assumes no closure invoke
      analyzer_->failure_ = true;
    } else {
      const Array<Expr>& args = node->args;
      Array<Var> vargs;
      for (const auto& arg : node->args) {
        if (arg.as<VarNode>() == nullptr && arg.as<ConstantNode>() == nullptr) {
          // assumes ANF
          analyzer_->failure_ = true;
        }
      }
    }
  }

  void VisitExpr_(const IfNode* node) override {
    if (node->cond.as<VarNode>() == nullptr) {
      // assumes ANF
      analyzer_->failure_ = true;
    }
  }

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

  void VisitExpr_(const FunctionNode* node) override {
    /*!
     * When a closure is used, the value of the captured variables are required.
     * For example, in
     * fn {
     *   let %closure = {
     *     %b1 = %a1 + %a1
     *     %b1
     *   }
     *   %closure  // here %a1 is used, and thus cannot be inplace rewritten
     * }
     * when the closure is invoked/returned, the value of %a1 (captured variables) is needed.
     */
    Function f = GetRef<Function>(node);
    Array<Var> free_vars = FreeVars(f);
    analyzer_->Init(let_var_, analyzer_->Merge(free_vars));
  }

  void VisitExpr_(const CallNode* node) override {
    if (node->op.as<OpNode>()) {
      Var dummy = analyzer_->CreateTensorVar(node->checked_type());
      analyzer_->Init(let_var_, dummy);
    } else {
      LOG(FATAL) << "NotImplementedError: Calling unsupported type: " << node->op->GetTypeKey();
    }
  }

  void VisitExpr_(const TupleNode* node) override {
    Array<Var> fields;
    for (const auto& field : node->fields) {
      Var var = Downcast<Var>(field);
      fields.push_back(var);
    }
    analyzer_->Init(let_var_, analyzer_->Merge(fields));
    analyzer_->vtuple_.Set(let_var_, fields);
  }

  void VisitExpr_(const TupleGetItemNode* node) override {
    Var var = analyzer_->vtuple_.at(Downcast<Var>(node->tuple))[node->index];
    analyzer_->Init(let_var_, var);
  }

  void VisitExpr_(const IfNode* node) override {
    Expr true_branch = node->true_branch;
    Expr false_branch = node->false_branch;
    Var true_ret = analyzer_->Forward(true_branch);
    Var false_ret = analyzer_->Forward(false_branch);
    Var ret = analyzer_->CreateTensorVar(node->checked_type());
    // mandatory memory sharing if condition is true
    Match(ret, true_ret);
    // mandatory memory sharing if condition is false
    Match(ret, false_ret);
    analyzer_->Init(let_var_, ret);
  }

  void Match(Var v1, Var v2) {
    if (analyzer_->vtuple_.count(v1) > 0) {
      Array<Var> v1t = analyzer_->vtuple_.at(v1);
      Array<Var> v2t = analyzer_->vtuple_.at(v2);
      Array<Var> fields;
      CHECK_EQ(v1t.size(), v2t.size());
      for (size_t i = 0; i < v1t.size(); ++i) {
        Match(v1t[i], v2t[i]);
      }
    } else {
      analyzer_->var_out_.push_back(v1);
      analyzer_->var_in_.push_back(v2);
    }
  }

  Var Run() {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();
    // forward analysis
    for (int i = 0; i < n; ++i) {
      let_var_ = vars[i];
      ExprVisitor::VisitExpr(exprs[i]);
    }
    return ell_->ret;
  }

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

  void VisitExpr_(const FunctionNode* node) override {
    Function f = GetRef<Function>(node);
    Array<Var> free_vars = FreeVars(f);
    analyzer_->live_[let_var_] = analyzer_->vset_[MergeLive(let_var_)];
  }

  void VisitExpr_(const CallNode* node) override {
    if (node->op.as<OpNode>()) {
      const Array<Expr>& args = node->args;
      Array<Var> vargs;
      for (const auto& arg : node->args) {
        if (arg.as<VarNode>()) {
          // use %arg
          vargs.push_back(Downcast<Var>(arg));
        } else if (arg.as<ConstantNode>()) {
          // use nothing
        } else {
          LOG(FATAL) << "NotImplementedError: unsupported args: " << arg->GetTypeKey();
        }
      }
      Var d1 = analyzer_->Merge(vargs);
      Var d2 = MergeLive(d1, let_var_);
      analyzer_->live_[let_var_] = analyzer_->vset_[d2];
    } else {
      LOG(FATAL) << "NotImplementedError: Calling unsupported type: " << node->op->GetTypeKey();
    }
  }

  void VisitExpr_(const TupleNode* node) override {
    analyzer_->live_[let_var_] = analyzer_->vset_[MergeLive(let_var_)];
  }

  void VisitExpr_(const TupleGetItemNode* node) override {
    analyzer_->live_[let_var_] = analyzer_->vset_[MergeLive(let_var_)];
  }

  void VisitExpr_(const IfNode* node) override {
    Var free_true = analyzer_->Merge(FreeVars(node->true_branch));
    Var free_false = analyzer_->Merge(FreeVars(node->false_branch));
    analyzer_->live_[let_var_] = analyzer_->vset_[MergeLive(
        analyzer_->Merge({free_true, free_false, Downcast<Var>(node->cond)}), let_var_)];
    VisitBranch(node->true_branch, let_var_);
    VisitBranch(node->false_branch, let_var_);
  }

  void VisitBranch(const Expr& branch, const Var& def) {
    Var total_next = analyzer_->CreateTensorVar("if");
    // get total live-out variables of true_branch
    analyzer_->vset_[total_next] = analyzer_->live_[next_var_];
    // remove the tensors defined at this line
    Var branch_next = analyzer_->Remove(total_next, def);
    analyzer_->live_[branch_next] = analyzer_->vset_[branch_next];
    analyzer_->Backward(branch, branch_next);
  }

  void Run(Var next_var) {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();
    // backward analysis
    next_var_ = next_var;
    Var dummy = analyzer_->CreateNull();
    analyzer_->live_[dummy] = analyzer_->vset_[MergeLive(ell_->ret)];
    for (int i = n - 1; i >= 0; --i) {
      let_var_ = vars[i];
      next_var_ = i == n - 1 ? dummy : vars[i + 1];
      ExprVisitor::VisitExpr(exprs[i]);
    }
  }

 private:
  /*! \brief returns live_[next_var_] - vset_[def] + vset_[cur]
             it's an instantiation of the following rule:
             live(l + 1, x) && !define(l, x) => live(l, x) */
  Var MergeLive(const Var& cur, const Var& def = Var()) {
    Var next_line_var = analyzer_->CreateTensorVar("ml");
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

class LivenessAnalyzer::Mutator : public ExprMutator {
 public:
  Mutator(const Expr& body, LivenessAnalyzer* analyzer)
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
    if (analyzer_->failure_) {
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
        Var fout = analyzer_->GetTensorVar(vars[i]);
        Var fin = analyzer_->GetTensorVar(var->may_share);
        CHECK(fout.defined());
        CHECK(fin.defined());
        fout = analyzer_->Find(fout);
        fin = analyzer_->Find(fin);
        if (fout != fin && analyzer_->Intersect(fout, fin)) {
          // invalidate the inplace update
          var->may_share = Var();
        } else {
          // the inplace update is valid
          analyzer_->Unite(fin, fout);
        }
      }
      if (const auto* node = exprs[i].as<IfNode>()) {
        // handle if branches recursively
        Expr true_branch = analyzer_->Mutate(node->true_branch);
        Expr false_branch = analyzer_->Mutate(node->false_branch);
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
  /*! \brief a variable that is set for each let expr */
  Var let_var_;
  /*! \brief the analyzer it belongs to */
  LivenessAnalyzer* analyzer_;
};

Var LivenessAnalyzer::Forward(const Expr& e) {
  return ForwardAnalyzer(e, this).Run();
}

void LivenessAnalyzer::Backward(const Expr& e, const Var& next_var) {
  BackwardAnalyzer(e, this).Run(next_var);
}

Expr LivenessAnalyzer::Mutate(const Expr& e) {
  return Mutator(e, this).Run();
}

void LivenessAnalyzer::FormCheck(const Expr& e) {
  FormChecker(e, this).Run();
}

Var LivenessAnalyzer::CreateTensorVar(const Type& type) {
  return VarCreator(this).Run(type);
}

}  // namespace memory_share

Pass MemShare() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return memory_share::LivenessAnalyzer(f).Run();
      };
  return CreateMNMFunctionPass(pass_func, 1, "MemShare", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.MemShare").set_body_typed(MemShare);

}  // namespace pass
}  // namespace mnm
