/*!
 * Copyright (c) 2019 by Contributors
 * \file gradient.cc
 * \brief Symbolic gradient pass
 */
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace gradient {

#define MNM_NODE_NOT_SUPPORT(NodeType)                                  \
  void VisitExpr_(const NodeType* node) final {                         \
    LOG(FATAL) << "ValueError: feature is not supported:" << #NodeType; \
    throw;                                                              \
  }

#define MNM_NODE_NOT_IMPL(NodeType)                     \
  void VisitExpr_(const NodeType* node) final {         \
    LOG(FATAL) << "NotImplementedError: " << #NodeType; \
    throw;                                              \
  }

using namespace mnm::ir;
using namespace mnm::op;
using mnm::value::NoGradValue;

class ANFNormalizer : public ExprMutator {
 public:
  explicit ANFNormalizer(LetList* ll) : ll_(ll) {
  }

  Expr Normalize(const Expr& expr) {
    if (expr.as<VarNode>() || expr.as<ConstantNode>() || expr.as<OpNode>()) {
      return expr;
    }
    if (vmap_.find(expr) != vmap_.end()) {
      return vmap_[expr];
    }
    return vmap_[expr] = ll_->Push(expr);
  }

  Expr VisitExpr(const Expr& expr) final {
    Expr ret = ExprMutator::VisitExpr(expr);
    return Normalize(ret);
  }

  Expr VisitExpr_(const TupleNode* node) final {
    tvm::Array<Expr> fields;
    bool all_fields_unchanged = true;
    for (auto field : node->fields) {
      if (field.defined()) {
        auto new_field = this->Mutate(field);
        fields.push_back(new_field);
        all_fields_unchanged &= new_field.same_as(field);
      }
    }
    if (all_fields_unchanged) {
      return GetRef<Expr>(node);
    } else {
      return Tuple(fields, node->span);
    }
  }

 private:
  LetList* ll_;
  std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> vmap_;
};

struct Gradient : public ExprVisitor {
 public:
  // Closures are not supported
  MNM_NODE_NOT_SUPPORT(FunctionNode);
  // The algorithm shouldn't generate or deal with references
  MNM_NODE_NOT_SUPPORT(RefCreateNode);
  MNM_NODE_NOT_SUPPORT(RefReadNode);
  MNM_NODE_NOT_SUPPORT(RefWriteNode);
  // MNM has not started to take care of ADTs yet
  MNM_NODE_NOT_SUPPORT(tvm::relay::ConstructorNode);
  MNM_NODE_NOT_SUPPORT(tvm::relay::MatchNode);
  // TODO(@junrushao1994): implement them
  // TODO(@junrushao1994): nested tuples are still problematic
  MNM_NODE_NOT_IMPL(OpNode);         // replace OpNode with its corresponding GlobalVar's adjoint
  MNM_NODE_NOT_IMPL(IfNode);         // normalize the program with tail call
  MNM_NODE_NOT_IMPL(VarNode);        // propagate copy/constant
  MNM_NODE_NOT_IMPL(GlobalVarNode);  // replace GlobalVar with adjoint

 public:
  explicit Gradient(const FunctionNode* func_, const ir::Array<tvm::Bool>& requires_grads_)
      : func(func_), ell(ExplicitLetList::make(func_->body)) {
    // If size of requires_grads_ is 0, check the inputs' datatype.
    if (requires_grads_.size() == 0) {
      for (const Var& param : func->params) {
        const VarNode* var = param.operator->();
        const auto tensor_type = Downcast<TensorType>(var->type_annotation);
        requires_grads[var] = tensor_type->dtype.is_float();
      }
    } else {
      CHECK_EQ(func->params.size(), requires_grads_.size());
      for (int i = 0; i < func->params.size(); ++i) {
        const VarNode* var = func->params[i].operator->();
        requires_grads[var] = requires_grads_[i];
      }
    }
    InitTuple();
  }

 public:
  void VisitExpr_(const RelayConstantNode* node) final {
    // Do nothing
  }

  void VisitExpr_(const TupleNode* node) final {
    // expr:
    //    let var = tuple(*node->fields);
    // situation:
    //    grad(var) is known
    // return:
    //    update `grad(node->fields[i])` with `+= grad(var)[i]`
    const Array<Expr>& ograds = GetOutputGrads();
    WriteBackInputGrads(node->fields, ograds);
  }

  void VisitExpr_(const TupleGetItemNode* node) final {
    // expr:
    //    let var = tuple[index]
    // situation:
    //    grad(var) are known
    // return:
    //    update `grad(tuple[index])` with `+= grad(var)`
    const VarNode* tuple = node->tuple.as<VarNode>();
    const Array<Expr>& ograds = GetOutputGrads();
    Array<Expr> tuple_igrads = tuple_grads[tuple];
    CHECK_EQ(ograds.size(), 1);
    CHECK_GT(tuple_igrads.size(), node->index);
    tuple_igrads.Set(node->index, AddTensor(ograds[0], tuple_igrads[node->index]));
    tuple_grads[tuple] = tuple_igrads;
  }

  void VisitExpr_(const CallNode* node) final {
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      const Op& op = Downcast<Op>(node->op);
      const Array<Expr>& ograds = GetOutputGrads();
      const Array<Expr>& igrads = GetInputGrads(node->args);
      const Array<Expr>& new_igrads = UpdateInputGrads(op, GetRef<Expr>(node), ograds, igrads);
      WriteBackInputGrads(node->args, new_igrads);
    } else {
      LOG(FATAL) << "NotImplementedError: Calling unsupported type: " << callee->GetTypeKey();
      throw;
    }
  }

 public:
  // Helper functions for the workflow:
  //    get igrads => update igrads => write back igrads
  Array<Expr> GetOutputGrads() {
    const VarNode* var = let_var.operator->();
    return tuple_grads[var];
  }

  Array<Expr> GetInputGrads(const Array<Expr>& vars) {
    Array<Expr> ret;
    for (const auto& expr : vars) {
      if (const auto* var = expr.as<VarNode>()) {
        ret.push_back(tuple_grads[var][0]);
      } else if (expr->IsInstance<RelayConstantNode>()) {
        ret.push_back(NullValue<Var>());
      } else {
        LOG(FATAL) << "Unsupported to get grads of type: " << expr->GetTypeKey();
        throw;
      }
    }
    return ret;
  }
  bool IsDefined(const Array<Expr>& exprs) {
    for (const auto& e : exprs) {
      if (!e.defined()) {
        return false;
      }
    }
    return true;
  }

  Array<Expr> UpdateInputGrads(const Op& op,      // the operator called
                               const Expr& orig,  // relay.Call that contains this expression
                               const Array<Expr>& ograds,  // grad(output)
                               const Array<Expr>& igrads) {
    // given: igrads
    // returns: new_igrads = igrads + grad-of-call-op<ograd>
    static const auto fpg = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");
    static const auto ffpg = Op::GetAttrMap<FFusedPrimalGradient>("FFusedPrimalGradient");
    const VarNode* var = let_var.operator->();
    const Expr& _ograds = tuple_length.count(var) ? Tuple(ograds) : ograds[0];
    if (!IsDefined(ograds)) {
      return {NullValue<Var>()};
    }
    Array<Expr> ret;
    auto call = Downcast<Call>(orig);
    if (ffpg.count(op)) {
      ret = ffpg[op](orig, let_var, _ograds, igrads);
    } else if (fpg.count(op)) {
      Array<Expr> orig_args;
      for (auto arg : call->args) {
        if (auto in_var = arg.as<VarNode>()) {
          orig_args.push_back(var_to_expr[in_var]);
        } else {
          CHECK(arg.as<ConstantNode>() != nullptr);
          orig_args.push_back(arg);
        }
      }
      ret = fpg[op](orig, orig_args, let_var, _ograds);
      // if not 'requires_grad', set return to null
      for (int i = 0, n = ret.size(); i < n; ++i) {
        if (auto in_var = call->args[i].as<VarNode>()) {
          if (requires_grads[in_var] == false) {
            ret.Set(i, NullValue<Var>());
          }
        }
      }
    } else {
      LOG(FATAL) << "Gradient is not registered for operator " << op->name;
      throw;
    }
    // ensure intermediate results are bound to a relay::var
    ANFNormalizer normalizer(ll);
    for (int i = 0, n = ret.size(); i < n; ++i) {
      // If a tuple contains null, we use tule_index to
      // store the non-null item index
      if (ret[i].defined() && !ret[i]->IsInstance<VarNode>()) {
        if (ret[i]->IsInstance<TupleNode>()) {
          auto tuple = ret[i].as<TupleNode>();
          bool has_null = false;
          std::vector<int> index;
          for (int i = 0; i < tuple->fields.size(); ++i) {
            if (tuple->fields[i].defined()) {
              index.push_back(i);
            } else {
              has_null = true;
            }
          }
          if (has_null) {
            auto args = call->args;
            tuple_index[args[i].as<VarNode>()] = index;
          }
        }
        ret.Set(i, normalizer(ret[i]));
      }
    }
    return ret;
  }

  void WriteBackInputGrads(const Array<Expr>& vars, const Array<Expr>& igrads) {
    CHECK_GE(vars.size(), igrads.size());
    int n = igrads.size();
    for (int i = 0; i < n; ++i) {
      const Expr& igrad = igrads[i];
      if (!igrad.defined()) {
        continue;
      }
      if (const auto* var = vars[i].as<VarNode>()) {
        if (tuple_length.count(var) == 0) {
          WriteBackInputGrad(var, 0, igrad);
        } else {
          CHECK_NE(tuple_length[var], -1);
          // If tuple_index contains var, the corresponding
          // gradient tuple has null value so we only take the
          // non-null item index and perform WriteBackInputGrad
          if (tuple_index.count(var)) {
            for (int i : tuple_index[var]) {
              WriteBackInputGrad(var, i, ll->Push(TupleGetItem(igrad, i)));
            }
          } else {
            for (int i = 0; i < tuple_length[var]; ++i) {
              WriteBackInputGrad(var, i, ll->Push(TupleGetItem(igrad, i)));
            }
          }
        }
      } else if (!vars[i]->IsInstance<RelayConstantNode>()) {
        LOG(FATAL) << "Assume ANF";
        throw;
      }
    }
  }

  void WriteBackInputGrad(const VarNode* var, int idx, const Expr& igrad) {
    Expr grad = AddTensor(tuple_grads[var][idx], igrad);
    tuple_grads[var].Set(idx, grad);
  }

 public:
  // helper functions for adding tensors
  Array<Expr> AddTensors(const Array<Expr>& x1s, const Array<Expr>& x2s) {
    int n1 = x1s.size();
    int n2 = x2s.size();
    int n = std::max(n1, n2);
    std::vector<Expr> ret;
    for (int i = 0; i < n; ++i) {
      const Expr& x1 = i < n1 ? x1s[i] : NullValue<Expr>();
      const Expr& x2 = i < n2 ? x2s[i] : NullValue<Expr>();
      ret.push_back(AddTensor(x1, x2));
    }
    return ret;
  }

  Expr AddTensor(const Expr& x1, const Expr& x2) {
    static Op op = Op::Get("mnm.op.add");
    if (!x1.defined() && !x2.defined()) {
      return NullValue<Var>();
    }
    if (!x1.defined()) {
      return x2->IsInstance<VarNode>() ? x2 : ll->Push(x2);
    }
    if (!x2.defined()) {
      return x1->IsInstance<VarNode>() ? x1 : ll->Push(x1);
    }
    const auto* t1 = x1.as<TupleNode>();
    const auto* t2 = x2.as<TupleNode>();
    if (t1 && t2) {
      return Tuple(AddTensors(t1->fields, t2->fields));
    }
    return ll->Push(Call(op, {x1, x2}));
  }

  // Initialize, running and finalize
  void InitTuple() {
    // grads for tuples
    const auto& vars = ell->vars;
    const auto& exprs = ell->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    for (const auto& var : vars) {
      requires_grads[var.operator->()] = true;
    }
    int n = exprs.size();
    for (int i = 0; i < n; ++i) {
      var_to_expr[vars[i].operator->()] = exprs[i];
    }
    for (int i = 0; i < n; ++i) {
      // a must-be tuple
      if (const auto* tuple = exprs[i].as<TupleNode>()) {
        const VarNode* var = vars[i].operator->();
        tuple_length[var] = tuple->fields.size();
        tuple_grads[var] = std::vector<Expr>(tuple->fields.size(), NullValue<Var>());
      } else if (const auto* tuple_get_item = exprs[i].as<TupleGetItemNode>()) {
        // This is a tuple, which however is not constructed inside this function
        // (i.e. input arguments or outputs of an operator/function)
        // Therefore, its length is unknown
        const VarNode* var = tuple_get_item->tuple.as<VarNode>();
        int size = tuple_get_item->index + 1;
        if (tuple_length.count(var) == 0) {
          tuple_length[var] = -1;
          tuple_grads[var] = std::vector<Expr>(size, NullValue<Var>());
        } else {
          // FIXME(comaniac): the ignored nn.dropout becomes a tuple-2 which has
          // tuple_length = 2 here.
          // CHECK_EQ(tuple_length[var], -1);
          int old_size = tuple_grads[var].size();
          if (size > old_size) {
            tuple_grads[var] = std::vector<Expr>(size, NullValue<Var>());
          }
        }
      }
    }
    // grads for non-tuples
    for (int i = 0; i < n; ++i) {
      const VarNode* var = vars[i].operator->();
      if (!tuple_grads.count(var)) {
        tuple_grads[var] = {NullValue<Var>()};
      }
    }
    // grad for input arguments
    for (const Var& param : func->params) {
      const VarNode* var = param.operator->();
      if (!tuple_grads.count(var)) {
        tuple_grads[var] = {NullValue<Var>()};
      }
    }
  }

  // The closure looks like:
  //   fn (dy) {
  //      let xxx = ...;
  //      ret_grads
  //   }
  void MakeClosureInputGrads(const Var& dy) {
    const VarNode* ret = ell->ret.operator->();
    if (tuple_length.count(ret)) {
      int n = tuple_grads[ret].size();
      for (int i = 0; i < n; ++i) {
        tuple_grads[ret].Set(i, ll->Push(TupleGetItem(dy, i)));
      }
    } else {
      CHECK_EQ(tuple_grads[ret].size(), 1);
      tuple_grads[ret] = {dy};
    }
  }

  Var MakeClosureRet() {
    const Array<Var>& targets = func->params;
    Array<Expr> grads;
    for (const Var& var : targets) {
      const VarNode* var_node = var.operator->();
      std::vector<Expr> var_grads(tuple_grads[var_node].begin(), tuple_grads[var_node].end());
      if (tuple_length.count(var_node)) {
        for (Expr& expr : var_grads) {
          if (!expr.defined()) {
            expr = MakeConstant(NoGradValue::make());
          }
        }
        grads.push_back(ll->Push(Tuple(var_grads)));
      } else {
        CHECK_EQ(var_grads.size(), 1);
        if (var_grads[0].defined()) {
          grads.push_back(var_grads[0]);
        } else {
          grads.push_back(MakeConstant(NoGradValue::make()));
        }
      }
    }
    if (targets.size() == 1) {
      return Downcast<Var>(grads[0]);
    }
    return ll->Push(Tuple(grads));
  }

  Var MakeOutputGrad() {
    Type ty = func->checked_type_;
    Type annotation;
    if (ty.defined()) {
      const auto* fty = ty.as<FuncTypeNode>();
      CHECK(fty != nullptr);
      annotation = fty->ret_type;
    }
    return mnm::ir::MakeVar("dy", annotation);
  }

  Function Run() {
    Var dy = MakeOutputGrad();
    Expr body = LetList::With([&](LetList* ll) {
      this->ll = ll;
      const auto& vars = ell->vars;
      const auto& exprs = ell->exprs;
      CHECK_EQ(vars.size(), exprs.size());
      int n = exprs.size();
      MakeClosureInputGrads(dy);
      for (int i = n - 1; i >= 0; --i) {
        let_var = vars[i];
        ExprVisitor::VisitExpr(exprs[i]);
      }
      return MakeClosureRet();
    });
    Var closure = mnm::ir::MakeVar("closure", {});
    Var ret = mnm::ir::MakeVar("ret", {});
    // let closure = fn(dy) {};
    ell->vars.push_back(closure);
    ell->exprs.push_back(Function({dy}, body, {}, {}));
    // let ret = tuple(y, closure)
    ell->vars.push_back(ret);
    ell->exprs.push_back(Tuple({ell->ret, closure}));
    ell->ret = ret;
    return Function(func->params, ell->AsExpr(), {}, {});
  }

 public:
  // initialized in constructor
  const FunctionNode* func;
  std::unique_ptr<ExplicitLetList> ell{nullptr};
  std::unordered_map<const VarNode*, int> tuple_length;
  std::unordered_map<const VarNode*, std::vector<int>> tuple_index;
  std::unordered_map<const VarNode*, Array<Expr>> tuple_grads;
  std::unordered_map<const VarNode*, Expr> var_to_expr;
  std::unordered_map<const VarNode*, bool> requires_grads;
  // initialized in Run
  LetList* ll = nullptr;
  // a variable that is set for each let expr
  Var let_var;
};

}  // namespace gradient

Pass AutoDiff(ir::Array<tvm::Bool> requires_grads) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto reverse_ad = gradient::Gradient(Downcast<Function>(f).get(), requires_grads);
        return Downcast<ir::Function>(reverse_ad.Run());
      };
  return CreateMNMFunctionPass(pass_func, 0, "AutoDiff", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.AutoDiff").set_body_typed(AutoDiff);

}  // namespace pass
}  // namespace mnm
