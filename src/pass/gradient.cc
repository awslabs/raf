/*!
 * Copyright (c) 2019 by Contributors
 * \file gradient.cc
 * \brief Symbolic gradient pass
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace gradient {

using namespace mnm::ir;
using namespace mnm::op;
using tvm::relay::LetList;

Expr TensorAdd(const Expr& x1, const Expr& x2) {
  static Op op = Op::Get("mnm.op.add");
  return CallNode::make(op, {x1, x2});
}

Array<Expr> AccGrad(const Array<Expr>& grads, const Array<Expr>& delta) {
  int n1 = grads.size();
  int n2 = delta.size();
  int n = std::max(n1, n2);
  std::vector<Expr> igrads;
  for (int i = 0; i < n; ++i) {
    const Expr& x1 = i < n1 ? grads[i] : NullValue<Expr>();
    const Expr& x2 = i < n2 ? delta[i] : NullValue<Expr>();
    if (!x1.defined()) {
      igrads.push_back(x2);
    } else if (!x2.defined()) {
      igrads.push_back(x1);
    } else {
      igrads.push_back(TensorAdd(x1, x2));
    }
  }
  return igrads;
}

Array<Expr> UpdatePrimalGrad(const Op& op, const Expr& orig, const Expr& ograd,
                             const Array<Expr>& old_igrads) {
  static const auto f_primal_grad = Op::GetAttr<FPrimalGradient>("FPrimalGradient");
  static const auto f_primal_grad_fused = Op::GetAttr<FFusedPrimalGradient>("FFusedPrimalGradient");
  if (f_primal_grad_fused.count(op)) {
    return f_primal_grad_fused[op](orig, ograd, old_igrads);
  } else if (f_primal_grad.count(op)) {
    return AccGrad(old_igrads, f_primal_grad[op](orig, ograd));
  }
  LOG(FATAL) << "Gradient is not registered for operator " << op->name;
  throw;
}

#define MNM_NODE_ASSUME_ANF(NodeType)                      \
  void VisitExpr_(const NodeType* node) final {            \
    LOG(FATAL) << "ValueError: Gradient pass assumes ANF"; \
    throw;                                                 \
  }

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

struct Gradient : public ExprVisitor {
 public:
  MNM_NODE_ASSUME_ANF(VarNode);
  MNM_NODE_ASSUME_ANF(RelayConstantNode);
  // The algorithm shouldn't generate or deal with references
  MNM_NODE_NOT_SUPPORT(RefCreateNode);
  MNM_NODE_NOT_SUPPORT(RefReadNode);
  MNM_NODE_NOT_SUPPORT(RefWriteNode);
  // MNM has not started to take care of ADTs yet
  MNM_NODE_NOT_SUPPORT(tvm::relay::ConstructorNode);
  MNM_NODE_NOT_SUPPORT(tvm::relay::MatchNode);
  // TODO(@junrushao1994): implement them
  // replace GlobalVar with adjoint
  MNM_NODE_NOT_IMPL(GlobalVarNode);
  // replace OpNode with its corresponding GlobalVar's adjoint (eta-expand)
  MNM_NODE_NOT_IMPL(OpNode);
  // normalize the program with tail call
  MNM_NODE_NOT_IMPL(IfNode);

  void UpdateAdjoints(const Array<Expr>& args, const Array<Expr>& new_ones) {
    int n = std::min(args.size(), new_ones.size());
    for (int i = 0; i < n; ++i) {
      const Expr& new_one = new_ones[i];
      const Expr& arg = args[i];
      if (!new_one.defined() || arg->IsInstance<RelayConstantNode>()) {
        continue;
      }
      CHECK(arg->IsInstance<VarNode>());
      const Var& var = Downcast<Var>(arg);
      adjoints.Set(var, ll->Push(new_one));
    }
  }

  Array<Expr> ExtractAdjoints(const Array<Expr>& args) {
    Array<Expr> result;
    for (const Expr& arg : args) {
      if (arg->IsInstance<RelayConstantNode>()) {
        result.push_back(NullValue<Expr>());
      } else if (arg->IsInstance<VarNode>()) {
        const Var& var = Downcast<Var>(arg);
        if (adjoints.count(var)) {
          result.push_back(adjoints[var]);
        } else {
          result.push_back(NullValue<Expr>());
        }
      } else if (arg->IsInstance<TupleNode>()) {
        LOG(INFO) << "NotImplementedError";
        throw;
      }
      LOG(FATAL) << "Cannot deal with adjoint of type: " << arg->GetTypeKey();
      throw;
    }
    return result;
  }

  // Entry
  void VisitExpr_(const FunctionNode* node) final {
    // TODO(@junrushao1994): check closure
    if (func_vistied) {
      LOG(FATAL) << "NotImplementedError: Closure";
      throw;
    }
    func_vistied = true;
    ExprVisitor::VisitExpr_(node);
  }

  // Let binding
  void VisitExpr_(const LetNode* node) final {
    CHECK(bound.defined()) << "ValueError: Gradient pass assumes ANF";
    VisitExpr(node->body);
    // If does exists gradient w.r.t. the bound var
    if (adjoints.count(node->var)) {
      bound = node->var;
      VisitExpr(node->value);
      bound = NullValue<Var>();
    }
  }

  void VisitExpr_(const CallNode* node) final {
    CHECK(bound.defined()) << "ValueError: Gradient pass assumes ANF";
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      const Op& op = Downcast<Op>(node->op);
      Array<Expr> old_igrads = this->ExtractAdjoints(node->args);
      Array<Expr> igrads = UpdatePrimalGrad(op, callee, adjoints[bound], old_igrads);
      this->UpdateAdjoints(node->args, igrads);
    } else {
      LOG(FATAL) << "Calling unsupported type: " << callee->GetTypeKey();
      throw;
    }
    LOG(FATAL) << "NotImplementedError: Call";
    throw;
  }

  void VisitExpr_(const TupleNode* node) final {
    CHECK(bound.defined()) << "ValueError: Gradient pass assumes ANF";
    const TupleNode* ograd = adjoints[bound].as<TupleNode>();
    CHECK(ograd);
    Array<Expr> delta = ograd->fields;
    Array<Expr> old_igrads = this->ExtractAdjoints(node->fields);
    Array<Expr> igrads = AccGrad(old_igrads, delta);
    this->UpdateAdjoints(node->fields, igrads);
  }

  void VisitExpr_(const TupleGetItemNode* node) final {
    CHECK(bound.defined()) << "ValueError: Gradient pass assumes ANF";
    const Expr& ograd = adjoints[bound];
    const Var& var = Downcast<Var>(node->tuple);
    int index = node->index;
    if (!adjoints.count(var)) {
      adjoints.Set(var, TupleNode::make(std::vector<Expr>(index)));
    }
    const TupleNode* igrad = adjoints[var].as<TupleNode>();
    CHECK(igrad);
    Array<Expr> items = igrad->fields;
    while (static_cast<int>(index) >= items.size()) {
      items.push_back(NullValue<Expr>());
    }
    if (items[index].defined()) {
      items.Set(index, ll->Push(TensorAdd(ograd, items[index])));
    } else {
      items.Set(index, ograd);
    }
    adjoints.Set(var, TupleNode::make(items));
  }

  Map<Var, Expr> adjoints;
  Var bound{nullptr};
  bool func_vistied = 0;
  LetList* ll;
};

}  // namespace gradient
}  // namespace pass
}  // namespace mnm
