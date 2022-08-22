/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file gradient.cc
 * \brief Symbolic gradient pass
 */
#include <sstream>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/op_utils.h"
#include "raf/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace gradient {

#define RAF_NODE_NOT_SUPPORT(NodeType)                                  \
  void VisitExpr_(const NodeType* node) final {                         \
    LOG(FATAL) << "ValueError: feature is not supported:" << #NodeType; \
    throw;                                                              \
  }

#define RAF_NODE_NOT_IMPL(NodeType)                     \
  void VisitExpr_(const NodeType* node) final {         \
    LOG(FATAL) << "NotImplementedError: " << #NodeType; \
    throw;                                              \
  }

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;

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

struct ReverseAD : public ExprVisitor {
 public:
  // Closures are not supported
  RAF_NODE_NOT_SUPPORT(FunctionNode);
  RAF_NODE_NOT_SUPPORT(LetNode);
  // The algorithm shouldn't generate or deal with references
  RAF_NODE_NOT_SUPPORT(RefCreateNode);
  RAF_NODE_NOT_SUPPORT(RefReadNode);
  RAF_NODE_NOT_SUPPORT(RefWriteNode);
  // RAF has not started to take care of ADTs yet
  RAF_NODE_NOT_SUPPORT(tvm::relay::ConstructorNode);
  RAF_NODE_NOT_SUPPORT(tvm::relay::MatchNode);
  // TODO(@junrushao1994): implement them
  // TODO(@junrushao1994): nested tuples are still problematic
  RAF_NODE_NOT_IMPL(OpNode);  // replace OpNode with its corresponding GlobalVar's adjoint

 public:
  explicit ReverseAD(std::unordered_map<const VarNode*, bool>& requires_grads_main_map)
      : requires_grads_main_map_(requires_grads_main_map) {
  }

 public:  // visitor functions
  /* These visitor functions compute the gradient for the ExprNode. For each node, the basic steps
   * are
   * 1) GetOgrads - Get the ograds - these are set by the reverse AD of operators that exist after
   * the current op.
   * 2) GetIgrads - Get the igrads placeholders. We will write the computed grads into these
   * locations.
   * 3) UpdateIgrads - Using the ograds, and depending on the node find the new igrads. This changes
   * from node to node, e.g. for CallNode with op of OpNode type, we use registered functions, but
   * for Tuple, we pass on the ograds to igrads as it is.
   * 4) WriteBack - Write the newly computed igrads into igrad placeholders.
   */

  /*!
   * \brief Handling the gradient for the variable node. For var node, we can pass the ograd to
   * igrads directly
   */
  void VisitExpr_(const VarNode* node) final {
    Var var = GetRef<Var>(node);
    var_to_primal_expr_[node] = var;
    const Array<Expr>& ograds = GetOutputGrads();
    WriteBackInputGrads({GetRef<Expr>(node)}, ograds);
  }

  /*!
   * \brief Handling the gradient for the relay constant node.
   * Constant nodes do not need any grad. Here, we are just saving the primal_expr for later use.
   */
  void VisitExpr_(const RelayConstantNode* node) final {
    var_to_primal_expr_[let_var_.get()] = GetRef<Expr>(node);
  }

  /*!
   * \brief Handling the gradient for the Tuple node.
   * Gradient is a passthrough. We can pass on the ograds directly to igrads.
   */
  void VisitExpr_(const TupleNode* node) final {
    var_to_primal_expr_[let_var_.get()] = GetRef<Expr>(node);
    const Array<Expr>& ograds = GetOutputGrads();
    WriteBackInputGrads(node->fields, ograds);
  }

  /*!
   * \brief Handling the gradient for the Tuple Get Item node.
   * Gradient is a passthrough. We can pass on the ograds directly to igrads.
   * We have to ensure that the right tuple item is set.
   */
  void VisitExpr_(const TupleGetItemNode* node) final {
    var_to_primal_expr_[let_var_.get()] = GetRef<Expr>(node);
    const VarNode* tuple = node->tuple.as<VarNode>();
    const Array<Expr>& ograds = GetOutputGrads();
    Array<Expr> tuple_igrads = tuple_grads[tuple];
    CHECK_EQ(ograds.size(), 1);
    CHECK_GT(tuple_igrads.size(), node->index);
    tuple_igrads.Set(node->index, AddTensor(ograds[0], tuple_igrads[node->index]));
    tuple_grads[tuple] = tuple_igrads;
  }

  /*!
   * \brief Handling the gradient for the if node.
   * An example of gradient for if node is as follows
   * Original
   *    let %a3 = if (%a2) {
   *      @true_branch_0(%x)
   *    } else {
   *      @false_branch_0(%x)
   *    };
   *    %a3
   *
   * After AD
   *   let %0 = if (%a2) {
   *     @true_branch_0_grad(%x) // The function returns (primal, adjoint)
   *   } else {
   *     @false_branch_0_grad(%x) // The function returns (primal, adjoint)
   *   };
   *   let %a3 = %0.0; // Primal Function
   *   let %a3_closure = %0.1 // Gradient closure
   *
   * The If node has true and false branches which have already been lifted into global functions
   * ealier using LiftBranchBody pass as shown above. We use the gradient functions of the branches
   * to construct the gradient function for if node
   *  1) Get the primal and closure for both branches.
   *  2) Set up the primal input by taking the 0th tuple item
   *  3) Set up the gradient by taking the 1st tuple item
   *
   */
  void VisitExpr_(const IfNode* node) final {
    auto get_ad_branch = [&](const Expr& branch) {
      auto branch_call = branch.as<CallNode>();
      CHECK(branch_call) << "If branches should be lifted to global functions. "
                         << "LiftBranchBody pass is not applied.";
      auto branch_gvar_node = branch_call->op.as<GlobalVarNode>();
      CHECK(branch_gvar_node) << "If branches should be lifted to global functions. "
                              << "LiftBranchBody pass is not applied.";

      auto branch_gvar = GetRef<GlobalVar>(branch_gvar_node);
      auto new_branch_call = Call(branch_gvar, branch_call->args);
      return new_branch_call;
    };

    // 1) Get the primal + gradient functions for both branches
    ExprVisitor::VisitExpr(node->cond);
    auto true_ad_branch = get_ad_branch(node->true_branch);
    auto false_ad_branch = get_ad_branch(node->false_branch);

    // 2) Set up the primal input by extracting the primal functions
    If primal_adjoint_if(node->cond, true_ad_branch, false_ad_branch);
    var_to_primal_expr_[let_var_.get()] = TupleGetItem(primal_adjoint_if, 0);
    var_to_primal_expr_[let_var_.get()]->checked_type_ = node->checked_type();

    // 3) Set up the gradient
    const Array<Expr>& ograds = GetOutputGrads();
    auto if_node_args = node->true_branch.as<CallNode>()->args;
    // Lambda branch body ensures that both branches have same args
    const Array<Expr>& igrads = GetInputGrads(if_node_args);

    Expr adjoint_func = adjoint_ll_->Push(TupleGetItem(primal_adjoint_if, 1));
    Expr _ograds =
        tuple_length.count(let_var_.get()) ? adjoint_ll_->Push(Tuple(ograds)) : ograds[0];

    // Calculating new_grad is easy as we just have to call the adjoint function on ograds.
    Expr new_igrad = adjoint_ll_->Push(Call(adjoint_func, {_ograds}));
    Array<Expr> new_igrads;
    if (igrads.size() == 1) {
      new_igrads.push_back(new_igrad);
    } else {
      for (int i = 0; i < igrads.size(); i++) {
        new_igrads.push_back(TupleGetItem(new_igrad, i));
      }
    }
    WriteBackInputGrads(if_node_args, new_igrads);
  }

  /*!
   * \brief Handling the gradient for the Call node.
   * For the call node, there are 2 scenarios
   * 1) Call op is an OpNode - In this case we use the registered op gradient functions.
   * 2) Call op is globalVar node - in this case we handle GlobalVar similar to IfNode handling
   * above
   */
  void VisitExpr_(const CallNode* node) final {
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      // Call node Op is of OpNode Type
      const Op& op = Downcast<Op>(node->op);
      const Array<Expr>& ograds = GetOutputGrads();
      const Array<Expr>& igrads = GetInputGrads(node->args);
      const Array<Expr>& new_igrads = UpdateInputGrads(op, GetRef<Expr>(node), ograds, igrads);
      WriteBackInputGrads(node->args, new_igrads);
      var_to_primal_expr_[let_var_.get()] = GetRef<Expr>(node);
    } else if (callee->IsInstance<GlobalVarNode>()) {
      // Call node op is global Var node
      // Original
      // %1 = @func(%a)
      //
      // After AD
      // %1 = @func(%a)
      // %primal = %1.0
      // %gradient = %1.1
      const GlobalVarNode* gvn = callee.as<GlobalVarNode>();
      GlobalVar gv = GetRef<GlobalVar>(gvn);

      // Get the primal value from the differentiated funtion and extracting 0th index
      // Get the adjoing closure func by extracting the first index
      auto primal_adjoint_fn = Call(GetRef<GlobalVar>(gvn), node->args);
      var_to_primal_expr_[let_var_.get()] = TupleGetItem(primal_adjoint_fn, 0);
      var_to_primal_expr_[let_var_.get()]->checked_type_ = node->checked_type();
      Expr adjoint_func = adjoint_ll_->Push(TupleGetItem(primal_adjoint_fn, 1));

      // Now extract ograds, igrads placeholder, update igrads using the adjoint func to get
      // new_igrads, and finally writeback the new_igrads to the placeholder
      const Array<Expr>& ograds = GetOutputGrads();
      Expr _ograds =
          tuple_length.count(let_var_.get()) ? adjoint_ll_->Push(Tuple(ograds)) : ograds[0];
      // new_igrads are just the application of extracted closure call on ograds.
      const Array<Expr>& igrads = GetInputGrads(node->args);
      Expr new_igrad = adjoint_ll_->Push(Call(adjoint_func, {_ograds}));
      Array<Expr> new_igrads;
      if (igrads.size() == 1) {
        new_igrads.push_back(new_igrad);
      } else {
        for (int i = 0; i < igrads.size(); i++) {
          new_igrads.push_back(TupleGetItem(new_igrad, i));
        }
      }
      WriteBackInputGrads(node->args, new_igrads);
    } else {
      LOG(FATAL) << "NotImplementedError: Calling unsupported type: " << callee->GetTypeKey();
      throw;
    }
  }

 public:
  /* For each node, the basic steps are
   * 1) GetOgrads - Get the ograds - these are set by the reverse AD of operators that exist after
   * the current op.
   * 2) GetIgrads - Get the igrads placeholders. We will write the computed grads into these
   * locations.
   * 3) UpdateIgrads - Using the ograds, and depending on the node find the new igrads. This changes
   * from node to node, e.g. for CallNode with op of OpNode type, we use registered functions, but
   * for Tuple, we pass on the ograds to igrads as it is.
   * 4) WriteBack - Write the newly computed igrads into igrad placeholders.
   */
  /*!
   * \brief Find the ograds. These have been set by the reverse AD of the operators that come after
   * the current op. Please take a look at the top of this class for high-level description.
   */
  Array<Expr> GetOutputGrads() {
    const VarNode* var = let_var_.operator->();
    return InitUndefinedGrads(tuple_grads[var]);
  }

  /*!
   * \brief Find the input grads.
   */
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

  Array<Expr> UpdateInputGrads(const Op& op,      // the operator called
                               const Expr& orig,  // relay.Call that contains this expression
                               const Array<Expr>& ograds,  // grad(output)
                               const Array<Expr>& igrads) {
    // given: igrads
    // returns: new_igrads = igrads + grad-of-call-op<ograd>
    static const auto fpg = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");
    static const auto ffpg = Op::GetAttrMap<FFusedPrimalGradient>("FFusedPrimalGradient");
    const VarNode* var = let_var_.operator->();
    const Expr& _ograds = tuple_length.count(var) ? Tuple(ograds) : ograds[0];
    CHECK(IsDefined(ograds)) << "Output grads are undefined for " << ir::AsText(ograds, false);
    Array<Expr> ret;
    auto call = Downcast<Call>(orig);
    if (ffpg.count(op)) {
      ret = ffpg[op](orig, let_var_, _ograds, igrads);
    } else if (fpg.count(op)) {
      Array<Expr> orig_args;
      for (auto arg : call->args) {
        if (auto in_var = arg.as<VarNode>()) {
          orig_args.push_back(var_to_primal_expr_[in_var]);
        } else {
          CHECK(arg.as<ConstantNode>() != nullptr);
          orig_args.push_back(arg);
        }
      }
      ret = fpg[op](orig, orig_args, let_var_, _ograds);
      // if not 'requires_grad', set return to null
      for (int i = 0, n = ret.size(); i < n; ++i) {
        if (auto in_var = call->args[i].as<VarNode>()) {
          if (requires_grads_map_[in_var] == false) {
            ret.Set(i, NullValue<Var>());
          }
        }
      }
    } else {
      LOG(FATAL) << "Gradient is not registered for operator " << op->name;
      throw;
    }

    // ensure intermediate results are bound to a relay::var
    ANFNormalizer normalizer(adjoint_ll_);
    for (int i = 0, n = ret.size(); i < n; ++i) {
      if (ret[i].defined() && !ret[i]->IsInstance<VarNode>()) {
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
          for (int i = 0; i < tuple_length[var]; ++i) {
            WriteBackInputGrad(var, i, adjoint_ll_->Push(TupleGetItem(igrad, i)));
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

  bool IsDefined(const Array<Expr>& exprs) {
    for (const auto& e : exprs) {
      if (!e.defined()) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Make zero tensors for undefined gradients of the expression binded by
   * the given var node. To make the IR more optimization friendly, this function
   * tries to use zeros instead of zeros_like to create zero tensors, when the target
   * type is available.
   */
  Expr MakeZero(const Var var) {
    if (var_to_primal_expr_.count(var.get()) > 0 && var_to_primal_expr_[var.get()].defined()) {
      auto expr = var_to_primal_expr_[var.get()];
      if (expr->checked_type_.defined()) {
        auto expr_type = expr->checked_type();

        if (auto tgi = expr.as<TupleGetItemNode>()) {
          auto tuple_var = Downcast<Var>(tgi->tuple);
          auto target_expr = var_to_primal_expr_[tuple_var.get()];
          if (auto tuple_node = target_expr.as<TupleNode>()) {
            // The target is a tuple node, keep tracing to the root tensor
            CHECK(tuple_node->fields[tgi->index]->IsInstance<VarNode>());
            return MakeZero(Downcast<Var>(tuple_node->fields[tgi->index]));
          } else {
            // The target is not a tuple node (e.g., a call node returns a tuple)
            if (target_expr->checked_type_.defined()) {
              // Pass its type to create zeros op
              auto target_type = target_expr->checked_type();
              auto tuple_type_node = target_type.as<TupleTypeNode>();
              CHECK(tuple_type_node != nullptr);
              expr_type = tuple_type_node->fields[tgi->index];
            } else {
              // Lack of type information, pass an undefined type to create zeros_like
              expr_type = Type();
            }
          }
        } else if (auto tuple = expr.as<TupleNode>()) {
          // Make zeros op for each tuple element. Note that we do not make a single
          // zeros_like op because it is hard to be optimized later.
          Array<Expr> zeros;
          for (size_t i = 0; i < tuple->fields.size(); ++i) {
            zeros.push_back(MakeZero(Downcast<Var>(tuple->fields[i])));
          }
          return adjoint_ll_->Push(Tuple(zeros));
        }

        if (auto ttype = expr_type.as<TensorTypeNode>()) {
          if (tvm::relay::IsDynamic(expr_type)) {
            static auto zeros_like = Op::Get("raf.op.zeros_like");
            return adjoint_ll_->Push(Call(zeros_like, {var}));
          }
          static auto zeros = Op::Get("raf.op.zeros");
          return adjoint_ll_->Push(
              Call(zeros, {MakeConstant(ArrayToIntTuple(ttype->shape)),
                           MakeConstant(StringValue::make(DLDataType2String(ttype->dtype)))}));
        } else {
          LOG(FATAL) << "Unsupported expression or type for making zeros: "
                     << raf::ir::AsText(expr);
          throw;
        }
      }
    }

    // Otherwise use zeros_like op.
    static auto zeros_like = Op::Get("raf.op.zeros_like");
    return adjoint_ll_->Push(Call(zeros_like, {var}));
  }

  /*!
   * \brief It is possible that some of the outputs of a CallNode are not used by the subsequent
   * operators. For example, we can split an operator and may use only one split tensor later on. In
   * this case, the ograd for the other tensor will not be set to a valid value from AutoDiff
   * because in the reverse mode there are no operators to set the ograd for the unused op.
   * Another common example is a while loop in Relay where we will pass on the counter and some
   * other float tensors that are involved only in condition evaluation. These sometimes become
   * output of the function, but they are never used by the subsequent graph.
   *
   * In such scenarios we have to define the ograd to have a valid ograd. There are 2 cases
   * 1) If tuple item, find the corresponding primal expr and tuple item and pass it through
   * zeros_like
   * 2) It is tensor type, pass it through zeros_like
   */
  Array<Expr> InitUndefinedGrads(const Array<Expr>& ograds) {
    if (IsDefined(ograds)) {
      return ograds;
    }
    Array<Expr> defined_ograds;
    for (int index = 0; index < ograds.size(); index++) {
      auto ograd = ograds[index];
      if (ograd.defined()) {
        defined_ograds.push_back(ograd);
      } else {
        // Output gradient is undefined. Generate a zero tensor as the placeholder
        Var target_var = let_var_;

        auto primal_expr = var_to_primal_expr_[let_var_.get()];
        if (primal_expr->checked_type_.defined()) {
          auto primal_expr_type = primal_expr->checked_type();
          if (primal_expr.as<TupleNode>() || primal_expr_type.as<TupleTypeNode>()) {
            // If primal expr is a tuple, find the corresponding primal expr and
            // create a tuple get item node.
            auto tuple_node = primal_expr.as<TupleNode>();
            if (tuple_node && tuple_node->fields[index].as<ConstantNode>()) {
              target_var = Var();  // Skip constants
            } else {
              auto tgi = TupleGetItem(let_var_, index);
              tgi->checked_type_ = primal_expr_type.as<TupleTypeNode>()->fields[index];
              target_var = adjoint_ll_->Push(tgi);
              var_to_primal_expr_[target_var.get()] = tgi;
            }
          }
        }

        if (target_var.defined() && !primal_expr.as<ConstantNode>()) {
          defined_ograds.push_back(MakeZero(target_var));
        } else {
          // Still put an undefined ograd as the placeholder of constants to make sure the
          // length of defined_ograds is same as ograds.
          defined_ograds.push_back(ograd);
        }
      }
    }
    return defined_ograds;
  }

 private:
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
    static Op op = Op::Get("raf.op.add");
    if (!x1.defined() && !x2.defined()) {
      return NullValue<Var>();
    }
    if (!x1.defined()) {
      return x2->IsInstance<VarNode>() ? x2 : adjoint_ll_->Push(x2);
    }
    if (!x2.defined()) {
      return x1->IsInstance<VarNode>() ? x1 : adjoint_ll_->Push(x1);
    }
    const auto* t1 = x1.as<TupleNode>();
    const auto* t2 = x2.as<TupleNode>();
    if (t1 && t2) {
      return Tuple(AddTensors(t1->fields, t2->fields));
    }
    return adjoint_ll_->Push(Call(op, {x1, x2, MakeNull(), MakeNull()}));
  }

 private:
  // Helper functions to assist in the creation of input and ret of the backward closure.
  /*!
   * \brief Make the "dy" variable which is the input variable of the grad closure.
   * The closure looks like this. We are creating the dy input variable.
   *   fn (dy) {
   *      let xxx = ...;
   *      ret_grads
   *   }
   */
  Var MakeOutputGrad() {
    Type ty = current_func_node_->checked_type_;
    Type annotation;
    if (ty.defined()) {
      const auto* fty = ty.as<FuncTypeNode>();
      CHECK(fty != nullptr);
      annotation = fty->ret_type;
    } else {
      LOG(WARNING) << "InferType is not run before AutoDiff pass";
    }
    return raf::ir::MakeVar("dy", annotation);
  }

  /*!
   * \brief Set the tuple_grad for the return variable of primal function to dy.
   * This initializes the backward gradient and we can start the gradient process.
   * The closure looks like this. We are connecting the primal_ret var with dy here.
   *   %primal_ret = ...
   *   %closure = fn (dy) {
   *      let xxx = ...;
   *      ret_grads
   *   }
   *   return (%primal_ret, %closure)
   */
  void MakeClosureInputGrads(const Var& dy) {
    const VarNode* ret = primal_ell_->ret.operator->();
    if (tuple_length.count(ret)) {
      int n = tuple_grads[ret].size();
      for (int i = 0; i < n; ++i) {
        tuple_grads[ret].Set(i, adjoint_ll_->Push(TupleGetItem(dy, i)));
      }
    } else {
      CHECK_EQ(tuple_grads[ret].size(), 1);
      tuple_grads[ret] = {dy};
    }
  }

  /*!
   * \brief The closure looks like this.
   *   %primal_ret = ...
   *   %closure = fn (dy) {
   *      let xxx = ...;
   *      ret_grads
   *   }
   *   return (%primal_ret, %closure)
   * In this function we are working on ret_grads. The gradients for all the expressions has already
   * finished. The relevant expressions have already populated the tuple_grads data structure
   * accordingly.
   */
  Expr MakeClosureRet() {
    const Array<Var>& targets = current_func_node_->params;
    Array<Expr> grads;
    for (const Var& var : targets) {
      const VarNode* var_node = var.operator->();
      std::vector<Expr> var_grads(tuple_grads[var_node].begin(), tuple_grads[var_node].end());
      if (tuple_length.count(var_node)) {
        for (Expr& expr : var_grads) {
          if (!expr.defined()) {
            // TODO (janimesh) - When is this if condition is satisfied? Should this be ZeroGrad or
            // NoGrad?
            expr = MakeConstant(NoGradValue::make());
          }
        }
        grads.push_back(adjoint_ll_->Push(Tuple(var_grads)));
      } else {
        CHECK_EQ(var_grads.size(), 1);
        if (var_grads[0].defined()) {
          grads.push_back(var_grads[0]);
        } else if (!requires_grads_map_[var.get()]) {
          // No grad required.
          grads.push_back(MakeConstant(NoGradValue::make()));
        } else {
          // This means that we need the gradient but nobody use this variable. This can happen in
          // Normalizing the if conditions. To handle such scenario, we can use actualy zero values.
          grads.push_back(adjoint_ll_->Push(MakeZero(var)));
        }
      }
    }
    if (targets.size() == 1) {
      if (requires_grads_map_[targets[0].get()]) {
        return Downcast<Var>(grads[0]);
      } else {
        return MakeConstant(NoGradValue::make());
      }
    }
    return adjoint_ll_->Push(Tuple(grads));
  }

 private:
  // Key functions that manange adjoint computation and bundling
  /*!
   * \brief This functions walks the graph in the reverse manner and computes the gradients.
   *             Primal                                      Adjoint
   *               x                                            dx --> igrad
   *               |                                            |
   *            operator                                  operator_grad
   *               |                                            |
   *               y                                            dy --> ograd
   *
   * As shown in the above example, while calculating the gradients, we have ograd as an input
   * and then find the igrad using the opertor_grad transformation. The AD works in reverse manner
   *
   * Primal  =            input -> x --> y --> output
   * Adjoint =            doutput --> dy --> dx ---> dinput
   *
   * By moving node-by-node, each node takes the ograd and finds the igrads. The next node in the
   * reverse order looks at the ograd (which is an igrad already set by the earlier operator in
   * adjoint computation).
   */
  Expr GetAdjoint(const Var& dy) {
    Expr adjoint_body = LetList::With([&](LetList* ll) {
      this->adjoint_ll_ = ll;
      // Set the igrad of last primal expression to dy. After this, we can now
      // call the gradient for the expressions in reverse order. It works like follows
      // igrad for expr[i] --> grad for expr[i] --> ograd for expr[i]
      // igrad for expr[i - 1] = ograd_for_expr[i]
      // igrad for expr[i - 1] --> grad for expr[i] --> ograd for expr[i - 1]
      // The next line basically sets igrad for last expr (expr[n-1]) = dy
      MakeClosureInputGrads(dy);

      const auto& vars = primal_ell_->vars;
      const auto& exprs = primal_ell_->exprs;
      CHECK_EQ(vars.size(), exprs.size());
      int n = exprs.size();

      // Walk through each expresion and set the tuple_grads accordingly.
      for (int i = n - 1; i >= 0; --i) {
        let_var_ = vars[i];
        ExprVisitor::VisitExpr(exprs[i]);
      }
      return MakeClosureRet();
    });
    return adjoint_body;
  }

  /*!
   * \brief This method bundles up the primal and adjoint computation together.
   */
  Function BundlePrimalAndAdjoint(const Var& dy, const Expr& adjoint_body) {
    // Create a new let list that captures both primal and adjoint closure.
    std::unique_ptr<ExplicitLetList> primal_adjoint_ell_{nullptr};

    // Add all the primal exprs in the let list
    primal_adjoint_ell_ = std::make_unique<ExplicitLetList>();
    for (int i = 0; i < primal_ell_->vars.size(); i++) {
      const Var& var = primal_ell_->vars[i];
      primal_adjoint_ell_->vars.push_back(var);
      CHECK(var_to_primal_expr_.count(var.get()) != 0);
      primal_adjoint_ell_->exprs.push_back(var_to_primal_expr_.at(var.get()));
    }

    // Now lets add the final reverse AD adjoint_closure into the primal_adjoint_ell_
    // let adjoint_closure = fn(dy) {};
    Var adjoint_closure = raf::ir::MakeVar("adjoint_closure", {});
    primal_adjoint_ell_->vars.push_back(adjoint_closure);
    primal_adjoint_ell_->exprs.push_back(Function({dy}, adjoint_body, {}, {}));

    // Now lets add the return value which is a tuple of primal and adjoint closure
    // let ret = tuple(y, adjoint_closure)
    Var ret = raf::ir::MakeVar("ret", {});
    primal_adjoint_ell_->vars.push_back(ret);
    primal_adjoint_ell_->exprs.push_back(Tuple({primal_ell_->ret, adjoint_closure}));

    // Set the return value of ell
    primal_adjoint_ell_->ret = ret;

    // Setup the types correctly to make InferType job easier
    Type primal_type = current_func_node_->ret_type;
    Type closure_args_type = primal_type;
    Array<Type> params_type;
    for (auto var : current_func_node_->params) {
      if (requires_grads_map_[var.get()]) {
        params_type.push_back(var->type_annotation);
      } else {
        // This is to help inferType when NoGradValue is set
        params_type.push_back(TensorType::Scalar(DataType::Int(64)));
      }
    }
    Type closure_ret_type = params_type.size() == 1 ? params_type[0] : TupleType(params_type);
    auto closure_type = FuncType({closure_args_type}, closure_ret_type, {}, {});
    auto func_ret_type = TupleType({primal_type, closure_type});
    return Function(current_func_node_->params, primal_adjoint_ell_->AsExpr(), func_ret_type, {});
  }

 private:  // Init functions
  void Init() {
    InitRequiresGrad();
    InitTuple();
  }
  /*!
   * \brief This function initializes the grad for func params and intermediate lets vars.
   * The tuple_grads basically point to the ograd object for a let var. Later on while computing the
   * grad for each expr, we write the grad funtion in these tuple_grads.
   */
  void InitTuple() {
    // grads for tuples
    const auto& vars = primal_ell_->vars;
    const auto& exprs = primal_ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    int n = exprs.size();
    for (const auto& var : vars) {
      requires_grads_map_[var.get()] = true;
    }

    // Save the var to expr mapping to use later
    for (int i = 0; i < n; ++i) {
      var_to_primal_expr_[vars[i].get()] = exprs[i];
    }

    // Initialize the grad entries for tuple first
    for (int i = 0; i < n; ++i) {
      if (const auto* tuple = exprs[i].as<TupleNode>()) {
        const VarNode* var_node = vars[i].get();
        tuple_length[var_node] = tuple->fields.size();
        tuple_grads[var_node] = std::vector<Expr>(tuple->fields.size(), NullValue<Var>());
      } else if (const auto* tuple_get_item = exprs[i].as<TupleGetItemNode>()) {
        // This is a tuple, which however is not constructed inside this function
        // (i.e. input arguments or outputs of an operator/function)
        // Therefore, its length is unknown
        const VarNode* var_node = tuple_get_item->tuple.as<VarNode>();
        int size;
        if (var_node->checked_type_.defined()) {
          const TupleType& tuple_type = Downcast<TupleType>(var_node->checked_type());
          size = tuple_type->fields.size();
        } else {
          size = tuple_get_item->index + 1;
        }
        if (tuple_length.count(var_node) == 0) {
          tuple_length[var_node] = -1;
          tuple_grads[var_node] = std::vector<Expr>(size, NullValue<Var>());
        } else {
          // FIXME(comaniac): the ignored nn.dropout becomes a tuple-2 which has
          // tuple_length = 2 here.
          // CHECK_EQ(tuple_length[var], -1);
          int old_size = tuple_grads[var_node].size();
          if (size > old_size) {
            tuple_grads[var_node] = std::vector<Expr>(size, NullValue<Var>());
          }
        }
      }
    }

    // Revisit the let variables and init the ones that are not tuples
    for (const Var& var : vars) {
      const VarNode* var_node = var.operator->();
      if (!tuple_grads.count(var_node)) {
        tuple_grads[var_node] = {NullValue<Var>()};
      }
    }

    // Finally init the grads for func params
    for (const Var& param : current_func_node_->params) {
      const VarNode* var_node = param.operator->();
      if (!tuple_grads.count(var_node)) {
        tuple_grads[var_node] = {NullValue<Var>()};
      }
    }
  }

  /*!
   * \brief Find out which func params require input grads. Here, we use the main function
   * requires grads as the final qualifier. If it is not main, the assumption is that we need
   * gradients for all float variables.
   */
  void InitRequiresGrad() {
    for (const Var& var : current_func_node_->params) {
      const VarNode* var_node = var.get();
      requires_grads_map_[var_node] = true;
      if (requires_grads_main_map_.count(var_node)) {
        requires_grads_map_[var_node] = requires_grads_main_map_[var_node];
      }
    }
  }

 public:
  Function Run(ir::Function func) {
    auto body = func->body;
    if (body.as<RelayConstantNode>()) {
      Var var = raf::ir::MakeVar("v1", {});
      body = Let(var, body, var);
    }
    current_func_node_ = func.get();
    primal_ell_ = ExplicitLetList::make(body);
    Init();
    Var dy = MakeOutputGrad();
    Expr adjoint_body = GetAdjoint(dy);
    return BundlePrimalAndAdjoint(dy, adjoint_body);
  }

 public:
  // The map telling which func params of the main function require grads.
  std::unordered_map<const VarNode*, bool>& requires_grads_main_map_;

 private:
  /*!
   * \brief Each instance of the class works on a function node. The instance keeps tracks of many
   * members of the function like current let_var we are working on, the primal and gradient
   * expression for each let var-expr binding. The following members track this type of information.
   */

  /*! \brief A global function we are working on in this instance of the class. */
  const FunctionNode* current_func_node_;
  /*! \brief The Let variable that is tracking the var-expr binding we are working on.*/
  Var let_var_;
  /*! \brief The explicit let list for primal computation. */
  std::unique_ptr<ExplicitLetList> primal_ell_{nullptr};
  /*! \brief The let list for the adjoint computation. */
  LetList* adjoint_ll_ = nullptr;
  /*! \brief A map saving let var to its grad expr mapping. */
  std::unordered_map<const VarNode*, Expr> var_to_primal_expr_;
  /*! \brief A map telling which func params need grad. */
  std::unordered_map<const VarNode*, bool> requires_grads_map_;
  /*! \brief A map saving the lenghth of grads. */
  std::unordered_map<const VarNode*, int> tuple_length;
  /*! \brief A map that tracks the computed grads in reverse AD. */
  std::unordered_map<const VarNode*, Array<Expr>> tuple_grads;
};

/*! \brief A helper to canonicalize the IR with AutoDiff if its backward is going to be inlined.
 * Specifically, it removes direct let assignment (i.e., let %a = %b;) and the sum ops with
 * empty keepdims axis. The sum op used to calculate the gradient of certain ops (e.g., add/sub/etc)
 * may use "raf.op.get_kept_dims" to get the keep dims. InferType pass can infer the real keep
 * dims and simplify the "raf.op.get_kept_dims" op. If the keep dims are empty after InferType,
 * then the sum op has no effect and can be simplified.
 */
class Canonicalizer : public ExprMutator {
 public:
  explicit Canonicalizer() : sum_op_(Op::Get("raf.op.sum")) {
  }

 private:
  bool IsEmptyTupleValue(const Expr& expr) {
    if (auto const_node = expr.as<ConstantNode>()) {
      auto data = const_node->value;
      if (data.defined()) {
        if (auto tuple_data = data.as<TupleValueObj>()) {
          if (tuple_data->fields.size() == 0) {
            return true;
          }
        }
      }
    }
    return false;
  }

  Expr SimplifySum(const CallNode* call_node) {
    if (call_node->args.size() > 2) {
      if (IsEmptyTupleValue(call_node->args[1]) && IsEmptyTupleValue(call_node->args[2])) {
        return call_node->args[0];
      }
    }
    return GetRef<Expr>(call_node);
  }

 public:
  Expr Canonicalize(const Expr& expr) {
    return this->Mutate(expr);
  }

  Expr VisitExpr_(const VarNode* var_node) final {
    Var var = GetRef<Var>(var_node);
    if (mapping_.count(var)) {
      return mapping_.at(var);
    }
    return var;
  }

  Expr VisitExpr_(const CallNode* node) final {
    Expr expr = ExprMutator::VisitExpr_(node);
    if (node->op.same_as(sum_op_)) {
      return SimplifySum(expr.as<CallNode>());
    }
    return Downcast<Call>(expr);
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto pre_visit = [this](const LetNode* op) {
      auto var = op->var;
      auto value = this->Mutate(op->value);
      // Simplify let %a = %b .. scenarios
      if (auto value_var = value.as<VarNode>()) {
        mapping_[var] = GetRef<Var>(value_var);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      auto var = op->var;
      auto value = this->Mutate(op->value);
      auto body = this->Mutate(op->body);

      if (auto value_var = value.as<VarNode>()) {
        this->memo_[GetRef<Expr>(op)] = body;
      } else {
        this->memo_[GetRef<Expr>(op)] = Let(var, value, body);
      }
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

 private:
  const Op& sum_op_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> mapping_;
};

}  // namespace gradient

// Parse the requires_grad and create a map from VarNode to boolean flag.
std::unordered_map<const VarNode*, bool> ParseRequireGradsMain(
    IRModule mod, ir::Array<tvm::Bool> requires_grads) {
  std::unordered_map<const VarNode*, bool> requires_grads_main_map;
  auto func = Downcast<Function>(mod->Lookup("main"));
  if (requires_grads.size() == 0) {
    for (const Var& param : func->params) {
      requires_grads_main_map[param.get()] = true;
    }
  } else {
    CHECK_EQ(func->params.size(), requires_grads.size());
    for (int i = 0; i < func->params.size(); ++i) {
      const VarNode* var = func->params[i].operator->();
      requires_grads_main_map[var] = requires_grads[i];
    }
  }
  return requires_grads_main_map;
}

Pass AutoDiff(ir::Array<tvm::Bool> requires_grads) {
  TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m, PassContext pc) {
    // Copy the module to avoid any in-place module update
    ir::IRModule mod = ir::IRModule(m->functions);

    // Collect the global function to their gradient map. We will walk through the mod functions
    // and collect the gradient for each global function and store in this map. After all the global
    // vars are visited, we would update the global functions using map.
    std::unordered_map<const GlobalVarNode*, Function> gvar_to_grad_func;

    // Parse the requires_grad array to a map for the main function.
    auto requires_grads_main_map = ParseRequireGradsMain(mod, requires_grads);

    auto grad_computer = gradient::ReverseAD(requires_grads_main_map);

    // Traverse through the functions and call Gradient on everyone
    for (auto gvar_func_pair : mod->functions) {
      const GlobalVarNode* gvn = gvar_func_pair.first.get();
      if (gvar_func_pair.second.as<ir::FunctionNode>()) {
        const Function& func = Downcast<Function>(gvar_func_pair.second);
        // Perform AD on the function
        auto grad_func = grad_computer.Run(func);

        // Add the grad function to the our map
        gvar_to_grad_func.emplace(gvn, grad_func);
      }
    }

    // Now create the new module with updated grad functions
    for (const auto& it : gvar_to_grad_func) {
      auto gv = GetRef<GlobalVar>(it.first);
      auto func = it.second;
      mod->Add(gv, func, true);
    }

    // Run InferType to get the collapse_ais and canonicalize the IR.
    // TODO: Infer type may fail if the module parameters do not have type information.
    // This is a legacy issue, and all modules with this case were only tested by interpreter,
    // which does not require strong types. We should fix them to remove this exception.
    try {
      mod = InferType()(mod);
      for (const auto& it : mod->functions) {
        auto func = Downcast<Function>(gradient::Canonicalizer().Canonicalize(it.second));
        mod->Update(it.first, func);
      }
    } catch (const dmlc::Error& e) {
      LOG(WARNING) << "Failed to infer type after AutoDiff. This may lead to errors later with VM ";
    }
    return mod;
  };
  return CreateModulePass(pass_func, 1, "AutoDiff", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.AutoDiff").set_body_typed(AutoDiff);

}  // namespace pass
}  // namespace raf
