/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file partition_graph.cc
 * \brief Partition an input function into multiple functions according based
 * on the inserted annotation nodes (i.e. compiler_begin and compiler_end).
 * These nodes are used as boundaries to partition the Relay function into
 * multiple regions that can be offloaded to different accelerators/backends.
 *
 * Each of these paritioned functions, a.k.a regions, will be viewed as
 * external functions, and they will use the provided compiler for codegen.
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "./common.h"
#include "../op/dialect/tvm/tvm_attrs.h"

namespace raf {
namespace pass {
namespace partition_graph {

using namespace raf::ir;
using raf::op::tvm_dialect::CompilerAttrs;

static const Op& begin_op = CompilerBeginOp();
static const Op& end_op = CompilerEndOp();

class PartitionFunction : public Object {
 public:
  explicit PartitionFunction(std::string target, int target_cnt) : target_(target) {
    func_name_ = target + "_" + std::to_string(target_cnt);
  }

  /*!
   * \brief Add a non-annotated Expr to the Function, find out the inputs and outputs of
   * this partition function.
   * \param var The Var that the expr bind to.
   * \param expr The Expr to be inserted.
   */
  void AddExpr(Var var, Expr expr) {
    // Push the inputs and outputs into ins_ and outs_.
    outs_.push_back(var);
    if (expr.as<CallNode>()) {
      const CallNode* call = expr.as<CallNode>();
      for (auto& arg : call->args) {
        if (arg.as<VarNode>()) {
          ins_.insert(arg);
        } else if (arg.as<ConstantNode>()) {
          continue;
        } else {
          LOG(FATAL) << "NotImplementedError: only support Var and Constant as input for now";
        }
      }
    } else if (expr.as<TupleNode>()) {
      const TupleNode* tuple = expr.as<TupleNode>();
      for (auto& field : tuple->fields) {
        if (field.as<VarNode>()) {
          ins_.insert(field);
        } else if (field.as<ConstantNode>()) {
          continue;
        } else {
          LOG(FATAL) << "NotImplementedError: only support Var and Constant as input for now";
        }
      }
    }
    // Push the Expr into ell_.
    ell_.vars.push_back(var);
    ell_.exprs.push_back(expr);
  }

  /*!
   * \brief Export the partition functions into ExplicitLetList.
   * \param part_func_vars The map from old vars into func_named vars
   * \return The ExplicitLetList with partition function packed.
   */
  ExplicitLetList Export(
      std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> part_func_vars) {
    // Because anf will auto-capture the global vars, we don't need to push ins_ into params.
    // If the Var inside ins_ is inside the outs_, which indicate that this input is given by
    // the expr inside this function. Then replace the usage of old vars with func_named vars.
    Array<Var> params_array;
    for (auto& in_ : ins_) {
      if (find(outs_.begin(), outs_.end(), in_) == outs_.end()) {
        continue;
      } else {
        // Find all usages of old vars and replace them with func_named vars.
        for (size_t i = 0; i < ell_.exprs.size(); ++i) {
          auto expr = ell_.exprs[i];
          if (expr.as<CallNode>()) {
            auto call = expr.as<CallNode>();
            Array<Expr> new_args;
            for (auto arg : call->args) {
              if (arg.as<VarNode>() && Downcast<Var>(arg) == in_) {
                new_args.push_back(part_func_vars[Downcast<Var>(in_)]);
              } else {
                new_args.push_back(arg);
              }
            }
            Call new_call = Call(call->op, new_args, call->attrs);
            new_call->checked_type_ = call->checked_type_;
            ell_.exprs[i] = new_call;
          } else if (expr.as<TupleNode>()) {
            auto tuple = expr.as<TupleNode>();
            Array<Expr> new_fields;
            for (auto field : tuple->fields) {
              if (field.as<VarNode>() && Downcast<Var>(field) == in_) {
                new_fields.push_back(part_func_vars[Downcast<Var>(in_)]);
              } else {
                new_fields.push_back(field);
              }
            }
            Tuple new_tuple = Tuple(new_fields);
            new_tuple->checked_type_ = tuple->checked_type_;
            ell_.exprs[i] = new_tuple;
          } else {
            LOG(FATAL)
                << "NotImplementedError: only support rewrite args for CallNode and TupleNode";
          }
        }
      }
    }

    // Replace the vars with part_func named vars.
    CHECK_EQ(ell_.vars.size(), ell_.exprs.size());
    for (size_t i = 0; i < ell_.vars.size(); ++i) {
      ell_.vars[i] = part_func_vars[ell_.vars[i]];
    }

    // Surround the Values in the outs_ with a TupleNode. And replace the old
    // vars with part_func named vars.
    std::vector<Expr> out_tuple_fields;
    for (auto out_ : outs_) {
      out_tuple_fields.push_back(part_func_vars[Downcast<Var>(out_)]);
    }
    Tuple outs_tuple = Tuple(out_tuple_fields);
    std::string outs_var_name = func_name_ + "_outs";
    Var outs_var = Var(outs_var_name, {});
    ell_.vars.push_back(outs_var);
    ell_.exprs.push_back(outs_tuple);
    ell_.ret = outs_var;
    // Assemble the partition function.
    Expr body = ell_.AsExpr();
    auto func = Function(params_array, body, {}, {});
    func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, tvm::runtime::String(func_name_));
    func = WithAttr(std::move(func), attr::kCompiler, tvm::runtime::String(target_));

    // Insert the CallNode fot the function and TupleGetItemNode to get the outputs from partition
    // functions.
    ExplicitLetList ret_ell_;
    // Call the partition function and get the outputs TupleNode.
    auto func_call = Call(func, {}, Attrs());
    std::string ret_var_name = func_name_ + "_ret";
    Var ret_var = Var(ret_var_name, {});
    ret_ell_.vars.push_back(ret_var);
    ret_ell_.exprs.push_back(func_call);
    for (size_t i = 0; i < outs_.size(); ++i) {
      int index = i;
      // Check if the index is correctly correspond to the func_named vars and old vars.
      String expected_var_name = String(func_name_ + "_" + std::to_string(index));
      CHECK_EQ(part_func_vars[Downcast<Var>(outs_[i])].as<VarNode>()->name_hint(),
               expected_var_name);
      TupleGetItem tgi = TupleGetItem(ret_var, index, {});
      Var tgi_var = Downcast<Var>(outs_[i]);
      ret_ell_.vars.push_back(tgi_var);
      ret_ell_.exprs.push_back(tgi);
      if (i == outs_.size() - 1) {
        ret_ell_.ret = tgi_var;
      }
    }
    return ret_ell_;
  }

  /*! \brief The target that the partition function supported. */
  std::string target_;
  /*! \brief The function name of the partition function. */
  std::string func_name_;
  /*! \brief The LetNodes to construct the partition function. */
  ExplicitLetList ell_;
  /*! \brief The inputs to this partition function. */
  std::set<Expr> ins_;
  /*! \brief The outputs of this partition function. */
  std::vector<Expr> outs_;
};

/*!
 * \brief Check if an Expr is the boundary of an annotated region and get its target.
 * \param expr The Expr to be checked.
 * \return The pair of boundary type and its target.
 * begin - The begin boundary of a partition function.
 * end - The end boundary of a partition function.
 * body - The body of a partition function.
 * single - Single Node.
 */
std::pair<std::string, std::string> CheckAnnotationBoundary(Expr expr) {
  std::pair<std::string, std::string> boundary;
  if (expr.as<CallNode>()) {
    const CallNode* call = expr.as<CallNode>();
    if (call->op == end_op) {
      CHECK_EQ(call->args.size(), 1U);
      std::string end_target = call->attrs.as<CompilerAttrs>()->compiler;
      const CallNode* op_call = call->args[0].as<CallNode>();
      if (op_call->args[0].as<CallNode>() && op_call->args[0].as<CallNode>()->op == begin_op) {
        const CallNode* begin_call = op_call->args[0].as<CallNode>();
        std::string begin_target = begin_call->attrs.as<CompilerAttrs>()->compiler;
        CHECK_EQ(end_target, begin_target);
        boundary.first = "single";
        boundary.second = begin_target;
        return boundary;
      }
      boundary.first = "end";
      boundary.second = end_target;
      return boundary;
    } else if (call->args[0].as<CallNode>() && call->args[0].as<CallNode>()->op == begin_op) {
      const CallNode* begin_call = call->args[0].as<CallNode>();
      std::string begin_target = begin_call->attrs.as<CompilerAttrs>()->compiler;
      boundary.first = "begin";
      boundary.second = begin_target;
      return boundary;
    } else {
      boundary.first = "body";
      boundary.second = "";
      return boundary;
    }
  } else if (expr.as<TupleNode>()) {
    const TupleNode* tuple = expr.as<TupleNode>();
    if (tuple->fields[0].as<CallNode>() && tuple->fields[0].as<CallNode>()->op == begin_op) {
      const CallNode* begin_call = tuple->fields[0].as<CallNode>();
      std::string begin_target = begin_call->attrs.as<CompilerAttrs>()->compiler;
      boundary.first = "begin";
      boundary.second = begin_target;
      return boundary;
    } else {
      boundary.first = "body";
      boundary.second = "";
      return boundary;
    }
  } else {
    LOG(FATAL) << "NotImplementedError: only support CallNode and TupleNode for now";
  }
}

class Partitioner : public ExprRewriter {
 public:
  explicit Partitioner() {
  }

  Expr Rewrite_(const FunctionNode* func, const Expr& post) final {
    // Store the new generated function with ExplicitLetList ell.
    ExplicitLetList ell;
    std::unique_ptr<ExplicitLetList> ref_ell = ExplicitLetList::make(func->body);
    ell.ret = ref_ell->ret;

    size_t ell_n = ref_ell->exprs.size();
    std::list<PartitionFunction*> part_funcs;
    std::unordered_map<std::string, int> target_map;

    // Push the LetNodes into PartitionFunction.
    for (size_t i = 0; i < ell_n; ++i) {
      auto boundary = CheckAnnotationBoundary(ref_ell->exprs[i]);
      std::string target = boundary.second;
      if (boundary.first == "single") {
        continue;
      } else if (boundary.first == "begin") {
        if (target_map.find(boundary.second) == target_map.end()) {
          // The target has not appared before. Initiate the target and
          // its target_cnt.
          target_map[target] = -1;
        }
        ++target_map[target];
        int target_cnt = target_map[target];
        PartitionFunction* part_func = new PartitionFunction(target, target_cnt);
        Expr begin_expr = RemoveAnnotation(ref_ell->exprs[i], begin_op);
        part_func->AddExpr(ref_ell->vars[i], begin_expr);
        part_funcs.push_back(part_func);
      } else if (boundary.first == "body") {
        auto part_func = part_funcs.back();
        part_func->AddExpr(ref_ell->vars[i], ref_ell->exprs[i]);
      } else if (boundary.first == "end") {
        auto part_func = part_funcs.back();
        Expr end_expr = RemoveAnnotation(ref_ell->exprs[i], end_op);
        part_func->AddExpr(ref_ell->vars[i], end_expr);
      } else {
        LOG(FATAL) << "Unknown boundary indicator";
      }
    }

    // Mapping the outputs of PartitionFunctions. Key is the var inside the previous
    // IR, and Value is the func_named var inside the partition function.
    std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> part_func_vars;
    for (auto part_func : part_funcs) {
      for (size_t i = 0; i < part_func->outs_.size(); ++i) {
        auto out = Downcast<Var>(part_func->outs_[i]);
        CHECK_EQ(part_func_vars.count(out), 0U);
        std::string var_name = part_func->func_name_ + "_" + std::to_string(i);
        Var func_var = Var(var_name, {});
        part_func_vars[out] = func_var;
      }
    }

    // Reassemble these partition functions back into IR.
    auto part_func_iter = part_funcs.begin();
    for (size_t i = 0; i < ell_n; ++i) {
      auto boundary = CheckAnnotationBoundary(ref_ell->exprs[i]);
      if (boundary.first == "begin") {
        CHECK(!part_funcs.empty());
        auto part_func = part_funcs.front();
        auto part_func_ell = part_func->Export(part_func_vars);
        for (size_t j = 0; j < part_func_ell.vars.size(); ++j) {
          ell.vars.push_back(part_func_ell.vars[j]);
          ell.exprs.push_back(part_func_ell.exprs[j]);
        }
        // Insert LetNodes to pass the func_named vars back into old vars.
        part_funcs.pop_front();
      } else if (boundary.first == "body" || boundary.first == "end") {
        continue;
      } else if (boundary.first == "single") {
        auto single_expr = RemoveAnnotation(ref_ell->exprs[i], begin_op);
        single_expr = RemoveAnnotation(single_expr, end_op);
        ell.vars.push_back(ref_ell->vars[i]);
        ell.exprs.push_back(single_expr);
      } else {
        LOG(FATAL) << "Unknown boundary indicator";
      }
    }

    Expr new_body = ell.AsExpr();
    return Function(func->params, new_body, func->ret_type, func->type_params, func->attrs);
  }
};

Expr PartitionGraph(const Expr& expr) {
  Partitioner partitioner = Partitioner();
  return PostOrderRewrite(expr, &partitioner);
}

}  // namespace partition_graph

Pass PartitionGraph() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(partition_graph::PartitionGraph(f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "PartitionGraph", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.PartitionGraph").set_body_typed(PartitionGraph);

}  // namespace pass
}  // namespace raf
