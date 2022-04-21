/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file group_allgather.cc
 * \brief ZeRO optimzed graph, this pass group the cast, if there is, and allgather ops.
 */
#include "raf/pass.h"
#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"
#include "raf/op_utils.h"
#include "raf/dist_config.h"
#include "raf/communicator.h"
#include <string>

namespace raf {
namespace pass {
namespace group_comm {

using namespace raf::op;
using raf::distributed::DistConfig;
using namespace raf::distributed::communicator;
class CommGrouper : public ExprMutator {
 public:
  CommGrouper(const Function& func) : func_(func) {
    auto dcfg = DistConfig::Global();
    auto comm = GetGlobalCommunicator();
    local_rank_ = comm->local_rank;
    bucket_size_ = dcfg->group_bucket_size;
    auto ell = ExplicitLetList::make(func->body);
    auto ret = ell->exprs.back().as<TupleNode>();
    ret_var_ = ell->vars.back();
    for (int i = 2; i < ret->fields.size(); ++i) {
      params_.Set(Downcast<Var>(ret->fields[i]), Expr());
    }
    scopes_.emplace_back(new LetList);
  }

  Function Group() {
    if (params_.empty()) {
      return func_;
    }
    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    return Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
  }

  Expr VisitExpr_(const LetNode* node) {
    static Op zeros_op = Op::Get("raf.op.zeros");
    static Op group_cast = Op::Get("raf.op.group_cast");
    static Op group_allgather = Op::Get("raf.op._group_allgather");

    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      auto curr_var = node->var;
      auto value = VisitExpr(node->value);

      bool comm_node = false;

      Nodes re_nodes = MatchParamUpdateWithAllGather(node);

      Var update_var = re_nodes.update_var;
      if (update_var.defined()) {
        comm_node = true;
        auto slice_node = re_nodes.slice_node;
        auto gather_node = re_nodes.gather_node;
        auto cast_node = re_nodes.cast_node;
        auto add_node = re_nodes.add_node;

        auto gather_var = gather_node->var;
        int64_t size = common::shape_utils::GetElementNum(update_var);
        auto var_type = gather_var->checked_type_.as<TensorTypeNode>();
        CHECK(var_type != nullptr);
        Var zero_input;

        Expr allgather_input;
        Expr allgather_output;
        Call cast_call;
        Call slice_call;

        Call gather_call = Downcast<Call>(gather_node->value);
        Call add_call = Downcast<Call>(add_node->value);

        if (cast_node) {
          cast_allgather_ = true;
          cast_call = Downcast<Call>(cast_node->value);
          allgather_input = cast_call->args[0];
        } else {
          allgather_input = gather_call->args[0];
        }

        if (slice_node) {
          slice_call = Downcast<Call>(slice_node->value);
          slice_dic_[allgather_inputs_.size()] = slice_call->args[2];
          zero_input = scope->Push(
              Call(zeros_op,
                   {MakeConstant(ArrayToIntTuple(var_type->shape)),
                    MakeConstant(StringValue::make(DLDataType2String(var_type->dtype))),
                    MakeConstant(StringValue::make("cuda(" + std::to_string(local_rank_) + ")"))}));
          allgather_output = zero_input;
        } else {
          allgather_output = add_call->args[2];
        }

        if (curr_size_ + size < bucket_size_) {
          curr_size_ += size;
          allgather_inputs_.push_back(allgather_input);
          allgather_outputs_.push_back(allgather_output);
          update_params_.push_back(update_var);
        } else {
          curr_size_ = size;
          Var gather_input;
          if (cast_node) {
            auto cast_input = scope->Push(Tuple(allgather_inputs_));
            gather_input = scope->Push(Call(group_cast, {cast_input, cast_call->args[1]}));
          } else {
            gather_input = scope->Push(Tuple(allgather_inputs_));
          }
          auto gather_output = scope->Push(Tuple(allgather_outputs_));
          auto output = scope->Push(
              Call(group_allgather, {gather_input, gather_call->args[1], gather_output}));
          for (int i = 0; i < allgather_inputs_.size(); ++i) {
            auto out_tensor = scope->Push(TupleGetItem(output, i));
            if (slice_dic_.count(i)) {
              out_tensor = scope->Push(
                  Call(slice_op_,
                       {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                        slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(1)}))}));
            }
            params_.Set(update_params_[i], out_tensor);
          }

          allgather_inputs_ = {allgather_input};
          allgather_outputs_ = {allgather_output};
          if (slice_node) {
            slice_dic_.clear();
            slice_dic_[allgather_inputs_.size() - 1] = slice_call->args[2];
          }
          update_params_ = {update_var};
        }
        node = add_node;
      } else if (curr_var == ret_var_) {
        comm_node = true;
        if (allgather_inputs_.size() > 1) {
          auto gather_input = scope->Push(Tuple(allgather_inputs_));
          auto gather_output = scope->Push(Tuple(allgather_outputs_));
          if (cast_allgather_) {
            gather_input = scope->Push(
                Call(group_cast, {gather_input, MakeConstant(StringValue::make("float16"))}));
          }
          auto output = scope->Push(Call(
              group_allgather, {gather_input, MakeConstant(ScalarValue::make(0)), gather_output}));
          for (int i = 0; i < allgather_inputs_.size(); ++i) {
            auto out_tensor = scope->Push(TupleGetItem(output, i));
            if (slice_dic_.count(i)) {
              out_tensor = scope->Push(
                  Call(slice_op_,
                       {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                        slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(1)}))}));
            }
            params_.Set(update_params_[i], out_tensor);
          }
        }
        Array<Expr> tuple;
        auto ret_value = value.as<TupleNode>();
        tuple.push_back(ret_value->fields[0]);
        tuple.push_back(ret_value->fields[1]);
        for (int j = 2; j < ret_value->fields.size(); ++j) {
          auto key = Downcast<Var>(ret_value->fields[j]);
          if (params_[key].defined()) {
            tuple.push_back(params_[key]);
          } else {
            tuple.push_back(key);
          }
        }
        scope->Push(curr_var, Tuple(tuple));
      }
      if (comm_node == false) {
        scope->Push(curr_var, value);
      }
      body = node->body;
      node = body.as<LetNode>();

    } while (node);
    auto ret = scopes_.back()->Get(this->Mutate(body));
    scopes_.pop_back();
    return ret;
  }

 private:
  struct Nodes {
    Var update_var;
    const LetNode* cast_node;
    const LetNode* slice_node;
    const LetNode* gather_node;
    const LetNode* add_node;
  };
  // TODO @zhen-jia we will have an ANF-based pattern matching mechanism in the future.
  inline Nodes MatchParamUpdateWithAllGather(const LetNode* node) {
    const LetNode* gather_node = nullptr;
    const LetNode* cast_node = nullptr;
    const LetNode* visit_node = nullptr;
    if (IsOp(node, cast_op_)) {
      // Matching cast -> allgather -> update parameter throguht in place update add
      cast_node = node;
      gather_node = node->body.as<LetNode>();
      if (IsOp(gather_node, allgather_op_)) {
        visit_node = gather_node->body.as<LetNode>();
      }
    } else if (IsOp(node, allgather_op_)) {
      // Matching allgather -> update parameter throguht in place update add
      gather_node = node;
      visit_node = node->body.as<LetNode>();
    }
    if (visit_node) {
      auto result_nodes = FindUpdateVar(visit_node);
      Var update_var = result_nodes.update_var;
      auto slice_node = result_nodes.slice_node;
      auto add_node = result_nodes.add_node;
      return Nodes{update_var, cast_node, slice_node, gather_node, add_node};
    }
    return Nodes{NullValue<Var>(), nullptr, nullptr, nullptr, nullptr};
  }

  inline Nodes FindUpdateVar(const LetNode* node) {
    if (IsOp(node, slice_op_)) {
      // Machinig stride_slice followed by add node
      auto add_node = node->body.as<LetNode>();
      if (IsOp(add_node, add_op_)) {
        auto update_var = add_node->var;
        if (params_.count(update_var)) {
          return Nodes{update_var, nullptr, node, nullptr, add_node};
        }
      }
    } else {
      // no slice op, only matching add node
      if (IsOp(node, add_op_)) {
        auto update_var = node->var;
        if (params_.count(update_var)) {
          return Nodes{update_var, nullptr, nullptr, nullptr, node};
        }
      }
    }
    return Nodes{NullValue<Var>(), nullptr, nullptr, nullptr, nullptr};
  }

  inline bool IsOp(const LetNode* node, Op op) {
    if (node->value.as<CallNode>()) {
      auto call = Downcast<Call>(node->value);
      auto opn = Downcast<Op>(call->op);
      if (opn == op) {
        return true;
      }
    }
    return false;
  }

  /*! \brief The target function. */
  Function func_;
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The parameters of the target function. */
  Map<Var, Expr> params_;
  /*! \brief The inputs of allgather. */
  std::vector<Expr> allgather_inputs_;
  /*! \brief The outputs of allgather. */
  std::vector<Expr> allgather_outputs_;
  /*! \brief The parameters need to be updated. */
  std::vector<Var> update_params_;
  /*! \brief Track the tensors that need to be sliced. */
  std::unordered_map<size_t, Expr> slice_dic_;
  /*! \brief Group bucket size. */
  size_t bucket_size_;
  /*! \brief The current bucket size for the group. */
  size_t curr_size_ = 0;
  /*! \brief whether has cast op before allgather. */
  bool cast_allgather_ = false;
  /*! \brief The return var. */
  Var ret_var_;
  /*! \brief Local rank. */
  int local_rank_;
  // ops using in this pass
  Op add_op_ = Op::Get("raf.op.add");
  Op allgather_op_ = Op::Get("raf.op._allgather");
  Op slice_op_ = Op::Get("raf.op.strided_slice");
  Op cast_op_ = Op::Get("raf.op.cast");
};
}  // namespace group_comm

Pass GroupAllgather() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return group_comm::CommGrouper(f).Group(); };
  auto group_allgather_pass = CreateRAFFunctionPass(pass_func, 0, "GroupAllgather", {});

  return RAFSequential({InferType(), group_allgather_pass, InferType()}, "GroupAllgather");
}  // namespace pass

RAF_REGISTER_GLOBAL("raf.pass_.GroupAllgather").set_body_typed(GroupAllgather);

}  // namespace pass
}  // namespace raf
