/*!
 * Copyright (c) 2020 by Contributors
 * \file from_relay.cc
 * \brief Build meta ir from Relay
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace from_relay {

using namespace mnm::ir;
using namespace mnm::value;
using namespace tvm;
using namespace ::tvm::relay;

// We set the parameters to be Meta model attributes, so their names
// have to be valid variable names in Python.
String ValidateRelayParamName(const String var_name) {
  auto name_str = std::string(var_name->data);
  std::replace(name_str.begin(), name_str.end(), '.', '_');
  return String(name_str);
}

struct FromRelayMutator : public ExprMutator {
 public:
  FromRelayMutator() {
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map_.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const RelayConstantNode* node) final {
    // check if it is a Meta Constant
    static const auto fake_tensor = MakeConstant(NullValue<Value>());
    if (node->data->data == fake_tensor->data->data) {
      return GetRef<Expr>(node);
    }
    static const auto& from_tvm = registry::GetPackedFunc("mnm.value.FromTVM");
    auto tv = from_tvm(node->data);
    return MakeConstant(tv);
  }

  Expr VisitExpr_(const LetNode* node) final {
    auto pre_visit = [this](const LetNode* op) {
      const Var& var = op->var;
      CHECK_EQ(var_map_.count(var), 0) << "IR is malformed: cannot bind the same var twice";
      Var new_var = mnm::ir::MakeVar("a" + std::to_string(++num_bound_var_), var->type_annotation);
      var_map_.Set(var, new_var);
      curr_let_var_ = new_var;
      this->Mutate(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Expr value = this->Mutate(op->value);
      Expr body = this->Mutate(op->body);
      if (body.as<VarNode>()) {
        body = MakeRet(Downcast<Var>(body));
      }
      this->memo_[GetRef<Expr>(op)] = Let(var_map_[op->var], value, body);
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

  Expr VisitExpr_(const CallNode* node) final {
    static auto fmap = Op::GetAttrMap<op::FMNMFromRelay>("FMNMFromRelay");
    static auto fmutation = Op::GetAttrMap<op::FMNMMutationFromRelay>("FMNMMutationFromRelay");
    if (node->op.as<OpNode>() == nullptr) {
      tvm::Array<Expr> call_args;
      for (auto arg : node->args) {
        auto new_arg = this->Mutate(arg);
        call_args.push_back(new_arg);
      }
      return Call(this->Mutate(node->op), call_args, node->attrs);
    }

    const Op& op = Downcast<Op>(node->op);
    Call res;
    if (fmap.count(op)) {
      try {
        auto new_expr = fmap[op](node->attrs, node->args);
        if (new_expr.as<CallNode>()) {
          Call new_call = Downcast<Call>(new_expr);
          tvm::Array<Expr> call_args;
          for (auto arg : new_call->args) {
            auto new_arg = this->Mutate(arg);
            call_args.push_back(new_arg);
          }
          res = Call(new_call->op, call_args);
        } else {
          return this->Mutate(new_expr);
        }
      } catch (const dmlc::Error& e) {
        LOG(WARNING) << e.what();
        // Return the orignial Relay call and make a record for unsupported ops
        unsupported_ops_[op->name]++;
        return Call(node->op, node->args, node->attrs);
      }
    }
    CHECK(res.defined());
    if (fmutation.count(op)) {
      Array<Array<Expr>> mutations = fmutation[op](curr_let_var_, res);
      for (const auto& mutation : mutations) {
        CHECK_EQ(mutation.size(), 2U);
        mutation_.Set(Downcast<Var>(mutation[0]), mutation[1]);
      }
    }
    return res;
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    Array<Var> params;
    for (auto param : node->params) {
      auto name_hint = ValidateRelayParamName(param->name_hint());
      Var new_param = mnm::ir::MakeVar(name_hint, param->type_annotation);
      params.push_back(new_param);
      var_map_.Set(param, new_param);
    }
    return Function(params, Mutate(node->body), {}, {});
  }

  Expr MakeRet(Var ret) {
    if (mutation_.size() == 0U) {
      return ret;
    }
    Expr body = LetList::With([&](LetList* ll) {
      Array<Expr> res{ret};
      for (const auto& kv : mutation_) {
        Var var = mnm::ir::MakeVar("a" + std::to_string(++num_bound_var_), {}, kv.first);
        ll->Push(var, kv.second);
        res.push_back(var);
      }
      return ll->Push(mnm::ir::MakeVar("a" + std::to_string(++num_bound_var_), {}), Tuple(res));
    });
    return body;
  }

  /*!
   * \brief Concat unsupported ops and their appearance to a string.
   * \return A string of unsupported ops, or empty if none.
   */
  std::string ListUnsupportedOps() {
    if (unsupported_ops_.empty()) {
      return "";
    }

    std::stringstream ss;
    for (auto pair : unsupported_ops_) {
      ss << "Failed to convert " << pair.first << " (appear " << pair.second << " times)\n";
    }
    return ss.str();
  }

 private:
  /*! \brief The counter of bound variables. */
  int num_bound_var_ = 0;
  /*! \brief Map from var in Relay graph to the converted Meta graph. */
  Map<Var, Var> var_map_;
  /*! \brief Map from unsupported op name to the appearance. */
  std::unordered_map<String, int> unsupported_ops_;
  /*! \brief Map from function parameters to their updated values */
  Map<Var, Expr> mutation_;
  /*! \brief The current let variable */
  Var curr_let_var_;
};
}  // namespace from_relay
using namespace tvm;
using namespace tvm::relay;

tvm::ObjectRef FromRelay(tvm::ObjectRef obj) {
  if (obj->IsInstance<tvm::IRModuleNode>()) {
    auto mod = Downcast<tvm::IRModule>(obj);
    auto relay_mod = tvm::relay::transform::ToANormalForm()(mod);
    tvm::Map<ir::GlobalVar, ir::Function> functions;
    std::stringstream unsupported_ops_ss;
    for (auto& kv : relay_mod->functions) {
      auto mutator = from_relay::FromRelayMutator();
      auto expr = mutator.Mutate(kv.second);
      functions.Set(kv.first, tvm::Downcast<ir::Function>(expr));
      unsupported_ops_ss << mutator.ListUnsupportedOps();
    }
    if (unsupported_ops_ss.rdbuf()->in_avail() > 0) {
      LOG(FATAL) << "One or more ops cannot be converted:\n" << unsupported_ops_ss.str();
      throw;
    }
    return ir::Module::make(functions);
  } else if (obj->IsInstance<ExprNode>()) {
    auto expr = Downcast<Expr>(obj);
    auto new_expr = tvm::relay::transform::ToANormalForm(expr);
    Let let = Downcast<Let>(new_expr);
    auto mutator = from_relay::FromRelayMutator();
    auto ret = mutator.Mutate(let->value);
    auto unsupported_ops_str = mutator.ListUnsupportedOps();
    if (!unsupported_ops_str.empty()) {
      LOG(FATAL) << "One or more ops cannot be converted:\n" << unsupported_ops_str;
      throw;
    }
    return ret;
  } else {
    LOG(FATAL) << "Unknown object type: " << obj->GetTypeKey() << ". Expected IRModule or Expr";
    throw;
  }
}

MNM_REGISTER_GLOBAL("mnm.pass_.FromRelay").set_body_typed(FromRelay);
MNM_REGISTER_GLOBAL("mnm.pass_.validate_relay_param_name")
    .set_body_typed(from_relay::ValidateRelayParamName);
}  // namespace pass
}  // namespace mnm
