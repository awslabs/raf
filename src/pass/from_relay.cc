/*!
 * Copyright (c) 2020 by Contributors
 * \file from_relay.cc
 * \brief Build meta ir from Relay
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {
namespace from_relay {

using namespace mnm::ir;
using namespace mnm::value;
using namespace tvm;
using namespace ::tvm::relay;

struct FromRelayMutator : public ExprMutator {
 public:
  FromRelayMutator() {
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map_.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const LetNode* node) final {
    const Var& var = node->var;
    CHECK_EQ(var_map_.count(var), 0) << "IR is malformed: cannot bind the same var twice";
    Var new_var = mnm::ir::MakeVar("a" + std::to_string(++num_bound_var_), var->type_annotation);
    var_map_.Set(var, new_var);
    return mnm::ir::Let(new_var, Mutate(node->value), Mutate(node->body));
  }

  Expr VisitExpr_(const CallNode* node) final {
    static auto fmap = Op::GetAttrMap<op::FMNMFromRelay>("FMNMFromRelay");
    if (node->op.as<OpNode>() == nullptr) {
      return Call(node->op, node->args, node->attrs);
    }

    const Op& op = Downcast<Op>(node->op);
    if (fmap.count(op)) {
      try {
        Call new_call = Downcast<Call>(fmap[op](node->attrs, node->args));
        tvm::Array<Expr> call_args;
        for (auto arg : new_call->args) {
          auto new_arg = this->Mutate(arg);
          call_args.push_back(new_arg);
        }
        return Call(new_call->op, call_args);
      } catch (const dmlc::Error& e) {
        LOG(WARNING) << e.what();
      }
    }

    // Return the orignial Relay call and make a record for unsupported ops
    unsupported_ops_[op->name]++;
    return Call(node->op, node->args, node->attrs);
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    Array<Var> params;
    for (auto param : node->params) {
      Var new_param = mnm::ir::MakeVar(param->name_hint(), param->type_annotation);
      params.push_back(new_param);
      var_map_.Set(param, new_param);
    }
    return Function(params, Mutate(node->body), node->ret_type, node->type_params);
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
  Map<Var, Expr> var_map_;
  /*! \brief Map from unsupported op name to the appearance. */
  std::unordered_map<String, int> unsupported_ops_;
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
      functions.Set(kv.first, tvm::Downcast<ir::Function>(mutator.Mutate(kv.second)));
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
}  // namespace pass
}  // namespace mnm
