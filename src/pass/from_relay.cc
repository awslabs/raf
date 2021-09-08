/*!
 * Copyright (c) 2020 by Contributors
 * \file from_relay.cc
 * \brief Build meta ir from Relay
 */
#include <map>
#include <tvm/relay/transform.h>
#include <tvm/support/with.h>
#include <relay/transforms/pattern_utils.h>
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "../op/dialect/tvm/tvm_attrs.h"

namespace mnm {
namespace pass {
namespace from_relay {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::tvm_dialect;
using tvm::TVMArgs;
using tvm::TVMRetValue;

struct RelayPattern {
 public:
  /*! \brief The converter to convert the matched Relay composite function to a Meta call.
   *  \param func The composite function to be converted.
   *  \param args The argument array of the converted op.
   */
  virtual Call converter(Function func, Array<Expr> args) = 0;

  /*! \brief The data flow pattern to match the Relay graph. */
  DFPattern pattern;
  /*! \brief The customized checker to further check if the matched pattern is valid. */
  PackedFunc check;
};

struct ConvertDense : RelayPattern {
 public:
  ConvertDense() {
    // The dense pattern that matches transposes of its inputs.
    DFPattern in_1 = IsWildcard();
    DFPattern in_2 = IsWildcard();
    in_1 = IsOp("transpose")({IsWildcard()}) || in_1;
    in_2 = IsOp("transpose")({IsWildcard()}) || in_2;
    this->pattern = IsOp("nn.dense")({in_1, in_2});

    // Checker always returns true
    this->check = PackedFunc([](TVMArgs args, TVMRetValue* rv) { *rv = true; });
  }

  Call converter(Function func, Array<Expr> args) {
    // nn.dense is equal to matmul_nt so the second input is already transposed
    bool transposes[] = {false, true};

    // Check possible transposes
    static Op op_transpose = Op::Get("transpose");
    auto dense = Downcast<Call>(func->body);
    for (size_t i = 0; i < 2; ++i) {
      auto trans_call = dense->args[i].as<CallNode>();
      if (trans_call && Downcast<Op>(trans_call->op) == op_transpose) {
        auto attrs = GetRef<Call>(trans_call)->attrs.as<TransposeAttrs>();
        if (attrs->axes.defined()) {
          auto axes = mnm::op::ArrayToInt(attrs->axes);
          for (size_t j = 0; j < 2; ++j) {  // Support negative axis
            axes[j] += (axes[j] < 0) ? 2 : 0;
          }
          transposes[i] = (axes[0] == 1 && axes[1] == 0) ? !transposes[i] : transposes[i];
        } else {  // Empty means reverse.
          transposes[i] = !transposes[i];
        }
      }
    }

    // Dispatch to the proper matmul based on the transposes
    std::string op_name = "mnm.op.matmul";
    if (transposes[0] || transposes[1]) {
      op_name += '_';
      op_name += (transposes[0]) ? 't' : 'n';
      op_name += (transposes[1]) ? 't' : 'n';
    }

    const Op& op = Op::Get(op_name);
    return Call(op, args);
  }
};

struct CompositeGelu : RelayPattern {
 public:
  CompositeGelu() {
    // Match Patterns
    DFPattern x_float64 = IsWildcard().HasDtype(DataType::Float(64));
    auto gelu_float64 = IsOp("multiply")({IsFloatExpr(64, 1 / sqrt(2)), x_float64});
    gelu_float64 = IsOp("erf")({gelu_float64});
    gelu_float64 = IsOp("multiply")({IsFloatExpr(64, 0.5), gelu_float64});
    gelu_float64 = IsOp("add")({IsFloatExpr(64, 0.5), gelu_float64});
    gelu_float64 = IsOp("multiply")({x_float64, gelu_float64});

    DFPattern x_float32 = IsWildcard().HasDtype(DataType::Float(32));
    auto gelu_float32 = IsOp("multiply")({IsFloatExpr(32, 1 / sqrt(2)), x_float32});
    gelu_float32 = IsOp("erf")({gelu_float32});
    gelu_float32 = IsOp("multiply")({IsFloatExpr(32, 0.5), gelu_float32});
    gelu_float32 = IsOp("add")({IsFloatExpr(32, 0.5), gelu_float32});
    gelu_float32 = IsOp("multiply")({x_float32, gelu_float32});

    // Match Patterns
    this->pattern = gelu_float64 || gelu_float32;

    // Checker always returns true
    this->check = PackedFunc([](TVMArgs args, TVMRetValue* rv) { *rv = true; });
  }

  Call converter(Function func, Array<Expr> args) {
    static const Op& op = Op::Get("mnm.op.gelu");
    return Call(op, args);
  }

 private:
  inline DFPattern IsFloatExpr(int bits, double value) {
    if (bits == 32) {
      return IsExpr(tvm::relay::MakeConstantScalar(DataType::Float(32), static_cast<float>(value)));
    }
    CHECK_EQ(bits, 64);
    return IsExpr(tvm::relay::MakeConstantScalar(DataType::Float(64), value));
  }
};

struct ConvertEmbedding : RelayPattern {
 public:
  ConvertEmbedding() {
    // First match all take ops with the second input casted or constant.
    auto cast_or_const = IsOp("cast")({IsWildcard()}) || IsConstant();
    this->pattern = IsOp("take")({IsWildcard(), cast_or_const});

    // Then check if it is from an embedding op.
    // TODO(comaniac): We do not have a better way to differentiate embedding ops from
    // Relay take ops. For example, in Relay PyTorch frontend, both embedding and size ops
    // may have mode=clip and axis=0. It means we may mis-convert a size op to an embedding op.
    // A long-term solution is introducing embedding op in Relay, or use our own PyTorch frontend
    // instead of relying on Relay frontned.
    this->check = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1U);
      Call take_call = args[0];

      *rv = false;

      // Check if the second argument is casted to int32. This is the pattern
      // specified in the Relay PyTorch frontend. Note that if the second argument
      // is generated by an init op, it has been folded by FoldConstant.
      auto dtype = take_call->args[1]->checked_type().as<TensorTypeNode>()->dtype;
      if (dtype.code() != kDLInt || dtype.bits() != 32) {
        return;
      }

      // Check take op attributes.
      const auto* relay_attrs = take_call->attrs.as<TakeAttrs>();
      *rv = relay_attrs->mode == "clip" &&
            (!relay_attrs->axis.defined() || relay_attrs->axis->value == 0);
    });
  }

  Call converter(Function func, Array<Expr> args) {
    LOG(WARNING) << "Converted a relay.take(data, indices, axis=0, mode=clip) to mnm.embedding, "
                 << "which requires all values in indices within the range; otherwise it will "
                 << "encounter memory error on GPUs";
    static const Op& op = Op::Get("mnm.op.embedding");

    // The second argument is a constant and has been embedded into the composite function.
    // In this case, we retrieve the second argument from the function body.
    if (args.size() == 1) {
      auto const_arg = Downcast<Call>(func->body)->args[1].as<ConstantNode>();
      CHECK(const_arg != nullptr);
      args.push_back(GetRef<Constant>(const_arg));
    }
    return Call(op, args);
  }
};

static const std::unordered_map<String, std::shared_ptr<RelayPattern>> composite_patterns{
    {String("dense"), std::shared_ptr<RelayPattern>(new ConvertDense)},
    {String("gelu"), std::shared_ptr<RelayPattern>(new CompositeGelu)},
    {String("embedding"), std::shared_ptr<RelayPattern>(new ConvertEmbedding)}};

// We set the parameters to be Meta model attributes, so their names
// have to be valid variable names in Python.
String ValidateRelayParamName(const String var_name) {
  auto name_str = std::string(var_name->data);
  std::replace(name_str.begin(), name_str.end(), '.', '_');
  return String(name_str);
}

struct FromRelayMutator : public ExprMutator {
 public:
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
      var_value_map_.Set(var, this->Mutate(op->value));
    };
    auto post_visit = [this](const LetNode* op) {
      Expr value = this->Mutate(op->value);
      Expr body = this->Mutate(op->body);
      if (body.as<VarNode>()) {
        body = MakeRet(Downcast<Var>(body));
      }

      auto expr = GetRef<Expr>(op);
      if (value->IsInstance<FunctionNode>() &&
          Downcast<Function>(value)->GetAttr<String>(attr::kComposite)) {
        // Discard composite functions because their callers will be substitited to an op
        this->memo_[expr] = body;
      } else {
        this->memo_[expr] = Let(var_map_[op->var], value, body);
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

  Expr VisitExpr_(const CallNode* node) final {
    // If this is a call node inside a composite function, then do not touch but just updating
    // its arguments.
    if (is_in_composite_func_) {
      tvm::Array<Expr> args;
      for (auto arg : node->args) {
        args.push_back(VisitExpr(arg));
      }
      return Call(node->op, args, node->attrs);
    }

    // If this node is calling a composite op, convert it to a Meta op using pattern converter
    bool is_composite_op = true;

    // Try to trace back to find the composite function
    Expr curr_op = node->op;
    const VarNode* var_node;
    while ((var_node = curr_op.as<VarNode>())) {
      auto var = GetRef<Var>(var_node);
      auto it = var_value_map_.find(var);
      if (it != var_value_map_.end()) {
        curr_op = (*it).second;
      } else {
        is_composite_op = false;
        break;
      }
    }
    // Convert the composite function call with the Meta op
    if (is_composite_op) {
      if (auto func = curr_op.as<FunctionNode>()) {
        if (auto comp_name = func->GetAttr<String>(attr::kComposite)) {
          auto comp_name_str = comp_name.value();
          CHECK_GT(composite_patterns.count(comp_name_str), 0)
              << "Unrecognized composite: " << comp_name_str;

          tvm::Array<Expr> call_args;
          for (auto arg : node->args) {
            auto new_arg = VisitExpr(arg);
            call_args.push_back(new_arg);
          }
          return composite_patterns.at(comp_name_str)->converter(GetRef<Function>(func), call_args);
        }
      }
    }

    // This node is calling a single op so convert it to a Meta op using the op converter
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
        auto new_expr = fmap[op](node->attrs, node->args, var_value_map_);
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

    // Mutate the function body. Note that if this is a composite function, then
    // we only visit the body without mutating the call nodes. The purpose is to make
    // sure we visit and mutate the embedded constant nodes inside composite functions.
    auto body = node->body;
    is_in_composite_func_ = bool(node->GetAttr<String>(attr::kComposite));
    body = Mutate(body);
    is_in_composite_func_ = false;
    return Function(params, body, {}, {}, node->attrs, node->span);
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
  /*! \brief Map from var in Relay graph to the converted Meta graph value. */
  Map<Var, Expr> var_value_map_;
  /*! \brief Map from unsupported op name to the appearance. */
  std::unordered_map<String, int> unsupported_ops_;
  /*! \brief Map from function parameters to their updated values */
  Map<Var, Expr> mutation_;
  /*! \brief The current let variable */
  Var curr_let_var_;
  /*! \brief Whether we are currently visiting a composite function body. */
  bool is_in_composite_func_ = false;
};

}  // namespace from_relay

Function PartitionPatterns(Function func) {
  Function ret = func;
  for (auto name_n_pattern : mnm::pass::from_relay::composite_patterns) {
    auto pattern = name_n_pattern.second;
    Map<String, ObjectRef> attrs;
    attrs.Set("Composite", name_n_pattern.first);
    attrs.Set("Primitive", Integer(1));
    ret = Downcast<Function>(PartitionPattern(pattern->pattern, ret, attrs, pattern->check));
  }
  return ret;
}

IRModule ApplyTransformSeq(const IRModule& mod) {
  std::vector<tvm::relay::transform::Pass> passes = {
      tvm::relay::transform::SimplifyExpr(), tvm::relay::transform::EliminateCommonSubexpr(),
      tvm::relay::transform::FoldConstant(), tvm::relay::transform::CombineParallelBatchMatmul()};
  auto seq = tvm::relay::transform::Sequential(passes);
  return seq(mod);
}

Pass FromRelay(Array<String> disabled_pass) {
  TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m, PassContext pc) {
    IRModule updated_mod = IRModule(m->functions, m->type_definitions, m->Imports(), m->source_map);

    // Apply Relay optimization passes before conversion
    auto pass_ctx = pass::PassContext::Create();
    pass_ctx->opt_level = 4;
    pass_ctx->disabled_pass = disabled_pass;
    {
      tvm::With<pass::PassContext> ctx_scope(pass_ctx);
      updated_mod = ApplyTransformSeq(updated_mod);
    }

    // Convert each function
    std::vector<std::pair<GlobalVar, Function>> updates;
    for (const auto& it : updated_mod->functions) {
      if (auto* n = it.second.as<FunctionNode>()) {
        Function func = GetRef<Function>(n);

        // Partition Meta-specific Relay simplify patterns
        auto updated_func = PartitionPatterns(func);

        // Transform to ANF and convert Relay ops to Meta ops
        auto anf_expr = tvm::relay::transform::ToANormalForm(updated_func);
        Let let = Downcast<Let>(anf_expr);
        auto mutator = from_relay::FromRelayMutator();
        updated_func = Downcast<Function>(mutator.Mutate(let->value));

        // Check unsupported ops
        auto unsupported_ops_str = mutator.ListUnsupportedOps();
        if (!unsupported_ops_str.empty()) {
          LOG(FATAL) << "One or more ops cannot be converted:\n" << unsupported_ops_str;
          throw;
        }
        updates.push_back({it.first, updated_func});
      }
    }

    for (const auto& pair : updates) {
      updated_mod->Add(pair.first, pair.second, true);
    }
    {
      tvm::With<pass::PassContext> ctx_scope(pass_ctx);
      updated_mod = DeadCodeElimination()(updated_mod);
    }
    return updated_mod;
  };

  return CreateModulePass(pass_func, 2, "FromRelay", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.FromRelay").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  Array<String> disabled_pass;
  if (args.size() == 1) {
    disabled_pass = args[0];
  }
  *rv = FromRelay(disabled_pass);
});

MNM_REGISTER_GLOBAL("mnm.pass_.validate_relay_param_name")
    .set_body_typed(from_relay::ValidateRelayParamName);
}  // namespace pass
}  // namespace mnm
