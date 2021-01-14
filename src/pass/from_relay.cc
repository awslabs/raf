/*!
 * Copyright (c) 2020 by Contributors
 * \file from_relay.cc
 * \brief Build meta ir from Relay
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "../op/dispatch/tvmjit/tvm_attrs.h"

namespace mnm {
namespace pass {
namespace from_relay {

using namespace mnm::ir;
using namespace mnm::value;
using namespace tvm;
using namespace ::tvm::relay;

#define MNM_OP_FROM_RELAY(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<op::FMNMFromRelay>("FMNMFromRelay", body)

TupleValue ArrarToIntTuple(const Array<IndexExpr> arr) {
  Array<Value> ret;
  for (const auto i : arr) {
    int64_t val = i.as<IntImmNode>()->value;
    ret.push_back(IntValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

Expr BinaryConverter(const Attrs& attrs, const Array<Expr>& args) {
  static const Op& op = Op::Get("mnm.op.add");
  Array<Expr> new_args = args;
  new_args.push_back(MakeConstant(NullValue<Value>()));
  new_args.push_back(MakeConstant(NullValue<Value>()));
  return Call(op, new_args);
}

MNM_OP_FROM_RELAY("add", BinaryConverter);

Expr UnaryConverter(const Attrs& attrs, const Array<Expr>& args) {
  static const Op& op = Op::Get("mnm.op.relu");
  return Call(op, args);
}

MNM_OP_FROM_RELAY("nn.relu", UnaryConverter);

MNM_OP_FROM_RELAY("nn.conv2d", [](const Attrs& attrs, const Array<Expr>& args) {
  static const Op& op = Op::Get("mnm.op.conv2d");
  Array<Expr> new_args = args;
  const auto* conv_2d_attr = attrs.as<Conv2DAttrs>();
  new_args.push_back(MakeConstant(ArrarToIntTuple(conv_2d_attr->strides)));
  new_args.push_back(MakeConstant(ArrarToIntTuple(conv_2d_attr->padding)));
  new_args.push_back(MakeConstant(ArrarToIntTuple(conv_2d_attr->dilation)));
  new_args.push_back(MakeConstant(IntValue::make(conv_2d_attr->groups)));
  new_args.push_back(MakeConstant(StringValue::make(conv_2d_attr->data_layout)));
  new_args.push_back(MakeConstant(StringValue::make(conv_2d_attr->kernel_layout)));
  if (conv_2d_attr->out_layout != "") {
    new_args.push_back(MakeConstant(StringValue::make(conv_2d_attr->out_layout)));
  } else {
    new_args.push_back(MakeConstant(StringValue::make(conv_2d_attr->data_layout)));
  }
  return Call(op, new_args);
});

MNM_OP_FROM_RELAY("nn.max_pool2d", [](const Attrs& attrs, const Array<Expr>& args) {
  static const Op& op = Op::Get("mnm.op.max_pool2d");
  Array<Expr> new_args = args;
  const auto* maxpool_2d_attr = attrs.as<MaxPool2DAttrs>();
  new_args.push_back(MakeConstant(ArrarToIntTuple(maxpool_2d_attr->pool_size)));
  new_args.push_back(MakeConstant(ArrarToIntTuple(maxpool_2d_attr->strides)));
  new_args.push_back(MakeConstant(ArrarToIntTuple(maxpool_2d_attr->padding)));
  new_args.push_back(MakeConstant(TupleValue::make({IntValue::make(1)})));
  new_args.push_back(MakeConstant(BoolValue::make(maxpool_2d_attr->ceil_mode)));
  new_args.push_back(MakeConstant(BoolValue::make(true)));
  new_args.push_back(MakeConstant(StringValue::make(maxpool_2d_attr->layout)));
  return Call(op, new_args);
});

MNM_OP_FROM_RELAY("nn.avg_pool2d", [](const Attrs& attrs, const Array<Expr>& args) {
  static const Op& op = Op::Get("mnm.op.avg_pool2d");
  Array<Expr> new_args = args;
  const auto* avgpool_2d_attr = attrs.as<AvgPool2DAttrs>();
  new_args.push_back(MakeConstant(ArrarToIntTuple(avgpool_2d_attr->pool_size)));
  new_args.push_back(MakeConstant(ArrarToIntTuple(avgpool_2d_attr->strides)));
  new_args.push_back(MakeConstant(ArrarToIntTuple(avgpool_2d_attr->padding)));
  new_args.push_back(MakeConstant(TupleValue::make({IntValue::make(1)})));
  new_args.push_back(MakeConstant(BoolValue::make(avgpool_2d_attr->ceil_mode)));
  new_args.push_back(MakeConstant(BoolValue::make(true)));
  new_args.push_back(MakeConstant(StringValue::make(avgpool_2d_attr->layout)));
  return Call(op, new_args);
});

struct FromRelayMutator : public ExprMutator {
 public:
  FromRelayMutator() {
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const LetNode* node) final {
    const Var& var = node->var;
    CHECK_EQ(var_map.count(var), 0) << "IR is malformed: cannot bind var twice";
    Var new_var = mnm::ir::MakeVar("a" + std::to_string(++num_bound_var), var->type_annotation);
    var_map.Set(var, new_var);
    return mnm::ir::Let(new_var, Mutate(node->value), Mutate(node->body));
  }

  Expr VisitExpr_(const CallNode* node) final {
    static auto fmap = Op::GetAttrMap<op::FMNMFromRelay>("FMNMFromRelay");
    CHECK(node->op.as<OpNode>() != nullptr) << "Callee is not an operator!";
    const Op& op = Downcast<Op>(node->op);
    if (fmap.count(op)) {
      Call new_call = Downcast<Call>(fmap[op](node->attrs, node->args));
      tvm::Array<Expr> call_args;
      for (auto arg : new_call->args) {
        auto new_arg = this->Mutate(arg);
        call_args.push_back(new_arg);
      }
      return Call(new_call->op, call_args);
    }
    LOG(FATAL) << "Cannot convert this operator '" << op->name << "'!";
    throw;
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    Array<Var> params;
    for (auto param : node->params) {
      Var new_param = mnm::ir::MakeVar(param->name_hint(), param->type_annotation);
      params.push_back(new_param);
      var_map.Set(param, new_param);
    }
    return Function(params, Mutate(node->body), node->ret_type, node->type_params);
  }

 private:
  int num_bound_var = 0;
  Map<Var, Expr> var_map;
};
}  // namespace from_relay
using namespace tvm;
using namespace tvm::relay;

tvm::ObjectRef FromRelay(tvm::ObjectRef obj) {
  if (obj->IsInstance<tvm::IRModuleNode>()) {
    auto mod = Downcast<tvm::IRModule>(obj);
    auto relay_mod = tvm::relay::transform::ToANormalForm()(mod);
    tvm::Map<ir::GlobalVar, ir::Function> functions;
    for (auto& kv : relay_mod->functions) {
      functions.Set(kv.first,
                    tvm::Downcast<ir::Function>(from_relay::FromRelayMutator().Mutate(kv.second)));
    }
    return ir::Module::make(functions);
  } else if (obj->IsInstance<ExprNode>()) {
    auto expr = Downcast<Expr>(obj);
    auto new_expr = tvm::relay::transform::ToANormalForm(expr);
    Let let = Downcast<Let>(new_expr);
    return from_relay::FromRelayMutator().Mutate(let->value);
  } else {
    LOG(FATAL) << "Unknown object type!";
    throw;
  }
}

MNM_REGISTER_GLOBAL("mnm.pass_.FromRelay").set_body_typed(FromRelay);
}  // namespace pass
}  // namespace mnm
