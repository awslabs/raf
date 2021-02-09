/*!
 * Copyright (c) 2020 by Contributors
 * \file ./src/op/dispatch/tvmjit/tvmjit_fusion.cc
 * \brief Implementation of tvmjit dispatch for fused functions
 */
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/pass.h"
#include "tvm/ir/type_functor.h"
#include "./tvmjit_utils.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::value;
using namespace mnm::ir;
using tvm::TypeFunctor;
using tvm::relay::TensorType;
using tvm::relay::TupleType;
using tvm::relay::Type;

class FakeValueCreator : public TypeFunctor<Value(const Type& n)> {
 public:
  FakeValueCreator(Device dev) : device_(dev) {
  }

  Value VisitType_(const TensorTypeNode* node) override {
    std::vector<int64_t> shape;
    DType dtype = DType(DLDataType(node->dtype));
    shape.reserve(node->shape.size());
    for (const auto& dim : node->shape) {
      if (const auto* idim = dim.as<IntImmNode>()) {
        shape.push_back(idim->value);
      } else {
        LOG(FATAL) << "unsupported type " << idim->GetTypeKey();
      }
    }
    return TensorValue::Assemble(device_, dtype, shape);
  }

  Value VisitType_(const TupleTypeNode* node) override {
    std::vector<Value> ret;
    for (const auto& field : node->fields) {
      ret.push_back(VisitType(field));
    }
    return TupleValue::make(Array<Value>(ret.begin(), ret.end()));
  }

 private:
  Device device_;
};

Value GetFakeValue(const Type& type, const Device& dev) {
  FakeValueCreator creator(dev);
  return creator(type);
}

/*! \brief Convert CallNode in a primitive function to CallValues,
           Intermediate results are filled with fake data */
class CallValuesGetter : public ExprMutator {
 public:
  CallValuesGetter(const CallValues& call)
      : func_(Downcast<ClosureValue>(call->callee)->func), device_(call->device) {
    Array<Value> args = GetListArgs(call->args);
    CHECK_EQ(args.size(), func_->params.size());
    size_t num = args.size();
    for (size_t i = 0; i < num; ++i) {
      vmap_[func_->params[i]] = args[i];
    }
  }

  void operator()() {
    VisitExpr(func_->body);
  }

  Expr VisitExpr_(const VarNode* node) override {
    Var var = GetRef<Var>(node);
    if (vmap_.find(var) != vmap_.end()) {
      RelayConstant constant = MakeConstant(vmap_.at(var));
      constant->checked_type_ = node->checked_type();
      return std::move(constant);
    }
    return var;
  }

  Expr VisitExpr_(const CallNode* node) override {
    const Op& op = Downcast<Op>(node->op);
    auto fschema = Op::GetAttrMap<op::FMNMSchema>("FMNMSchema")[op];
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(node));
    std::vector<Value> fake_args;
    for (size_t i = 0; i < node->args.size(); ++i) {
      Expr new_arg = VisitExpr(node->args[i]);
      if (const auto* cn = new_arg.as<ConstantNode>()) {
        fake_args.push_back(Downcast<Value>(cn->value));
      } else if (vmap_.find(new_arg) != vmap_.end()) {
        fake_args.push_back(vmap_.at(new_arg));
      } else {
        fake_args.push_back(GetFakeValue(node->args[i]->checked_type(), device_));
      }
    }
    CallValues op_call_values = CallValues::make({}, fschema(fake_args));
    CHECK(call_values.find(call) == call_values.end());
    call_values[GetRef<Call>(node)] = op_call_values;
    return std::move(call);
  }

  Expr VisitExpr_(const TupleGetItemNode* node) override {
    const auto* cn = node->tuple.as<ConstantNode>();
    if (cn) {
      auto value = Downcast<TupleValue>(cn->value);
      return MakeConstant(value->fields[node->index]);
    }
    return ExprMutator::VisitExpr_(node);
  }

 private:
  /*! \brief create a dummy value of a given type */
  Value CreateFakeValue(Type type);

 public:
  /*! \brief maps CallNode to CallValues, with fake intermediate results */
  std::unordered_map<Expr, CallValues, ObjectPtrHash, ObjectPtrEqual> call_values;

 private:
  /*! \brief maps from params to values */
  std::unordered_map<Expr, Value, ObjectPtrHash, ObjectPtrEqual> vmap_;
  /*! \brief The device to compile for. */
  Device device_;
  /*! \brief the primitive function to be analyzed */
  Function func_;
};

/*!
 * \brief Converter from meta style (all inputs are arguments) to
 *        tvm style (inputs are explicitly marked as arguments or attrs)
 */
class Meta2TVM : public ExprMutator {
 public:
  Meta2TVM(const CallValues& call)
      : func_(Downcast<ClosureValue>(call->callee)->func), call_values_getter_(call) {
  }

  Expr operator()() {
    call_values_getter_();
    Expr ret = VisitExpr(func_);
    return ret;
  }

  Expr VisitExpr(const Expr& expr) override {
    auto ret = ExprMutator::VisitExpr(expr);
    ret->checked_type_ = expr->checked_type();
    return ret;
  }

  Expr VisitExpr_(const VarNode* node) override {
    input_.insert(GetRef<Var>(node));
    return GetRef<Var>(node);
  }

  Expr VisitExpr_(const CallNode* node) override {
    CallValues op_call_values = call_values_getter_.call_values.at(GetRef<Call>(node));
    const Op& op = Downcast<Op>(node->op);
    auto farg_indices = Op::GetAttrMap<FMNMArgIndices>("FMNMArgIndices")[op];
    auto fschema = Op::GetAttrMap<op::FMNMSchema>("FMNMSchema")[op];
    auto fattr = Op::GetAttrMap<FMNMAttr>("FMNMAttr")[op];
    Attrs op_tvm_attr = fattr(op_call_values);
    ir::Array<tvm::IntImm> arg_indices = farg_indices(op_call_values);
    std::vector<Expr> inputs;
    for (const auto& i : arg_indices) {
      Expr arg = VisitExpr(node->args[i->value]);
      inputs.push_back(arg);
      if (const auto* vn = arg.as<VarNode>()) {
        input_.insert(GetRef<Var>(vn));
      }
    }
    return Call(node->op, inputs, op_tvm_attr);
  }

  Expr VisitExpr_(const FunctionNode* node) override {
    Expr new_body = VisitExpr(node->body);
    std::vector<Var> new_params;
    size_t num = node->params.size();
    for (size_t i = 0; i < num; ++i) {
      const Var& param = node->params[i];
      if (input_.find(param) != input_.end()) {
        // param is a tensor input
        new_params.push_back(param);
        arg_indices.push_back(i);
      }
    }
    return Function(Array<Var>(new_params), new_body, node->ret_type, {});
  }

 public:
  /*! \brief the indices of fused function params that correspond to tvm non-attr */
  std::vector<int> arg_indices;

 private:
  /*! \brief convert CallNode to CallValues */
  CallValuesGetter call_values_getter_;
  /*! \brief params that are tvm op inputs, instead of attrs */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> input_;
  /*! \brief the primitive function to be analyzed */
  Function func_;
};

OpEnv* FusedFuncBuild(const op::CallValues& call) {
  static auto engine = registry::GetPackedFunc("relay.backend._CompileEngineGlobal")();
  static auto c_cache_key = registry::GetPackedFunc("relay.backend._make_CCacheKey");
  static auto jit = registry::GetPackedFunc("relay.backend._CompileEngineJIT");
  static auto engine_clear = registry::GetPackedFunc("relay.backend._CompileEngineClear");
  auto env = std::make_unique<TVMOpEnv>();
  Device dev = call->device;
  tvm::Target target;
  if (dev.device_type == DevType::kCPU()) {
    target = tvm::Target("llvm");
  } else if (dev.device_type == DevType::kCUDA()) {
    target = tvm::Target("cuda");
  } else {
    LOG(FATAL) << "NotImplementedError: target is not supported " << dev.device_type.c_str();
    throw;
  }
  Meta2TVM meta_to_tvm(call);
  Function func = Downcast<Function>(meta_to_tvm());
  engine_clear(engine);
  env->f = jit(engine, c_cache_key(func, target));
  env->arg_indices = meta_to_tvm.arg_indices;
  Array<Value> args = GetListArgs(call->args);
  for (const int& i : env->arg_indices) {
    GetDLTensor(args[i], &env->inputs);
  }
  GetDLTensor(call->out, &env->outputs);
  return env.release();
}

MNM_FUNC_DISPATCH(FusedFuncBuild, DevType::kCPU(), "tvmjit");
MNM_FUNC_DISPATCH(FusedFuncBuild, DevType::kCUDA(), "tvmjit");

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
