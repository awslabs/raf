/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/tvm_fusion.cc
 * \brief Implementation of tvm dispatch for fused functions
 */
#include "raf/value.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "tvm/auto_scheduler/compute_dag.h"
#include "relay/backend/te_compiler.h"
#include "relay/backend/te_compiler_cache.h"
#include "./tvm_utils.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::value;
using namespace raf::ir;

class FakeValueCreator : public tvm::TypeFunctor<Value(const Type& n)> {
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
    readable_name_stream << "fused";
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
    auto fschema = GetOpAttr<op::FRAFSchema>(op, "FRAFSchema");
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(node));
    std::vector<Value> fake_args;
    std::string op_name = op->name;
    if (!op_name.compare(0, 11, "raf.op.tvm.")) {
      op_name = op_name.substr(11);
    }
    readable_name_stream << "_" << op_name;
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
  /*! \brief String stream for function name */
  std::ostringstream readable_name_stream;

 private:
  /*! \brief maps from params to values */
  std::unordered_map<Expr, Value, ObjectPtrHash, ObjectPtrEqual> vmap_;
  /*! \brief The device to compile for. */
  Device device_;
  /*! \brief the primitive function to be analyzed */
  Function func_;
};

/*! \brief Cast base and dialect ops to TVM dialect if possible. */
class Cast2TVMDialect : public ExprMutator {
 public:
  Expr VisitExpr(const Expr& expr) override {
    auto ret = ExprMutator::VisitExpr(expr);
    if (expr->checked_type_.defined()) {
      ret->checked_type_ = expr->checked_type_;
    }
    return ret;
  }

  Expr VisitExpr_(const OpNode* node) override {
    auto op = GetRef<Op>(node);
    auto base_op = IsDialectOp(op) ? GetBaseOp(op) : op;
    auto tvm_op = OpDialect::Lower(base_op, "tvm");
    if (tvm_op.defined()) {
      return tvm_op;
    }
    // No TVM op registered for this base op, just return the original op
    return op;
  }
};

/*!
 * \brief Converter from raf style (all inputs are arguments) to
 *        tvm style (inputs are explicitly marked as arguments or attrs)
 */
class RAF2TVM : public ExprMutator {
 public:
  RAF2TVM(const CallValues& call, const DevType& dev_type)
      : func_(Downcast<ClosureValue>(call->callee)->func),
        call_values_getter_(call),
        device_type_(dev_type) {
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
    ICHECK_EQ(GetDialect(op), "tvm")
        << "Encountered a non-TVM op in fused TVM closure: " << op->name;
    auto farg_indices = GetOpAttr<FRAFArgIndices>(op, "FRAFArgIndices");
    auto fattr = GetOpAttr<FRAFAttr>(op, "FRAFAttr");
    Attrs op_tvm_attr = fattr(op_call_values);
    Array<IntImm> arg_indices = farg_indices(op_call_values);
    std::vector<Expr> inputs;
    for (const auto& i : arg_indices) {
      Expr arg = VisitExpr(node->args[i->value]);
      inputs.push_back(arg);
      if (const auto* vn = arg.as<VarNode>()) {
        input_.insert(GetRef<Var>(vn));
      }
    }
    return Call(op, inputs, op_tvm_attr);
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
    func_name = call_values_getter_.readable_name_stream.str();
    return Function(Array<Var>(new_params), new_body, node->ret_type, {});
  }

 public:
  /*! \brief the indices of fused function params that correspond to tvm non-attr */
  std::vector<int> arg_indices;
  /*! \brief readable function name */
  std::string func_name;

 private:
  /*! \brief convert CallNode to CallValues */
  CallValuesGetter call_values_getter_;
  /*! \brief params that are tvm op inputs, instead of attrs */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> input_;
  /*! \brief the primitive function to be analyzed */
  Function func_;
  /*! \brief The device type */
  DevType device_type_;
};

HashKey HashFusedFunc(const Function& func) {
  HashKey key;
  key << raf::ir::AsText(func, true);
  return key;
}

OpEnv* FusedFuncBuild(const op::CallValues& call) {
  tvm::relay::tec::TECompiler te_compiler;
  auto env = std::make_unique<TVMOpEnv>();
  Device dev = call->device;

  // Determine cache
  MetaPersistCache<TVMModuleCacheEntry>* cache;
  if (dev.device_type() == DevType::kCPU()) {
    cache = &CacheBuildCpu;
  } else if (dev.device_type() == DevType::kCUDA()) {
    cache = &CacheBuildCuda;
  } else {
    LOG(FATAL) << "NotImplementedError: device is not supported " << dev.device_type().c_str();
    throw;
  }

  tvm::Target target = dev.tvm_target();
  CHECK(dev.device_type() == DevType::kCPU() || dev.device_type() == DevType::kCUDA())
      << "NotImplementedError: target is not supported " << dev.device_type().c_str();
  RAF2TVM raf_to_tvm(call, dev.device_type());
  Function func = Downcast<Function>(raf_to_tvm());
  env->env_name = TruncateName(GetUniqueName(raf_to_tvm.func_name));

  auto key = HashFusedFunc(Downcast<ClosureValue>(call->callee)->func);
  TVMModuleCacheEntry entry;
  if (const auto* compiled = cache->Get(key.byte_vector)) {
    entry = *compiled;
  } else {
    te_compiler->Clear();
    try {
      auto cached_key = tvm::relay::tec::CCacheKey(func, target);
      auto cached_func = te_compiler->Lower(cached_key);
      auto mod = tvm::build(cached_func->funcs, cached_key->target, Target(nullptr));
      entry = TVMModuleCacheEntry(mod, cached_func->prim_fn_var->name_hint);
      cache->Set(key.byte_vector, entry);
    } catch (const dmlc::Error& e) {
      if (!AllowJitFailure()) {
        LOG(FATAL) << "Failed to build a fused op " << env->env_name << ": " << e.what();
      }
    }
  }

  env->f = entry.GetFunction();
  env->arg_indices = raf_to_tvm.arg_indices;
  Array<Value> args = GetListArgs(call->args);
  for (const int& i : env->arg_indices) {
    GetDLTensor(args[i], &env->inputs);
  }
  GetDLTensor(call->out, &env->outputs);
  return env.release();
}

/*!
 * \brief Calculate the total computation GFLOPS required by a function.
 * \param call The call values, which callee is a ClosureValue that includes the target function.
 * \param param_types The type of function parameters.
 * \param ret_type The function return type.
 * \param device The device.
 * \return The calculated GFLOPS.
 */
float CalcFuncGFLOPS(const op::CallValues& call, const Array<Type>& param_types,
                     const Type& ret_type, const Device& device) {
  tvm::relay::tec::TECompiler compiler;
  // Create a new call value and cast ops in callee to TVM dialect
  auto new_call = op::CallValues::make();
  auto callee = Downcast<ClosureValue>(call->callee)->func;
  callee = Downcast<Function>(Cast2TVMDialect().Mutate(callee));
  new_call->callee = ClosureValue::make({}, callee);
  new_call->args = call->args;
  new_call->out = call->out;
  new_call->device = call->device;

  RAF2TVM raf_to_tvm(new_call, device.device_type());
  Function tvm_func = Downcast<Function>(raf_to_tvm());
  tvm::Target target = device.tvm_target();

  auto cache_key = tvm::relay::tec::CCacheKey(tvm_func, target);
  try {
    auto tensors = compiler->Lower(cache_key, "mod_calc_flops")->outputs;
    auto dag = tvm::auto_scheduler::ComputeDAG(tensors);
    return dag->flop_ct / 1e9;
  } catch (dmlc::Error& e) {
    LOG(WARNING) << "Failed to create ComputeDAG for " << raf::ir::AsText(tvm_func) << "\n"
                 << e.what();
  }
  return -1;
}

RAF_OP_ENV_MAKER("raf.op.tvm._fused_op", FusedFuncBuild);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
