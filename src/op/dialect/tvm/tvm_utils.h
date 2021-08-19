/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dialect/tvm/tvm_utils.h
 * \brief Utility methods for TVM dialect.
 */
#pragma once
#include <vector>
#include <memory>
#include "dlpack/dlpack.h"
#include "tvm/relay/transform.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/c_runtime_api.h"
#include "relay/backend/te_compiler.h"
#include "relay/backend/te_compiler_cache.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::value;

void GetDLTensor(const Value& v, std::vector<DLTensor>* tensors);
ir::Type GetTensorType(const DLTensor& dlt);
ir::Type GetTupleType(const std::vector<DLTensor>& dlts);
ir::Function LowerOp(const ir::Op& op, const ir::Attrs& attrs,
                     const std::vector<ir::Type>& param_types, const ir::Type& ret_type);
float CalcFuncGFLOPS(const op::CallValues& call, const Array<Type>& param_types,
                     const Type& ret_type, const Device& device);

class TVMOpEnv : public op::OpEnv {
 public:
  std::string env_name;
  std::vector<DLTensor> inputs;
  std::vector<DLTensor> outputs;
  registry::PackedFunc f{nullptr};

  TVMOpEnv() = default;
  virtual ~TVMOpEnv() = default;
  std::string name() const override {
    return env_name;
  }
  void Execute(const op::CallValues& call) override;
  void Execute(const std::vector<Value>& inputs, Value outputs) override;
};

template <class Unused>
HashKey GenericHasher(const std::vector<ir::Type>& param_types, const ir::Type& ret_type,
                      const Unused* args) {
  HashKey key;
  for (int i = 0, n = param_types.size(); i < n; ++i) {
    key << ir::Downcast<ir::TensorType>(param_types[i]);
  }
  if (const auto tuple = ret_type.as<ir::TupleTypeNode>()) {
    for (int i = 0, n = tuple->fields.size(); i < n; ++i) {
      key << ir::Downcast<ir::TensorType>(tuple->fields[i]);
    }
  } else {
    key << ir::Downcast<ir::TensorType>(ret_type);
  }
  return key;
}

/*!
 * \brief Return whether the auto scheduler task extraction mode is enabled in the pass context.
 */
inline bool AllowJitFailure() {
  return tvm::relay::transform::PassContext::Current()
      ->GetConfig<tvm::Bool>("mnm.tvm.allow_jit_failure", tvm::Bool(false))
      .value();
}

using FMNMLower = registry::TypedPackedFunc<ir::Function(const CallValues& call)>;
using FMNMAttr = registry::TypedPackedFunc<ir::Attrs(const CallValues& call)>;
using FMNMArgIndices =
    registry::TypedPackedFunc<ir::Array<tvm::IntImm>(const op::CallValues& call)>;

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm

#define MNM_TVM_PLEVEL(OP, FUNC, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH,        \
                       OP_PATTERN, PLEVEL)                                                         \
  MetaCache<registry::PackedFunc> FUNC##CacheBuildCpu;                                             \
  MetaCache<registry::PackedFunc> FUNC##CacheBuildCuda;                                            \
  MetaCache<ir::Function> FUNC##CacheLoweredFunc;                                                  \
  template <typename RType>                                                                        \
  inline RType FUNC##CacheCompile(TVMOpEnv* env, const op::CallValues& call,                       \
                                  MetaCache<RType>* cache,                                         \
                                  std::function<RType(const ir::Function&)> f_post_lower) {        \
    static const auto op = Op::Get(MNM_DIALECT_OP_NAME(tvm, OP));                                  \
    const auto* schema = call->args.as<SCHEMA>();                                                  \
    CHECK(schema != nullptr);                                                                      \
    Attrs attrs = SCHEMA2ATTRS(schema);                                                            \
    for (auto arg : SCHEMA2ARGS(schema)) {                                                         \
      GetDLTensor(arg, &env->inputs);                                                              \
    }                                                                                              \
    GetDLTensor(call->out, &env->outputs);                                                         \
    std::vector<Type> param_types;                                                                 \
    Type ret_type;                                                                                 \
    for (auto tensor : env->inputs) {                                                              \
      param_types.push_back(GetTensorType(tensor));                                                \
    }                                                                                              \
    if (env->outputs.size() == 1) {                                                                \
      ret_type = GetTensorType(env->outputs[0]);                                                   \
    } else {                                                                                       \
      ret_type = GetTupleType(env->outputs);                                                       \
    }                                                                                              \
    RType ret{nullptr};                                                                            \
    HashKey key = HASH(param_types, ret_type, schema);                                             \
    if (const auto* compiled = cache->Get(key.byte_vector)) {                                      \
      ret = *compiled;                                                                             \
    } else {                                                                                       \
      auto lowered = LowerOp(op, attrs, param_types, ret_type);                                    \
      ret = f_post_lower(lowered);                                                                 \
      cache->Set(key.byte_vector, ret);                                                            \
    }                                                                                              \
    return ret;                                                                                    \
  }                                                                                                \
  OpEnv* FUNC##Build(const op::CallValues& call) {                                                 \
    tvm::relay::tec::TECompiler te_compiler;                                                       \
    const auto& dev = call->device;                                                                \
    static const auto base_op = Op::Get(MNM_BASE_OP_NAME(OP));                                     \
    auto env = new TVMOpEnv();                                                                     \
    auto fschema_index = Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");         \
    for (auto field : SCHEMA_ARG_NAMES(call)) {                                                    \
      int idx = fschema_index[base_op](field);                                                     \
      CHECK_GE(idx, 0) << "Cannot find " << field << " in the schema for OP";                      \
      env->arg_indices.push_back(idx);                                                             \
    }                                                                                              \
    /* Determine cache */                                                                          \
    MetaCache<registry::PackedFunc>* cache;                                                        \
    if (dev.device_type() == DevType::kCPU()) {                                                    \
      cache = &FUNC##CacheBuildCpu;                                                                \
    } else if (dev.device_type() == DevType::kCUDA()) {                                            \
      cache = &FUNC##CacheBuildCuda;                                                               \
    } else {                                                                                       \
      LOG(FATAL) << "NotImplementedError: device is not supported " << dev.device_type().c_str();  \
      throw;                                                                                       \
    }                                                                                              \
    tvm::Target target = dev.tvm_target();                                                         \
    env->env_name = TruncateName(GetUniqueName(MNM_DIALECT_OP_NAME(tvm, OP)));                     \
    std::function<registry::PackedFunc(const ir::Function&)> f_post_lower(                         \
        [&](const ir::Function& f) {                                                               \
          te_compiler->Clear();                                                                    \
          return te_compiler->JIT(tvm::relay::tec::CCacheKey(f, target));                          \
        });                                                                                        \
    try {                                                                                          \
      env->f = FUNC##CacheCompile(env, call, cache, f_post_lower);                                 \
    } catch (const dmlc::Error& e) {                                                               \
      /* Invalid implementation. Return nullptr to let dispatcher select the next one */           \
      if (!AllowJitFailure()) {                                                                    \
        DLOG(ERROR) << "Failed to JIT " << env->env_name << ": " << e.what();                      \
        return nullptr;                                                                            \
      }                                                                                            \
    }                                                                                              \
    return env;                                                                                    \
  }                                                                                                \
  Attrs FUNC##Attr(const op::CallValues& call) {                                                   \
    static const auto op = Op::Get(MNM_BASE_OP_NAME(OP));                                          \
    const auto* schema = call->args.as<SCHEMA>();                                                  \
    CHECK(schema != nullptr);                                                                      \
    return SCHEMA2ATTRS(schema);                                                                   \
  }                                                                                                \
  Array<tvm::IntImm> FUNC##ArgIndices(const op::CallValues& call) {                                \
    static const auto op = Op::Get(MNM_BASE_OP_NAME(OP));                                          \
    static const auto fschema_index =                                                              \
        Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex")[op];                      \
    std::vector<tvm::IntImm> ret;                                                                  \
    for (const auto& field : SCHEMA_ARG_NAMES(call)) {                                             \
      ret.push_back(tvm::IntImm(DataType::Int(32), fschema_index(field)));                         \
    }                                                                                              \
    return Array<tvm::IntImm>(ret);                                                                \
  }                                                                                                \
  ir::Function FUNC##Lower(const op::CallValues& call) {                                           \
    static const std::function<ir::Function(const ir::Function&)> identity(                        \
        [](const ir::Function& f) { return f; });                                                  \
    MetaCache<ir::Function>* cache;                                                                \
    cache = &FUNC##CacheLoweredFunc;                                                               \
    auto env = std::make_unique<TVMOpEnv>();                                                       \
    return FUNC##CacheCompile(env.get(), call, cache, identity);                                   \
  }                                                                                                \
  MNM_REGISTER_DIALECT_OP(tvm, OP)                                                                 \
      .set_attr<::mnm::op::TOpPattern>("TOpPattern", OP_PATTERN)                                   \
      .set_attr<::mnm::op::tvm_dialect::FMNMLower>("FMNMLower", FUNC##Lower)                       \
      .set_attr<::mnm::op::tvm_dialect::FMNMAttr>("FMNMAttr", FUNC##Attr)                          \
      .set_attr<::mnm::op::tvm_dialect::FMNMArgIndices>("FMNMArgIndices", FUNC##ArgIndices);       \
  MNM_REGISTER_OP(MNM_BASE_OP_NAME(OP)).set_attr<::mnm::op::TOpPattern>("TOpPattern", OP_PATTERN); \
  MNM_OP_ENV_MAKER(MNM_DIALECT_OP_NAME(tvm, OP), FUNC##Build);                                     \
  MNM_OP_DISPATCH_DIALECT_PLEVEL(OP, tvm, DevType::kCPU(), PLEVEL);                                \
  MNM_OP_DISPATCH_DIALECT_PLEVEL(OP, tvm, DevType::kCUDA(), PLEVEL);

#define MNM_TVM(FUNC, OP, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH, OP_PATTERN)  \
  MNM_TVM_PLEVEL(FUNC, OP, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH, OP_PATTERN, \
                 10)
