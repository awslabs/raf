/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/tvmjit/tvmjit_utils.h
 * \brief Utility methods for TVM JIT.
 */
#pragma once
#include <vector>
#include <memory>
#include "dlpack/dlpack.h"
#include "tvm/relay/transform.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/c_runtime_api.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::value;

void GetDLTensor(const Value& v, std::vector<DLTensor>* tensors);
ir::Type GetTensorType(const DLTensor& dlt);
ir::Type GetTupleType(const std::vector<DLTensor>& dlts);
ir::Function LowerOp(const ir::Op& op, const ir::Attrs& attrs,
                     const std::vector<ir::Type>& param_types, const ir::Type& ret_type);
/*!
 * \brief Find an unallocated name for the given name.
 * \param name The given name
 * \return An unallocated name with a unique suffix attached
 */
std::string GetUniqueName(std::string name);

/*!
 * \brief Truncate the given name to fit in 80 characters
 * \param name The given name
 * \return The truncated name
 */
std::string TruncateName(std::string name);

class TVMOpEnv : public op::OpEnv {
 public:
  std::vector<DLTensor> inputs;
  std::vector<DLTensor> outputs;
  registry::PackedFunc f{nullptr};

  TVMOpEnv() = default;
  virtual ~TVMOpEnv() = default;
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
inline bool IsAutoSchedulerTaskExtractionEnabled() {
  return tvm::relay::transform::PassContext::Current()
      ->GetConfig<tvm::Bool>("mnm.tvmjit.extract_task", tvm::Bool(false))
      .value();
}

using FMNMLower = registry::TypedPackedFunc<ir::Function(const CallValues& call)>;
using FMNMAttr = registry::TypedPackedFunc<ir::Attrs(const CallValues& call)>;
using FMNMArgIndices =
    registry::TypedPackedFunc<ir::Array<tvm::IntImm>(const op::CallValues& call)>;

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm

#define MNM_TVMJIT_PLEVEL(FUNC, OP, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH,  \
                          PLEVEL)                                                               \
  MetaCache<registry::PackedFunc> FUNC##CacheBuildCpu;                                          \
  MetaCache<registry::PackedFunc> FUNC##CacheBuildCuda;                                         \
  MetaCache<ir::Function> FUNC##CacheLoweredFunc;                                               \
  template <typename RType>                                                                     \
  inline RType FUNC##CacheCompile(TVMOpEnv* env, const op::CallValues& call,                    \
                                  MetaCache<RType>* cache,                                      \
                                  std::function<RType(const ir::Function&)> f_post_lower) {     \
    static const auto op = Op::Get(OP);                                                         \
    const auto* schema = call->args.as<SCHEMA>();                                               \
    CHECK(schema != nullptr);                                                                   \
    Attrs attrs = SCHEMA2ATTRS(schema);                                                         \
    for (auto arg : SCHEMA2ARGS(schema)) {                                                      \
      GetDLTensor(arg, &env->inputs);                                                           \
    }                                                                                           \
    GetDLTensor(call->out, &env->outputs);                                                      \
    std::vector<Type> param_types;                                                              \
    Type ret_type;                                                                              \
    for (auto tensor : env->inputs) {                                                           \
      param_types.push_back(GetTensorType(tensor));                                             \
    }                                                                                           \
    if (env->outputs.size() == 1) {                                                             \
      ret_type = GetTensorType(env->outputs[0]);                                                \
    } else {                                                                                    \
      ret_type = GetTupleType(env->outputs);                                                    \
    }                                                                                           \
    RType ret{nullptr};                                                                         \
    HashKey key = HASH(param_types, ret_type, schema);                                          \
    if (const auto* compiled = cache->Get(key.byte_vector)) {                                   \
      ret = *compiled;                                                                          \
    } else {                                                                                    \
      auto lowered = LowerOp(op, attrs, param_types, ret_type);                                 \
      ret = f_post_lower(lowered);                                                              \
      cache->Set(key.byte_vector, ret);                                                         \
    }                                                                                           \
    return ret;                                                                                 \
  }                                                                                             \
  OpEnv* FUNC##Build(const op::CallValues& call) {                                              \
    static auto engine = registry::GetPackedFunc("relay.backend._CompileEngineGlobal")();       \
    static auto c_cache_key = registry::GetPackedFunc("relay.backend._make_CCacheKey");         \
    static auto jit = registry::GetPackedFunc("relay.backend._CompileEngineJIT");               \
    static auto engine_clear = registry::GetPackedFunc("relay.backend._CompileEngineClear");    \
    const auto& dev = call->device;                                                             \
    static const auto op = Op::Get(OP);                                                         \
    auto env = new TVMOpEnv();                                                                  \
    auto fschema_index = Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");      \
    for (auto field : SCHEMA_ARG_NAMES(call)) {                                                 \
      int idx = fschema_index[op](field);                                                       \
      CHECK_GE(idx, 0) << "Cannot find " << field << " in the schema for OP";                   \
      env->arg_indices.push_back(idx);                                                          \
    }                                                                                           \
    tvm::Target target;                                                                         \
    /* Determine cache and target */                                                            \
    MetaCache<registry::PackedFunc>* cache;                                                     \
    if (dev.device_type == DevType::kCPU()) {                                                   \
      cache = &FUNC##CacheBuildCpu;                                                             \
      target = tvm::Target("llvm");                                                             \
    } else if (dev.device_type == DevType::kCUDA()) {                                           \
      cache = &FUNC##CacheBuildCuda;                                                            \
      target = tvm::Target("cuda");                                                             \
    } else {                                                                                    \
      LOG(FATAL) << "NotImplementedError: target is not supported " << dev.device_type.c_str(); \
      throw;                                                                                    \
    }                                                                                           \
    std::function<registry::PackedFunc(const ir::Function&)> f_post_lower(                      \
        [&](const ir::Function& f) {                                                            \
          engine_clear(engine);                                                                 \
          return jit(engine, c_cache_key(f, target));                                           \
        });                                                                                     \
    try {                                                                                       \
      env->f = FUNC##CacheCompile(env, call, cache, f_post_lower);                              \
    } catch (const dmlc::Error& e) {                                                            \
      if (!IsAutoSchedulerTaskExtractionEnabled()) {                                            \
        /* Invalid implementation. Return nullptr to let dispatcher select the next one */      \
        return nullptr;                                                                         \
      }                                                                                         \
    }                                                                                           \
    env->env_name = TruncateName(GetUniqueName(op->name.operator std::string()));               \
    return env;                                                                                 \
  }                                                                                             \
  Attrs FUNC##Attr(const op::CallValues& call) {                                                \
    static const auto op = Op::Get(OP);                                                         \
    const auto* schema = call->args.as<SCHEMA>();                                               \
    CHECK(schema != nullptr);                                                                   \
    return SCHEMA2ATTRS(schema);                                                                \
  }                                                                                             \
  Array<tvm::IntImm> FUNC##ArgIndices(const op::CallValues& call) {                             \
    static const auto op = Op::Get(OP);                                                         \
    static const auto fschema_index =                                                           \
        Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex")[op];                   \
    std::vector<tvm::IntImm> ret;                                                               \
    for (const auto& field : SCHEMA_ARG_NAMES(call)) {                                          \
      ret.push_back(tvm::IntImm(DataType::Int(32), fschema_index(field)));                      \
    }                                                                                           \
    return Array<tvm::IntImm>(ret);                                                             \
  }                                                                                             \
  ir::Function FUNC##Lower(const op::CallValues& call) {                                        \
    static const std::function<ir::Function(const ir::Function&)> identity(                     \
        [](const ir::Function& f) { return f; });                                               \
    MetaCache<ir::Function>* cache;                                                             \
    cache = &FUNC##CacheLoweredFunc;                                                            \
    auto env = std::make_unique<TVMOpEnv>();                                                    \
    return FUNC##CacheCompile(env.get(), call, cache, identity);                                \
  }                                                                                             \
  RELAY_REGISTER_OP(OP).set_attr<::mnm::op::tvmjit::FMNMLower>("FMNMLower", FUNC##Lower);       \
  RELAY_REGISTER_OP(OP).set_attr<::mnm::op::tvmjit::FMNMAttr>("FMNMAttr", FUNC##Attr);          \
  RELAY_REGISTER_OP(OP).set_attr<::mnm::op::tvmjit::FMNMArgIndices>("FMNMArgIndices",           \
                                                                    FUNC##ArgIndices);          \
  MNM_OP_DISPATCH_PLEVEL(OP, FUNC##Build, DevType::kCPU(), "tvmjit", PLEVEL);                   \
  MNM_OP_DISPATCH_PLEVEL(OP, FUNC##Build, DevType::kCUDA(), "tvmjit", PLEVEL);

#define MNM_TVMJIT(FUNC, OP, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH) \
  MNM_TVMJIT_PLEVEL(FUNC, OP, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH, 10)
