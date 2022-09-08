/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/tvm_utils.h
 * \brief Utility methods for TVM dialect.
 */
#pragma once
#include <vector>
#include <memory>
#include <dmlc/filesystem.h>
#include <tvm/node/serialization.h>
#include "dlpack/dlpack.h"
#include "tvm/relay/transform.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/c_runtime_api.h"
#include "relay/backend/te_compiler.h"
#include "relay/backend/te_compiler_cache.h"
#include "raf/cache.h"
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/serialization.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::value;

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

/*! \brief The persist cache entry of TVM modules. */
class TVMModuleCacheEntry {
 public:
  explicit TVMModuleCacheEntry() {
  }

  TVMModuleCacheEntry(tvm::runtime::Module& mod, std::string func_name)
      : mod_(mod), func_name_(func_name) {
  }

  static TVMModuleCacheEntry Load(const std::string path) {
    static auto f_load = registry::GetPackedFunc("raf._tvm_op.utils.load_module");
    tvm::runtime::Module mod = f_load(path + "/" + MOD_SO_FILE);

    std::ifstream ifs(path + "/" + FUNC_NAME_FILE);
    std::string func_name;
    ifs >> func_name;

    return TVMModuleCacheEntry(mod, func_name);
  }

  registry::PackedFunc GetFunction() {
    return mod_->GetFunction(func_name_);
  }

  bool Save(const std::string& path) {
    static auto f_export = registry::GetPackedFunc("raf._tvm_op.utils.export_library");
    auto bin_path = path + "/" + MOD_SO_FILE;
    bool success = f_export(mod_, bin_path);
    if (!success) {
      return false;
    }

    std::ofstream ofs(path + "/" + FUNC_NAME_FILE);
    ofs << func_name_;
    return true;
  }

 private:
  /*! \brief The persist TVM module name. */
  static constexpr const char* MOD_SO_FILE = "tvm_module.so";
  /*! \brief The persist function name file. */
  static constexpr const char* FUNC_NAME_FILE = "func_name.txt";
  /*! \brief The built module to be used and persist. */
  tvm::runtime::Module mod_;
  /*! \brief The corresponding function name. */
  std::string func_name_;
};

class RelayFuncCacheEntry {
 public:
  explicit RelayFuncCacheEntry() {
  }

  RelayFuncCacheEntry(const ir::Function& func) : func_(func) {
  }

  ir::Function GetFunction() {
    return func_;
  }

  static RelayFuncCacheEntry Load(const std::string path) {
    std::ifstream ifs(path + "/" + FUNC_FILE, std::ios::in);
    if (!ifs.is_open()) {
      LOG(FATAL) << "Function JSON file does not exist: " << path + "/" + FUNC_FILE;
      throw;
    }

    std::string func_json;
    ifs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    func_json.resize(size);
    ifs.read(&func_json[0], size);
    ifs.close();
    auto func = Downcast<Function>(tvm::LoadJSON(func_json));
    return RelayFuncCacheEntry(func);
  }

  bool Save(const std::string& path) {
    auto json_str = ir::serialization::SaveJSON(func_);
    std::ofstream ofs(path + "/" + FUNC_FILE, std::ios::out);
    if (!ofs.is_open()) {
      return false;
    }
    ofs << json_str;
    ofs.close();
    return true;
  }

 private:
  /*! \brief The persist function file name. */
  static constexpr const char* FUNC_FILE = "func.json";
  /*! \brief The cached function. */
  ir::Function func_;
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
      ->GetConfig<tvm::Bool>("raf.tvm.allow_jit_failure", tvm::Bool(false))
      .value();
}

/*!
 * \brief Modify the configs of the current PassContext to enable auto-scheduler for TVM ops.
 */
inline void ForceEnableAutoScheduler() {
  tvm::relay::transform::PassContext::Current()->config.Set("relay.backend.use_auto_scheduler",
                                                            tvm::Bool(true));
}

using FRAFLower = registry::TypedPackedFunc<ir::Function(const CallValues& call)>;
using FRAFAttr = registry::TypedPackedFunc<ir::Attrs(const CallValues& call)>;
using FRAFArgIndices =
    registry::TypedPackedFunc<ir::Array<tvm::IntImm>(const op::CallValues& call)>;

extern MetaPersistCache<TVMModuleCacheEntry> CacheBuildCpu;
extern MetaPersistCache<TVMModuleCacheEntry> CacheBuildCuda;
extern MetaPersistCache<RelayFuncCacheEntry> CacheLoweredFunc;

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf

#define RAF_TVM_PLEVEL(OP, FUNC, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH,        \
                       OP_PATTERN, PLEVEL)                                                         \
  template <typename RType>                                                                        \
  inline RType FUNC##CacheCompile(TVMOpEnv* env, const op::CallValues call,                        \
                                  MetaPersistCache<RType>* cache,                                  \
                                  std::function<RType(const ir::Function&)> f_post_lower) {        \
    raf::op::tvm_dialect::ForceEnableAutoScheduler();                                              \
    static const auto op = Op::Get(RAF_DIALECT_OP_NAME(tvm, OP));                                  \
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
    RType ret;                                                                                     \
    HashKey key;                                                                                   \
    key << #OP << HASH(param_types, ret_type, schema);                                             \
    if (const auto* compiled = cache->Get(key.byte_vector)) {                                      \
      ret = *compiled;                                                                             \
    } else {                                                                                       \
      auto lowered = LowerOp(op, attrs, param_types, ret_type);                                    \
      ret = f_post_lower(lowered);                                                                 \
      cache->Set(key.byte_vector, ret);                                                            \
    }                                                                                              \
    return ret;                                                                                    \
  }                                                                                                \
  OpEnv* FUNC##Build(const op::CallValues call) {                                                  \
    tvm::relay::tec::TECompiler te_compiler;                                                       \
    const auto& dev = call->device;                                                                \
    static const auto base_op = Op::Get(RAF_BASE_OP_NAME(OP));                                     \
    auto env = new TVMOpEnv();                                                                     \
    auto fschema_index = op::GetOpAttr<op::FRAFSchemaFieldIndex>(base_op, "FRAFSchemaFieldIndex"); \
    for (auto field : SCHEMA_ARG_NAMES(call)) {                                                    \
      int idx = fschema_index(field);                                                              \
      CHECK_GE(idx, 0) << "Cannot find " << field << " in the schema for OP";                      \
      env->arg_indices.push_back(idx);                                                             \
    }                                                                                              \
    /* Determine cache */                                                                          \
    MetaPersistCache<TVMModuleCacheEntry>* cache;                                                  \
    if (dev.device_type() == DevType::kCPU()) {                                                    \
      cache = &CacheBuildCpu;                                                                      \
    } else if (dev.device_type() == DevType::kCUDA()) {                                            \
      cache = &CacheBuildCuda;                                                                     \
    } else {                                                                                       \
      LOG(FATAL) << "NotImplementedError: device is not supported " << dev.device_type().c_str();  \
      throw;                                                                                       \
    }                                                                                              \
    tvm::Target target = dev.tvm_target();                                                         \
    env->env_name = TruncateName(GetUniqueName(RAF_DIALECT_OP_NAME(tvm, OP)));                     \
    std::function<TVMModuleCacheEntry(const ir::Function&)> f_post_lower(                          \
        [&](const ir::Function& f) {                                                               \
          te_compiler->Clear();                                                                    \
          auto key = tvm::relay::tec::CCacheKey(f, target);                                        \
          auto cached_func = te_compiler->Lower(key);                                              \
          auto mod = tvm::build(cached_func->funcs, key->target, Target(nullptr));                 \
          return TVMModuleCacheEntry(mod, cached_func->prim_fn_var->name_hint);                    \
        });                                                                                        \
    try {                                                                                          \
      auto module_cache_entry = FUNC##CacheCompile(env, call, cache, f_post_lower);                \
      env->f = module_cache_entry.GetFunction();                                                   \
    } catch (const dmlc::Error& e) {                                                               \
      /* Invalid implementation. Return nullptr to let dispatcher select the next one */           \
      if (!AllowJitFailure()) {                                                                    \
        std::stringstream ss;                                                                      \
        ss << "[TVM] Failed to JIT: " << env->env_name << ": " << e.what();                        \
        auto msg = ss.str();                                                                       \
        env->error_msgs.push_back(msg);                                                            \
        DLOG(WARNING) << msg;                                                                      \
        return env;                                                                                \
      }                                                                                            \
    }                                                                                              \
    return env;                                                                                    \
  }                                                                                                \
  Attrs FUNC##Attr(const op::CallValues call) {                                                    \
    static const auto op = Op::Get(RAF_BASE_OP_NAME(OP));                                          \
    const auto* schema = call->args.as<SCHEMA>();                                                  \
    CHECK(schema != nullptr);                                                                      \
    return SCHEMA2ATTRS(schema);                                                                   \
  }                                                                                                \
  Array<tvm::IntImm> FUNC##ArgIndices(const op::CallValues call) {                                 \
    static const auto op = Op::Get(RAF_BASE_OP_NAME(OP));                                          \
    static const auto fschema_index =                                                              \
        op::GetOpAttr<op::FRAFSchemaFieldIndex>(op, "FRAFSchemaFieldIndex");                       \
    std::vector<tvm::IntImm> ret;                                                                  \
    for (const auto& field : SCHEMA_ARG_NAMES(call)) {                                             \
      ret.push_back(tvm::IntImm(DataType::Int(32), fschema_index(field)));                         \
    }                                                                                              \
    return Array<tvm::IntImm>(ret);                                                                \
  }                                                                                                \
  ir::Function FUNC##Lower(const op::CallValues call) {                                            \
    static const std::function<RelayFuncCacheEntry(const ir::Function&)> identity(                 \
        [](const ir::Function& f) { return RelayFuncCacheEntry(f); });                             \
    MetaPersistCache<RelayFuncCacheEntry>* cache;                                                  \
    cache = &CacheLoweredFunc;                                                                     \
    auto env = std::make_unique<TVMOpEnv>();                                                       \
    return FUNC##CacheCompile(env.get(), call, cache, identity).GetFunction();                     \
  }                                                                                                \
  RAF_REGISTER_DIALECT_OP(tvm, OP, PLEVEL)                                                         \
      .set_attr<::raf::op::TOpPattern>("TOpPattern", OP_PATTERN)                                   \
      .set_attr<::raf::op::tvm_dialect::FRAFLower>("FRAFLower", FUNC##Lower)                       \
      .set_attr<::raf::op::tvm_dialect::FRAFAttr>("FRAFAttr", FUNC##Attr)                          \
      .set_attr<::raf::op::tvm_dialect::FRAFArgIndices>("FRAFArgIndices", FUNC##ArgIndices);       \
  RAF_OP_ENV_MAKER(RAF_DIALECT_OP_NAME(tvm, OP), FUNC##Build);

#define RAF_TVM(OP, FUNC, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH, OP_PATTERN)  \
  RAF_TVM_PLEVEL(OP, FUNC, SCHEMA, SCHEMA2ARGS, SCHEMA_ARG_NAMES, SCHEMA2ATTRS, HASH, OP_PATTERN, \
                 10)
