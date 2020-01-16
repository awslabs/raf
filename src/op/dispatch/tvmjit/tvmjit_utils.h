/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/tvmjit/tvmjit_utils.h
 * \brief Utility methods for TVM JIT.
 */
#pragma once
#include <vector>
#include <memory>
#include "dlpack/dlpack.h"
#include "tvm/runtime/c_runtime_api.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"

#include "../../op_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {
DLTensor GetDLTensor(const value::Value& v);
void GetOut(const value::Value& out, std::vector<DLTensor>* ret);
ir::Type GetTensorType(const DLTensor& dlt);
ir::Type GetTupleType(const std::vector<DLTensor>& dlts);
registry::PackedFunc CompileOp(const ir::Op& op,                          //
                               const ir::Attrs& attrs,                    //
                               const std::vector<ir::Type>& param_types,  //
                               const ir::Type& ret_type,                  //
                               Context ctx);

class TVMOpEnv : public op::OpEnv {
 public:
  std::vector<DLTensor> inputs;
  std::vector<DLTensor> outputs;
  std::vector<TVMValue> values;
  std::vector<int> codes;
  std::vector<std::vector<int64_t>> shape_slots;
  registry::PackedFunc f{nullptr};

  TVMOpEnv() = default;
  virtual ~TVMOpEnv() = default;
  void Setup();
  void Execute(const op::CallValues& call) override;
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

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm

#define MNM_TVMJIT(FUNC, OP, SCHEMA, NORM, TYPE, HASH)             \
  MetaCache<registry::PackedFunc> FUNC##CacheCpu;                  \
  MetaCache<registry::PackedFunc> FUNC##CacheCuda;                 \
  OpEnv* FUNC(const op::CallValues& call) {                        \
    static const auto op = Op::Get(OP);                            \
    const auto* args = call->args.as<SCHEMA>();                    \
    const auto& ctx = call->ctx;                                   \
    auto env = std::make_unique<TVMOpEnv>();                       \
    /* Normalize inputs and outputs */                             \
    GetOut(call->out, &env->outputs);                              \
    Attrs attrs = NORM(env.get(), args);                           \
    /* Normalize types */                                          \
    std::vector<Type> param_types;                                 \
    Type ret_type;                                                 \
    TYPE(env.get(), &param_types, &ret_type);                      \
    /* Determine which cache to look up */                         \
    MetaCache<registry::PackedFunc>* cache;                        \
    if (call->ctx.device_type == DevType::kCPU()) {                \
      cache = &FUNC##CacheCpu;                                     \
    } else if (call->ctx.device_type == DevType::kCUDA()) {        \
      cache = &FUNC##CacheCuda;                                    \
    } else {                                                       \
      LOG(FATAL) << "NotImplementedError: ";                       \
      throw;                                                       \
    }                                                              \
    /* Look up hash */                                             \
    HashKey key = HASH(param_types, ret_type, args);               \
    {                                                              \
      std::lock_guard<std::mutex> lock(cache->mu);                 \
      if (const auto *compiled = cache->Get(key.byte_vector)) {    \
        env->f = *compiled;                                        \
      } else {                                                     \
        env->f = CompileOp(op, attrs, param_types, ret_type, ctx); \
        cache->Set(key.byte_vector, env->f);                       \
      }                                                            \
    }                                                              \
    /* Setup other parts of the environment */                     \
    env->Setup();                                                  \
    return env.release();                                          \
  }                                                                \
  MNM_OP_DISPATCH(OP, FUNC, DevType::kCPU(), "tvmjit");            \
  MNM_OP_DISPATCH(OP, FUNC, DevType::kCUDA(), "tvmjit");
