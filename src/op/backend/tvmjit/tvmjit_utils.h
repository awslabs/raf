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

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm

#define MNM_TVMJIT(FUNC, OP, SCHEMA, NORM, TYPE)               \
  OpEnv* FUNC(const op::CallValues& call) {                    \
    static const auto op = Op::Get(OP);                        \
    const auto* args = call->args.as<SCHEMA>();                \
    const auto& ctx = call->ctx;                               \
    auto env = std::make_unique<TVMOpEnv>();                   \
    /* Normalize inputs and outputs */                         \
    GetOut(call->out, &env->outputs);                          \
    Attrs attrs = NORM(env.get(), args);                       \
    /* Normalize types */                                      \
    std::vector<Type> param_types;                             \
    Type ret_type;                                             \
    TYPE(env.get(), &param_types, &ret_type);                  \
    /* Compile. TODO(@junrushao1994): cache */                 \
    env->f = CompileOp(op, attrs, param_types, ret_type, ctx); \
    env->Setup();                                              \
    return env.release();                                      \
  }                                                            \
  MNM_OP_DISPATCH(OP, FUNC, DevType::kCPU(), "tvm-cpu");       \
  MNM_OP_DISPATCH(OP, FUNC, DevType::kCUDA(), "tvm-cuda");
