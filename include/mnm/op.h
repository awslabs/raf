/*!
 * Copyright (c) 2019 by Contributors
 * \file op.h
 * \brief Operator interface
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "./base.h"
#include "./ir.h"
#include "./registry.h"
#include "./value.h"
#include "./memory_pool.h"

namespace mnm {
namespace executor {
class Executor;
}  // namespace executor
namespace requests {
class Requests;
}  // namespace requests
}  // namespace mnm

namespace mnm {
namespace op {

class CallValuesNode : public ir::Object {
 public:
  mutable value::Value callee;
  mutable ir::Attrs args;
  mutable value::Value out;
  mutable Context ctx;

 public:
  static constexpr const char* _type_key = "mnm.op.CallValues";
  MNM_FINAL_OBJECT(CallValuesNode, ir::Object);
};

class CallValues : public ir::ObjectRef {
 public:
  static CallValues make(value::Value callee = {}, ir::Attrs args = {});
  MNM_OBJECT_REF(CallValues, ir::ObjectRef, CallValuesNode);
};

class OpEnv {
  class Impl;
  std::shared_ptr<Impl> impl;

 public:
  OpEnv();
  virtual ~OpEnv();
  virtual void Execute(const CallValues& call) = 0;

  void RequestWorkspace(void** dest, const Context& ctx, int64_t nbytes);
  void RequestStream(void** dest, const Context& ctx, int tag_idx);
  void RequestDistributed(void** dest);

  void BindExecutor(executor::Executor* executor);
  std::shared_ptr<requests::Requests> GetRequests() const;
  void SetOutputBuffer(std::vector<std::shared_ptr<memory_pool::Memory>> out_buf);

 private:
  // Keep reference to output buffer so that the memory is released when OpEnv is destructed.
  // TODO(wuwei,haichen): we should move this part to a better place
  std::vector<std::shared_ptr<memory_pool::Memory>> out_buf_;
};

class OpDispatch {
  using FMakeOpEnv = std::function<OpEnv*(const CallValues& call)>;
  using TDispatchList = std::unordered_map<std::string, FMakeOpEnv>;
  using TRegistry = ::dmlc::Registry<OpDispatch>;

 public:
  OpDispatch() = default;
  OpDispatch& set_name(const std::string& name);
  OpDispatch& add_dispatch(DevType device_type, const std::string& backend_name,
                           const FMakeOpEnv& op_env_maker);

 public:
  static TRegistry* Registry();
  static TDispatchList* Get(const ir::Op& op, DevType device_type);
  static std::unique_ptr<OpEnv> Dispatch(const CallValues& call);

 public:
  std::string name;
  registry::PerDevTypeStore<TDispatchList> dispatch;
};

using tvm::relay::kBroadcast;
using tvm::relay::kCommReduce;
using tvm::relay::kElemWise;
using tvm::relay::kInjective;
using tvm::relay::kOpaque;
using tvm::relay::kOutEWiseFusable;
using tvm::relay::kTuple;
using tvm::relay::OpPatternKind;
using tvm::relay::TOpPattern;

using FMNMDeclare = registry::TypedPackedFunc<void(const CallValues& call)>;

using FMNMSchema = registry::TypedPackedFunc<ir::Attrs(const ir::Array<value::Value>&)>;

using FPrimalGradient = registry::TypedPackedFunc<
    // returns: op's contribution to igrads
    ir::Array<ir::Expr>(
        // orig_call: a relay::Call which invokes this operator
        const ir::Expr& orig_call,
        // orig_args: Array of original args of the call node
        const ir::Array<ir::Expr> orig_args,
        // computed_output: (optional) to which var the op's output binds to
        const ir::Var& computed_output,
        // out_grad: ograds
        const ir::Expr& out_grad)>;

using FFusedPrimalGradient = registry::TypedPackedFunc<
    // returns: the updated igrads
    ir::Array<ir::Expr>(
        // orig_call: a relay::Call which invokes this operator
        const ir::Expr& orig_call,
        // computed_output: (optional) to which var the op's output binds to
        const ir::Var& computed_output,
        // out_grad: ograds
        const ir::Expr& out_grad,
        // in_grad: old igrads
        const ir::Array<ir::Expr>& in_grad)>;

void RunDeclare(const CallValues& call);
ir::Attrs MakeListArgs(const ir::Array<value::Value>& values);
ir::Array<value::Value> GetListArgs(const ir::Attrs& attrs);

}  // namespace op
}  // namespace mnm

#define _MNM_OP_DISPATCH_DEF static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDispatch& __make_##OpDispatch

#define MNM_OP_REGISTER(op_name) RELAY_REGISTER_OP(op_name)

#define MNM_OP_DECLARE(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<::mnm::op::FMNMDeclare>("FMNMDeclare", body)

#define MNM_OP_DISPATCH(op_name, op_env_maker, device_type, backend_name) \
  DMLC_STR_CONCAT(_MNM_OP_DISPATCH_DEF, __COUNTER__) =                    \
      ::mnm::op::OpDispatch::Registry()                                   \
          ->__REGISTER_OR_GET__(op_name)                                  \
          .set_name(op_name)                                              \
          .add_dispatch(device_type, backend_name, op_env_maker)

#define MNM_OP_SCHEMA(class_name, type_key)          \
  static constexpr const char* _type_key = type_key; \
  MNM_FINAL_OBJECT(class_name, ::tvm::BaseAttrsNode) \
  template <typename FVisit>                         \
  void __VisitAttrs__(FVisit& __fvisit__) {          \
  }

#define MNM_OP_GRAD(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<::mnm::op::FPrimalGradient>("FPrimalGradient", body);

#define MNM_OP_FUSED_GRAD(op_name, body)                                                       \
  RELAY_REGISTER_OP(op_name).set_attr<::mnm::op::FFusedPrimalGradient>("FFusedPrimalGradient", \
                                                                       body);
