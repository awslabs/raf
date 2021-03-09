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
  mutable Device device;

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
  virtual void Execute(const std::vector<value::Value>& inputs, value::Value output) = 0;

  void RequestWorkspace(void** dest, const Device& device, int64_t nbytes);
  void RequestStream(void** dest, const Device& device, int tag_idx);
  void RequestDistributed(void** dest);

  void BindExecutor(executor::Executor* executor);
  std::shared_ptr<requests::Requests> GetRequests() const;
  /*! \brief Input indices in the argument array. This is used by VM executor. */
  std::vector<int> arg_indices;
  /*! \brief OpEnv name*/
  std::string env_name;
};

class OpDispatch {
  using FMakeOpEnv = std::function<OpEnv*(const CallValues& call)>;

  struct OpEnvMaker {
    int plevel;
    std::string backend;
    FMakeOpEnv maker;
  };
  using TDispatchList = std::list<OpEnvMaker>;
  using TRegistry = ::dmlc::Registry<OpDispatch>;

 public:
  OpDispatch() = default;
  OpDispatch& set_name(const std::string& name);
  OpDispatch& add_dispatch(DevType device_type, const std::string& backend_name,
                           const FMakeOpEnv& op_env_maker, int plevel = 10);

 public:
  static TRegistry* Registry();
  static TDispatchList* Get(const ir::Op& op, DevType device_type);
  static std::shared_ptr<OpEnv> Dispatch(const CallValues& call);

 public:
  std::string name;
  registry::PerDevTypeStore<TDispatchList> dispatch;
};

/*!
 * \brief Singleton dispatcher for fused ops
 */
class FusedOpDispatch {
  using FMakeFuncEnv = std::function<OpEnv*(const CallValues& call)>;
  using TDispatchList = std::unordered_map<std::string, FMakeFuncEnv>;

 private:
  FusedOpDispatch() = default;

 public:
  /*!
   * \brief add a dispatch for a specific backend
   * \param device_type the device type
   * \param backend_name the backend name
   * \param op_env_maker a function that converts a call value into the corresponding op env
   */
  FusedOpDispatch& add_dispatch(DevType device_type, const std::string& backend_name,
                                const FMakeFuncEnv& op_env_maker);

 public:
  /*! \brief get the FusedOpDispatch instance */
  static FusedOpDispatch* Get();
  /*! \brief get the list of all available backends on a specific device */
  static TDispatchList* Get(DevType device_type);
  /*! \brief dispatch call to some backend according to its device */
  static std::shared_ptr<OpEnv> Dispatch(const CallValues& call);

 public:
  /*! \brief the list of backends and their dispatch mathods available on different devices */
  registry::PerDevTypeStore<TDispatchList> dispatch;
};

/*! \brief dispatch (fused or un-fused) ops to backend implementation */
std::shared_ptr<OpEnv> Dispatch(const CallValues& call);

using tvm::Attrs;
using tvm::relay::Expr;
using tvm::relay::kBroadcast;
using tvm::relay::kCommReduce;
using tvm::relay::kElemWise;
using tvm::relay::kInjective;
using tvm::relay::kOpaque;
using tvm::relay::kOutEWiseFusable;
using tvm::relay::kTuple;
using tvm::relay::OpPatternKind;
using tvm::relay::TOpPattern;
using tvm::runtime::Array;

using FMNMDeclare = registry::TypedPackedFunc<void(const CallValues& call)>;

using FMNMSchema = registry::TypedPackedFunc<ir::Attrs(const ir::Array<value::Value>&)>;

using FMNMCastRule = registry::TypedPackedFunc<Array<ir::Integer>(const Array<Expr>&)>;

using FMNMAnnotateTarget =
    registry::TypedPackedFunc<bool(const Attrs& attrs, const Array<Expr>& args)>;

using FMNMSchemaFieldIndex = registry::TypedPackedFunc<int(const std::string&)>;

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

using FMNMFromRelay = registry::TypedPackedFunc<Expr(const Attrs& attrs, const Array<Expr>& args)>;
using FMNMMutationFromRelay =
    registry::TypedPackedFunc<Array<Array<Expr>>(const ir::Var& var, const ir::Call& call)>;

void RunDeclare(const CallValues& call);
ir::Attrs MakeListArgs(const ir::Array<value::Value>& values);
ir::Array<value::Value> GetListArgs(const ir::Attrs& attrs);

}  // namespace op
}  // namespace mnm

#define _MNM_OP_DISPATCH_DEF static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDispatch& __make_##OpDispatch

#define _MNM_FUNC_DISPATCH_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::mnm::op::FusedOpDispatch& __make_##FusedOpDispatch

#define MNM_OP_REGISTER(op_name) RELAY_REGISTER_OP(op_name)

#define MNM_OP_DECLARE(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<::mnm::op::FMNMDeclare>("FMNMDeclare", body)

#define MNM_OP_DISPATCH_PLEVEL(op_name, op_env_maker, device_type, backend_name, plevel) \
  DMLC_STR_CONCAT(_MNM_OP_DISPATCH_DEF, __COUNTER__) =                                   \
      ::mnm::op::OpDispatch::Registry()                                                  \
          ->__REGISTER_OR_GET__(op_name)                                                 \
          .set_name(op_name)                                                             \
          .add_dispatch(device_type, backend_name, op_env_maker, plevel)

#define MNM_OP_DISPATCH(op_name, op_env_maker, device_type, backend_name) \
  MNM_OP_DISPATCH_PLEVEL(op_name, op_env_maker, device_type, backend_name, 10)

#define MNM_FUNC_DISPATCH(func_env_maker, device_type, backend_name) \
  DMLC_STR_CONCAT(_MNM_FUNC_DISPATCH_DEF, __COUNTER__) =             \
      ::mnm::op::FusedOpDispatch::Get()->add_dispatch(device_type, backend_name, func_env_maker)

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
#define MNM_OP_GRAD_SKIP_INPUTS(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<std::string>("GradientInputSkip", body);
