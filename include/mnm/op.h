#pragma once

#include <memory>
#include <string>

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/registry.h>
#include <mnm/value.h>

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

class CallValuesNode : public ir::Node {
 public:
  mutable value::Value callee;
  mutable ir::Attrs args;
  mutable value::Value out;
  mutable Context ctx;

 public:
  static constexpr const char* _type_key = "mnm.op.CallValues";
  MNM_DEF_NODE_TYPE_INFO(CallValuesNode, ir::Node);
};

class CallValues : public ir::NodeRef {
 public:
  static CallValues make();
  MNM_DEF_NODE_REF_METHODS(CallValues, ir::NodeRef, CallValuesNode);
};

class OpEnv {
  class Impl;
  std::shared_ptr<Impl> impl;

 public:
  OpEnv();
  virtual ~OpEnv();
  virtual void Execute(const CallValues& call) = 0;

  void RequestMemory(void** dest, const Context& ctx, int64_t nbytes);
  void RequestWorkspace(void** dest, const Context& ctx, int64_t nbytes);
  void RequestStream(void** dest, const Context& ctx, int tag_idx);
  void RequestDistributed(void** dest) {
    LOG(FATAL) << "NotImplementedError: RequestDistributed";
    throw;
  }

  void BindExecutor(executor::Executor* executor);
  std::shared_ptr<requests::Requests> GetRequests() const;
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

// TODO: change it to FOpDeclare
using FMNMDeclare = registry::TypedPackedFunc<void(const CallValues& call)>;
using FMNMSchema = registry::TypedPackedFunc<ir::Attrs(const ir::Array<value::Value>&)>;

void RunDeclare(const CallValues& call);

}  // namespace op
}  // namespace mnm

#define _MNM_REGISTER_OP_DISPATCH_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDispatch& __make_##OpDispatch

#define MNM_REGISTER_OP_DISPATCH(op_name, ctx, backend_name, op_env_maker) \
  DMLC_STR_CONCAT(_MNM_REGISTER_OP_DISPATCH_DEF, __COUNTER__) =            \
      ::mnm::op::OpDispatch::Registry()                                    \
          ->__REGISTER_OR_GET__(op_name)                                   \
          .set_name(op_name)                                               \
          .add_dispatch(ctx, backend_name, op_env_maker)

#define MNM_OP_SCHEMA(ClassName, TypeKey)                     \
  static constexpr const char* _type_key = TypeKey;           \
  TVM_DECLARE_NODE_TYPE_INFO(ClassName, ::tvm::BaseAttrsNode) \
  template <typename FVisit>                                  \
  void __VisitAttrs__(FVisit& __fvisit__) {                   \
  }                                                           \
  void Init(const ::mnm::ir::Array<::mnm::value::Value>& args)

#define MNM_ARG_OPTIONAL(i, type, name)    \
  if (static_cast<int>(args.size()) > i) { \
    this->name = type(args[i]);            \
  }

#define MNM_ARG_REQUIRED(i, type, name)                                       \
  {                                                                           \
    CHECK(static_cast<int>(args.size()) > i) << "Missing argument " << #name; \
    this->name = type(args[i]);                                               \
  }

#define MNM_REGISTER_OP(OpName, ArgsName)                                                \
  RELAY_REGISTER_OP(OpName).set_attr<::mnm::op::FMNMSchema>(                             \
      "FMNMSchema", [](const ::mnm::ir::Array<::mnm::value::Value>& args) -> ir::Attrs { \
        auto attrs = ::mnm::ir::make_node<ArgsName>();                                   \
        attrs->Init(args);                                                               \
        return ir::Attrs(attrs);                                                         \
      })
