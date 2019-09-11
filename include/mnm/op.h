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

class OpInfoNode : public ir::Node {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("output", &output);
  }

 public:
  mutable value::Value output;
  Context ctx;
  bool computational = true;

  static constexpr const char* _type_key = "mnm.op.OpInfo";
  MNM_DEF_NODE_TYPE_INFO(OpInfoNode, Node);
};

class OpInfo : public ir::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(OpInfo, NodeRef, OpInfoNode);
  static OpInfo make(value::Value output, Context ctx, bool computational = true);
};

class OpEnv {
  class Impl;
  std::shared_ptr<Impl> impl;

 public:
  OpEnv();
  virtual ~OpEnv();
  virtual void Execute(ir::Array<value::Value> args, const OpInfo& info, ir::Attrs attrs) = 0;

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
  using FMakeOpEnv = std::function<OpEnv*(ir::Array<value::Value>, const OpInfo&, ir::Attrs)>;
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
  static std::unique_ptr<OpEnv> Dispatch(const ir::Op& op, const OpInfo& info,
                                         const ir::Array<value::Value>& args,
                                         const ir::Attrs& attrs);

 public:
  std::string name;
  registry::PerDevTypeStore<TDispatchList> dispatch;
};

using FOpMakeOutput =
    registry::TypedPackedFunc<OpInfo(const ir::Array<value::Value>&, const ir::Attrs& attrs)>;

OpInfo MakeOutput(const ir::Op& op, const ir::Array<value::Value>& args, const ir::Attrs& attrs);

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
