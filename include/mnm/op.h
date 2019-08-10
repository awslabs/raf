#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/registry.h>
#include <mnm/value.h>

namespace mnm {
namespace executor {
class Executor;
}  // namespace executor
}  // namespace mnm

namespace mnm {
namespace op {

using Op = tvm::relay::Op;

using FOpMakeOutput =
    ir::TypedPackedFunc<value::Value(const ir::Array<value::Value>&, const ir::Attrs& attrs)>;

class OpBackend {
  using TRegistry = ::dmlc::Registry<OpBackend>;
  using TDeviceRegistry = registry::PerDevTypeStore<std::vector<OpBackend*> >;

 public:
  OpBackend() = default;

  OpBackend& set_name(const std::string& name) {
    this->name = name;
    return *this;
  }

  // TODO(@junrushao1994): this is not good to allow the library itself to set its own priority
  OpBackend& set_priority(int priority) {
    this->priority = priority;
    return *this;
  }

  // For heteregenous, make device = kExtDev
  OpBackend& set_device(DevType device);

 public:
  static TRegistry* Registry();

  static TDeviceRegistry* DeviceRegistry();

  static OpBackend* Get(const std::string& name);

 public:
  std::string name;
  int priority = 0;
  DevType device = DevType::kUnknown();
};

class OpDispatch {
  using FMakeOpEnv = std::function<void*(ir::Array<value::Value>, ir::Attrs)>;
  using TDispatchList = std::vector<std::pair<OpBackend*, FMakeOpEnv> >;
  using TRegistry = ::dmlc::Registry<OpDispatch>;

 public:
  OpDispatch() = default;

  OpDispatch& set_name(const std::string& name) {
    this->name = name;
    return *this;
  }

  OpDispatch& add_dispatch(DevType device_type, const std::string& backend_name,
                           const FMakeOpEnv& op_env_maker);

 public:
  static TRegistry* Registry();

  static TDispatchList* Get(const std::string& op_name, DevType device_type);

 public:
  std::string name;
  registry::PerDevTypeStore<TDispatchList> dispatch;
};

class OpEnv {
 public:
  /*
    A concrete operator provides
    1) an interface indicating if it accepts the given inputs, and request resources prior to
      computation, including streams, memory and distributed primitives
    2) an interface that does the actual computation using both the resources pre-allocated and
      requested on the fly.
    The OpEnv doesn't have any ownership to any resource, which is held by the executor.

    TODO: replace void* with concrete type
    TODO: make it accommodate with TVM node system
  */
 public:
  OpEnv();
  virtual ~OpEnv();

  void RequestMemory(void** dest, Context ctx, int64_t nbytes);
  void RequestWorkspace(void** dest, Context ctx, int64_t nbytes);
  void RequestStream(void** dest, Context ctx);
  void RequestDistributed(void** dest);

 public:
  // TODO: try TVMArgs
  virtual void Execute(ir::Array<value::Value> args, ir::Attrs attrs) = 0;

 private:
  /*
   * When executor is nullptr, resource allocation is in lazy phase; Otherwise it is eager.
   * Returns type-erased `requests::Requests *` because don't want to expose it in header
   */
  void* SetExecutor(executor::Executor* executor);

  class Impl;
  std::unique_ptr<Impl> impl;

  friend ::mnm::executor::Executor;
};

}  // namespace op
}  // namespace mnm

#define _MNM_REGISTER_OP_BACKEND_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpBackend& __make_##OpBackend

#define MNM_REGISTER_OP_BACKEND(name)                          \
  DMLC_STR_CONCAT(_MNM_REGISTER_OP_BACKEND_DEF, __COUNTER__) = \
      ::mnm::op::OpBackend::Registry()->__REGISTER_OR_GET__(name).set_name(name)

#define _MNM_REGISTER_OP_DISPATCH_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDispatch& __make_##OpDispatch

#define MNM_REGISTER_OP_DISPATCH(op_name, ctx, backend_name, op_env_maker) \
  DMLC_STR_CONCAT(_MNM_REGISTER_OP_DISPATCH_DEF, __COUNTER__) =            \
      ::mnm::op::OpDispatch::Registry()                                    \
          ->__REGISTER_OR_GET__(op_name)                                   \
          .set_name(op_name)                                               \
          .add_dispatch(ctx, backend_name, op_env_maker)
