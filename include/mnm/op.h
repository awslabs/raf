#pragma once

#include <utility>
#include <vector>

#include <mnm/registry.h>
#include <mnm/rly.h>
#include <mnm/types.h>
#include <mnm/value.h>

namespace mnm {
namespace executor {
class Executor;
}  // namespace executor
}  // namespace mnm

namespace mnm {
namespace op {

using Op = tvm::relay::Op;

using FOpMakeOutput = mnm::rly::TypedPackedFunc<mnm::value::Value(
    const mnm::rly::Array<mnm::value::Value>&, const mnm::rly::Attrs& attrs)>;

class OpBackend {
  using TRegistry = ::dmlc::Registry<OpBackend>;
  using TDeviceRegistry = mnm::registry::PerDeviceTypeStorage<std::vector<OpBackend*>>;

 public:
  OpBackend() = default;

  inline OpBackend& set_name(const std::string& name);

  // For heteregenous, make device = kExtDev
  inline OpBackend& set_device(mnm::types::DeviceType device);

  // TODO(@junrushao1994): this is not good to allow the library itself to set its own priority
  inline OpBackend& set_priority(int priority);

 public:
  static TRegistry* Registry();

  static TDeviceRegistry* DeviceRegistry();

  static OpBackend* Get(const std::string& name);

 public:
  std::string name;
  mnm::types::DeviceType device = mnm::types::DeviceType::kUnknown();
  int priority = 0;
};

class OpDispatch {
  using FMakeOpEnv = std::function<void*(mnm::rly::Array<mnm::value::Value>, mnm::rly::Attrs)>;
  using TDispatchList = std::vector<std::pair<OpBackend*, FMakeOpEnv>>;
  using TRegistry = ::dmlc::Registry<OpDispatch>;

 public:
  OpDispatch() = default;

  inline OpDispatch& set_name(const std::string& name);

  inline OpDispatch& add_dispatch(mnm::types::DeviceType device_type,  //
                                  const std::string& backend_name,     //
                                  const FMakeOpEnv& op_env_maker);

 public:
  static TRegistry* Registry();

  static TDispatchList& Get(const std::string& op_name, mnm::types::DeviceType device_type);

 public:
  std::string name;
  mnm::registry::PerDeviceTypeStorage<TDispatchList> dispatch;
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

  void RequestMemory(void** dest, mnm::types::Context ctx, int64_t nbytes);
  void RequestWorkspace(void** dest, mnm::types::Context ctx, int64_t nbytes);
  void RequestStream(void** dest, mnm::types::Context ctx);
  void RequestDistributed(void** dest);

 public:
  // TODO: try TVMArgs
  virtual void Execute(mnm::rly::Array<mnm::value::Value> args, mnm::rly::Attrs attrs) = 0;

 private:
  /*
   * When executor is nullptr, resource allocation is in lazy phase; Otherwise it is eager.
   * Returns type-erased `mnm::requests::Requests *` because don't want to expose it in header
   */
  void* SetExecutor(mnm::executor::Executor* executor);

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

// implementation

namespace mnm {
namespace op {

inline OpBackend& OpBackend::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

inline OpBackend& OpBackend::set_device(mnm::types::DeviceType device) {
  CHECK(this->device == mnm::types::DeviceType::kUnknown()) << "Cannot set backend's device twice";
  this->device = device;
  DeviceRegistry()->Write(device.operator int(),
                          [this](auto list) -> void { list->push_back(this); });
  return *this;
}

inline OpBackend& OpBackend::set_priority(int priority) {
  this->priority = priority;
  return *this;
}

inline OpDispatch& OpDispatch::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

inline OpDispatch& OpDispatch::add_dispatch(mnm::types::DeviceType device_type,  //
                                            const std::string& backend_name,     //
                                            const FMakeOpEnv& op_env_maker) {
  OpBackend* backend = OpBackend::Get(backend_name);
  dispatch.Write(device_type, [&](TDispatchList* list) {
    for (const auto& e : *list) {
      if (e.first == backend) {
        LOG(FATAL) << "InternalError: operator " << name
                   << " already has an implementation on backend " << backend_name;
      }
    }
    list->push_back(std::make_pair(backend, op_env_maker));
  });
  return *this;
}

}  // namespace op
}  // namespace mnm
