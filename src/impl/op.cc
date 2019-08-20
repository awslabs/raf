#include <dmlc/registry.h>

#include <mnm/executor.h>
#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/registry.h>
#include <mnm/value.h>

#include "../requests.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mnm::op::OpBackend);
DMLC_REGISTRY_ENABLE(::mnm::op::OpDispatch);
}  // namespace dmlc

namespace mnm {
namespace op {

using executor::Executor;
using ir::Array;
using ir::Attrs;
using ir::make_node;
using ir::NodePtr;
using ir::Op;
using requests::Requests;
using value::Value;

// Implementation: OpBackend

OpBackend& OpBackend::set_device(DevType device) {
  CHECK(this->device == DevType::kUnknown()) << "Cannot set backend's device twice";
  this->device = device;
  TDeviceRegistry::EntryPtr& ptr = DeviceRegistry()->Get(device);
  {
    std::unique_lock<std::mutex> lock = DeviceRegistry()->GrabLock();
    ptr->push_back(this);
  }
  return *this;
}

OpBackend::TDeviceRegistry* OpBackend::DeviceRegistry() {
  static OpBackend::TDeviceRegistry* registry = new OpBackend::TDeviceRegistry();
  return registry;
}

OpBackend* OpBackend::Get(const std::string& name) {
  return &Registry()->__REGISTER_OR_GET__(name);
}

OpBackend::TRegistry* OpBackend::Registry() {
  return TRegistry::Get();
}

// Implementation: OpDispatch

OpDispatch::TDispatchList* OpDispatch::Get(const std::string& op_name, DevType device_type) {
  OpDispatch& op_dispatch = Registry()->__REGISTER_OR_GET__(op_name);
  std::shared_ptr<TDispatchList>& list = op_dispatch.dispatch.Get(device_type);
  return list.get();
}

OpDispatch::TDispatchList* OpDispatch::Get(const Op& op, DevType device_type) {
  return OpDispatch::Get(op->name, device_type);
}

OpDispatch::TRegistry* OpDispatch::Registry() {
  return TRegistry::Get();
}

std::unique_ptr<OpEnv> OpDispatch::Dispatch(const ir::Op& op, const OpInfo& info,
                                            const ir::Array<value::Value>& args,
                                            const ir::Attrs& attrs) {
  for (const auto& e : *OpDispatch::Get(op, info->ctx.device_type)) {
    const auto& maker = e.second;
    std::unique_ptr<OpEnv> op_env(static_cast<OpEnv*>(maker(args, info->output, attrs)));
    if (op_env) {
      return op_env;
    }
  }
  LOG(FATAL) << "Cannot dispatch " << op->name << "@" << info->ctx.c_str();
  throw;
}

OpDispatch& OpDispatch::add_dispatch(DevType device_type,              //
                                     const std::string& backend_name,  //
                                     const FMakeOpEnv& op_env_maker) {
  OpBackend* backend = OpBackend::Get(backend_name);
  auto& list = dispatch.Get(device_type);
  {
    std::unique_lock<std::mutex> lock = dispatch.GrabLock();
    for (const auto& e : *list) {
      if (e.first == backend) {
        LOG(FATAL) << "InternalError: operator " << name
                   << " already has an implementation on backend " << backend_name;
      }
    }
    list->push_back(std::make_pair(backend, op_env_maker));
  }
  return *this;
}

// Implementation: OpInfo
OpInfo OpInfo::make(Value output, Context ctx) {
  NodePtr<OpInfoNode> n = make_node<OpInfoNode>();
  n->output = std::move(output);
  n->ctx = std::move(ctx);
  return OpInfo(n);
}

// Implementation: OpEnv

class OpEnv::Impl {
 public:
  friend class OpEnv;

  Impl() : requests(new Requests()) {
  }

  ~Impl() = default;

  void RequestMemory(Value& value, const Context& ctx, const int64_t& nbytes) {
    if (executor == nullptr) {
      requests->memory.push_back({value, ctx, nbytes});
    } else {
      executor->RequestMemory(value, ctx, nbytes);
    }
  }

  void RequestWorkspace(void** dest, const Context& ctx, const int64_t& nbytes) {
    if (executor == nullptr) {
      requests->workspace.push_back({dest, ctx, nbytes});
    } else {
      executor->RequestWorkspace(dest, ctx, nbytes);
    }
  }

  void RequestStream(void** dest, const Context& ctx, int tag_idx, int index) {
    if (executor == nullptr) {
      requests->stream.push_back({dest, ctx, tag_idx, index});
    } else {
      executor->RequestStream(dest, ctx, tag_idx, index);
    }
  }

  std::unique_ptr<Requests> SetExecutor(Executor* exec) {
    CHECK(this->executor == nullptr);
    this->executor = exec;
    return std::unique_ptr<Requests>(requests.release());
  }

 public:
  Executor* executor = nullptr;
  std::unique_ptr<Requests> requests;
};

OpEnv::OpEnv() : impl(new OpEnv::Impl()) {
}

OpEnv::~OpEnv() = default;

void OpEnv::RequestMemory(Value& value, const Context& ctx, int64_t nbytes) {
  impl->RequestMemory(value, ctx, nbytes);
}

void OpEnv::RequestWorkspace(void** dest, const Context& ctx, int64_t nbytes) {
  impl->RequestWorkspace(dest, ctx, nbytes);
}

void OpEnv::RequestStream(void** dest, const Context& ctx, int tag_idx, int index) {
  impl->RequestStream(dest, ctx, tag_idx, index);
}

std::unique_ptr<Requests> OpEnv::SetExecutor(Executor* executor) {
  return impl->SetExecutor(executor);
}

OpInfo _MakeOutput(std::string op_name, Array<Value> args, Attrs attrs) {
  return MakeOutput(Op::Get(op_name), args, attrs);
}

OpInfo MakeOutput(const Op& op, const Array<Value>& args, const Attrs& attrs) {
  static const auto f_op_make_output = Op::GetAttr<FOpMakeOutput>("FOpMakeOutput");
  const auto& f = f_op_make_output[op];
  return f(args, attrs);
}

Op GetOp(const std::string& op_name) {
  return Op::Get(op_name);
}

MNM_REGISTER_GLOBAL("mnm.op.MakeOutput").set_body_typed(_MakeOutput);

MNM_REGISTER_GLOBAL("mnm.op.GetOp").set_body_typed(GetOp);

}  // namespace op
}  // namespace mnm
