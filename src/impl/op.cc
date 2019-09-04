#include <dmlc/registry.h>

#include <mnm/executor.h>
#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/registry.h>
#include <mnm/value.h>

#include "../requests.h"

namespace dmlc {
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

// Implementation: OpDispatch

OpDispatch::TDispatchList* OpDispatch::Get(const Op& op, DevType device_type) {
  OpDispatch& op_dispatch = TRegistry::Get()->__REGISTER_OR_GET__(op->name);
  std::shared_ptr<TDispatchList>& list = op_dispatch.dispatch.Get(device_type);
  return list.get();
}

std::unique_ptr<OpEnv> OpDispatch::Dispatch(const Op& op, const OpInfo& info,
                                            const Array<Value>& args, const Attrs& attrs) {
  for (const auto& e : *OpDispatch::Get(op, info->ctx.device_type)) {
    const auto& maker = e.second;
    std::unique_ptr<OpEnv> op_env(static_cast<OpEnv*>(maker(args, info, attrs)));
    if (op_env) {
      return op_env;
    }
  }
  LOG(FATAL) << "Cannot dispatch " << op->name << "@" << info->ctx.c_str();
  throw;
}

OpDispatch& OpDispatch::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

OpDispatch& OpDispatch::add_dispatch(DevType device_type, const std::string& backend_name,
                                     const FMakeOpEnv& op_env_maker) {
  std::shared_ptr<TDispatchList> list = dispatch.Get(device_type);
  {
    std::unique_lock<std::mutex> lock = dispatch.GrabLock();
    if (list->count(backend_name)) {
      LOG(FATAL) << "InternalError: operator " << name
                 << " already has an implementation on backend " << backend_name;
    }
    (*list)[backend_name] = op_env_maker;
  }
  return *this;
}

OpDispatch::TRegistry *OpDispatch::Registry() {
  return TRegistry::Get();
}

// Implementation: OpInfo
OpInfo OpInfo::make(Value output, Context ctx, bool computational) {
  NodePtr<OpInfoNode> n = make_node<OpInfoNode>();
  n->output = std::move(output);
  n->ctx = std::move(ctx);
  n->computational = computational;
  return OpInfo(n);
}

// Implementation: OpEnv
class OpEnv::Impl : public Requests {
 public:
  executor::Executor* executor = nullptr;
};

OpEnv::OpEnv() : impl(new OpEnv::Impl()) {
}

OpEnv::~OpEnv() {
  if (impl->executor != nullptr) {
    impl->executor->OnDestruct(this);
  }
}

void OpEnv::RequestMemory(void** dest, const Context& ctx, int64_t nbytes) {
  int index = impl->memory.size();
  impl->memory.push_back({dest, ctx, nbytes, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestMemory(impl.get(), index);
  }
}

void OpEnv::RequestWorkspace(void** dest, const Context& ctx, int64_t nbytes) {
  int index = impl->workspace.size();
  impl->workspace.push_back({dest, ctx, nbytes, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestWorkspace(impl.get(), index);
  }
}

void OpEnv::RequestStream(void** dest, const Context& ctx, int tag_idx, int stream_index) {
  int index = impl->stream.size();
  impl->stream.push_back({dest, ctx, tag_idx, index, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestStream(impl.get(), index);
  }
}

void OpEnv::BindExecutor(Executor* executor) {
  CHECK(impl->executor != nullptr);
  impl->executor = executor;
  executor->OnBind(this);
}

std::shared_ptr<Requests> OpEnv::GetRequests() const {
  return this->impl;
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
