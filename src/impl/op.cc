/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/op.cc
 * \brief MNM operator interface underlying implementation
 */
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
using ir::make_node;
using ir::NodePtr;
using ir::Downcast;
using ir::Op;
using requests::Requests;
using value::Value;
using value::OpValue;

CallValues CallValues::make() {
  NodePtr<CallValuesNode> n = make_node<CallValuesNode>();
  return CallValues(n);
}

// Implementation: OpDispatch

OpDispatch::TDispatchList* OpDispatch::Get(const Op& op, DevType device_type) {
  OpDispatch& op_dispatch = TRegistry::Get()->__REGISTER_OR_GET__(op->name);
  std::shared_ptr<TDispatchList>& list = op_dispatch.dispatch.Get(device_type);
  return list.get();
}

std::unique_ptr<OpEnv> OpDispatch::Dispatch(const CallValues& call) {
  const Op &op = Downcast<OpValue>(call->callee)->op;
  for (const auto& e : *OpDispatch::Get(op, call->ctx.device_type)) {
    const auto& maker = e.second;
    std::unique_ptr<OpEnv> op_env(static_cast<OpEnv*>(maker(call)));
    if (op_env) {
      return op_env;
    }
  }
  return nullptr;
}

OpDispatch& OpDispatch::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

OpDispatch& OpDispatch::add_dispatch(DevType device_type, const std::string& backend_name,
                                     const FMakeOpEnv& op_env_maker) {
  std::shared_ptr<TDispatchList> list = dispatch.Get(device_type);
  {
    std::lock_guard<std::mutex> lock(dispatch.mutex_);
    if (list->count(backend_name)) {
      LOG(FATAL) << "InternalError: operator " << name
                 << " already has an implementation on backend " << backend_name;
    }
    (*list)[backend_name] = op_env_maker;
  }
  return *this;
}

OpDispatch::TRegistry* OpDispatch::Registry() {
  return TRegistry::Get();
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

void OpEnv::RequestStream(void** dest, const Context& ctx, int tag_idx) {
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

void RunDeclare(const CallValues &call) {
  static const auto f_op_make_output = Op::GetAttr<FMNMDeclare>("FMNMDeclare");
  const Op &op = Downcast<OpValue>(call->callee)->op;
  const auto& f = f_op_make_output[op];
  f(call);
}

Op GetOp(const std::string& op_name) {
  return Op::Get(op_name);
}

MNM_REGISTER_GLOBAL("mnm.op.GetOp").set_body_typed(GetOp);

}  // namespace op
}  // namespace mnm
