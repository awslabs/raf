/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/op.cc
 * \brief MNM operator interface underlying implementation
 */
#include "dmlc/registry.h"
#include "mnm/executor.h"
#include "mnm/ir.h"
#include "mnm/op.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "../requests.h"
#include "../op/schema/list_args.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mnm::op::OpDispatch);
}  // namespace dmlc

namespace mnm {
namespace op {

using executor::Executor;
using ir::Array;
using ir::Attrs;
using ir::Downcast;
using ir::make_object;
using ir::ObjectPtr;
using ir::Op;
using requests::Requests;
using value::OpValue;
using value::Value;

CallValues CallValues::make(value::Value callee, ir::Attrs args) {
  ObjectPtr<CallValuesNode> n = make_object<CallValuesNode>();
  n->callee = std::move(callee);
  n->args = std::move(args);
  return CallValues(n);
}

// Implementation: OpDispatch

OpDispatch::TDispatchList* OpDispatch::Get(const Op& op, DevType device_type) {
  OpDispatch& op_dispatch = TRegistry::Get()->__REGISTER_OR_GET__(op->name);
  std::shared_ptr<TDispatchList>& list = op_dispatch.dispatch.Get(device_type);
  return list.get();
}

std::shared_ptr<OpEnv> OpDispatch::Dispatch(const CallValues& call) {
  const Op& op = Downcast<OpValue>(call->callee)->op;
  TDispatchList* list = OpDispatch::Get(op, call->ctx.device_type);
  for (const auto e : *list) {
    const auto& maker = e.maker;
    std::shared_ptr<OpEnv> op_env(static_cast<OpEnv*>(maker(call)));
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
                                     const FMakeOpEnv& op_env_maker, int plevel) {
  std::shared_ptr<TDispatchList> list = dispatch.Get(device_type);
  {
    std::lock_guard<std::mutex> lock(dispatch.mutex_);
    for (auto e : *list) {
      if (e.backend == backend_name) {
        LOG(FATAL) << "InternalError: operator " << name
                   << " already has an implementation on backend " << backend_name;
      }
    }
    OpEnvMaker maker = OpEnvMaker{plevel, backend_name, op_env_maker};
    auto it = list->begin();
    for (; it != list->end(); ++it) {
      if (plevel >= it->plevel) {
        list->insert(it, maker);
        break;
      }
    }
    if (it == list->end()) {
      list->push_back(maker);
    }
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

void OpEnv::RequestDistributed(void** dest) {
  int index = impl->distributed.size();
  impl->distributed.push_back({dest});
  if (impl->executor != nullptr) {
    impl->executor->RequestDistributed(impl.get(), index);
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

void OpEnv::SetOutputBuffer(std::vector<std::shared_ptr<memory_pool::Memory>> out_buf) {
  out_buf_ = std::move(out_buf);
}

void RunDeclare(const CallValues& call) {
  static const auto f_op_make_output = Op::GetAttrMap<FMNMDeclare>("FMNMDeclare");
  const Op& op = Downcast<OpValue>(call->callee)->op;
  const auto& f = f_op_make_output[op];
  f(call);
}

Op GetOp(const std::string& op_name) {
  return Op::Get(op_name);
}

Attrs MakeListArgs(const Array<Value>& values) {
  auto attrs = make_object<schema::ListArgs>();
  attrs->args = values;
  return Attrs(attrs);
}

Array<Value> GetListArgs(const Attrs& attrs) {
  return attrs.as<schema::ListArgs>()->args;
}

MNM_REGISTER_GLOBAL("mnm.op.GetOp").set_body_typed(GetOp);

}  // namespace op
}  // namespace mnm
