/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/op.cc
 * \brief RAF operator interface underlying implementation
 */
#include <tvm/runtime/device_api.h>
#include "dmlc/registry.h"
#include "raf/executor.h"
#include "raf/ir_ext.h"
#include "raf/op.h"
#include "raf/dialect.h"
#include "raf/registry.h"
#include "raf/pass.h"
#include "raf/value.h"
#include "raf/device_api.h"
#include "../requests.h"
#include "../op/schema/list_args.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::raf::op::OpEnvMaker);
}  // namespace dmlc

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using executor::Executor;
using requests::Requests;

CallValues CallValues::make(value::Value callee, ir::Attrs args) {
  ObjectPtr<CallValuesNode> n = make_object<CallValuesNode>();
  n->callee = std::move(callee);
  n->args = std::move(args);
  return CallValues(n);
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

void OpEnv::RequestWorkspace(void** dest, const Device& dev, int64_t nbytes) {
  int index = impl->workspace.size();
  impl->workspace.push_back({dest, dev, nbytes, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestWorkspace(impl.get(), index);
  }
}

void OpEnv::RequestStream(void** dest, const Device& dev, int tag_idx) {
  int index = impl->stream.size();
  impl->stream.push_back({dest, dev, tag_idx, index, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestStream(impl.get(), index);
  }
}

void OpEnv::RequestDistributed(void** dest, const std::string& name, const value::Value rank_list) {
  int index = impl->distributed.size();
  impl->distributed.push_back({dest, name, rank_list});
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

void OpEnv::SetStreamForAllBackends(Device device, void* stream) {
#ifdef RAF_USE_CUDA
  device_api::DeviceAPI::Get(DevType::kCUDA())->SetStream(device, stream);
#endif
}

// Implementation: OpEnvMaker

OpEnvMaker& OpEnvMaker::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

OpEnvMaker& OpEnvMaker::set_func(FMakeOpEnv func) {
  func_ = func;
  return *this;
}

OpEnvMaker::TRegistry* OpEnvMaker::Registry() {
  return TRegistry::Get();
}

const OpEnvMaker* OpEnvMaker::Get(const std::string& op_name) {
  return TRegistry::Get()->Find(op_name);
}

OpEnvPtr OpEnvMaker::Make(const std::string& op_name, const CallValues& call) {
  auto maker = OpEnvMaker::Get(op_name);
  CHECK(maker) << "Cannot find an OpEnvMaker registered to " << op_name;
  auto env = (*maker)(call);
  return OpEnvPtr(env);
}

// Implementation : helper functions

OpEnvPtr DispatchSingleOp(const CallValues& call) {
  std::vector<std::string> dispatch_error_msgs;

  Op op = Downcast<OpValue>(call->callee)->op;
  std::string skip_dialect = "";
  // Try dispatch directly
  auto maker = OpEnvMaker::Get(op->name);
  if (maker != nullptr) {
    auto env = OpEnvPtr((*maker)(call));
    if (env && !env->HasError()) {
      DLOG(INFO) << "Dispatch to " << op->name;
      return env;
    } else if (env) {
      for (auto msg : env->error_msgs) {
        dispatch_error_msgs.push_back(msg);
      }
    }
  }
  if (IsDialectOp(op)) {
    skip_dialect = GetDialect(op);
    auto base_op = GetBaseOp(op);
    base_op->op_type = op->op_type;
    op = base_op;
  }
  // Iterate over all dialect ops based on plevel.
  auto dialect_list = OpDialect::GetDispatchList(op, call->device.device_type());
  for (const auto& entry : dialect_list) {
    if (entry.dialect == skip_dialect) {
      continue;
    }
    auto dialect_op = Op::Get(entry.dialect_op);
    dialect_op->op_type = op->op_type;
    auto maker = OpEnvMaker::Get(dialect_op->name);
    if (maker != nullptr) {
      auto env = OpEnvPtr((*maker)(call));
      if (env && !env->HasError()) {
        DLOG(INFO) << "Dispatch to " << dialect_op->name;
        return env;
      } else if (env) {
        for (auto msg : env->error_msgs) {
          dispatch_error_msgs.push_back(msg);
        }
      }
    }
  }

  std::stringstream ss;
  ss << "Cannot find a valid dispatch for op " << op->name << ":";
  for (auto msg : dispatch_error_msgs) {
    ss << "\n" << msg;
  }
  LOG(FATAL) << ss.str();
  return nullptr;
}

OpEnvPtr DispatchFusedOp(const CallValues& call) {
  auto clo = Downcast<ClosureValue>(call->callee);
  auto func = clo->func;
  ICHECK(func->HasNonzeroAttr(attr::kPrimitive))
      << "Encountered a non-primitive function when dispatching a call";
  auto dialect = func->GetAttr<String>(attr::kDialect);
  ICHECK(dialect.defined()) << "Fused function doesn't have dialect attribute: "
                            << ir::AsText(func);
  std::ostringstream os;
  os << "raf.op." << dialect.value() << "._fused_op";
  auto op_env = OpEnvMaker::Make(os.str(), call);
  if (!op_env || op_env->HasError()) {
    std::stringstream ss;
    ss << "Failed to dispatch fused op:";
    if (op_env) {
      for (auto msg : op_env->error_msgs) {
        ss << "\n\t" << msg;
      }
    }
    ss << "\nName: " << os.str() << "\n" << ir::AsText(func);
    LOG(FATAL) << ss.str();
  }
  return op_env;
}

OpEnvPtr Dispatch(const CallValues& call) {
  if (call->callee.as<value::OpValueObj>()) {
    return DispatchSingleOp(call);
  } else if (call->callee.as<value::ClosureValueObj>()) {
    return DispatchFusedOp(call);
  }
  LOG(FATAL) << "call->op type " << call->callee->GetTypeKey() << " unsupported";
  return nullptr;
}

CallValues CreateDummyCallValues(Call call, Device device) {
  auto call_node = call.as<CallNode>();
  CHECK(call_node != nullptr);
  std::vector<Value> inputs(call_node->args.size());
  for (int i = 0; i < inputs.size(); i++) {
    const Expr& arg_expr = call_node->args[i];
    if (auto relay_const_node = arg_expr.as<RelayConstantNode>()) {
      const auto* node = static_cast<const ConstantNode*>(relay_const_node);
      CHECK(node != nullptr);
      inputs[i] = Downcast<Value>(node->value);
    } else {
      inputs[i] = value::CreateDummyValueFromType(arg_expr->checked_type(), device);
    }
  }
  Value output = value::CreateDummyValueFromType(call->checked_type(), device);
  CallValues call_values = CallValues::make();
  Expr callee = call_node->op;
  if (auto fused_op_node = callee.as<FunctionNode>()) {
    auto fused_op = GetRef<Function>(fused_op_node);
    Array<Var> free_vars = pass::FreeVars(fused_op);
    CHECK_EQ(free_vars.size(), 0)
        << "Closure function call with captured vars has not been implemented.";
    call_values->callee = ClosureValue::make({}, fused_op);
    call_values->args = MakeListArgs(inputs);
  } else {
    auto op_node = callee.as<OpNode>();
    auto op = GetRef<Op>(op_node);
    CHECK_NOTNULL(op_node);
    call_values->callee = OpValue::make(GetRef<Op>(op_node));
    call_values->args = GetOpAttr<FRAFSchema>(op, "FRAFSchema")(inputs);
  }
  call_values->device = device;
  call_values->out = output;
  return call_values;
}

Attrs MakeListArgs(const Array<Value>& values) {
  auto attrs = make_object<schema::ListArgs>();
  attrs->args = values;
  return Attrs(attrs);
}

Array<Value> GetListArgs(const Attrs& attrs) {
  return attrs.as<schema::ListArgs>()->args;
}

std::string GetUniqueName(std::string name) {
  static std::unordered_map<std::string, int> name_map;
  for (size_t i = 0; i < name.length(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  while (true) {
    auto it = name_map.find(name);
    if (it == name_map.end()) {
      name_map[name] = 1;
      return name;
    } else {
      std::ostringstream os;
      os << name << "_" << it->second;
      ++(it->second);
      name = os.str();
    }
  }
  return name;
}

std::string TruncateName(std::string name) {
  constexpr static size_t kMaxFuncNameLength = 80;
  if (name.size() > kMaxFuncNameLength) {
    std::stringstream truncated_name;
    truncated_name << name.substr(0, kMaxFuncNameLength);
    truncated_name << "_" << std::hash<std::string>{}(name) << "_";
    name = truncated_name.str();
  }
  return name;
}

Op GetOp(const std::string& op_name) {
  return Op::Get(op_name);
}

RAF_REGISTER_GLOBAL("raf.op.GetOp").set_body_typed(GetOp);

RAF_REGISTER_OBJECT_REFLECT(CallValuesNode);

}  // namespace op
}  // namespace raf
