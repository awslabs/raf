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
DMLC_REGISTRY_ENABLE(::mnm::op::OpDialect);
DMLC_REGISTRY_ENABLE(::mnm::op::OpEnvMaker);
}  // namespace dmlc

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using executor::Executor;
using requests::Requests;

CallValues CallValues::make(value::Value callee, ir::Attrs args) {
  ObjectPtr<CallValuesNode> n = make_object<CallValuesNode>();
  n->callee = std::move(callee);
  n->args = std::move(args);
  return CallValues(n);
}

template <typename TList>
TList get_preferred_backends(TList* default_list) {
  const static auto* f = registry::Registry::Get("mnm.backend.preferred_backends");
  if (f == nullptr) {
    return *default_list;
  }
  ObjectRef preferred_backends_obj = (*f)();
  TList ret;
  if (preferred_backends_obj.defined()) {
    Array<String> preferred_backends = Downcast<Array<String>>(preferred_backends_obj);
    for (const auto e : *default_list) {
      for (const auto& backend : preferred_backends) {
        if (e.backend == backend) {
          ret.push_back(e);
          break;
        }
      }
    }
  } else {
    ret = *default_list;
  }
  return ret;
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

const OpEnvMaker* OpEnvMaker::Get(const ir::Op& op) {
  return TRegistry::Get()->Find(op->name);
}

std::shared_ptr<OpEnv> OpEnvMaker::Make(const ir::Op& op, const CallValues& call) {
  auto maker = OpEnvMaker::Get(op);
  CHECK(maker) << "Cannot find an OpEnvMaker registered to " << op->name;
  auto env = (*maker)(call);
  return std::shared_ptr<OpEnv>(env);
}

// Implementation: OpDialect

OpDialect& OpDialect::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

OpDialect& OpDialect::add_dialect(DevType device_type, const std::string& dialect_name,
                                  const std::string& dialect_op, int plevel) {
  std::shared_ptr<TDialectList> list = dialects.Get(device_type);
  {
    std::lock_guard<std::mutex> lock(dialects.mutex_);
    // sanity check
    for (const auto& e : *list) {
      CHECK_NE(e.backend, dialect_name)
          << "InternalError: base operator " << this->name
          << " already has a registration on dialect " << dialect_name;
      CHECK_NE(plevel, e.plevel) << "InternalError: base operator " << this->name
                                 << " already has a registration on dialect " << e.backend
                                 << " with same plevel=" << plevel << " as this dialect "
                                 << dialect_name;
    }
    // insert the new dialect op entry
    auto entry = DialectOpEntry{dialect_name, dialect_op, plevel};
    auto it = list->begin();
    for (; it != list->end(); ++it) {
      if (plevel > it->plevel) {
        list->insert(it, entry);
        break;
      }
    }
    if (it == list->end()) {
      list->push_back(entry);
    }
  }
  return *this;
}

OpDialect::TRegistry* OpDialect::Registry() {
  return TRegistry::Get();
}

OpDialect::TDialectList OpDialect::GetDispatchList(const ir::Op& op, DevType device_type) {
  OpDialect& dialect = TRegistry::Get()->__REGISTER_OR_GET__(op->name);
  std::shared_ptr<TDialectList>& list = dialect.dialects.Get(device_type);
  return get_preferred_backends<TDialectList>(list.get());
}

ir::Op OpDialect::Dispatch(const ir::Op& base_op, DevType device_type,
                           std::vector<std::string> skip_dialects) {
  TDialectList list = OpDialect::GetDispatchList(base_op, device_type);
  for (const auto& e : list) {
    bool skip = false;
    for (const auto& dialect : skip_dialects) {
      if (dialect == e.backend) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      auto dialect_op = Op::Get(e.dialect_op);
      dialect_op->op_type = base_op->op_type;
      return dialect_op;
    }
  }
  return ir::Op();
}

ir::Op OpDialect::Dispatch(const ir::Op& base_op, DevType device_type, const std::string& dialect) {
  TDialectList list = OpDialect::GetDispatchList(base_op, device_type);
  for (const auto& e : list) {
    if (dialect == e.backend) {
      auto dialect_op = Op::Get(e.dialect_op);
      dialect_op->op_type = base_op->op_type;
      return dialect_op;
    }
  }
  return ir::Op();
}

// Implementation: FusedOpDispatch

FusedOpDispatch* FusedOpDispatch::Get() {
  static FusedOpDispatch inst;
  return &inst;
}

FusedOpDispatch::TDispatchList* FusedOpDispatch::Get(DevType device_type) {
  FusedOpDispatch* func_dispatch = Get();
  std::shared_ptr<TDispatchList>& list = func_dispatch->dispatch.Get(device_type);
  return list.get();
}

std::shared_ptr<OpEnv> FusedOpDispatch::Dispatch(const CallValues& call) {
  TDispatchList* default_list = FusedOpDispatch::Get(call->device.device_type);
  TDispatchList list = get_preferred_backends<TDispatchList>(default_list);
  for (const auto& e : list) {
    const auto& maker = e.maker;
    std::shared_ptr<OpEnv> func_env(static_cast<OpEnv*>(maker(call)));
    if (func_env) {
      return func_env;
    }
  }
  return nullptr;
}

FusedOpDispatch& FusedOpDispatch::add_dispatch(DevType device_type, const std::string& backend_name,
                                               const OpEnvMaker::FMakeOpEnv& func_env_maker,
                                               int plevel) {
  std::shared_ptr<TDispatchList> list = dispatch.Get(device_type);
  {
    std::lock_guard<std::mutex> lock(dispatch.mutex_);
    for (auto e : *list) {
      if (e.backend == backend_name) {
        LOG(FATAL) << "InternalError: fused functions "
                   << "already have an implementation on backend " << backend_name;
      }
    }
    FuncEnvMaker maker = FuncEnvMaker{plevel, backend_name, OpEnvMaker(func_env_maker)};
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

std::shared_ptr<OpEnv> DispatchOp(const CallValues& call) {
  static auto fbase_op = Op::GetAttrMap<TMNMBaseOp>("TMNMBaseOp");
  Op op = Downcast<OpValue>(call->callee)->op;
  std::string skip_dialect = "";
  if (IsDialectOp(op)) {
    // dialect op, directly call the OpEnvMaker registered to it
    auto env = OpEnvMaker::Make(op, call);
    if (env != nullptr) {
      return env;
    }
    // failed to generate OpEnv, lift back to base op and try other dialects
    skip_dialect = GetDialect(op);
    CHECK(fbase_op.count(op)) << "Dialect op " << op->name << " does not have attribute TMNMBaseOp";
    auto base_op = Op::Get(fbase_op[op]);
    base_op->op_type = op->op_type;
    op = base_op;
  }
  // Iterate over all dialect ops based on plevel.
  auto dialect_list = OpDialect::GetDispatchList(op, call->device.device_type);
  for (const auto& entry : dialect_list) {
    if (entry.backend == skip_dialect) {
      continue;
    }
    auto dialect_op = Op::Get(entry.dialect_op);
    dialect_op->op_type = op->op_type;
    if (auto env = OpEnvMaker::Make(dialect_op, call)) {
      return env;
    }
  }
  LOG(FATAL) << "Cannot find a valid dispatch for op " << op->name;
  return nullptr;
}

std::shared_ptr<OpEnv> Dispatch(const CallValues& call) {
  if (call->callee.as<value::OpValueObj>()) {
    return DispatchOp(call);
  } else if (call->callee.as<value::ClosureValueObj>()) {
    return FusedOpDispatch::Dispatch(call);
  }
  LOG(FATAL) << "call->op type " << call->callee->GetTypeKey() << " unsupported";
  return nullptr;
}

Attrs MakeListArgs(const Array<Value>& values) {
  auto attrs = make_object<schema::ListArgs>();
  attrs->args = values;
  return Attrs(attrs);
}

Array<Value> GetListArgs(const Attrs& attrs) {
  return attrs.as<schema::ListArgs>()->args;
}

std::string GetDialect(const Op& op) {
  static auto fdialect = Op::GetAttrMap<TMNMDialect>("TMNMDialect");
  if (fdialect.count(op)) {
    return fdialect[op];
  }
  return "";
}

bool IsDialectOp(const Op& op) {
  static auto fdialect = Op::GetAttrMap<TMNMDialect>("TMNMDialect");
  return fdialect.count(op) > 0;
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

MNM_REGISTER_GLOBAL("mnm.op.GetOp").set_body_typed(GetOp);

}  // namespace op
}  // namespace mnm
