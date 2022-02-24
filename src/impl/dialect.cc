/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/dialect.cc
 * \brief The implementation for dialect and dialect operator.
 */
#include <stack>
#include <dmlc/registry.h>
#include <dmlc/thread_local.h>

#include "raf/executor.h"
#include "raf/ir.h"
#include "raf/op.h"
#include "raf/dialect.h"
#include "raf/dialect.h"
#include "raf/registry.h"
#include "raf/value.h"
#include "../requests.h"
#include "../op/schema/list_args.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::raf::op::Dialect);
DMLC_REGISTRY_ENABLE(::raf::op::OpDialect);
}  // namespace dmlc

namespace raf {
namespace op {

using namespace raf::ir;

// Implementation: DialectPreference

RAF_REGISTER_OBJECT_REFLECT(DialectPreferenceObj);

struct DialectPrefThreadLocalEntry {
  /*! \brief The dialect scope stack */
  std::stack<DialectPreference> context_stack;
};

/*! \brief Thread local store to hold the DialectPreference context stack. */
using DialectPrefThreadLocalStore = dmlc::ThreadLocalStore<DialectPrefThreadLocalEntry>;

DialectPreference::DialectPreference(Array<String> dialects) {
  auto n = make_object<DialectPreferenceObj>();
  n->preferred_dialects = dialects;
  data_ = std::move(n);
}

void DialectPreference::EnterWithScope() {
  DialectPrefThreadLocalEntry* entry = DialectPrefThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void DialectPreference::ExitWithScope() {
  DialectPrefThreadLocalEntry* entry = DialectPrefThreadLocalStore::Get();
  ICHECK(!entry->context_stack.empty());
  entry->context_stack.pop();
}

const DialectPreference* DialectPreference::Current() {
  DialectPrefThreadLocalEntry* entry = DialectPrefThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return &entry->context_stack.top();
  }
  return nullptr;
}

// Implementation: Dialect

Dialect& Dialect::set_name(std::string&& name) {
  this->name = std::move(name);
  return *this;
}

Dialect& Dialect::set_enable(DevType device_type) {
  enable_devices_.push_back(device_type);
  return *this;
}

bool Dialect::is_enabled(DevType device_type) const {
  for (auto dev : enable_devices_) {
    if (dev == device_type) {
      return true;
    }
  }
  return false;
}

Dialect::TRegistry* Dialect::Registry() {
  return TRegistry::Get();
}

const Dialect* Dialect::Get(const std::string& dialect_name) {
  return TRegistry::Get()->Find(dialect_name);
}

bool Dialect::IsEnabled(const std::string& dialect, DevType device_type) {
  const Dialect* d = TRegistry::Get()->Find(dialect);
  ICHECK(d) << "Dialect " << dialect << " is not registered.";
  return d->is_enabled(device_type);
}

std::vector<std::string> Dialect::GetEnabledDialects(DevType device_type) {
  std::vector<std::string> ret;
  for (auto dialect : TRegistry::List()) {
    if (dialect->is_enabled(device_type)) {
      ret.push_back(dialect->name);
    }
  }
  return ret;
}

// Implementation: OpDialect

OpDialect& OpDialect::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

OpDialect& OpDialect::add_dialect(const std::string& dialect_name, const std::string& dialect_op,
                                  int plevel) {
  std::lock_guard<std::mutex> lock(dialect_ops_mu_);
  // sanity check
  for (const auto& e : dialect_ops_) {
    CHECK_NE(e.dialect, dialect_name) << "InternalError: base operator " << this->name
                                      << " already has a registration on dialect " << dialect_name;
    CHECK_NE(plevel, e.plevel) << "InternalError: base operator " << this->name
                               << " already has a registration on dialect " << e.dialect
                               << " with same plevel=" << plevel << " as this dialect "
                               << dialect_name;
  }
  // insert the new dialect op entry
  auto entry = DialectOpEntry{dialect_name, dialect_op, plevel};
  auto it = dialect_ops_.begin();
  for (; it != dialect_ops_.end(); ++it) {
    if (plevel > it->plevel) {
      dialect_ops_.insert(it, entry);
      break;
    }
  }
  if (it == dialect_ops_.end()) {
    dialect_ops_.push_back(entry);
  }
  return *this;
}

OpDialect::TRegistry* OpDialect::Registry() {
  return TRegistry::Get();
}

OpDialect::TDialectList OpDialect::GetDispatchList(const ir::Op& op, DevType device_type) {
  const OpDialect& op_dialect = TRegistry::Get()->__REGISTER_OR_GET__(op->name);
  OpDialect::TDialectList default_list;
  for (auto entry : op_dialect.dialect_ops_) {
    auto dialect = Dialect::Get(entry.dialect);
    ICHECK(dialect) << "Dialect " << entry.dialect << " is not registered.";
    if (dialect->is_enabled(device_type)) {
      default_list.push_back(entry);
    }
  }

  auto dialect_pref = DialectPreference::Current();
  if (dialect_pref == nullptr) {
    return default_list;
  }
  OpDialect::TDialectList preferred_list;
  for (auto dialect : (*dialect_pref)->preferred_dialects) {
    for (auto& e : default_list) {
      if (e.dialect == dialect) {
        preferred_list.push_back(e);
        break;
      }
    }
  }
  return preferred_list;
}

ir::Op OpDialect::Dispatch(const ir::Op& base_op, DevType device_type,
                           const std::vector<std::string>& skip_dialects) {
  TDialectList list = OpDialect::GetDispatchList(base_op, device_type);
  for (const auto& e : list) {
    if (e.plevel <= 0) {
      continue;
    }
    bool skip = false;
    for (const auto& dialect : skip_dialects) {
      if (dialect == e.dialect) {
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

ir::Op OpDialect::Lower(const ir::Op& base_op, const std::string& dialect) {
  const OpDialect& op_dialect = TRegistry::Get()->__REGISTER_OR_GET__(base_op->name);
  for (const auto& e : op_dialect.dialect_ops_) {
    if (dialect == e.dialect) {
      auto dialect_op = Op::Get(e.dialect_op);
      dialect_op->op_type = base_op->op_type;
      return dialect_op;
    }
  }
  return ir::Op();
}

// Implementation: DialectPattern

void DialectFusePattern::AddPattern(const DFPattern& pattern, const std::string& dialect,
                                    int plevel, const std::string& name) {
  auto pattern_list = DialectFusePattern::Get();
  auto it = pattern_list->begin();
  DialectFusePattern entry{pattern, dialect, plevel, name};
  for (; it != pattern_list->end(); ++it) {
    ICHECK_NE(plevel, it->plevel) << "InternalError: dialect pattern with plevel " << plevel
                                  << " is already registered";
    if (plevel > it->plevel) {
      pattern_list->insert(it, entry);
      break;
    }
  }
  if (it == pattern_list->end()) {
    pattern_list->push_back(entry);
  }
}

DialectFusePattern::PatternList* DialectFusePattern::Get() {
  static DialectFusePattern::PatternList dialect_fuse_patterns;
  return &dialect_fuse_patterns;
}

// Implementation: helper functions

std::string GetDialect(const Op& op) {
  static auto fdialect = Op::GetAttrMap<TRAFDialect>("TRAFDialect");
  if (fdialect.count(op)) {
    return fdialect[op];
  }
  return "";
}

bool IsDialectOp(const Op& op) {
  static auto fdialect = Op::GetAttrMap<TRAFDialect>("TRAFDialect");
  return fdialect.count(op) > 0;
}

ir::Op GetBaseOp(const ir::Op& dialect_op) {
  static auto fbase_op = Op::GetAttrMap<TRAFBaseOp>("TRAFBaseOp");
  CHECK(fbase_op.count(dialect_op))
      << "Dialect op " << dialect_op->name << " does not have attribute TRAFBaseOp";
  return Op::Get(fbase_op[dialect_op]);
}

bool DialectEnabled(const std::string& dialect, int dev_type) {
  return Dialect::IsEnabled(dialect, DevType(dev_type));
}

Array<String> GetAllDialects() {
  Array<String> dialects;
  for (auto name : Dialect::Registry()->ListAllNames()) {
    dialects.push_back(name);
  }
  return dialects;
}

RAF_REGISTER_GLOBAL("raf.op.DialectPreference").set_body_typed([](Array<String> dialects) {
  return DialectPreference(dialects);
});
RAF_REGISTER_GLOBAL("raf.op.DialectPrefEnterScope").set_body_typed([](DialectPreference pref) {
  pref.EnterWithScope();
});
RAF_REGISTER_GLOBAL("raf.op.DialectPrefExitScope").set_body_typed([](DialectPreference pref) {
  pref.ExitWithScope();
});
RAF_REGISTER_GLOBAL("raf.op.AddDialectPattern").set_body_typed(DialectFusePattern::AddPattern);
RAF_REGISTER_GLOBAL("raf.op.DialectEnabled").set_body_typed(DialectEnabled);
RAF_REGISTER_GLOBAL("raf.op.GetAllDialects").set_body_typed(GetAllDialects);

}  // namespace op
}  // namespace raf
