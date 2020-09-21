/*!
 * Copyright (c) 2020 by Contributors
 * \file include/vm/value.h
 * \brief The values that are specific to VM.
 */
#pragma once
#include "mnm/ir.h"
#include "mnm/memory_pool.h"
#include "mnm/value.h"

namespace mnm {
namespace executor {
namespace vm {
using namespace mnm::ir;
using namespace mnm::value;

/*!
 * \brief An object representing a vm closure.
 */
class VMClosureValueObj final : public ValueObj {
 public:
  /*!
   * \brief The index into the function list. The function could be any
   * function object that is compatible to the VM runtime.
   */
  int64_t func_index;
  /*! \brief The free variables of the closure. */
  Array<Value> free_vars;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_func_index", &func_index);
    v->Visit("_free_vars", &free_vars);
  }
  static constexpr const char* _type_key = "mnm.value.vm.VMClosureValue";
  MNM_FINAL_OBJECT(ClosureValueObj, ValueObj);
};

/*! \brief reference to vm closure. */
class VMClosureValue final : public Value {
 public:
  static VMClosureValue make(size_t func_index, std::vector<Value> free_vars);
  MNM_OBJECT_REF(VMClosureValue, Value, VMClosureValueObj);
};

/*! \brief An object representing a storage allocation. */
class StorageValueObj final : public ValueObj {
 public:
  /*! \brief The index into the VM function table. */
  std::shared_ptr<memory_pool::Memory> buffer;

  static constexpr const char* _type_key = "mnm.value.vm.StorageValue";
  MNM_FINAL_OBJECT(StorageValueObj, ValueObj);
};

/*! \brief reference to storage. */
class StorageValue final : public Value {
 public:
  static StorageValue make(std::shared_ptr<memory_pool::Memory> buffer);

  MNM_OBJECT_REF(StorageValue, Value, StorageValueObj);
};

}  // namespace vm
}  // namespace executor
}  // namespace  mnm