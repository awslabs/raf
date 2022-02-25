/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/vm/value.h
 * \brief The values that are specific to VM.
 */
#pragma once
#include "raf/ir.h"
#include "raf/memory_pool.h"
#include "raf/value.h"

namespace raf {
namespace executor {
namespace vm {
using namespace raf::ir;
using namespace raf::value;

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
  static constexpr const char* _type_key = "raf.value.vm.VMClosureValue";
  RAF_FINAL_OBJECT(VMClosureValueObj, ValueObj);
};

/*! \brief reference to vm closure. */
class VMClosureValue final : public Value {
 public:
  static VMClosureValue make(size_t func_index, std::vector<Value> free_vars);
  RAF_OBJECT_REF(VMClosureValue, Value, VMClosureValueObj);
};

/*! \brief An object representing a storage allocation. */
class StorageValueObj final : public ValueObj {
 public:
  /*! \brief The index into the VM function table. */
  std::shared_ptr<memory_pool::Memory> buffer;

  static constexpr const char* _type_key = "raf.value.vm.StorageValue";
  RAF_FINAL_OBJECT(StorageValueObj, ValueObj);
};

/*! \brief reference to storage. */
class StorageValue final : public Value {
 public:
  static StorageValue make(std::shared_ptr<memory_pool::Memory> buffer);

  RAF_OBJECT_REF(StorageValue, Value, StorageValueObj);
};

}  // namespace vm
}  // namespace executor
}  // namespace  raf
