/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/vm/value.cc
 * \brief The implementation for vm values.
 */

#include "raf/vm/value.h"

namespace raf {
namespace executor {
namespace vm {

VMClosureValue VMClosureValue::make(size_t func_index, std::vector<Value> free_vars) {
  auto ptr = make_object<VMClosureValueObj>();
  ptr->func_index = func_index;
  ptr->free_vars = std::move(free_vars);
  return VMClosureValue(ptr);
}

StorageValue StorageValue::make(std::shared_ptr<memory_pool::Memory> buffer) {
  auto node = make_object<StorageValueObj>();
  node->buffer = std::move(buffer);
  return StorageValue(node);
}

RAF_REGISTER_OBJECT_REFLECT(VMClosureValueObj);

}  // namespace vm
}  // namespace executor
}  // namespace raf
