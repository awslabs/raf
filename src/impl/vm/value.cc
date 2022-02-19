/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/impl/vm/value.cc
 * \brief The implementation for vm values.
 */

#include "mnm/vm/value.h"

namespace mnm {
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

MNM_REGISTER_OBJECT_REFLECT(VMClosureValueObj);

}  // namespace vm
}  // namespace executor
}  // namespace mnm
