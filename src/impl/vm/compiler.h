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
 * \file src/impl/vm/compiler.h
 * \brief The Meta virtual machine compiler.
 */
#pragma once
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "mnm/device.h"
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/vm/bytecode.h"
#include "mnm/vm/executable.h"
#include "mnm/vm/vm.h"

namespace mnm {
namespace executor {
namespace vm {

using namespace mnm::ir;
using namespace mnm::registry;
using namespace mnm::value;

template <typename T, typename U>
using NodeMap = std::unordered_map<T, U, ObjectPtrHash, ObjectPtrEqual>;
using TagMap = NodeMap<tvm::relay::Constructor, Index>;
using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;
using GlobalMap = NodeMap<GlobalVar, Index>;
using ConstMap = NodeMap<Constant, Index>;
using ConstTensorShapeMap = NodeMap<TensorType, std::pair<Index, NDArray>>;
using DeviceMap = Map<tvm::Integer, Device>;

struct VMCompilerContext {
  // The module context for the compilation
  IRModule module;
  // Error reporter
  tvm::ErrorReporter err_reporter;
  // Map from a unique integer to ADT constructor tag
  TagNameMap tag_index_map;
  // Map from ADT constructor tag to a unique integer
  TagMap tag_map;
  // Map from global var to a unique integer
  GlobalMap global_map;
  // List of constants
  std::vector<Value> constants;
};

class VMCompiler : public tvm::runtime::ModuleNode {
 public:
  virtual ~VMCompiler() {
  }

  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  const char* type_key() const {
    return "VMCompiler";
  }

  /*!
   * \brief Set the parameters
   *
   * \param name name of parameter
   * \param data_in input value
   */
  void SetParam(const std::string& name, Value data_in);

  /*!
   * \brief Lower the functions in a Module
   *
   * \param mod Relay Module
   * \param device_map Mapping from context to device mapping. If it has more than one entries,
   * then it means the heterogeneous compilation.
   */
  void Lower(IRModule mod, const DeviceMap& device_map);

  // /*! \brief Generate the machine code for lowered functions. */
  // void Codegen();

 protected:
  IRModule OptimizeModule(const IRModule& mod, const DeviceMap& device_map);

  void PopulateGlobalMap();

 protected:
  /*! \brief Device map. */
  DeviceMap device_map_;
  /*! \brief Global shared meta data */
  VMCompilerContext context_;
  /*! \brief Compiled executable. */
  ObjectPtr<Executable> exec_;
  /*! \brief parameters */
  std::unordered_map<std::string, Value> params_;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
