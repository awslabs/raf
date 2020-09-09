/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/vm/compiler.h
 * \brief The Meta virtual machine compiler.
 */
#pragma once
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/support/logging.h>
#include <tvm/relay/transform.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
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
using TargetsMap = Map<tvm::Integer, tvm::Target>;

struct VMCompilerContext {
  // The module context for the compilation
  Module module;
  // Error reporter
  tvm::ErrorReporter err_reporter;
  // Map from a unique integer to ADT constructor tag
  TagNameMap tag_index_map;
  // Map from ADT constructor tag to a unique integer
  TagMap tag_map;
  // Map from global var to a unique integer
  GlobalMap global_map;
  // List of constants
  std::vector<ObjectRef> constants;
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
   * \param targets For heterogeneous compilation, it is a dictionary indicating context
                    to target mapping. For homogeneous compilation, it is a build target.
   * \param target_host Host compilation target, if target is device.
   */
  void Lower(Module mod, const TargetsMap& targets, const tvm::Target& target_host);

  // /*! \brief Generate the machine code for lowered functions. */
  // void Codegen();

 protected:
  Module OptimizeModule(const Module& mod, const TargetsMap& targets);

  void PopulateGlobalMap();

 protected:
  /*! \brief Target devices. */
  TargetsMap targets_;
  /*! \brief Target host device. */
  tvm::Target target_host_;
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
