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
 * \file src/pass/pass_manager.cc
 * \brief Infrastructure for transformation passes.
 */

#include <tvm/ir/transform.h>
#include <tvm/node/repr_printer.h>

#include "mnm/pass.h"
#include "mnm/pass_manager.h"
#include "mnm/registry.h"

namespace mnm {
namespace pass {

using namespace mnm::ir;
using tvm::ReprPrinter;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

/*!
 * \brief The MNMSequentialNode contains a set of passes that transform Meta
 * programs from one AST to another semantically equivalent one.
 *
 * One example of this level of pass is that the pass manager needs to correctly
 * perform a host of optimizations with a given optimization level and disabled
 * passes.
 */
class MNMSequentialNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief A list of passes that used to compose a sequential pass. */
  tvm::Array<Pass> passes;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
    v->Visit("passes", &passes);
  }

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override {
    return pass_info;
  }

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   *        typical pass manager jobs could be done by it. This function could
   *        be overloaded to focus on different metrics, i.e. performance,
   *        memory footprint, etc.
   *
   * \param mod The module that these passes are applied on.
   * \param pass_ctx The context that these passes execute on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  static constexpr const char* _type_key = "mnm.pass_.MNMSequential";
  MNM_FINAL_OBJECT(MNMSequentialNode, PassNode);
};

MNMSequential::MNMSequential(tvm::Array<Pass> passes, PassInfo pass_info) {
  auto n = make_object<MNMSequentialNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

MNMSequential::MNMSequential(tvm::Array<Pass> passes, String name) {
  auto n = make_object<MNMSequentialNode>();
  n->passes = std::move(passes);
  PassInfo pass_info = PassInfo(2, std::move(name), {});
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

const MNMSequentialNode* MNMSequential::operator->() const {
  return static_cast<const MNMSequentialNode*>(get());
}

inline Pass GetPass(const String& pass_name) {
  const PackedFunc* f;
  if (pass_name.operator std::string().find("mnm.pass_.") != std::string::npos) {
    f = tvm::runtime::Registry::Get(pass_name);
  } else {
    f = tvm::runtime::Registry::Get("mnm.pass_." + pass_name);
  }
  ICHECK(f != nullptr) << "Cannot use " << pass_name << " to create the pass";
  return (*f)();
}

// TODO(zhiics): we currenlty only sequentially execute each pass in
// a MNMSequential without the consideration of their orders. The phase
// ordering problem needs to be handled in the future.
IRModule MNMSequentialNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!pass_ctx.PassEnabled(pass_info)) continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      mod = GetPass(it)(std::move(mod), pass_ctx);
    }
    mod = pass(std::move(mod), pass_ctx);
  }
  return mod;
}

MNM_REGISTER_OBJECT_REFLECT(MNMSequentialNode);

class MNMFunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given IRModule. It fetches one function at a time
 * from the function list in the module for optimization.
 *
 * Note that the scope of passes at this level is a Meta function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class MNMFunctionPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Meta function as a
   * `pass_func` and let it run on a given module. The same `pass_func` will
   * then be applied on each function in the module.
   */
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func;

  MNMFunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
  }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override {
    return pass_info;
  }

  static constexpr const char* _type_key = "mnm.pass_.MNMFunctionPass";
  MNM_FINAL_OBJECT(MNMFunctionPassNode, PassNode);

 private:
  /*
   * \brief Check if a function should be skipped for optimization.
   *
   * \param func The target function to be checked.
   *
   * \return Return true if the function will be skipped, otherwise false.
   */
  bool SkipFunction(const Function& func) const;
};

class MNMFunctionPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  MNMFunctionPass(TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
                  PassInfo pass_info);

  MNM_OBJECT_REF(MNMFunctionPass, Pass, MNMFunctionPassNode);
};

MNMFunctionPass::MNMFunctionPass(
    TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func, PassInfo pass_info) {
  auto n = make_object<MNMFunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Perform Module -> Module optimizations at the Function level.
IRModule MNMFunctionPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();

  ICHECK(mod.defined());

  DLOG(INFO) << "Executing function pass : " << pass_info->name
             << " with opt level: " << pass_info->opt_level;

  // Execute the pass function and return a new module.
  IRModule updated_mod =
      IRModule(mod->functions, mod->type_definitions, mod->Imports(), mod->source_map);

  std::vector<std::pair<GlobalVar, Function>> updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relay::Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      auto updated_func = SkipFunction(func) ? func : pass_func(func, updated_mod, pass_ctx);
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  return updated_mod;
}

bool MNMFunctionPassNode::SkipFunction(const Function& func) const {
  return (func->GetAttr<String>(attr::kCompiler).defined()) ||
         func->GetAttr<Integer>(attr::kSkipOptimization, 0) != 0;
}

Pass CreateMNMFunctionPass(
    const TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func, int opt_level,
    String name, tvm::Array<String> required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return MNMFunctionPass(pass_func, pass_info);
}

TVM_REGISTER_NODE_TYPE(MNMFunctionPassNode);

TVM_REGISTER_GLOBAL("mnm.pass_.MakeMNMFunctionPass")
    .set_body_typed([](TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
                       PassInfo pass_info) { return MNMFunctionPass(pass_func, pass_info); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MNMFunctionPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MNMFunctionPassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Function pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });

MNM_REGISTER_GLOBAL("mnm.pass_.MNMSequential").set_body([](TVMArgs args, TVMRetValue* ret) {
  tvm::Array<Pass> passes = args[0];
  int opt_level = args[1];
  std::string name = args[2];
  tvm::Array<String> required = args[3];
  PassInfo pass_info = PassInfo(opt_level, name, required);
  *ret = MNMSequential(passes, pass_info);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MNMSequentialNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MNMSequentialNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run MNMSequential pass: " << info->name << " at the optimization level "
                << info->opt_level << ". ";
      p->stream << "The passes will be executed are: [";
      for (const auto& it : node->passes) {
        const PassInfo pass_info = it->Info();
        p->stream << pass_info->name << " ";
      }
      p->stream << "]";
    });

}  // namespace pass
}  // namespace mnm
