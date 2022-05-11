/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/pass_manager.cc
 * \brief Infrastructure for transformation passes.
 */
#include <tvm/node/repr_printer.h>

#include "raf/file.h"
#include "raf/pass.h"
#include "raf/pass_manager.h"
#include "raf/registry.h"

namespace raf {
namespace pass {

using namespace raf::ir;
using tvm::ReprPrinter;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

/*!
 * \brief The RAFSequentialNode contains a set of passes that transform RAF
 * programs from one AST to another semantically equivalent one.
 *
 * One example of this level of pass is that the pass manager needs to correctly
 * perform a host of optimizations with a given optimization level and disabled
 * passes.
 */
class RAFSequentialNode : public PassNode {
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

  static constexpr const char* _type_key = "raf.pass_.RAFSequential";
  RAF_FINAL_OBJECT(RAFSequentialNode, PassNode);
};

RAFSequential::RAFSequential(tvm::Array<Pass> passes, PassInfo pass_info) {
  auto n = make_object<RAFSequentialNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

RAFSequential::RAFSequential(tvm::Array<Pass> passes, String name) {
  auto n = make_object<RAFSequentialNode>();
  n->passes = std::move(passes);
  PassInfo pass_info = PassInfo(2, std::move(name), {});
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

const RAFSequentialNode* RAFSequential::operator->() const {
  return static_cast<const RAFSequentialNode*>(get());
}

inline Pass GetPass(const String& pass_name) {
  const PackedFunc* f;
  if (pass_name.operator std::string().find("raf.pass_.") != std::string::npos) {
    f = tvm::runtime::Registry::Get(pass_name);
  } else {
    f = tvm::runtime::Registry::Get("raf.pass_." + pass_name);
  }
  ICHECK(f != nullptr) << "Cannot use " << pass_name << " to create the pass";
  return (*f)();
}

std::string DumpAfterPassIRToFile(std::string dump_ir_path, const IRModule& mod, size_t idx,
                                  std::string pass_name) {
  if (dump_ir_path.empty()) {
    return "";
  }
  // Dump the IR to the folder.
  std::ofstream ofs(dump_ir_path + "/" + std::to_string(idx) + "_" + pass_name + ".txt");
  ofs << raf::ir::AsText(mod);
  return dump_ir_path;
}

// TODO(zhiics): we currenlty only sequentially execute each pass in
// a RAFSequential without the consideration of their orders. The phase
// ordering problem needs to be handled in the future.
IRModule RAFSequentialNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const char* raf_dump_ir_to = getenv("RAF_DUMP_IR_TO");
  std::string dump_ir_path = "";
  if (raf_dump_ir_to != nullptr) {
    dump_ir_path = std::string(raf_dump_ir_to);
    // Create parent directory if it doesn't exist.
    CreateDir(dump_ir_path);

    // Create a unique sequence directory.
    dump_ir_path += "/" + pass_info->name;
    if (DirExists(dump_ir_path)) {
      dump_ir_path += "_1";
    }
    CreateDir(dump_ir_path);
    DumpAfterPassIRToFile(dump_ir_path, mod, 0, "init");
  }

  size_t pass_cnt = 1;
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!pass_ctx.PassEnabled(pass_info)) continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      mod = GetPass(it)(std::move(mod), pass_ctx);
    }
    mod = pass(std::move(mod), pass_ctx);
    DumpAfterPassIRToFile(dump_ir_path, mod, pass_cnt++, pass_info->name);
  }
  return mod;
}

RAF_REGISTER_OBJECT_REFLECT(RAFSequentialNode);

class RAFFunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given IRModule. It fetches one function at a time
 * from the function list in the module for optimization.
 *
 * Note that the scope of passes at this level is a RAF function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class RAFFunctionPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a RAF function as a
   * `pass_func` and let it run on a given module. The same `pass_func` will
   * then be applied on each function in the module.
   */
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func;

  RAFFunctionPassNode() = default;

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

  static constexpr const char* _type_key = "raf.pass_.RAFFunctionPass";
  RAF_FINAL_OBJECT(RAFFunctionPassNode, PassNode);

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

class RAFFunctionPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  RAFFunctionPass(TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
                  PassInfo pass_info);

  RAF_OBJECT_REF(RAFFunctionPass, Pass, RAFFunctionPassNode);
};

RAFFunctionPass::RAFFunctionPass(
    TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func, PassInfo pass_info) {
  auto n = make_object<RAFFunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Perform Module -> Module optimizations at the Function level.
IRModule RAFFunctionPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
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

bool RAFFunctionPassNode::SkipFunction(const Function& func) const {
  return (func->GetAttr<String>(attr::kCompiler).defined()) ||
         func->GetAttr<Integer>(attr::kSkipOptimization, 0) != 0;
}

Pass CreateRAFFunctionPass(
    const TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func, int opt_level,
    String name, tvm::Array<String> required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return RAFFunctionPass(pass_func, pass_info);
}

RAF_REGISTER_OBJECT_REFLECT(RAFFunctionPassNode);

TVM_REGISTER_GLOBAL("raf.pass_.MakeRAFFunctionPass")
    .set_body_typed([](TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
                       PassInfo pass_info) { return RAFFunctionPass(pass_func, pass_info); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RAFFunctionPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RAFFunctionPassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Function pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });

RAF_REGISTER_GLOBAL("raf.pass_.RAFSequential").set_body([](TVMArgs args, TVMRetValue* ret) {
  tvm::Array<Pass> passes = args[0];
  int opt_level = args[1];
  std::string name = args[2];
  tvm::Array<String> required = args[3];
  PassInfo pass_info = PassInfo(opt_level, name, required);
  *ret = RAFSequential(passes, pass_info);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RAFSequentialNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RAFSequentialNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run RAFSequential pass: " << info->name << " at the optimization level "
                << info->opt_level << ". ";
      p->stream << "The passes will be executed are: [";
      for (const auto& it : node->passes) {
        const PassInfo pass_info = it->Info();
        p->stream << pass_info->name << " ";
      }
      p->stream << "]";
    });

}  // namespace pass
}  // namespace raf
