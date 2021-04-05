/*!
 * Copyright (c) 2021 by Contributors
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

using namespace tvm;
using namespace mnm::ir;
using namespace tvm::transform;
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
  const runtime::PackedFunc* f = tvm::runtime::Registry::Get("mnm.pass_." + pass_name);
  ICHECK(f != nullptr) << "Cannot use " << pass_name << "to create the pass";
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

MNM_REGISTER_GLOBAL("mnm.pass_.MNMSequential").set_body([](TVMArgs args, TVMRetValue* ret) {
  tvm::Array<Pass> passes = args[0];
  int opt_level = args[1];
  std::string name = args[2];
  tvm::Array<runtime::String> required = args[3];
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
