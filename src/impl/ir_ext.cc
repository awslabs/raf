#include <mnm/ir_ext.h>
#include <mnm/registry.h>

namespace mnm {
namespace ir {

void ModuleNode::Add(const GlobalVar& var, const Function& func) {
  auto it = functions.find(var);
  CHECK(it == functions.end() || !(*it).second.defined())
      << "Duplicate definition of " << var->name_hint;
  functions.Set(var, func);
}

Function ModuleNode::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  CHECK(it != functions.end()) << "There is no definition of " << var->name_hint;
  return (*it).second;
}

Module Module::make(Map<GlobalVar, Function> functions) {
  NodePtr<ModuleNode> n = make_node<ModuleNode>();
  n->functions = std::move(functions);
  return Module(n);
}

MNM_REGISTER_GLOBAL("mnm._make.Module").set_body_typed(Module::make);

MNM_REGISTER_GLOBAL("mnm.module.Module_Add")
    .set_body_typed<void(Module, GlobalVar, Function)>([](Module mod, GlobalVar var,
                                                          Function func) { mod->Add(var, func); });

MNM_REGISTER_GLOBAL("mnm.module.Module_Lookup")
    .set_body_typed<Function(Module, GlobalVar)>([](Module mod, GlobalVar var) {
      return mod->Lookup(var);
    });

}  // namespace ir
}  // namespace mnm
