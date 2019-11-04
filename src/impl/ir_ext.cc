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

void ModuleAdd(Module mod, GlobalVar var, Function func) {
  mod->Add(var, func);
}

Function ModuleLookup(Module mod, GlobalVar var, Function func) {
  return mod->Lookup(var);
}

RelayConstant MakeConstant(NodeRef node_ref) {
  NodePtr<ConstantNode> n = make_node<ConstantNode>();
  n->value = std::move(node_ref);
  return RelayConstant(n);
}

NodeRef ConstantExtractValue(RelayConstant _node) {
  const ConstantNode* node = static_cast<const ConstantNode*>(_node.get());
  return node->value;
}

MNM_REGISTER_GLOBAL("mnm.ir._make.Module").set_body_typed(Module::make);
MNM_REGISTER_GLOBAL("mnm.ir._make.Constant").set_body_typed(MakeConstant);
MNM_REGISTER_GLOBAL("mnm.ir.constant.ExtractValue").set_body_typed(ConstantExtractValue);
MNM_REGISTER_GLOBAL("mnm.ir.module.Add").set_body_typed(ModuleAdd);
MNM_REGISTER_GLOBAL("mnm.ir.module.Lookup").set_body_typed(ModuleLookup);

}  // namespace ir
}  // namespace mnm
