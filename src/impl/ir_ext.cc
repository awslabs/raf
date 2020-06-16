/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/ir_ext.cc
 * \brief MNM extension to TVM/Relay IR.
 */
#include "mnm/ir_ext.h"
#include "mnm/registry.h"

namespace mnm {
namespace ir {

void ModuleObj::Add(const GlobalVar& var, const Function& func) {
  auto it = functions.find(var);
  CHECK(it == functions.end() || !(*it).second.defined())
      << "Duplicate definition of " << var->name_hint;
  functions.Set(var, func);
}

Function ModuleObj::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  CHECK(it != functions.end()) << "There is no definition of " << var->name_hint;
  return (*it).second;
}

Module Module::make(Map<GlobalVar, Function> functions) {
  ObjectPtr<ModuleObj> n = make_object<ModuleObj>();
  n->functions = std::move(functions);
  return Module(n);
}

Module Module::Global() {
  static Module inst = Module::make({});
  return inst;
}

void ModuleAdd(Module mod, GlobalVar var, Function func) {
  mod->Add(var, func);
}

Function ModuleLookup(Module mod, GlobalVar var, Function func) {
  return mod->Lookup(var);
}

tvm::runtime::NDArray MakeFakeTensor() {
  static int64_t a[1] = {-114514};
  static int64_t b[1] = {1};
  Context ctx = mnm::Context(mnm::DevType::kCPU(), 0);
  DType dtype = mnm::DType(mnm::DTypeCode::kInt(), 64, 1);
  DLTensor tensor;
  tensor.data = a;
  tensor.ctx = ctx;
  tensor.dtype = dtype;
  tensor.shape = b;
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty({}, dtype, ctx);
  array.CopyFrom(&tensor);
  return array;
}

RelayConstant MakeConstant(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = MakeFakeTensor();
  n->value = std::move(node_ref);
  return RelayConstant(n);
}

ObjectRef ConstantExtractValue(RelayConstant _node) {
  const ConstantNode* node = static_cast<const ConstantNode*>(_node.get());
  return node->value;
}

MNM_REGISTER_GLOBAL("mnm.ir._make.Module").set_body_typed(Module::make);
MNM_REGISTER_GLOBAL("mnm.ir._make.Constant").set_body_typed(MakeConstant);
MNM_REGISTER_GLOBAL("mnm.ir.constant.ExtractValue").set_body_typed(ConstantExtractValue);
MNM_REGISTER_GLOBAL("mnm.ir.module.Add").set_body_typed(ModuleAdd);
MNM_REGISTER_GLOBAL("mnm.ir.module.Lookup").set_body_typed(ModuleLookup);
MNM_REGISTER_GLOBAL("mnm.ir.module.Global").set_body_typed(Module::Global);

MNM_REGISTER_OBJECT_REFLECT(ModuleObj);

}  // namespace ir
}  // namespace mnm
