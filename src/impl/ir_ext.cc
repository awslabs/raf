/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/ir_ext.cc
 * \brief MNM extension to TVM/Relay IR.
 */
#include "mnm/ir_ext.h"
#include "mnm/registry.h"

namespace mnm {
namespace ir {

void ModuleObj::Add(const GlobalVar& var, const Function& func, bool update) {
  auto it = functions.find(var);
  if (functions.find(var) != functions.end()) {
    CHECK(update) << "Duplicate definition of " << var->name_hint;
  }
  functions.Set(var, func);
  global_var_map_.Set(var->name_hint, var);
}

Function ModuleObj::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  CHECK(it != functions.end()) << "There is no definition of " << var->name_hint;
  return (*it).second;
}

Function ModuleObj::Lookup(const std::string& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

bool ModuleObj::ContainGlobalVar(const std::string& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

GlobalVar ModuleObj::GetGlobalVar(const std::string& name) const {
  auto it = global_var_map_.find(name);
  if (it == global_var_map_.end()) {
    std::ostringstream msg;
    msg << "ValueError: Cannot find global var \"" << name << "\" in the Module\n"
        << "candidates are: [";
    int counter = 0;
    for (auto kv : global_var_map_) {
      if (counter++ != 0) {
        msg << ", ";
      }
      msg << "\"" << kv.first << "\"";
    }
    msg << "]";
    LOG(FATAL) << msg.str();
  }
  return (*it).second;
}

Module Module::make(Map<GlobalVar, Function> functions) {
  ObjectPtr<ModuleObj> n = make_object<ModuleObj>();
  n->functions = std::move(functions);
  n->global_var_map_ = {};

  for (const auto& kv : n->functions) {
    // set global var map
    CHECK(n->global_var_map_.count(kv.first->name_hint) == 0)
        << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }

  return Module(n);
}

Module Module::Global() {
  static Module inst = Module::make({});
  return inst;
}

void ModuleAdd(Module mod, GlobalVar var, Function func) {
  mod->Add(var, func);
}

Function ModuleLookup(Module mod, GlobalVar var) {
  return mod->Lookup(var);
}

Function ModuleLookupStr(Module mod, const std::string& name) {
  return mod->Lookup(name);
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

ObjectPtr<ConstantNode> MakeConstantNode(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = MakeFakeTensor();
  n->value = std::move(node_ref);
  return n;
}

RelayConstant MakeConstant(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = MakeConstantNode(node_ref);
  return RelayConstant(n);
}

ObjectRef ConstantExtractValue(RelayConstant _node) {
  const ConstantNode* node = static_cast<const ConstantNode*>(_node.get());
  return node->value;
}

Var MakeVar_(Id vid, Type type_annotation, Var may_share = Var()) {
  ObjectPtr<ExtendedVarNode> n = make_object<ExtendedVarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->may_share = may_share;
  return Var(n);
}

Var MakeVar(const std::string& name_hint, Type type_annotation, Var may_share) {
  return MakeVar_(Id(name_hint), type_annotation, may_share);
}

/*!
 * \brief Set may_share field of an extended variable
 * \param var the variable
 * \param may_share the may_share field
 * \return the variable with may_share set
 */
Var SetMayShare(Var var, Var may_share) {
  const auto* vn = var.as<ExtendedVarNode>();
  vn->may_share = may_share;
  return GetRef<Var>(vn);
}

/*!
 * \brief Extract the may_share field of an extended variable
 * \param var the variable
 * \return the may_share field of this variable
 */
Var GetMayShare(Var var) {
  const auto* vn = var.as<ExtendedVarNode>();
  return vn->may_share;
}

MNM_REGISTER_GLOBAL("mnm.ir._make.Module").set_body_typed(Module::make);
MNM_REGISTER_GLOBAL("mnm.ir._make.Constant").set_body_typed(MakeConstant);
MNM_REGISTER_GLOBAL("mnm.ir._make.Var").set_body_typed(MakeVar);
MNM_REGISTER_GLOBAL("mnm.ir.constant.ExtractValue").set_body_typed(ConstantExtractValue);
MNM_REGISTER_GLOBAL("mnm.ir.variable.SetMayShare").set_body_typed(SetMayShare);
MNM_REGISTER_GLOBAL("mnm.ir.variable.GetMayShare").set_body_typed(GetMayShare);
MNM_REGISTER_GLOBAL("mnm.ir.module.Add").set_body_typed(ModuleAdd);
MNM_REGISTER_GLOBAL("mnm.ir.module.Lookup").set_body_typed(ModuleLookup);
MNM_REGISTER_GLOBAL("mnm.ir.module.LookupStr").set_body_typed(ModuleLookupStr);
MNM_REGISTER_GLOBAL("mnm.ir.module.Global").set_body_typed(Module::Global);

MNM_REGISTER_OBJECT_REFLECT(ModuleObj);

}  // namespace ir
}  // namespace mnm
