/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/serialization.cc
 * \brief MNM serialization underlying implementation
 */
#include <tvm/node/serialization.h>
#include "mnm/serialization.h"
#include "mnm/registry.h"

namespace mnm {
namespace ir {
namespace serialization {

class IRRewrite4Loader : public ir::ExprMutator {
 public:
  ir::Expr VisitExpr_(const tvm::relay::ConstantNode* _node) override {
    const ir::ConstantNode* node = static_cast<const ir::ConstantNode*>(_node);
    ir::ObjectPtr<serialization::ConstantNode> n = ir::make_object<serialization::ConstantNode>();
    n->data = node->data;
    n->value = node->value;
    return ir::Expr(n);
  }
};

std::string SaveJSON(const ir::Module& mod) {
  ir::Module inst = ir::Module::make({});
  for (auto kv : mod->functions) {
    ir::Expr func = IRRewrite4Loader()(kv.second);
    inst->Add(kv.first, Downcast<ir::Function>(func));
  }
  return tvm::SaveJSON(inst);
}

std::string SaveJSON(const ir::Expr& expr) {
  auto e = IRRewrite4Loader().VisitExpr(expr);
  return tvm::SaveJSON(e);
}

std::string SaveJSON(const ir::ObjectRef& n) {
  if (const ir::ModuleObj* m = n.as<ir::ModuleObj>()) {
    return SaveJSON(ir::GetRef<ir::Module>(m));
  } else if (const ir::FunctionNode* f = n.as<ir::FunctionNode>()) {
    return SaveJSON(ir::GetRef<ir::Function>(f));
  } else if (const ir::ExprNode* e = n.as<ir::ExprNode>()) {
    return SaveJSON(ir::GetRef<ir::Expr>(e));
  }
  return tvm::SaveJSON(n);
}

ir::ObjectPtr<ir::Object> CreateConstantNode(const std::string& s) {
  return ir::MakeConstantNode(tvm::LoadJSON(s));
}

MNM_REGISTER_OBJECT_REFLECT(ConstantNode)
    .set_creator(CreateConstantNode)
    .set_repr_bytes([](const ir::Object* n) -> std::string {
      return tvm::SaveJSON(static_cast<const ConstantNode*>(n)->value);
    });

MNM_REGISTER_GLOBAL("mnm.ir.serialization.SaveJSON")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
      CHECK(args.size() == 1);
      ir::ObjectRef obj = args[0].operator ir::ObjectRef();
      *ret = SaveJSON(obj);
    });

}  // namespace serialization
}  // namespace ir
}  // namespace mnm
