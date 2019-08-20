#pragma once

#include <mnm/ir.h>

/****** mnm::ir::Float ******/
namespace mnm {
namespace ir {

// TODO(@were): not sure it is necessary, let's discuss later.
class Float : public tvm::Expr {
 public:
  Float() : tvm::Expr() {
  }
  explicit Float(NodePtr<Node> node) : tvm::Expr(node) {
  }
  Float(double value) : tvm::Expr(tvm::ir::FloatImm::make(tvm::Float(64), value)) {
  }
  Float(float value) : tvm::Expr(value) {
  }
  Float& operator=(const Float& other) {
    node_ = other.node_;
    return *this;
  }
  const tvm::ir::FloatImm* operator->() const {
    return static_cast<const tvm::ir::FloatImm*>(node_.get());
  }
  operator double() const {
    CHECK(node_ != nullptr) << " Trying get reference a null Integer";
    return (*this)->value;
  }
  operator float() const {
    CHECK(node_ != nullptr) << " Trying get reference a null Integer";
    return (*this)->value;
  }
  /*! \brief type indicate the container type */
  using ContainerType = tvm::ir::FloatImm;
};

}  // namespace ir
}  // namespace mnm

/****** mnm::ir::Module ******/
namespace mnm {
namespace ir {

class ModuleNode : public Node {
 public:
  Map<GlobalVar, Function> functions;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("functions", &functions);
  }

  void Add(const GlobalVar& var, const Function& func);

  Function Lookup(const GlobalVar& var) const;

 public:
  static constexpr const char* _type_key = "mnm.ir.Module";
  MNM_DEF_NODE_TYPE_INFO(ModuleNode, Node);
};

class Module : public NodeRef {
 public:
  static Module make(Map<GlobalVar, Function> functions);
  MNM_DEF_NODE_REF_METHODS(Module, NodeRef, ModuleNode);
};

}  // namespace ir
}  // namespace mnm

/****** mnm::ir::Module ******/
namespace mnm {
namespace ir {

using RelayConstantNode = tvm::relay::ConstantNode;
using RelayConstant = tvm::relay::Constant;

class ConstantNode : public RelayConstantNode {
 public:
  NodeRef value{nullptr};
};
using Constant = tvm::relay::Constant;

}  // namespace ir
}  // namespace mnm
