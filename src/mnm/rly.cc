#include <mnm/rly.h>
#include <mnm/value.h>

#include "./commons.h"

namespace mnm {
namespace rly {

ASSERT_SAME_CLASS(tvm::Node, HalideIR::Node);
ASSERT_SAME_CLASS(tvm::relay::NodeRef, tvm::NodeRef);
ASSERT_SAME_CLASS(tvm::relay::NodeHash, tvm::NodeHash);
ASSERT_SAME_CLASS(tvm::relay::NodeEqual, tvm::NodeEqual);
ASSERT_SAME_CLASS(tvm::relay::IndexExpr, tvm::Expr);
ASSERT_SAME_CLASS(tvm::relay::DataType, tvm::Type);

ASSERT_DERIVED_FROM(SourceName, NodeRef);
ASSERT_DERIVED_FROM(SourceNameNode, Node);

ASSERT_DERIVED_FROM(Span, NodeRef);
ASSERT_DERIVED_FROM(SpanNode, Node);

ASSERT_DERIVED_FROM(Id, NodeRef);
ASSERT_DERIVED_FROM(IdNode, Node);

ASSERT_DERIVED_FROM(RelayNode, Node);

ASSERT_DERIVED_FROM(Module, NodeRef);
ASSERT_DERIVED_FROM(ModuleNode, RelayNode);

// Relay Expression
ASSERT_DERIVED_FROM(Expr, NodeRef);
ASSERT_DERIVED_FROM(ExprNode, RelayNode);

ASSERT_DERIVED_FROM(Constant, Expr);
ASSERT_DERIVED_FROM(ConstantNode, ExprNode);

ASSERT_DERIVED_FROM(Tuple, Expr);
ASSERT_DERIVED_FROM(TupleNode, ExprNode);

ASSERT_DERIVED_FROM(Var, Expr);
ASSERT_DERIVED_FROM(VarNode, ExprNode);

ASSERT_DERIVED_FROM(GlobalVar, Expr);
ASSERT_DERIVED_FROM(GlobalVarNode, ExprNode);

ASSERT_DERIVED_FROM(Function, Expr);
ASSERT_DERIVED_FROM(FunctionNode, ExprNode);

ASSERT_DERIVED_FROM(Call, Expr);
ASSERT_DERIVED_FROM(CallNode, ExprNode);

ASSERT_DERIVED_FROM(Let, Expr);
ASSERT_DERIVED_FROM(LetNode, ExprNode);

ASSERT_DERIVED_FROM(If, Expr);
ASSERT_DERIVED_FROM(IfNode, ExprNode);

ASSERT_DERIVED_FROM(TupleGetItem, Expr);
ASSERT_DERIVED_FROM(TupleGetItemNode, ExprNode);

ASSERT_DERIVED_FROM(RefCreate, Expr);
ASSERT_DERIVED_FROM(RefCreateNode, ExprNode);

ASSERT_DERIVED_FROM(RefRead, Expr);
ASSERT_DERIVED_FROM(RefReadNode, ExprNode);

ASSERT_DERIVED_FROM(RefWrite, Expr);
ASSERT_DERIVED_FROM(RefWriteNode, ExprNode);

ASSERT_DERIVED_FROM(TempExpr, Expr);
ASSERT_DERIVED_FROM(TempExprNode, ExprNode);

// Relay Types
ASSERT_DERIVED_FROM(Type, NodeRef);
ASSERT_DERIVED_FROM(TypeNode, RelayNode);

ASSERT_DERIVED_FROM(BaseTensorType, Type);
ASSERT_DERIVED_FROM(BaseTensorTypeNode, TypeNode);

ASSERT_DERIVED_FROM(TensorType, Type);
ASSERT_DERIVED_FROM(TensorTypeNode, BaseTensorTypeNode);

ASSERT_DERIVED_FROM(TypeVar, Type);
ASSERT_DERIVED_FROM(TypeVarNode, TypeNode);

ASSERT_DERIVED_FROM(GlobalTypeVar, Type);
ASSERT_DERIVED_FROM(GlobalTypeVarNode, TypeNode);

ASSERT_DERIVED_FROM(TypeCall, Type);
ASSERT_DERIVED_FROM(TypeCallNode, TypeNode);

ASSERT_DERIVED_FROM(IncompleteType, Type);
ASSERT_DERIVED_FROM(IncompleteTypeNode, TypeNode);

ASSERT_DERIVED_FROM(FuncType, Type);
ASSERT_DERIVED_FROM(FuncTypeNode, TypeNode);

ASSERT_DERIVED_FROM(TupleType, Type);
ASSERT_DERIVED_FROM(TupleTypeNode, TypeNode);

ASSERT_DERIVED_FROM(TupleType, Type);
ASSERT_DERIVED_FROM(TupleTypeNode, TypeNode);

ASSERT_DERIVED_FROM(TypeConstraint, Type);
ASSERT_DERIVED_FROM(TypeConstraintNode, TypeNode);

ASSERT_DERIVED_FROM(TypeConstraint, TypeConstraint);
ASSERT_DERIVED_FROM(TypeConstraintNode, TypeConstraintNode);

}  // namespace rly
}  // namespace mnm
