/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file graph_utils.cc
 * \brief The implementation of graph utilities.
 */
#include "./graph_utils.h"

namespace raf {
namespace pass {

using namespace raf::ir;

std::string IndexedForwardGraph::Node::DebugDump() const {
  std::ostringstream os;
  os << "node[" << index << "], " << GetRef<ObjectRef>(ref) << ", pattern=" << pattern
     << ", outputs=[";
  for (auto* link = outputs.head; link != nullptr; link = link->next) {
    os << link->value.node->index << "(" << link->value.pattern << "), ";
  }
  os << "], extern_ref=" << extern_ref << ", inplace_update=" << inplace_update;
  return os.str();
}

std::string IndexedForwardGraph::DebugDump() const {
  std::ostringstream os;
  for (size_t i = 0; i < post_dfs_order.size(); ++i) {
    Node* node = post_dfs_order[i];
    os << "node[" << i << "], " << GetRef<ObjectRef>(node->ref) << " outputs=[";
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      os << link->value.node->index << ", ";
    }
    os << "]\n";
  }
  return os.str();
}

// Creator of post dominator tree of the dataflow
class IndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(Arena* arena) : arena_(arena) {
  }

  IndexedForwardGraph Prepare(const Expr& body) {
    this->Update(body, nullptr, kOpaque);
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  Arena* arena_;
  // The output.
  IndexedForwardGraph graph_;
  // attribute equal comparator
  tvm::StructuralEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern) {
    const tvm::Object* key = node.get();
    IndexedForwardGraph::Node* current;
    auto it = graph_.node_map.find(key);
    if (it != graph_.node_map.end()) {
      current = it->second;
    } else {
      current = arena_->make<IndexedForwardGraph::Node>();
      graph_.node_map[key] = current;
    }
    if (parent != nullptr) {
      auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge> >();
      link->value.node = parent;
      link->value.pattern = pattern;
      current->outputs.Push(link);
    } else {
      current->extern_ref = true;
    }
  }

  void AddNode(const tvm::Object* key) {
    auto it = graph_.node_map.find(key);
    ICHECK(it != graph_.node_map.end()) << "Cannot find node " << GetRef<ObjectRef>(key);
    IndexedForwardGraph::Node* node = it->second;
    ICHECK(node->ref == nullptr);
    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  // Post order tree
  void VisitExpr_(const FunctionNode* op) final {
    for (auto param : op->params) {
      this->Update(param, nullptr, kOpaque);
    }
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RelayConstantNode* _op) final {
    const auto* op = static_cast<const ConstantNode*>(_op);
    this->AddNode(op);
    Node* node = graph_.node_map.at(op);
    if (op->IsScalar()) {
      node->pattern = kElemWise;
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      node->pattern = kOpaque;
    }
  }

  void VisitExpr_(const CallNode* call) final {
    ICHECK(graph_.node_map.count(call));
    Node* node = graph_.node_map.at(call);
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    static auto finplace = Op::GetAttrMap<TRAFInplaceUpdate>("TRAFInplaceUpdate");
    static auto add_op = Op::Get("raf.op.add");
    static auto subtract_op = Op::Get("raf.op.subtract");

    // Now we set the pattern of this call.
    //
    // If we see a call mentioning an operator we should mark it with its
    // annotated pattern.
    //
    // If the pattern is not annotated we will default to opaque.
    //
    // Finally if the operator position is not a call node we will
    // need to call Update, as it may be an arbitrary expression.
    OpPatternKind op_pattern = kOpaque;
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto op = GetRef<Op>(opnode);
      if (!IsDialectOp(op)) {
        auto tvm_op = OpDialect::Lower(op, "tvm");
        if (tvm_op.defined()) {
          // TODO(@icemelon9): Check if the op has data dependant shape inference
          op_pattern = static_cast<OpPatternKind>(fpattern[tvm_op]);
        }
        // Check if this op performs inplace update to the outputs
        // No need to check inplace update for dialect ops
        if (finplace.count(op)) {
          node->inplace_update = true;
        } else if (op == add_op || op == subtract_op) {
          CHECK_GT(call->args.size(), 2);
          auto out = call->args[2];
          if (out.defined()) {
            auto konst = out.as<ConstantNode>();
            // Inplace update when out is not constant or konst->value is defined
            if (!konst || konst->value.defined()) {
              node->inplace_update = true;
            }
          }
        }
      }
    } else {
      this->Update(call->op, node, kOpaque);
    }

    node->pattern = op_pattern;
    this->Update(call->op, nullptr, kOpaque);
    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    // pass the analysis back to all the children it references.
    for (size_t i = 0; i < call->args.size(); ++i) {
      const auto* arg_type = call->args[i]->checked_type().as<TensorTypeNode>();
      // specifically check if result type is the same as arguments type
      OpPatternKind edge_pattern = op_pattern;
      if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
          attr_equal_(rtype->shape, arg_type->shape)) {
        edge_pattern = kElemWise;
      }
      this->Update(call->args[i], node, edge_pattern);
    }
    ExprVisitor::VisitExpr_(call);
    this->AddNode(call);
  }

  void VisitExpr_(const TupleNode* op) final {
    ICHECK(graph_.node_map.count(op));
    Node* tuple_node = graph_.node_map.at(op);
    tuple_node->pattern = kTuple;
    for (const Expr& field : op->fields) {
      if (field->checked_type().as<TensorTypeNode>()) {
        this->Update(field, tuple_node, kInjective);
      } else {
        this->Update(field, nullptr, kOpaque);
      }
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    auto tuple_type = op->tuple->checked_type().as<TupleTypeNode>();
    ICHECK(tuple_type);
    // When TVM lowers a fused function, it expects all arguments to be a Tensor or
    // a tuple containing only Tensors. But this tuple may contain a reference or
    // another tuple. To avoid modifying codegen logic, we do not allow fusing through this node
    // if the tuple contains such non Tensor fields. However, all fields will be recursively
    // visited via call to ExprVisitor::VisitExpr_(op) below and corresponding visitor methods.
    bool has_non_tensor = false;
    for (auto ty : tuple_type->fields) {
      if (!ty.as<TensorTypeNode>()) {
        has_non_tensor = true;
        break;
      }
    }
    if (has_non_tensor) {
      this->Update(op->tuple, nullptr, kOpaque);
    } else {
      ICHECK(graph_.node_map.count(op));
      Node* node = graph_.node_map.at(op);
      node->pattern = kInjective;
      this->Update(op->tuple, node, kInjective);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const VarNode* op) final {
    this->AddNode(op);
  }

  void VisitExpr_(const LetNode* op) final {
    // do not fuse through let.
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      this->Update(op->var, nullptr, kOpaque);
      this->Update(op->value, nullptr, kOpaque);
      this->Update(op->body, nullptr, kOpaque);
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
      this->AddNode(op);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const IfNode* op) final {
    // do not fuse through if.
    this->Update(op->cond, nullptr, kOpaque);
    this->Update(op->true_branch, nullptr, kOpaque);
    this->Update(op->false_branch, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefCreateNode* op) final {
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefReadNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefWriteNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }
};

IndexedForwardGraph IndexedForwardGraph::Create(Arena* arena, const Expr& body) {
  return Creator(arena).Prepare(body);
}

std::string DominatorTree::Node::DebugDump() const {
  std::ostringstream os;
  os << "gnode=<" << gnode->DebugDump() << ">, depth=" << depth << ", pattern=" << pattern
     << ", parent=";
  if (parent) {
    os << parent->gnode->index;
  } else {
    os << "null";
  }
  return os.str();
}

DominatorTree DominatorTree::PostDom(Arena* arena, const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    tree.nodes[index] = tree.GetNode(arena, graph.post_dfs_order[index]);
  }
  return tree;
}

DominatorTree::Node* DominatorTree::LeastCommonAncestor(Node* lhs, Node* rhs,
                                                        OpPatternKind* edge_pattern) {
  while (lhs != rhs) {
    if (lhs == nullptr) return nullptr;
    if (rhs == nullptr) return nullptr;
    if (lhs->depth < rhs->depth) {
      edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
      rhs = rhs->parent;
    } else if (rhs->depth < lhs->depth) {
      edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
      lhs = lhs->parent;
    } else {
      edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
      edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }
  return lhs;
}

DominatorTree::Node* DominatorTree::LeastCommonAncestor(
    const LinkedList<IndexedForwardGraph::Edge>& input_nodes, OpPatternKind* edge_pattern) {
  auto link = input_nodes.head;
  if (link == nullptr) {
    return nullptr;
  }
  auto get_node = [&](const IndexedForwardGraph::Edge& edge) {
    size_t oindex = edge.node->index;
    ICHECK_LT(oindex, nodes.size());
    Node* onode = nodes[oindex];
    ICHECK(onode != nullptr);
    return onode;
  };
  DominatorTree::Node* parent = get_node(link->value);
  *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
  link = link->next;
  for (; link != nullptr; link = link->next) {
    parent = LeastCommonAncestor(parent, get_node(link->value), edge_pattern);
    *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
  }
  return parent;
}

DominatorTree::Node* DominatorTree::GetNode(Arena* arena, IndexedForwardGraph::Node* gnode) {
  DominatorTree::Node* tnode = arena->make<DominatorTree::Node>();
  tnode->gnode = gnode;
  if (gnode->extern_ref) {
    tnode->depth = 1;
    tnode->parent = nullptr;
    tnode->pattern = kOpaque;
  } else {
    // find the LCAs of all outputs.
    OpPatternKind pattern = kElemWise;
    DominatorTree::Node* parent = LeastCommonAncestor(gnode->outputs, &pattern);
    tnode->depth = parent ? parent->depth + 1 : 1;
    tnode->parent = parent;
    tnode->pattern = pattern;
  }
  return tnode;
}

}  // namespace pass
}  // namespace raf
