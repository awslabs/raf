/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file deduplicate.h
 * \brief Deduplicate the same structure in a GNF IR.
 */

#include "raf/ir_ext.h"
#include "raf/registry.h"
#include "raf/pass.h"

namespace raf {
namespace pass {

namespace dataflow_graph {

class DataflowGraph {
 public:
  /*! \brief A Node that wraps the input type and represents the indexed graph and dominator tree */
  struct Node {
    /*! \brief Node Constructor
     *  \param ref The input graph node
     *  \param index The index of the node in toplogical order
     */
    Node(const Expr& ref, const size_t index) : ref_(ref), index_(index) {
    }

    /*! \brief The input node */
    const Expr ref_;
    /*! \brief The topological order index */
    const size_t index_;

    /*! \brief A boolean to determine if this node is external to the graph */
    bool is_external_ = false;
    /*! \brief The forward inputs of the node */
    std::vector<Node*> inputs_;
    /*! \brief The forward outputs/users of the node */
    std::vector<Node*> outputs_;

    /*! \brief The depth of the node in the dominator tree */
    size_t depth_ = 0;
    /*! \brief The dominator parent/final user of the outputs of this node */
    Node* dominator_parent_;
    /*! \brief The nodes this node dominates */
    std::vector<Node*> dominator_children_;

    bool Dominates(const Node* other) {
      std::stack<const Node*> stack;
      std::unordered_set<const Node*> visited;
      stack.push(this);
      while (!stack.empty()) {
        const Node* current = stack.top();
        stack.pop();
        for (auto node : current->dominator_children_) {
          if (visited.count(node) == 0) {
            if (other == node) {
              return true;
            } else {
              stack.push(node);
            }
            visited.insert(node);
          }
        }
      }
      return false;
    }
  };
  /*! \brief Construct the domination tree inside IndexedGraph */
  void PostDom() {
    for (size_t i = topological_order_.size(); i != 0; --i) {
      size_t index = i - 1;
      auto* current = topological_order_[index].get();
      if (current->is_external_) {
        current->depth_ = 1;
        current->dominator_parent_ = nullptr;
      } else {
        auto parent = LeastCommonAncestor(current->outputs_);
        current->depth_ = parent ? parent->depth_ + 1 : 1;
        current->dominator_parent_ = parent;
        parent->dominator_children_.push_back(current);
      }
    }
  }
  /*! \brief Map of input nodes to IndexedGraph Nodes */
  std::unordered_map<Expr, std::shared_ptr<Node>, ObjectPtrHash, ObjectPtrEqual> node_map_;
  /*! \brief Topological IndexedGraph Nodes */
  std::vector<std::shared_ptr<Node>> topological_order_;

 protected:
  /*! \brief Find the least common ancestor of all outputs of a node */
  Node* LeastCommonAncestor(const std::vector<Node*>& outputs) {
    if (outputs.size() == 0) {
      return nullptr;
    }
    auto parent = outputs.at(0);
    for (size_t i = 1; i < outputs.size(); ++i) {
      parent = LeastCommonAncestor(parent, outputs.at(i));
    }
    return parent;
  }

  /*! \brief Find the least common ancestor of two nodes */
  Node* LeastCommonAncestor(Node* lhs, Node* rhs) {
    if (lhs == nullptr || rhs == nullptr) {
      return nullptr;
    }
    while (lhs != rhs) {
      ICHECK(lhs);
      ICHECK(rhs);
      if (lhs->depth_ < rhs->depth_) {
        rhs = rhs->dominator_parent_;
      } else if (lhs->depth_ > rhs->depth_) {
        lhs = lhs->dominator_parent_;
      } else {
        rhs = rhs->dominator_parent_;
        lhs = lhs->dominator_parent_;
      }
    }
    return lhs;
  }
};

}  // namespace dataflow_graph

namespace deduplicate {

using namespace raf::ir;

using DataflowGraph = pass::dataflow_graph::DataflowGraph;
using Node = DataflowGraph::Node;
using Nodes = std::vector<Node*>;
using DomMask = std::vector<int>;
using DomMasks = std::vector<DomMask>;

/*!
 * \brief Create a Dataflow Graph from a GNF expression.
 *
 * Dataflow Graph is a kind of Indexed Graph based on an Expr,
 * but it only keeps the dataflow expr types:
 *   Call
 *   Tuple
 *   TupleGetItem
 * TODO(@guangtai): Add support of the following exprs.
 * We do not support the following exprs now.
 *   If
 *   Let (for BBNF)
 */
DataflowGraph CreateDataflowGraph(const Expr& expr) {
  using NodePtr = std::shared_ptr<DataflowGraph::Node>;
  /*! \brief Creator Creates an DataflowGraph and determintes Topological order */
  class Creator : public MixedModeVisitor {
   public:
    DataflowGraph CreateGraph(const Expr& expr) {
      VisitExpr(expr);
      graph_.node_map_[expr]->is_external_ = true;
      return std::move(graph_);
    }

   protected:
    void AddNode(const Expr& expr) {
      auto node = std::make_shared<DataflowGraph::Node>(expr, index_++);
      graph_.node_map_[expr] = node;
      graph_.topological_order_.push_back(node);
    }
    void VisitExpr_(const CallNode* op) override {
      AddNode(GetRef<Expr>(op));
    }
    void VisitExpr_(const TupleNode* op) override {
      AddNode(GetRef<Expr>(op));
    }
    void VisitExpr_(const TupleGetItemNode* op) override {
      AddNode(GetRef<Expr>(op));
    }
    void VisitExpr_(const FunctionNode* op) override {
      // do nothing to avoid traversing the function
    }
    DataflowGraph graph_;
    size_t index_ = 0;
  };
  /*! \brief Annotator takes an DataflowGraph, fills it's forward outputs, and does dominator tree
   * analysis.
   *
   *  Annotator use ExprFunctor to visit nodes, but iterates over them in pre-determined
   * topological order instead of recursing.
   */
  class Annotator : public ExprFunctor<void(const Expr&, NodePtr)> {
   public:
    Annotator(const DataflowGraph& graph) : graph_(graph) {
    }
    DataflowGraph Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (const auto& node : graph_.topological_order_) {
        ExprFunctor::VisitExpr(node->ref_, nullptr);
      }
      // do the dominator analysis
      graph_.PostDom();
      return std::move(graph_);
    }

    /*! Default visitation pushes the parent to the child's outputs and the child to the parent's
     * inputs*/
    void VisitExpr(const Expr& expr, NodePtr parent) override {
      if (graph_.node_map_.count(expr)) {
        auto current = graph_.node_map_[expr];
        if (parent) {
          current->outputs_.push_back(parent.get());
          parent->inputs_.push_back(current.get());
        }
      }
    }

   protected:
    DataflowGraph graph_;
    void VisitExpr_(const TupleNode* op, NodePtr parent) override {
      for (auto field : op->fields) {
        this->VisitExpr(field, graph_.node_map_[GetRef<Expr>(op)]);
      }
    }

    void VisitExpr_(const CallNode* op, NodePtr parent) override {
      for (auto arg : op->args) {
        this->VisitExpr(arg, graph_.node_map_[GetRef<Expr>(op)]);
      }
    }

    void VisitExpr_(const TupleGetItemNode* op, NodePtr parent) override {
      this->VisitExpr(op->tuple, graph_.node_map_[GetRef<Expr>(op)]);
    }
  };
  return Annotator(Creator().CreateGraph(expr)).Annotate();
}

/*!
 * \brief Enumerate valid subgraphs from a dataflow graph.
 *
 * A subgraph is represented by a vector of nodes, and the first node in it is the root.
 * In addition, if not `must_dominate`, each subgraph has a domination mask of the same size,
 * values in it represent if a node is dominated by the root node.
 *
 * This function uses a variant of the Exact Subgraph Enumeration (ESU) algorithm, it only
 * chooses the valid nodes as candidates:
 *   1. If node is the root, a candidate must be some other's child.
 *   2. If `must_dominate`, a node must be dominated by the root node.
 *   3. If a tuple node, its size of fields should not more than 10. It is to avoid including
 *      large tuple (usually generated by `with_autodiff`) that will cause huge number of
 *      combinations, the number 10 is randomly chosen.
 *   4. If a call node, its op should not be a non-primitive function or an inplace op.
 * And then check if an obtained subgraph is valid:
 *   1. If a node is dominated by the root, its all output(s) should in the subgraph in
 *      order to avoid recomputing
 *   2. If not `must_dominate`, all input nodes' index of the subgraph should smaller than
 *      the not dominated node with the smallest index in the subgraph. It is to fit the
 *      visiting order of Relay MixedModeMutator, so that when an intermediate output is
 *      visited, all the inputs of the call of extracted function have been mutated.
 *   3. If a node is a leaf node of the subgraph, it should not be a tuplegetitem node for
 *      it is usually meaningless.
 *
 * The variant ESU algorithm works as follows:
 * 1. for each valid node `v` in `graph` do:
 *      extension <- v's valid childs
 *      call ExtendSubgraph({v}, extension, v)
 * 2. ExtendSubgraph(subgraph, extension, v):
 *      if |subgraph| = k, output subgraph and return
 *      while extension is not empty do
 *        remove an arbitrary node w from extension
 *        extension' <- extension union {valid nodes in w's exclusive neighborhood}
 *        call ExtendSubgraph(subgraph union {w}, extension', v)
 *  P.S.: exclusive neighborhood means all nodes neighboring w but not in subgraph or
 *        neighborhoods of subgraph
 *
 * References:
 * [1] http://snap.stanford.edu/class/cs224w-2018/handouts/05-motifs.pdf
 *
 * \param graph The dataflow graph.
 * \param k The size of each subgraph.
 * \param must_dominate Whether the root node of a subgraph must dominate other nodes in the
 * subgraph.
 * \return The subgraphs and corresponding domination masks.
 */
std::pair<std::vector<Nodes>, DomMasks> EnumerateValidSubgraph(const DataflowGraph& graph, int k,
                                                               bool must_dominate) {
  std::vector<Nodes> ret_nodes;
  DomMasks ret_masks;
  auto is_valid_node = [](Node* n) {
    bool ret = true;
    if (auto* tuple = n->ref_.as<TupleNode>()) {
      if (tuple->fields.size() > 10) {
        ret = false;
      }
    } else if (auto* call = n->ref_.as<CallNode>()) {
      if (const FunctionNode* fn = call->op.as<FunctionNode>()) {
        if (!fn->HasNonzeroAttr(attr::kPrimitive)) {
          ret = false;
        }
      } else if (const OpNode* opnode = call->op.as<OpNode>()) {
        // TODO(@hgt312): let's revisit this part after having new inplace mechanism
        // Do not merge call nodes with inplace op
        static auto finplace = Op::GetAttrMap<op::TRAFInplaceUpdate>("TRAFInplaceUpdate");
        static auto add_op = Op::Get("raf.op.add");
        static auto subtract_op = Op::Get("raf.op.subtract");
        auto op = GetRef<Op>(opnode);
        if (op::IsDialectOp(op)) {
          op = op::GetBaseOp(op);
        }
        if (finplace.count(op)) {
          ret = false;
        } else if (op == add_op || op == subtract_op) {
          CHECK_GT(call->args.size(), 2);
          auto out = call->args[2];
          if (out.defined()) {
            auto konst = out.as<ConstantNode>();
            // Inplace update when out is not constant or konst->value is defined
            if (!konst || konst->value.defined()) {
              ret = false;
            }
          }
        }
      }
    }
    return ret;
  };
  std::function<void(Nodes subgraph, Nodes extension, Node * v)> extend_subgraph =
      [&](Nodes subgraph, Nodes extension, Node* v) {
        if (subgraph.size() == k) {
          // check if the subgraph is valid
          bool is_valid_subgraph = true;
          Node* root = subgraph[0];
          for (size_t i = 1; i < k; ++i) {
            if (must_dominate || root->Dominates(subgraph[i])) {
              for (auto parent : subgraph[i]->outputs_) {
                auto it = std::find(subgraph.begin(), subgraph.end(), parent);
                if (it == subgraph.end()) {
                  is_valid_subgraph = false;
                  break;
                }
              }
            }
            if (!is_valid_subgraph) {
              break;
            }
          }
          if (!must_dominate && is_valid_subgraph) {
            size_t index = std::numeric_limits<size_t>::max();
            for (size_t i = 1; i < k; ++i) {
              if (!root->Dominates(subgraph[i])) {
                index = subgraph[i]->index_ < index ? subgraph[i]->index_ : index;
              }
            }
            if (index != std::numeric_limits<size_t>::max()) {
              for (size_t i = 0; i < k; ++i) {
                for (auto child : subgraph[i]->inputs_) {
                  auto it = std::find(subgraph.begin(), subgraph.end(), child);
                  if (it == subgraph.end() && child->index_ > index) {
                    is_valid_subgraph = false;
                    break;
                  }
                }
                if (!is_valid_subgraph) {
                  break;
                }
              }
            }
          }
          if (!must_dominate && is_valid_subgraph) {
            for (size_t i = 1; i < k; ++i) {
              bool is_leaf = true;
              for (auto child : subgraph[i]->inputs_) {
                auto it = std::find(subgraph.begin(), subgraph.end(), child);
                if (it != subgraph.end()) {
                  is_leaf = false;
                  break;
                }
              }
              if (is_leaf && subgraph[i]->ref_->IsInstance<TupleGetItemNode>()) {
                is_valid_subgraph = false;
                break;
              }
            }
          }
          if (is_valid_subgraph) {
            ret_nodes.push_back(subgraph);
            if (!must_dominate) {
              // generate domination mask
              DomMask mask(k, 0);
              mask[0] = 1;
              for (size_t i = 1; i < k; ++i) {
                if (!root->Dominates(subgraph[i])) {
                  mask[i] = 1;
                }
              }
              ret_masks.push_back(mask);
            }
          }
          return;
        }
        while (extension.size() != 0) {
          Node* w = extension.back();
          extension.pop_back();
          std::unordered_set<Node*> excludes;
          for (auto node : subgraph) {
            excludes.insert(node);
            for (auto child : node->inputs_) {
              excludes.insert(child);
            }
          }
          Nodes new_extension = extension;
          for (auto child : w->inputs_) {
            if (is_valid_node(child) && excludes.count(child) == 0) {
              if (!must_dominate || v->Dominates(child)) {
                new_extension.push_back(child);
              }
            }
          }
          Nodes new_subgraph = subgraph;
          new_subgraph.push_back(w);
          extend_subgraph(new_subgraph, new_extension, v);
        }
      };

  for (auto it = graph.topological_order_.rbegin(); it != graph.topological_order_.rend(); ++it) {
    auto node = *it;
    if (!is_valid_node(node.get()) || node->ref_->IsInstance<TupleGetItemNode>()) {
      continue;
    }
    Nodes extenstion;
    for (auto child : node->inputs_) {
      if (is_valid_node(child)) {
        if (!must_dominate || node->Dominates(child)) {
          extenstion.push_back(child);
        }
      }
    }
    extend_subgraph({node.get()}, extenstion, node.get());
  }
  return std::make_pair(ret_nodes, ret_masks);
}

// Class for computing hash value of a subgraph and corresponding domination mask.
class DedupHasher : public ExprVisitor {
 public:
  explicit DedupHasher(bool consider_type) : consider_type_(consider_type) {
  }
  size_t GetHashKey(const Nodes& nodes, const DomMask& mask, const ir::Optional<ir::String>& salt) {
    for (auto v : nodes) {
      exprs_.insert(v->ref_);
    }
    VisitExpr(nodes[0]->ref_);
    for (auto i : mask) {
      HashCombine(std::hash<int>()(i));
    }
    if (salt != nullptr) {
      HashCombine(std::hash<std::string>()(salt.value()));
    }
    return hashkey;
  }

 protected:
  void HashCombine(const uint64_t value) {
    // Do not use std::hash in this function. This hash must be stable
    // across different platforms and std::hash is implementation dependent.
    hashkey = hashkey ^ (value + 0x9e3779b9 + (hashkey << 6) + (hashkey >> 2));
  }
  void HashForPlaceHolder(const ExprNode* op) {
    static const size_t placeholder_hash = std::hash<std::string>()("PlaceHolder");
    HashCombine(std::hash<size_t>()(node_counter_++));
    HashCombine(placeholder_hash);
    if (consider_type_) {
      HashCombine(tvm::StructuralHash()(op->checked_type_));
    }
  }
  void VisitExpr_(const TupleNode* op) override {
    if (exprs_.count(GetRef<Expr>(op))) {
      HashCombine(std::hash<size_t>()(node_counter_++));
      HashCombine(op->GetTypeKeyHash());
      ExprVisitor::VisitExpr_(op);
    } else {
      HashForPlaceHolder(op);
    }
  }
  void VisitExpr_(const CallNode* op) override {
    if (exprs_.count(GetRef<Expr>(op))) {
      HashCombine(std::hash<size_t>()(node_counter_++));
      HashCombine(op->GetTypeKeyHash());
      if (op->op->IsInstance<OpNode>()) {
        HashCombine(std::hash<std::string>()(Downcast<Op>(op->op)->name));
      } else if (op->op->IsInstance<FunctionNode>()) {
        ICHECK(op->op.as<FunctionNode>()->HasNonzeroAttr(attr::kPrimitive));
        HashCombine(ObjectPtrHash()(op->op));
      }
      ExprVisitor::VisitExpr_(op);
    } else {
      HashForPlaceHolder(op);
    }
  }
  void VisitExpr_(const TupleGetItemNode* op) override {
    if (exprs_.count(GetRef<Expr>(op))) {
      HashCombine(std::hash<size_t>()(node_counter_++));
      HashCombine(op->GetTypeKeyHash());
      HashCombine(std::hash<int>()(op->index));
      ExprVisitor::VisitExpr_(op);
    } else {
      HashForPlaceHolder(op);
    }
  }
  void VisitExpr_(const VarNode* op) override {
    HashForPlaceHolder(op);
  }
  void VisitExpr_(const RelayConstantNode* op) override {
    const ConstantNode* node = static_cast<const ConstantNode*>(op);
    HashCombine(std::hash<size_t>()(node_counter_++));
    HashCombine(op->GetTypeKeyHash());
    HashCombine(tvm::StructuralHash()(node->value));
  }
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> exprs_;
  int node_counter_{0};
  size_t hashkey{0};
  bool consider_type_;
};

// Class for extracting the function body of a given subgraph.
class BodyExtractor : public ExprMutator {
 public:
  BodyExtractor(const std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual>& inputs,
                const std::vector<Expr>& masked_exprs)
      : inputs_(inputs), masked_exprs_(masked_exprs) {
  }
  Expr Extract(const Expr& expr) {
    Expr body = Mutate(expr);  // mutate the expr mapped to the root node
    if (masked_exprs_.size() != 0) {
      // put intermediate exprs into output tuple
      Array<Expr> exprs{body};
      for (auto e : masked_exprs_) {
        exprs.push_back(memo_[e]);
      }
      return Tuple(exprs);
    }
    return body;
  }

 protected:
  Expr VisitExpr(const Expr& pre) override {
    // replace exprs by generated vars
    if (inputs_.count(pre)) {
      return inputs_.at(pre);
    }
    return ExprMutator::VisitExpr(pre);
  }
  const std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> inputs_;
  std::vector<Expr> masked_exprs_;
};

// Struct that stores subgraphs, domination mask, and some on-demand generated auxiliary
// information.
struct SubgraphGroup {
  std::vector<Nodes> subgraphs;
  DomMask mask;
  std::unordered_set<Node*> used_nodes;
  // extracted function
  Function function;
  // input expressions of each subgraph
  std::vector<Array<Expr>> group_of_args;

  bool IsValid() {
    return subgraphs.size() > 1;
  }

  void GenHelpInfo() {
    // generate function
    int var_number = 0;
    std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> inputs;
    Array<Var> params;
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> exprs;
    for (auto node : subgraphs[0]) {
      exprs.insert(node->ref_);
    }
    for (auto node : subgraphs[0]) {
      auto make_input = [&](const Expr& input) {
        if (!input->IsInstance<RelayConstantNode>() && exprs.count(input) == 0) {
          inputs[input] = MakeVar("v_" + std::to_string(var_number++), input->checked_type_);
          params.push_back(inputs[input]);
        }
      };
      if (auto tuple = node->ref_.as<TupleNode>()) {
        for (auto field : tuple->fields) {
          make_input(field);
        }
      } else if (auto call = node->ref_.as<CallNode>()) {
        for (auto arg : call->args) {
          make_input(arg);
        }
      } else if (auto tgi = node->ref_.as<TupleGetItemNode>()) {
        make_input(tgi->tuple);
      }
    }
    std::vector<Expr> masked_exprs;
    if (mask.size() != 0) {
      // additional processing for domination mask
      for (size_t i = 1; i < mask.size(); ++i) {
        if (mask[i] != 0) {
          masked_exprs.push_back(subgraphs[0][i]->ref_);
        }
      }
    }
    auto extractor = BodyExtractor(inputs, masked_exprs);
    auto body = extractor.Extract(subgraphs[0][0]->ref_);
    function = Function(params, body, {}, {});
    // generate group of args
    for (auto subgraph : subgraphs) {
      std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> exprs;
      Array<Expr> args;
      for (auto node : subgraph) {
        exprs.insert(node->ref_);
      }
      for (auto node : subgraph) {
        auto make_arg = [&](const Expr& input) {
          if (!input->IsInstance<RelayConstantNode>() && exprs.count(input) == 0) {
            args.push_back(input);
          }
        };
        if (auto tuple = node->ref_.as<TupleNode>()) {
          for (auto field : tuple->fields) {
            make_arg(field);
          }
        } else if (auto call = node->ref_.as<CallNode>()) {
          for (auto arg : call->args) {
            make_arg(arg);
          }
        } else if (auto tgi = node->ref_.as<TupleGetItemNode>()) {
          make_arg(tgi->tuple);
        }
      }
      group_of_args.push_back(args);
    }
  }
};

// Class that uses an extract function to merge a function's body
class DeduplicateMutator : public MixedModeMutator {
 public:
  explicit DeduplicateMutator(const SubgraphGroup* group) : group_(group) {
    if (group_->mask.size() != 0) {
      for (size_t i = 0; i < group_->subgraphs.size(); ++i) {
        int count = 0;
        for (size_t j = 0; j < group_->mask.size(); ++j) {
          if (group_->mask[j] != 0) {
            expr_map[group_->subgraphs[i][j]->ref_] = i;
            mask_map[group_->subgraphs[i][j]->ref_] = count;
            ++count;
          }
        }
      }
    } else {
      for (size_t i = 0; i < group_->subgraphs.size(); ++i) {
        expr_map[group_->subgraphs[i][0]->ref_] = i;
      }
    }
  }

 protected:
  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (expr_map.count(pre)) {
      if (group_->mask.size() != 0 && group_->subgraphs.size() != expr_map.size()) {
        if (func_call_cache_.count(expr_map[pre]) == 0) {
          Array<Expr> args = group_->group_of_args[expr_map[pre]];
          Array<Expr> new_args;
          for (size_t i = 0; i < args.size(); ++i) {
            if (args[i]->IsInstance<VarNode>()) {
              new_args.push_back(args[i]);
            } else {
              new_args.push_back(memo_[args[i]]);
            }
          }
          func_call_cache_[expr_map[pre]] = Call(group_->function, new_args);
        }
        post = TupleGetItem(func_call_cache_[expr_map[pre]], mask_map[pre]);
      } else {
        Array<Expr> args = group_->group_of_args[expr_map[pre]];
        Array<Expr> new_args;
        for (size_t i = 0; i < args.size(); ++i) {
          new_args.push_back(memo_[args[i]]);
        }
        post = Call(group_->function, new_args);
      }
    }
    return post;
  }

  // map of expr to index of its corresponding subgraph's input args
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> expr_map;
  // map of expr to index of function's output tuple
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> mask_map;
  // input subgraph group
  const SubgraphGroup* group_;
  // cache of the func call for each subgraph in group
  std::unordered_map<int, Expr> func_call_cache_;
};

/*!
 * \brief Extract one function and use it to merge the origin IR.
 */
Expr MergeOneFunction(const Expr& expr, int forward_steps, bool consider_type, bool must_dominate,
                      const ir::Optional<ir::String>& salt) {
  DataflowGraph graph = CreateDataflowGraph(expr);

  int k = 2;
  std::shared_ptr<SubgraphGroup> current_group{nullptr};
  std::shared_ptr<SubgraphGroup> next_group{nullptr};

  auto loop_body = [&](int k) {
    DLOG(INFO) << "Size of subgraph (k): " << k;
    std::vector<Nodes> subgraphs;
    DomMasks masks;
    std::tie(subgraphs, masks) = EnumerateValidSubgraph(graph, k, must_dominate);
    DLOG(INFO) << "Num of subgraphs: " << subgraphs.size();
    std::unordered_map<size_t, std::shared_ptr<SubgraphGroup>> group_map;
    for (size_t i = 0; i < subgraphs.size(); ++i) {
      Nodes subgraph = subgraphs[i];
      DomMask mask = must_dominate ? DomMask{} : masks[i];
      size_t hashkey = DedupHasher(consider_type).GetHashKey(subgraph, mask, salt);
      if (group_map.count(hashkey) == 0) {
        group_map[hashkey] = std::make_shared<SubgraphGroup>();
        group_map[hashkey]->mask = mask;
      }
      auto group = group_map[hashkey];
      bool no_overlap = true;
      for (auto node : subgraph) {
        if (group->used_nodes.count(node)) {
          no_overlap = false;
          break;
        }
      }
      if (no_overlap) {
        group->subgraphs.push_back(subgraph);
        for (auto node : subgraph) {
          group->used_nodes.insert(node);
        }
      }
    }
    for (auto it = group_map.begin(); it != group_map.end();) {
      if (!it->second->IsValid()) {
        it = group_map.erase(it);
      } else {
        ++it;
      }
    }
    for (auto pair : group_map) {
      if (next_group) {
        int largest_s = next_group->subgraphs.size() * (next_group->subgraphs[0].size() - 1);
        int current_s = pair.second->subgraphs.size() * (pair.second->subgraphs[0].size() - 1);
        DLOG(INFO) << "Current score: " << current_s;
        if (current_s >= largest_s) {
          next_group = pair.second;
        }
      } else {
        next_group = pair.second;
      }
    }
  };

  do {
    if (next_group) {
      if (next_group == current_group) {
        break;
      } else {
        current_group = next_group;
      }
    }
    loop_body(k);
    ++k;
  } while (next_group || current_group);

  if (!next_group) {
    return expr;
  }

  for (int i = k; i < k + forward_steps; ++i) {
    loop_body(i);
  }
  DLOG(INFO) << "Max score: "
             << next_group->subgraphs.size() * (next_group->subgraphs[0].size() - 1);

  Expr updated_expr;
  next_group->GenHelpInfo();
  updated_expr = DeduplicateMutator(next_group.get()).Mutate(expr);
  return updated_expr;
}

}  // namespace deduplicate

/*!
 * \brief Deduplicate a GNF IR (merge the same patterns into function calls).
 *
 * For example:
 *   Original:
 *   fn (%x) {
 *     %0 = raf.op.relu(%x);
 *     %1 = raf.op.relu(%0);
 *     %2 = raf.op.relu(%1);
 *     raf.op.relu(%2)
 *   }
 *   Deduplicated:
 *   fn (%x) {
 *     %1 = fn (%v_0) {
 *       %0 = raf.op.relu(%v_0);
 *       raf.op.relu(%0)
 *     };
 *     %2 = %1(%x);
 *     %1(%2)
 *   }
 *
 * Mechanism of this pass:
 * 1. Get the dataflow graph of input IR (body of the function).
 * 2. Enumerate valid subgraphs with size k from dataflow graph.
 * 3. Calculate hash (with an option to decide if consider type) of all these subgraphs,
 *    put those with the same hash and no overlap to the same group.
 * 4. Remove the groups that only contain one subgraph.
 * 5. Choose the group with the largest score: (k - 1) * num of subgraphs.
 * 6. Go to 2 and increase k with 1, until no longer find larger group, then run this loop with
 *    continuously increased k, the number of times is determined by the value of `forward_steps`.
 * 7. Extract the function and use it to obtain the updated IR.
 * 8. 1-7 is the process of MergeOneFunction, run this function with the just updated IR until
 *    the IR no longer changed
 *
 * \param forward_steps The additional num of steps to search.
 * \param consider_type Whether considering the type information.
 * \param must_dominate Whether the root node of a subgraph must dominate other nodes in the
 * subgraph.
 * \param salt An optional hash salt.
 * \return The created pass.
 */
Pass Deduplicate(int forward_steps, bool consider_type, bool must_dominate,
                 ir::Optional<ir::String> salt) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto expr = f->body;
    auto post = expr;
    auto last = post;
    // Mutate the IR by using MergeOneFunction until it stops changing
    int count = 0;
    bool equal = true;
    static auto* structural_equal = tvm::runtime::Registry::Get("node.StructuralEqual");
    ICHECK(structural_equal) << "node.StructuralEqual is not registered.";
    do {
      last = post;
      post = deduplicate::MergeOneFunction(post, forward_steps, consider_type, must_dominate, salt);
      if (consider_type) {
        post = InferType(post);
      }
      equal = (*structural_equal)(last, post, false, true);
    } while (!equal && count < 100);
    if (count >= 100) {
      LOG(FATAL) << "Observed 100 MergeOneFunction runs, something must have gone wrong!";
    }
    Function updated_func = Function(f->params, post, f->ret_type, f->type_params);
    if (consider_type) {
      updated_func = Downcast<Function>(pass::InferType(updated_func));
    }
    return updated_func;
  };
  return CreateRAFFunctionPass(pass_func, 0, "Deduplicate", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.Deduplicate").set_body_typed(Deduplicate);

}  // namespace pass
}  // namespace raf
