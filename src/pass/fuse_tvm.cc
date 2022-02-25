/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/tvm_fuse.cc
 * \brief Fuse the operators using TVM op patterns.
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/binding.h"
#include "raf/pass.h"
#include "support/arena.h"
#include "tvm/relay/op_attr_types.h"
#include "./graph_utils.h"

namespace raf {
namespace pass {
namespace fuse_tvm {

using namespace raf::ir;
using namespace raf::op;
using namespace tvm::support;

/*
  Fusion level:
  - 0: No fusion.
  - 1: Only fuse elementwise, broadcast, and injective nodes.
  - 2: TBA.
  - 3: Fuse all fusable nodes.

  Note on Fusing algorithm:

  The main challenge of general fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  However, at the point of conv2d we do not necessarily know that all the future paths
  will merge at the elemwise add. The fusion algorithm applies post-dominator analysis.

  The immediate post-dominator of a node defined by the closest node where all the future path goes
  into. In the above case, the elemwise add is the post-dominator of conv2d. The general algorithm
  is as follows:

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.

  Note that, because we run analysis on a DAG, we use a single pass post-dominator
  tree construction algorithm via LCA, which is simpler than the full version that handles cycles.

  The fusion algorithm traverses from each node and checks if it can be fused to its
  immediate post dominator. It has to check the following things:

  - CheckPath: check all the path between a node and its immediate post-dominator
               satisfies the fuse condition.
  - Note that these intermediate node can already be fused with another nodes, the algorithm
      will still run correctly.
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/

constexpr uint32_t kMaxFusedOps = 256;

/*!
 * \brief A partition of the graph marked by union find data structure.
 */
class GraphPartitioner {
 public:
  explicit GraphPartitioner(Arena* arena) : arena_(arena) {
  }
  /*!
   * \brief Group as a union find data structure.
   */
  struct Group {
    /*! \brief The parent in the union find data structure. */
    Group* parent{nullptr};
    /*! \brief The pattern of the group */
    OpPatternKind pattern;
    /*! \brief reference to the root node. */
    const tvm::Object* root_ref{nullptr};
    /*!
     * \brief Reference to the master node,
     * this field is not nullptr only if pattern is kOutEWiseFusable.
     */
    const tvm::Object* master_ref{nullptr};
    /*!
     * \brief Find the group root, perform path compression
     * \return The root type node.
     */
    Group* FindRoot() {
      // fast path
      if (this->parent == nullptr) return this;
      // slow path with path compression.
      Group* root = this;
      while (root->parent != nullptr) {
        root = root->parent;
      }
      for (Group* p = this; p != root;) {
        Group* parent = p->parent;
        p->parent = root;
        p = parent;
      }
      return root;
    }

    std::string DebugDump() {
      std::ostringstream os;
      os << this;
      if (root_ref) {
        os << ", " << GetRef<ObjectRef>(root_ref);
      }
      os << ", #call_nodes=" << num_call_nodes << ", pattern=" << pattern
         << ", root=" << FindRoot();
      return os.str();
    }

    /*!
     * \brief The number of call nodes belonging to this group.
     */
    uint32_t num_call_nodes{0};
  };
  /*!
   * \brief Partition a graph.
   * \return group assignments of each node.
   */
  std::vector<Group*> Partition(const IndexedForwardGraph& graph);

  std::string DebugDump() {
    std::ostringstream os;
    for (auto i = 0; i < groups_.size(); ++i) {
      os << "group[" << i << "]: " << groups_[i]->DebugDump() << "\n";
    }
    return os.str();
  }

 private:
  /*! \brief The internal arena for temporary space. */
  Arena* arena_;
  /*! \brief The internal groups. */
  std::vector<Group*> groups_;
  /*! \brief internal field used for deduplication */
  std::unordered_set<IndexedForwardGraph::Node*> visited_;
  // Internal implelementation of CheckPath
  template <typename F>
  bool CheckPath_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    if (visited_.count(src)) return true;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    CHECK(gnode != nullptr);
    gnode = gnode->FindRoot();
    if (!fcond(gnode->pattern, src == sink)) {
      return false;
    }
    if (src == sink) return true;
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  /*!
   * \brief Check all the node and edge pattern
   *  between src and sink satisfies fcond.
   *
   * src is not checked.
   *
   * \param src The source node.
   * \param sink The termination node.
   * \param fcond The condition to be checked.
   * \tparam F the condition function, with signature
   * \note sink must be a post-dominator of src.
   */
  template <typename F>
  bool CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    CHECK(!src->extern_ref);
    visited_.clear();
    CHECK(src != sink);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  // Combine two patterns together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > kBroadcast && rhs > kBroadcast) {
      LOG(FATAL) << "Cannot merge two complex group together";
    }
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Merge the child group to the parent.
   * \param child The child group.
   * \param parent The parent group.
   */
  void MergeFromTo(Group* child, Group* parent) {
    // update the number of nodes of the parent group
    parent->num_call_nodes += child->num_call_nodes;
    child = child->FindRoot();
    parent = parent->FindRoot();
    if (child == parent) return;
    child->parent = parent;
    // update master ref and pattern
    if (child->master_ref != nullptr) {
      CHECK(parent->master_ref == nullptr);
      parent->master_ref = child->master_ref;
      parent->pattern = CombinePattern(child->pattern, parent->pattern);
    }
  }
  // Internal implelementation of CommitFuse
  void CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, Group* target) {
    if (src == sink) return;
    if (visited_.count(src)) return;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    CHECK(gnode != nullptr);
    // merge the current group to the parent if possible.
    MergeFromTo(gnode, target);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      CommitFuse_(link->value.node, sink, target);
    }
  }
  /*!
   * \brief Commit fusion operation.
   * \param src The source node.
   * \param sink The termination node.
   * \note sink must be a post-dominator of src.
   */
  void CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
    Group* target = groups_[sink->index];
    visited_.clear();
    CHECK(src != sink);
    CommitFuse_(src, sink, target);
  }

  // Initialize the groups.
  void InitGroups(const IndexedForwardGraph& graph) {
    groups_.resize(graph.post_dfs_order.size());
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      const auto* graph_node = graph.post_dfs_order[nid];
      auto* group_node = arena_->make<Group>();
      group_node->pattern = graph_node->pattern;
      group_node->root_ref = graph_node->ref;
      group_node->num_call_nodes += (group_node->root_ref->IsInstance<CallNode>()) ? 1 : 0;
      // set master ref if necessary.
      if (group_node->pattern == kOutEWiseFusable) {
        group_node->master_ref = graph_node->ref;
      }
      groups_[nid] = group_node;
    }
  }

  // execute the fusion algorithm.
  void RunFuse(const IndexedForwardGraph& graph, const DominatorTree& post_dom_tree, int phase) {
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      // the group of current node has been specified already.
      auto* graph_node = graph.post_dfs_order[nid];
      auto* dom_node = post_dom_tree.nodes[nid];
      Group* group_node = groups_[nid];
      CHECK(group_node != nullptr);
      // no actions for opaque nodes
      if (group_node->pattern == kOpaque) continue;
      // no nodes fuse into inplace update nodes
      if (graph_node->inplace_update) continue;
      // no actions needed if the current node have no dominator
      if (dom_node->parent == nullptr) continue;
      CHECK(!graph_node->extern_ref);
      size_t dom_parent_gindex = dom_node->parent->gnode->index;

      // refuse the fusion if too many ops are going to be fused together
      if (groups_[dom_parent_gindex]->num_call_nodes + group_node->num_call_nodes > kMaxFusedOps)
        continue;

      if (phase == 2) {
        // Fuse injective ops into intermediate tuples, if any
        if (group_node->pattern > kInjective) continue;
        Group* dom_parent_group = groups_[dom_parent_gindex];
        Group* dom_root_group = dom_parent_group->FindRoot();
        // TODO(@hzfan): remove this
        // If dom node group has a tuple as its root, we do not fuse tuple fields into it
        if (dom_root_group->pattern == kTuple) continue;
        if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= kInjective) {
          // Now we know the tuple has been fused into subsequent injective ops
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
          // dom_root_group can also be tuple, as in inception layers
          // CheckPath is needed to avoid fusing two intermediate tuples
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
        continue;
      }
      // TODO(@hzfan): remove this
      // Do not fuse into tuple for now
      if (groups_[dom_parent_gindex]->pattern == kTuple) continue;
      // Skip if current node is already fused to the parent.
      if (groups_[dom_parent_gindex] != nullptr &&
          group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
        continue;
      }
      // Try to fuse current node to its post-dominator.
      if (group_node->pattern == kOutEWiseFusable) {
        if (phase != 0) continue;
        // Path for OutEWiseFusable: conv2d
        // Check if the dominator relation is elemwise.
        if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
          CHECK(dom_node->parent->gnode != nullptr);
          // The fuse can be executed if all the intermediate ops are still broadcast.
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern <= kBroadcast) {
        // Pre-condition: can only be fused to parent which is injective or reduction.
        if (dom_node->parent != nullptr &&
            (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce)) {
          // Check if all the intermediate ops are still broadcast.
          // The final terminal node can already be fused to a OutEWiseFusable group.
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            if (!is_sink) {
              // Elemwise, broadcast, and injective ops on the parallel branches
              // are allowed be fused to the elemwise/broadcast master.
              return kind <= kInjective;
            } else {
              return (kind <= kBroadcast || kind == kCommReduce || kind == kInjective ||
                      kind == kOutEWiseFusable || kind == kTuple);
            }
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern == kInjective || group_node->pattern == kTuple) {
        // defer injective fusion to second phase.
        // so conv2d always finishes fusing.
        if (phase != 1) continue;
        // Check if all path are injective.
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      } else {
        // do nothing.
        CHECK(group_node->pattern == kCommReduce);
      }
    }
  }
};

std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  // get post dominator tree
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);
  // The following line can be used tofor the post dominator tree
  // LOG(INFO) << post_dom_tree.DebugDump();
  // run fusion algorithm.
  for (int phase = 0; phase < 3; ++phase) {
    this->RunFuse(graph, post_dom_tree, phase);
    // The following lines can be used to debug the fusion phase
    // LOG(INFO) << "PHASE " << phase;
    // for (int i = 0; i < groups_.size(); ++i) {
    //   LOG(INFO) << "group[" << i << "]: " << groups_[i]->DebugDump();
    // }
  }
  return std::move(groups_);
}

struct HasCallVisitor : ExprVisitor {
  bool has_call = false;
  void VisitExpr_(const CallNode* op) final {
    has_call = true;
  }
  void VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }
};

struct DispatchToTVMOps : ExprMutator {
  Expr VisitExpr_(const OpNode* node) {
    auto op = GetRef<Op>(node);
    ICHECK(!IsDialectOp(op)) << "Encountered dialect op " << op->name;
    auto tvm_op = OpDialect::Lower(op, "tvm");
    ICHECK(tvm_op.defined()) << "Cannot find tvm dialect op for " << op->name;
    return tvm_op;
  }
};

class FuseMutator : private ExprMutator {
 public:
  // Run the transform
  Expr Transform(const Expr& body) {
    // setup the group map.
    auto graph = IndexedForwardGraph::Create(&arena_, body);
    auto groups = GraphPartitioner(&arena_).Partition(graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      CHECK(graph.post_dfs_order[nid]->ref != nullptr);
      gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }
    // The following line can be used for debug.
    // this->DebugDumpGroup(body);
    return this->Mutate(body);
  }

 private:
  /*! \brief Temporary information from each group. */
  struct GroupInfo {
   public:
    // The parameters of the function.
    Array<Var> params;
    // The arguments to call the functions.
    Array<Expr> arguments;
    // Get a new parameter or allocate an old one
    Var GetOrAllocParam(const Expr& expr, const Type& type) {
      // run linear scan as most fused groups contain only a few inputs.
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (expr.same_as(arguments[i])) return params[i];
      }
      // create a new parameter.
      std::ostringstream os;
      os << "p" << params.size();
      auto var = MakeVar(os.str(), type);
      params.push_back(var);
      arguments.push_back(expr);
      return var;
    }
  };
  /*! \brief Internal arena. */
  Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Object*, GraphPartitioner::Group*> gmap_;
  /*! \brief Internal group information map. */
  std::unordered_map<GraphPartitioner::Group*, GroupInfo> ginfo_;
  /*! \brief Let binding map from variable to the pair of old value and new value. */
  std::unordered_map<Expr, std::pair<Expr, Expr>, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  /*! \brief A set of variables that are inlined into fused operators. */
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> let_inlined_;
  /*! \brief A cache of already created fused functions. */
  std::unordered_map<std::string, Function> func_cache_;

  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node) {
    if (fn_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Expr>(fn_node);
    } else {
      return ExprMutator::VisitExpr_(fn_node);
    }
  }

  // Transform calls.
  Expr VisitExpr_(const CallNode* call) {
    if (call->op.as<OpNode>()) {
      //      static auto fnoncomputational =
      //      Op::GetAttrMap<TNonComputational>("TNonComputational");
      //
      //      if (fnoncomputational.get(Downcast<Op>(call->op), false)) {
      //        return ExprMutator::VisitExpr_(call);
      //      }

      // If it is a primitive op call
      // then we must have a group assignment for it already.
      CHECK(gmap_.count(call));
      //      if (call->op == stop_fusion_op) {
      //        return ExprMutator::VisitExpr(call->args[0]);
      //      }
      auto* ret_group = gmap_.at(call)->FindRoot();

      if (ret_group->num_call_nodes == 1) {
        // Skip the group with only one call node.
        return ExprMutator::VisitExpr_(call);
      }

      Array<Expr> new_args = GetNewArguments(call->args, ret_group);

      auto new_call = Call(call->op, new_args, call->attrs, call->type_args);

      if (ret_group->root_ref == call) {
        // This is the root of the group
        // create the new call node.
        return MakeNewFunction(ret_group, call->checked_type(), new_call);
      } else {
        // This is an intermediate node of a fused function
        // simply return the new call.
        return std::move(new_call);
      }
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr VisitExpr_(const TupleNode* tuple) {
    auto* ret_group = gmap_.at(tuple)->FindRoot();

    if (ret_group->num_call_nodes <= 1) {
      // Skip the group with 0 or 1 call node.
      return ExprMutator::VisitExpr_(tuple);
    }

    Array<Expr> new_fields = GetNewArguments(tuple->fields, ret_group);
    if (ret_group->root_ref != tuple) {
      // This tuple is an intermediate node in the group
      return Tuple(new_fields);
    }
    // This tuple is the root of group
    HasCallVisitor visitor;
    visitor.VisitExpr(Tuple(new_fields));
    if (visitor.has_call) {
      // Other ops have been fused into this tuple
      return MakeNewFunction(ret_group, tuple->checked_type(), Tuple(new_fields));
    }
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const TupleGetItemNode* tuple_get) {
    auto* ret_group = gmap_.at(tuple_get)->FindRoot();
    if (ret_group->num_call_nodes == 1) {
      // Skip the group with only one call node.
      return ExprMutator::VisitExpr_(tuple_get);
    }

    auto new_tuple = GetNewArguments({tuple_get->tuple}, ret_group)[0];
    auto new_node = TupleGetItem(new_tuple, tuple_get->index);
    if (ret_group->root_ref == tuple_get) {
      if (gmap_.at(tuple_get->tuple.get())->FindRoot() != ret_group) {
        // Isolated. This case occurs when tuple is created by an Opaque op
        // e.g. multibox_transform_loc
        return ExprMutator::VisitExpr_(tuple_get);
      }
      // A new function whose output is a tuple field access
      return MakeNewFunction(ret_group, tuple_get->checked_type(), new_node);
    }
    // This is an intermediate node in the group
    return std::move(new_node);
  }

  Expr VisitExpr_(const LetNode* let) {
    auto pre_visit = [this](const LetNode* op) {
      auto ret_group = gmap_.at(op->var.get())->FindRoot();
      if (ret_group->num_call_nodes == 1) {
        // Skip the group with only one call node.
        Mutate(op->var);
        Mutate(op->value);
      } else {
        auto new_value = Mutate(op->value);
        let_binding_.emplace(op->var, std::make_pair(op->value, new_value));
      }
    };
    auto post_visit = [this](const LetNode* op) {
      auto expr = GetRef<Expr>(op);
      auto ret_group = gmap_.at(op->var.get())->FindRoot();
      auto var = Downcast<Var>(Mutate(op->var));
      auto value = Mutate(op->value);
      auto body = Mutate(op->body);
      if (ret_group->num_call_nodes == 1) {
        // Skip the group with only one call node.
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      } else {
        if (let_inlined_.count(op->var)) {
          // The var is already inlined into fused op, directly return the body.
          this->memo_[expr] = body;
        } else {
          this->memo_[expr] = Let(op->var, value, body);
        }
      }
    };
    ExpandANormalForm(let, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let)];
  }

  Expr MakeNewFunction(GraphPartitioner::Group* group, Type ret_type, Expr body) {
    // If the function has no call, it is not a primitive function.
    HasCallVisitor visitor;
    visitor(body);
    const GroupInfo& ginfo = ginfo_[group];
    auto func = Function(ginfo.params, body, ret_type, {});
    func = Downcast<Function>(DispatchToTVMOps().Mutate(func));
    func = WithAttr(std::move(func), attr::kPrimitive, Integer(visitor.has_call));
    func = WithAttr(std::move(func), attr::kDialect, String("tvm"));

    // If the identical function has been created before, reuse it.
    std::string func_cache_key = raf::ir::AsText(func);
    if (func_cache_.count(func_cache_key)) {
      func = func_cache_.at(func_cache_key);
    } else {
      func_cache_[func_cache_key] = func;
    }
    return Call(func, ginfo.arguments, Attrs());
  }

  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args,
                              GraphPartitioner::Group* current_group) {
    Array<Expr> new_args;
    for (auto arg : args) {
      auto type = arg->checked_type();
      auto it = let_binding_.find(arg);
      if (it != let_binding_.end()) {
        // The argument is bound to value via let
        Expr value, new_value;
        std::tie(value, new_value) = it->second;
        auto* arg_group = gmap_.at(value.get())->FindRoot();
        if (current_group != arg_group) {
          Var param = ginfo_[current_group].GetOrAllocParam(arg, type);
          new_args.push_back(param);
        } else {
          new_args.push_back(new_value);
          let_inlined_.insert(arg);
        }
      } else {
        Expr new_arg = this->Mutate(arg);
        auto* arg_group = gmap_.at(arg.get())->FindRoot();
        if (current_group != arg_group) {
          Var param = ginfo_[current_group].GetOrAllocParam(new_arg, type);
          new_args.push_back(param);
        } else {
          new_args.push_back(new_arg);
        }
      }
    }
    return new_args;
  }

  // Debug function, dump the group assignment in text.
  void DebugDumpGroup(const Expr& body) {
    std::string text = AsText(body, false, [this](const ObjectRef& expr) -> std::string {
      auto it = gmap_.find(expr.get());
      if (it == gmap_.end()) return "";
      std::ostringstream os;
      auto* group = it->second->FindRoot();
      os << " /* group=" << group << " */";
      return os.str();
    });
    LOG(INFO) << "Dump of group info:\n" << text;
  }
};

}  // namespace fuse_tvm

Pass FuseTVM() {
  PassContext pass_ctx = PassContext::Current();
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(fuse_tvm::FuseMutator().Transform(f));
  };

  Pass func_pass = CreateRAFFunctionPass(pass_func, 2, "FuseTVM", {});
  PassInfo pass_info(2, "FuseTVM", {});
  return RAFSequential({InferType(), func_pass}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.FuseTVM").set_body_typed(FuseTVM);

}  // namespace pass
}  // namespace raf
