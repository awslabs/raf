#pragma once

#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace trace_entry {

class TraceEntryNode : public mnm::rly::Node {
 public:
  mnm::rly::Expr expr;
  mnm::value::Value value;
  mnm::rly::Array<mnm::rly::NodeRef> dep;
  // TODO(@junrushao1994): maybe fine-grained lock here for multi-threaded inference

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("expr", &expr);
    v->Visit("value", &value);
    v->Visit("dep", &dep);
  }

  static constexpr const char* _type_key = "mnm.trace_entry.TraceEntry";
  MNM_DEF_NODE_TYPE_INFO(TraceEntryNode, mnm::rly::Node);
};

class TraceEntry final : public mnm::rly::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(TraceEntry, mnm::rly::NodeRef, TraceEntryNode);
};

}  // namespace trace_entry
}  // namespace mnm
