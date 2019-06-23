#pragma once

#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace trace_entry {

class TraceEntryNode : public rly::Node {
 public:
  rly::Expr expr;
  value::Value value;
  rly::Array<rly::NodeRef> dep;
  // TODO(@junrushao1994): maybe fine-grained lock here for multi-threaded inference

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("expr", &expr);
    v->Visit("value", &value);
    v->Visit("dep", &dep);
  }

  static constexpr const char* _type_key = "mnm.trace_entry.TraceEntry";
  MNM_DEF_NODE_TYPE_INFO(TraceEntryNode, rly::Node);
};

class TraceEntry final : public rly::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(TraceEntry, rly::NodeRef, TraceEntryNode);
};

}  // namespace trace_entry
}  // namespace mnm
