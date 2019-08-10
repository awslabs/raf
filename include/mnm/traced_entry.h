#pragma once

#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace trace_entry {

class TraceEntryNode : public ir::Node {
 public:
  ir::Expr expr;
  value::Value value;
  ir::Array<ir::NodeRef> dep;
  // TODO(@junrushao1994): maybe fine-grained lock here for multi-threaded inference

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("expr", &expr);
    v->Visit("value", &value);
    v->Visit("dep", &dep);
  }

  static constexpr const char* _type_key = "mnm.trace_entry.TraceEntry";
  MNM_DEF_NODE_TYPE_INFO(TraceEntryNode, ir::Node);
};

class TraceEntry final : public ir::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(TraceEntry, ir::NodeRef, TraceEntryNode);
};

}  // namespace trace_entry
}  // namespace mnm
