/*!
 * Copyright (c) 2021 by Contributors
 * \file stream_schedule.h
 * \brief Base class of all stream scheduler.
 */
#pragma once
#include "mnm/ir.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace stream_schedule {

/*!
 * The base class of all stream schedulers. A stream scheduler transforms an expr in GNF format to
 * scheduled expr in ANF format, injecting stream-related operators (set_stream, add_event, and
 * wait_event).
 */
class StreamSchedulerBase : public ExprMutator {
 public:
  Expr VisitExpr_(const VarNode* var) override {
    return GetRef<Expr>(var);
  }

  Expr VisitExpr_(const GlobalVarNode* var) override {
    return GetRef<Expr>(var);
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    CHECK(func->HasNonzeroAttr(attr::kPrimitive))
        << "Stream scheduler does not support nested function call now";
    return GetRef<Expr>(func);
  }

  Expr VisitExpr_(const RelayConstantNode* op) override {
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const OpNode* op) override {
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const TupleNode* op) override {
    std::vector<Expr> fields;
    for (auto field : op->fields) {
      fields.push_back(VisitExpr(field));
    }
    return let_list_.Push(Tuple(fields));
  }

  Expr VisitExpr_(const CallNode* c) override {
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return let_list_.Push(Call(VisitExpr(c->op), args, c->attrs, c->type_args));
  }

  Expr VisitExpr_(const TupleGetItemNode* op) override {
    return let_list_.Push(TupleGetItem(VisitExpr(op->tuple), op->index));
  }

 protected:
  Expr AnnotateSetStream(int64_t device_id, int64_t stream_id) {
    static Op op = Op::Get("mnm.op.set_stream");
    Expr device_id_e = MakeConstant(value::ScalarValue::make(device_id));
    Expr stream_id_e = MakeConstant(value::ScalarValue::make(stream_id));
    Array<Expr> args({device_id_e, stream_id_e});
    return let_list_.Push(Call(op, args));
  }

  Expr AnnotateAddEvent(int64_t event_id) {
    static Op op = Op::Get("mnm.op.add_event");
    Expr event_id_e = MakeConstant(value::ScalarValue::make(event_id));
    Array<Expr> args({event_id_e});
    return let_list_.Push(Call(op, args));
  }

  Expr AnnotateWaitEvent(int64_t event_id) {
    static Op op = Op::Get("mnm.op.wait_event");
    Expr event_id_e = MakeConstant(value::ScalarValue::make(event_id));
    Array<Expr> args({event_id_e});
    return let_list_.Push(Call(op, args));
  }

  LetList let_list_;
};

}  // namespace stream_schedule
}  // namespace pass
}  // namespace mnm