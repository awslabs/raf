/*!
 * Copyright (c) 2020 by Contributors
 * \file auto_cast.cc
 * \brief AutoCast pass
 */
#include <stack>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "mnm/executor.h"
#include "mnm/binding.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace auto_cast {

using namespace mnm::ir;
using namespace mnm::op;
using namespace tvm;
using namespace runtime;
using namespace mnm::value;

enum CastHintType {
  kSkip = 0,
  kFloat16 = 1,
  kFloat32 = 2,
};

inline Expr Cast(Expr x, DataType dtype) {
  static const Op& op = Op::Get("mnm.op.cast");
  static const RelayConstant& f16_constant = MakeConstant(StringValue::make("float16"));
  static const RelayConstant& f32_constant = MakeConstant(StringValue::make("float32"));
  if (dtype.is_float16()) {
    return Call(op, {x, f16_constant}, {});
  }
  return Call(op, {x, f32_constant}, {});
}

struct InsertCastVisitor : public ExprVisitor {
 public:
  InsertCastVisitor() {
  }

  void VisitExpr_(const VarNode* node) final {
    ell->ret = GetRef<Var>(node);
  }

  void VisitExpr_(const LetNode* node) final {
    static auto frule = Op::GetAttrMap<op::FMNMCastRule>("FMNMCastRule");
    if (node->value->IsInstance<CallNode>()) {
      const CallNode* call = node->value.as<CallNode>();
      if (call->op.as<OpNode>() != nullptr) {
        const Op op = Downcast<Op>(call->op);
        if (frule.count(op)) {
          // infertype
          ell->ret = node->var;
          Expr inferred_expr = InferType(ell->AsExpr());
          auto rules = frule[op](call->args);
          InsertCastCall(node, call, rules);
          return;
        }
      }
    }
    // insert a line
    ell->vars.push_back(node->var);
    ell->exprs.push_back(node->value);
    ExprVisitor::VisitExpr(node->body);
  }

  Function Run(const Expr& expr) {
    if (expr->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(expr);
      for (const auto& p : func->params) {
        InferType(p);
      }
      ExprVisitor::VisitExpr(func->body);
      return Function(func->params, ell->AsExpr(), Type(), {}, {});
    } else {
      LOG(FATAL) << "ValueError: Input of Insertcast pass should be function";
      throw;
    }
  }

 private:
  std::unique_ptr<ExplicitLetList> ell = std::make_unique<ExplicitLetList>();
  int num_casted_var = 0;

  void InsertCastCall(const LetNode* let, const CallNode* call, const Array<Integer>& rules) {
    Array<Expr> call_args;
    for (int i = 0; i < rules.size(); ++i) {
      auto arg = call->args[i];
      if (rules[i] == CastHintType::kSkip) {
        call_args.push_back(arg);
      } else {
        DataType dtype;
        switch (rules[i]) {
          case CastHintType::kFloat32:
            dtype = DataType::Float(32);
            break;
          case CastHintType::kFloat16:
            dtype = DataType::Float(16);
            break;
          default:
            LOG(FATAL) << "Unknown cast hint type!";
        }
        auto arg_type = arg->checked_type();
        if (arg_type->IsInstance<TensorTypeNode>()) {
          // Call arg is TensorType
          auto ttype = Downcast<TensorType>(arg_type);
          if (ttype->dtype == dtype) {
            call_args.push_back(arg);
          } else {
            Var new_var = MakeVar("c" + std::to_string(++num_casted_var), {});
            ell->vars.push_back(new_var);
            ell->exprs.push_back(Cast(arg, dtype));
            call_args.push_back(new_var);
          }
        } else if (arg_type->IsInstance<TupleTypeNode>()) {
          CastTupleElements(arg, dtype);
          call_args.push_back(arg);
        }
      }
    }
    Call new_call = Call(call->op, call_args, call->attrs, call->type_args);
    ell->vars.push_back(let->var);
    ell->exprs.push_back(new_call);
    ExprVisitor::VisitExpr(let->body);
  }

  void CastTupleElements(const Expr arg, DataType dtype) {
    // Call arg is TupleType
    // Find the location that defines the tuple,
    // then insert cast lines before if and
    // replace the variables in the tuple.
    auto it = std::find(ell->vars.begin(), ell->vars.end(), Downcast<Var>(arg));
    int idx = std::distance(ell->vars.begin(), it);
    auto tuple = Downcast<Tuple>(ell->exprs[idx]);
    Array<Expr> arr;
    for (const auto& v : tuple->fields) {
      if (v->IsInstance<ConstantNode>()) {
        arr.push_back(v);
      } else {
        auto v_type = v->checked_type();
        if (v_type->IsInstance<TensorTypeNode>()) {
          // find the location of `arg` every loop
          it = std::find(ell->vars.begin(), ell->vars.end(), Downcast<Var>(arg));
          idx = std::distance(ell->vars.begin(), it);
          Var new_var = MakeVar("c" + std::to_string(++num_casted_var), {});
          it = ell->vars.insert(it, new_var);
          ell->exprs.insert(ell->exprs.begin() + idx, Cast(Downcast<Var>(v), dtype));
          arr.push_back(new_var);
        } else if (v_type->IsInstance<TupleTypeNode>()) {
          CastTupleElements(v, dtype);
        }
      }
    }
    it = std::find(ell->vars.begin(), ell->vars.end(), Downcast<Var>(arg));
    idx = std::distance(ell->vars.begin(), it);
    ell->exprs[idx] = Tuple(arr);
  }
};

Expr InsertCast(const Expr& expr) {
  return InsertCastVisitor().Run(expr);
}
}  // namespace auto_cast

ir::Expr AutoCast(ir::Expr func) {
  auto f = auto_cast::InsertCast(func);
  f = InferType(f);
  return f;
}

MNM_REGISTER_GLOBAL("mnm.pass_.AutoCast").set_body_typed(AutoCast);
}  // namespace pass
}  // namespace mnm
