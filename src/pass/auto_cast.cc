/*!
 * Copyright (c) 2020 by Contributors
 * \file auto_cast.cc
 * \brief AutoCast pass
 */
#include <tvm/ir/transform.h>

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

struct InsertCastVisitor : public MixedModeVisitor {
 public:
  InsertCastVisitor() {
  }

  void VisitExpr_(const VarNode* node) final {
    ell->ret = GetRef<Var>(node);
  }

  void VisitExpr_(const LetNode* node) final {
    static auto frule = Op::GetAttrMap<op::FMNMCastRule>("FMNMCastRule");
    auto pre_visit = [this](const LetNode* node_) {
      if (node_->value->IsInstance<CallNode>()) {
        const CallNode* call = node_->value.as<CallNode>();
        if (call->op.as<OpNode>() != nullptr) {
          const Op op = Downcast<Op>(call->op);
          if (frule.count(op)) {
            // infertype
            ell->ret = node_->var;
            Expr inferred_expr = InferType(ell->AsExpr());
            auto rules = frule[op](call->args);
            InsertCastCall(node_, call, rules);
            return;
          }
        }
      }
      // insert a line
      ell->vars.push_back(node_->var);
      ell->exprs.push_back(node_->value);
    };
    auto post_visit = [this](const LetNode* node_) {
      this->VisitExpr(node_->body);
      this->visit_counter_[node_] += 1;
    };
    ExpandANormalForm(node, pre_visit, post_visit);
  }

  Function Run(const Expr& expr) {
    if (expr->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(expr);
      for (const auto& p : func->params) {
        if (p->type_annotation.defined()) {
          p->checked_type_ = p->type_annotation;
        } else {
          LOG(FATAL) << "Some param(s) is missing type annotation!";
        }
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

Pass AutoCast() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(auto_cast::InsertCast(f));
      };
  auto insert_cast = CreateMNMFunctionPass(pass_func, 0, "AutoCastFunc", {});
  return MNMSequential({insert_cast, InferType()}, "AutoCast");
}

MNM_REGISTER_GLOBAL("mnm.pass_.AutoCast").set_body_typed(AutoCast);
}  // namespace pass
}  // namespace mnm
