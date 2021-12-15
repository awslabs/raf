/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/fuse_dialect.cc
 * \brief Fuse the operators using registered dialect fusion patterns.
 */
#include <vector>
#include "mnm/device.h"
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {
namespace fuse_dialect {

using namespace mnm::ir;
using namespace mnm::op;
using tvm::TVMArgs;
using tvm::TVMRetValue;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;

class CallPatternExtractor : public DFPatternVisitor {
 public:
  std::vector<DFPattern> Extract(DFPattern pattern) {
    call_patterns_.clear();
    VisitDFPattern(pattern);
    return call_patterns_;
  }

  void VisitDFPattern_(const CallPatternNode* op) {
    call_patterns_.push_back(GetRef<CallPattern>(op));
    DFPatternVisitor::VisitDFPattern_(op);
  }

 private:
  std::vector<DFPattern> call_patterns_;
};

class FuseMutator : public ExprMutator {
 public:
  FuseMutator(const IRModule& mod, DevType dev_type, const std::string dialect,
              const ExprSet& call_set, const std::string& pattern_name)
      : mod_(mod),
        dev_type_(dev_type),
        dialect_(dialect),
        call_set_(call_set),
        pattern_name_(pattern_name) {
    single_call_ = (call_set.size() == 1);
  }

  Expr Rewrite(Expr expr) {
    auto body = Mutate(expr);
    Expr ret;
    if (single_call_) {
      // No need to create a fused function for a single call
      ret = body;
    } else {
      auto func = Function(fused_func_params_, body, Type(), {});
      func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(1));
      func = WithAttr(std::move(func), attr::kDialect, String(dialect_));
      if (!pattern_name_.empty()) {
        func = WithAttr(std::move(func), attr::kPatternName, String(pattern_name_));
      }
      ret = Call(func, fused_func_args_, Attrs());
    }
    return ret;
  }

  Expr VisitExpr_(const CallNode* call) final {
    ICHECK(call->op.as<OpNode>());
    auto op = Downcast<Op>(call->op);
    auto dialect_op = OpDialect::Lower(op, dialect_);
    ICHECK(dialect_op.defined()) << "Cannot find dialect \"" << dialect_
                                 << "\" registered to operator " << op->name;
    Array<Expr> new_args;

    if (single_call_) {
      // Single call, no need to lift args into function params
      new_args = call->args;
    } else {
      for (auto arg : call->args) {
        if (call_set_.count(arg)) {
          new_args.push_back(Mutate(arg));
        } else {
          Type ty;
          if (arg->checked_type_.defined()) {
            ty = arg->checked_type_;
          } else {
            ty = InferTypeWithModule(arg, mod_)->checked_type();
          }
          auto var = MakeVar("p", ty);
          new_args.push_back(var);
          fused_func_params_.push_back(var);
          fused_func_args_.push_back(arg);
        }
      }
    }
    auto ret = Call(dialect_op, new_args, call->attrs, call->type_args, call->span);
    return ret;
  }

 private:
  const IRModule& mod_;
  DevType dev_type_;
  std::string dialect_;
  std::string pattern_name_;
  ExprSet call_set_;
  bool single_call_;
  Array<Var> fused_func_params_;
  Array<Expr> fused_func_args_;
};

class DialectPatternRewrite {
 public:
  DialectPatternRewrite(const IRModule& mod, DevType dev_type, DialectFusePattern pattern)
      : mod_(mod), dev_type_(dev_type), pattern_(pattern) {
    call_patterns_ = CallPatternExtractor().Extract(pattern_.pattern);
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const {
    ExprSet call_set;
    for (auto call_pat : call_patterns_) {
      auto it = node_map.find(call_pat);
      if (it != node_map.end()) {
        call_set.insert((*it).second[0]);
      }
    }
    FuseMutator mutator(mod_, dev_type_, pattern_.dialect, call_set, pattern_.name);
    return mutator.Rewrite(post);
  }

  DFPatternCallback MakeCallback() const {
    auto func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = this->Callback(pre, post, node_map);
    };
    return DFPatternCallback(pattern_.pattern, PackedFunc(func), true);
  }

 private:
  const IRModule& mod_;
  DevType dev_type_;
  DialectFusePattern pattern_;
  std::vector<DFPattern> call_patterns_;
};

Expr FuseDialectPatterns(const Expr& expr, const IRModule& mod) {
  auto dev = Device::Current(true);
  if (dev->device_type == DevType::kUnknown() || dev->device_id < 0) {
    LOG(WARNING) << "Device is not specified, skip FuseDialect pass.";
    return expr;
  }
  DevType dev_type = dev.device_type();
  Expr ret = expr;
  for (auto pat : *DialectFusePattern::Get()) {
    if (!Dialect::IsEnabled(pat.dialect, dev_type)) {
      continue;
    }
    DLOG(INFO) << "Fuse pattern " << pat.name << " for " << pat.dialect;
    DialectPatternRewrite rewrite(mod, dev_type, pat);
    ret = MNMRewritePatterns({rewrite.MakeCallback()}, ret, mod);
  }
  return ret;
}

}  // namespace fuse_dialect

Pass FuseDialect() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(fuse_dialect::FuseDialectPatterns(f, m));
  };
  return CreateMNMFunctionPass(pass_func, 2, "FuseDialect", {"InferType"});
}

MNM_REGISTER_GLOBAL("mnm.pass_.FuseDialect").set_body_typed(FuseDialect);

}  // namespace pass
}  // namespace mnm
