/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/fuse_dialect.cc
 * \brief Fuse the operators using registered dialect fusion patterns.
 */
#include <string>
#include <unordered_map>
#include <vector>
#include "raf/device.h"
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"

namespace raf {
namespace pass {
namespace fuse_dialect {

using namespace raf::ir;
using namespace raf::op;
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
              const ExprSet& call_set, const std::string& pattern_name,
              std::unordered_map<std::string, Function>& cache)
      : mod_(mod),
        dev_type_(dev_type),
        dialect_(dialect),
        call_set_(call_set),
        pattern_name_(pattern_name),
        func_cache_(cache) {
    single_call_ = (call_set.size() == 1);
  }

  /*! \brief Rewrite the matched expression to a fused function. */
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

      // If the identical function has been created before, reuse it.
      std::string func_cache_key = raf::ir::AsText(func);
      if (func_cache_.count(func_cache_key)) {
        func = func_cache_.at(func_cache_key);
      } else {
        func_cache_[func_cache_key] = func;
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
      // No need to lift args into function params for a single call
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
  /*! \brief The working module. */
  const IRModule& mod_;
  /*! \brief The target device type. */
  DevType dev_type_;
  /*! \brief The matched dialect name. */
  std::string dialect_;
  /*! \brief The matched pattern name. */
  std::string pattern_name_;
  /*! \brief A set of matched call nodes. */
  ExprSet call_set_;
  /*! \brief Whether the matched pattern contains just a single call node. */
  bool single_call_;
  /*! \brief A list of fused function parameters. */
  Array<Var> fused_func_params_;
  /*! \brief A list of fused function caller arguments. */
  Array<Expr> fused_func_args_;
  /*! \brief A cache of already created fused functions. */
  std::unordered_map<std::string, Function>& func_cache_;
};

class DialectPatternRewrite {
 public:
  DialectPatternRewrite(const IRModule& mod, DevType dev_type, DialectFusePattern pattern)
      : mod_(mod), dev_type_(dev_type), pattern_(pattern) {
    call_patterns_ = CallPatternExtractor().Extract(pattern_.pattern);
  }

  Expr Callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    ExprSet call_set;
    for (auto call_pat : call_patterns_) {
      auto it = node_map.find(call_pat);
      if (it != node_map.end()) {
        call_set.insert((*it).second[0]);
      }
    }
    FuseMutator mutator(mod_, dev_type_, pattern_.dialect, call_set, pattern_.name, func_cache_);
    return mutator.Rewrite(post);
  }

  DFPatternCallback MakeCallback() {
    auto func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = this->Callback(pre, post, node_map);
    };
    return DFPatternCallback(pattern_.pattern, PackedFunc(func), true);
  }

 private:
  /*! \brief The working module. */
  const IRModule& mod_;
  /*! \brief The target device type. */
  DevType dev_type_;
  /*! \brief The pattern to be matched. */
  DialectFusePattern pattern_;
  /*! \brief A list of variant call patterns extracted from the pattern. */
  std::vector<DFPattern> call_patterns_;
  /*! \brief A cache of already created fused functions. */
  std::unordered_map<std::string, Function> func_cache_;
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
    ret = RAFRewritePatterns({rewrite.MakeCallback()}, ret, mod);
  }
  return ret;
}

}  // namespace fuse_dialect

Pass FuseDialect() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(fuse_dialect::FuseDialectPatterns(f, m));
  };
  return CreateRAFFunctionPass(pass_func, 2, "FuseDialect", {"InferType"});
}

RAF_REGISTER_GLOBAL("raf.pass_.FuseDialect").set_body_typed(FuseDialect);

}  // namespace pass
}  // namespace raf
