/*!
 * Copyright (c) 2021 by Contributors
 * \file  init_shardspec.cc
 * \brief Gradient operator input selection pass
 */
#include <sstream>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/sharding.h"
#include <string>
#include <vector>

namespace raf {
namespace pass {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::sharding;

namespace shard_pass {

class ShardOpCallAttrsSetter : public ExprMutator {
 public:
  explicit ShardOpCallAttrsSetter(const Map<Expr, Attrs>& attrs_map) : _attrs_map(attrs_map) {
  }

  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      auto ref = GetRef<Expr>(node);
      if (_attrs_map.count(ref)) {
        auto new_expr = Call(node->op, node->args, Attrs(_attrs_map[ref]));
        return ExprMutator::VisitExpr_(new_expr.as<CallNode>());
      }
    }
    return ExprMutator::VisitExpr_(node);
  }

 private:
  const Map<Expr, Attrs>& _attrs_map;
};

class ShardOpCallExpander : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& op = node->op;
    const Attrs& attrs = node->attrs;
    const auto* f = tvm::runtime::Registry::Get("raf.sharding._match_expansion_rule");
    if (attrs.defined() && op->IsInstance<OpNode>() && attrs->IsInstance<ShardOpCallAttrs>()) {
      auto call = GetRef<Call>(node);
      Expr new_expr = (*f)(call);
      // return call.same_as(new_expr) ? new_expr : ExprMutator::VisitExpr(new_expr);
      return new_expr;
    }
    return ExprMutator::VisitExpr_(node);
  }
};

 // // Step 1: Propagate ShardSpec
      // Array<BaseShardSpec> sin;
      // for (int64_t i = 0; i < sattr->sin.size(); ++i) {
      //   if (sattr->sin[i]->IsInstance<UnsetShardSpecObj>()) {
      //     LOG(INFO) << i << " is unset shardspec";
      //     bool flag_unchanged = true;
      //     if (args[i]->IsInstance<CallNode>()) {
      //       // Copy ShardSpec from previous output
      //       LOG(INFO) << i << " is call";
      //       const auto pcall = Downcast<Call>(args[i]);
      //       if (pcall->attrs->IsInstance<ShardOpCallAttrs>()) {
      //         const auto pattr = pcall->attrs.as<ShardOpCallAttrs>();
      //         sin.push_back(pattr->sout[0]);
      //         flag_unchanged = false;
      //       }
      //     } 
      //     if (flag_unchanged) {
      //       // sin[i] = ShardSpec::make()
      //     }
      //   } else {
      //     sin.push_back(sattr->sin[i]);
      //   }
      // }

class ShardSpecPropagator : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* node) override {
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(node));
    const Expr& op = call->op;
    const Attrs& attrs = call->attrs;
    const Array<Expr>& args = call->args;
    const auto* f = tvm::runtime::Registry::Get("raf.sharding._infer_shardspec");
    if (attrs.defined() && op->IsInstance<OpNode>() && attrs->IsInstance<ShardOpCallAttrs>()) {
      LOG(INFO) << op << " " << call->op;

      Expr new_expr = (*f)(call);
      return new_expr;
    }
    return call;
  }
};

}  // namespace shard_pass

Pass AnnotateShardOpCall(const Map<Expr, Attrs>& attrs_map) {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::AnnotateShardOpCall";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpCallAttrsSetter(attrs_map);
            auto func = tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "AnnotateShardOpCall", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.AnnotateShardOpCall").set_body_typed(AnnotateShardOpCall);

Pass ExpandShardOpCall() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::ExpandShardOpCall";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpCallExpander();
            auto func = tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "ExpandShardOpCall", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ExpandShardOpCall").set_body_typed(ExpandShardOpCall);

Pass InferShardSpec() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::InferShardSpec";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardSpecPropagator();
            auto func = tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "InferShardSpec", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.InferShardSpec").set_body_typed(InferShardSpec);

}  // namespace pass
}  // namespace raf
