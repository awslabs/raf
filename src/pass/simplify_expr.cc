// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
// Acknowledgement: The main logic originates from TVM
/*!
 * Copyright (c) 2021 by Contributors
 * \file simplify_expr.cc
 * \brief Simplifies the commonly seen patterns.
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/op_utils.h"
#include "mnm/pass.h"
#include "mnm/value.h"

namespace mnm {
namespace pass {
namespace simplify_expr {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;

using tvm::TVMArgs;
using tvm::TVMRetValue;

/*! \brief A wrapper class defining a rewrite matching a specific pattern. */
class DFPatternRewrite {
 public:
  /*! \brief Returns the rewritten expression. */
  virtual Expr Callback(const Expr& pre, const Expr& post,
                        const Map<DFPattern, Array<Expr>>& node_map) const = 0;

  virtual ~DFPatternRewrite() = default;

  /*! \brief Returns the pattern to be used for matching and rewriting. */
  inline DFPattern Pattern() const {
    return pattern_;
  }

  inline bool RequireType() const {
    return require_type_;
  }

  inline DFPatternCallback MakeCallback() const {
    auto func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = this->Callback(pre, post, node_map);
    };
    return DFPatternCallback(pattern_, PackedFunc(func), require_type_);
  }

 protected:
  /*! \brief The pattern for matching and rewriting. */
  DFPattern pattern_;
  /*! \brief Whether or not the rewrite requires types to be inferred. */
  bool require_type_ = true;
};

/*! \brief Helper class for composing rewrites and getting callbacks. */
class DFPatternRewriteComposer {
 public:
  template <typename T, typename... Args>
  inline void AddRewrite(Args... args) {
    rewrites_.push_back(std::make_shared<T, Args...>(&args...));
  }

  inline Array<DFPatternCallback> MakeCallbacks() const {
    Array<DFPatternCallback> callbacks;
    for (const auto& rewrite : rewrites_) {
      callbacks.push_back(rewrite->MakeCallback());
    }
    return callbacks;
  }

  inline void Clear() {
    rewrites_.clear();
  }

 private:
  /*! \brief the rewrites to be composed. */
  std::vector<std::shared_ptr<DFPatternRewrite>> rewrites_;
};

/*!
 * \brief The base class to convert `*_like` operators to their explicit shape equivalent
 * when the target shape is concrete.
 */
class ConcretizeLikeRewrite : public DFPatternRewrite {
 public:
  virtual bool Check(const Expr& pre, const Expr& post,
                     const Map<DFPattern, Array<Expr>>& node_map) const {
    const CallNode* call_node = pre.as<CallNode>();
    ICHECK(call_node);

    if (!call_node->checked_type().as<TensorTypeNode>()) {
      return false;
    }

    return true;
  }

  virtual Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                          DataType dtype) const = 0;

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    if (!Check(pre, post, node_map)) {
      return post;
    }

    const TensorTypeNode* like_ty = pre->checked_type().as<TensorTypeNode>();
    Array<Integer> cshape;
    for (const auto& dim : like_ty->shape) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        cshape.push_back(Integer(GetRef<IntImm>(imm)));
      } else {
        // shape is not static, don't concretize
        return post;
      }
    }
    auto ret = Concretize(node_map, cshape, like_ty->dtype);
    return ret;
  }
};

/*!
 * \brief Converts `*_like` unary operators to their explicit shape equivalent
 * (e.g. `zeros_like(x, y)` to `zeros(x, y.shape)`), when the target information is concrete.
 */
class ConcretizeUnaryLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeUnaryLikeRewrite(const std::string op_like, const std::string op) : op_(Op::Get(op)) {
    like_pat_ = IsWildcard();
    pattern_ = IsExpr(Op::Get(op_like))({like_pat_});
  }

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return Call(op_, {MakeConstant(ArrayToIntTuple(shape)),
                      MakeConstant(StringValue::make(DLDataType2String(dtype)))});
  }

 protected:
  DFPattern like_pat_;
  const Op op_;
};

class ConcretizeZerosLikeRewrite : public ConcretizeUnaryLikeRewrite {
 public:
  ConcretizeZerosLikeRewrite() : ConcretizeUnaryLikeRewrite("mnm.op.zeros_like", "mnm.op.zeros") {
  }
};

class ConcretizeOnesLikeRewrite : public ConcretizeUnaryLikeRewrite {
 public:
  ConcretizeOnesLikeRewrite() : ConcretizeUnaryLikeRewrite("mnm.op.ones_like", "mnm.op.ones") {
  }
};

/*!
 * \brief Converts `*_like` binary operators to their explicit shape equivalent
 * (e.g. `cast_like(x, y)` to `cast(x, y.dtype)`), when the target information is concrete.
 */
class ConcretizeBinaryLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeBinaryLikeRewrite(const Op& op) {
    like_pat_ = IsWildcard();
    data_pat_ = IsWildcard();
    pattern_ = IsExpr(op)({data_pat_, like_pat_});
  }

 protected:
  DFPattern data_pat_;
  DFPattern like_pat_;
};

class ConcretizeCastLikeRewrite : public ConcretizeBinaryLikeRewrite {
 public:
  ConcretizeCastLikeRewrite() : ConcretizeBinaryLikeRewrite(Op::Get("mnm.op.cast_like")) {
  }

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    static auto op = Op::Get("mnm.op.cast");
    return Call(
        op, {node_map[data_pat_][0], MakeConstant(StringValue::make(DLDataType2String(dtype)))});
  }
};

class ConcretizeBroadcastToLikeRewrite : public ConcretizeBinaryLikeRewrite {
 public:
  ConcretizeBroadcastToLikeRewrite()
      : ConcretizeBinaryLikeRewrite(Op::Get("mnm.op.broadcast_to_like")) {
  }

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    static auto op = Op::Get("mnm.op.broadcast_to");
    return Call(op, {node_map[data_pat_][0], MakeConstant(ArrayToIntTuple(shape))});
  }
};

/*! \brief Simplify useless cast ops. */
class SimplifyCast : public DFPatternRewrite {
 public:
  SimplifyCast() {
    data_pat_ = IsWildcard();
    pattern_ = IsOp("mnm.op.cast")({data_pat_, IsWildcard()});
    pattern_ = IsOp("mnm.op.cast")({pattern_, IsWildcard()}) || pattern_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    static auto cast_op = Op::Get("mnm.op.cast");
    const TensorTypeNode* out_ty = pre->checked_type().as<TensorTypeNode>();

    // Find the data node to get its type, because node_map[data_pat_] does not have checked type.
    auto arg = Downcast<Call>(pre)->args[0];
    if (auto prev_node = arg.as<CallNode>()) {
      if (prev_node->op->IsInstance<OpNode>() && Downcast<Op>(prev_node->op) == cast_op) {
        arg = prev_node->args[0];
      }
    }
    const TensorTypeNode* data_ty = arg->checked_type().as<TensorTypeNode>();

    if (out_ty->dtype == data_ty->dtype) {
      return node_map[data_pat_][0];
    }
    return post;
  }

 protected:
  DFPattern data_pat_;
};

class SimplifyReshape : public DFPatternRewrite {
 public:
  SimplifyReshape() {
    data_pat_ = IsWildcard();
    pattern_ = IsOp("mnm.op.reshape")({data_pat_, IsWildcard(), IsWildcard()});
    pattern_ = IsOp("mnm.op.reshape")({pattern_, IsWildcard(), IsWildcard()}) || pattern_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    static auto reshape_op = Op::Get("mnm.op.reshape");
    const CallNode* call = pre.as<CallNode>();
    auto data = node_map[data_pat_][0];
    const TensorTypeNode* out_ty = pre->checked_type().as<TensorTypeNode>();

    // Find the data node to get its type, because node_map[data_pat_] does not have checked type.
    auto arg = Downcast<Call>(pre)->args[0];
    if (auto prev_node = arg.as<CallNode>()) {
      if (prev_node->op->IsInstance<OpNode>() && Downcast<Op>(prev_node->op) == reshape_op) {
        arg = prev_node->args[0];
      }
    }
    const TensorTypeNode* data_ty = arg->checked_type().as<TensorTypeNode>();

    bool const_new_shape = true;
    Array<Integer> new_shape;
    for (auto dim : out_ty->shape) {
      if (dim.as<IntImmNode>() == nullptr) {
        const_new_shape = false;
        break;
      }
      new_shape.push_back(Downcast<Integer>(dim));
    }
    if (const_new_shape) {
      // Check input shape to determine whether we still need a reshape.
      Array<Integer> old_shape;
      bool need_reshape = false;
      for (auto dim : data_ty->shape) {
        if (dim.as<IntImmNode>() == nullptr) {
          need_reshape = true;
          break;
        }
        old_shape.push_back(Downcast<Integer>(dim));
      }
      need_reshape = need_reshape || old_shape.size() != new_shape.size();
      if (!need_reshape) {
        for (size_t i = 0; i < old_shape.size(); ++i) {
          if (old_shape[i]->value != new_shape[i]->value) {
            need_reshape = true;
            break;
          }
        }
      }

      if (need_reshape) {
        static auto op = Op::Get("mnm.op.reshape");
        auto ret = Call(op, {data, MakeConstant(ArrayToIntTuple(new_shape)), call->args[2]});
        ret->checked_type_ = GetRef<TensorType>(out_ty);
        return ret;
      } else {
        return data;
      }
    }
    return post;
  }

 private:
  /*! \brief Pattern input. */
  DFPattern data_pat_;
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  // Phase 1: Single-op patterns that only need to be applied once.
  DFPatternRewriteComposer composer;
  composer.AddRewrite<ConcretizeZerosLikeRewrite>();
  composer.AddRewrite<ConcretizeOnesLikeRewrite>();
  composer.AddRewrite<ConcretizeCastLikeRewrite>();
  composer.AddRewrite<ConcretizeBroadcastToLikeRewrite>();
  auto ret = mnm::ir::RewritePatterns(composer.MakeCallbacks(), expr, mod);

  // Phase 2: Sequence patterns that may need to be applied iteratively.
  composer.Clear();
  composer.AddRewrite<SimplifyCast>();
  composer.AddRewrite<SimplifyReshape>();
  return mnm::ir::RewritePatterns(composer.MakeCallbacks(), ret, mod);
}

}  // namespace simplify_expr

Pass SimplifyExpr() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(simplify_expr::SimplifyExpr(f, m));
  };
  return CreateMNMFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

MNM_REGISTER_GLOBAL("mnm.pass_.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace pass
}  // namespace mnm
