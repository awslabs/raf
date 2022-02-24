/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Acknowledgement: The main logic originates from TVM

/*!
 * \file simplify_expr.cc
 * \brief Simplifies the commonly seen patterns.
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/op_utils.h"
#include "raf/pass.h"
#include "raf/value.h"
#include "../op/ty/utils.h"

namespace raf {
namespace pass {
namespace simplify_expr {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;

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
 * (e.g. `zeros_like(x)` to `zeros(x.shape, x.dtype, device)`), when the target information
 * is known.
 * Note that the simplified *_like op becomes an init op and needs to specify the target device.
 * Hoever, the target device is not included in the type system, so we assume the target device
 * is the current device.
 */
class ConcretizeUnaryLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeUnaryLikeRewrite(const std::string op_like, const std::string op)
      : op_(Op::Get(op)), device_(std::string(Device::Current(false).c_str())) {
    like_pat_ = IsWildcard();
    pattern_ = IsExpr(Op::Get(op_like))({like_pat_});
  }

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return Call(op_, {MakeConstant(ArrayToIntTuple(shape)),
                      MakeConstant(StringValue::make(DLDataType2String(dtype))),
                      MakeConstant(StringValue::make(device_))});
  }

 protected:
  DFPattern like_pat_;
  const Op op_;
  const std::string device_;
};

class ConcretizeZerosLikeRewrite : public ConcretizeUnaryLikeRewrite {
 public:
  ConcretizeZerosLikeRewrite() : ConcretizeUnaryLikeRewrite("raf.op.zeros_like", "raf.op.zeros") {
  }
};

class ConcretizeOnesLikeRewrite : public ConcretizeUnaryLikeRewrite {
 public:
  ConcretizeOnesLikeRewrite() : ConcretizeUnaryLikeRewrite("raf.op.ones_like", "raf.op.ones") {
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
  ConcretizeCastLikeRewrite() : ConcretizeBinaryLikeRewrite(Op::Get("raf.op.cast_like")) {
  }

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    static auto op = Op::Get("raf.op.cast");
    return Call(
        op, {node_map[data_pat_][0], MakeConstant(StringValue::make(DLDataType2String(dtype)))});
  }
};

class ConcretizeBroadcastToLikeRewrite : public ConcretizeBinaryLikeRewrite {
 public:
  ConcretizeBroadcastToLikeRewrite()
      : ConcretizeBinaryLikeRewrite(Op::Get("raf.op.broadcast_to_like")) {
  }

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    static auto op = Op::Get("raf.op.broadcast_to");
    return Call(op, {node_map[data_pat_][0], MakeConstant(ArrayToIntTuple(shape))});
  }
};

/*! \brief Check whether the given expr is a constant scalar with the expected value.
 * It is also applicable when the constant is a 0-dim tensor.
 */
bool IsExpectedScalar(const Expr& arg, float expected) {
  if (auto node = arg.as<ConstantNode>()) {
    if (auto val_obj = node->value.as<IntValueObj>()) {
      return val_obj->value == (int64_t)expected;
    } else if (auto val_obj = node->value.as<FloatValueObj>()) {
      return val_obj->value == expected;
    } else if (auto val_obj = node->value.as<TensorValueObj>()) {
      tensor::Tensor tensor = val_obj->tensor;
      if (tensor->ndim != 0) {  // Not even a scalar.
        return false;
      }

      bool is_expected = false;
      if (DataType(tensor->dtype) == DataType::Float(32) ||
          DataType(tensor->dtype) == DataType::Float(16)) {
        float value = GetScalarValueData<float>(GetRef<TensorValue>(val_obj));
        is_expected = value == expected;
      } else if (DataType(tensor->dtype) == DataType::Int(32)) {
        int32_t value = GetScalarValueData<int32_t>(GetRef<TensorValue>(val_obj));
        is_expected = value == (int32_t)expected;
      } else if (DataType(tensor->dtype) == DataType::Int(64)) {
        int64_t value = GetScalarValueData<int64_t>(GetRef<TensorValue>(val_obj));
        is_expected = value == (int64_t)expected;
      } else {
        LOG(WARNING) << "Unsupported type: " << DataType(tensor->dtype);
      }
      return is_expected;
    }
  }
  return false;
}

/*! \brief A helper function to free a constant tensor. */
void FreeConstTensor(const Expr& expr) {
  auto node = expr.as<ConstantNode>();
  CHECK(node != nullptr) << "Expected a ConstantNode, but got " << expr->GetTypeKey();
  if (auto val_obj = node->value.as<TensorValueObj>()) {
    val_obj->mem.reset();
  }
}

/*! \brief Remove *1 and fold *0. */
class ConcretizeMultiplyRewrite : public DFPatternRewrite {
 public:
  ConcretizeMultiplyRewrite() {
    in1_pat_ = IsWildcard();
    in2_pat_ = IsWildcard();
    pattern_ = IsOp("raf.op.multiply")({in1_pat_, in2_pat_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    // Remove *1.
    if (IsExpectedScalar(node_map[in1_pat_][0], 1)) {
      FreeConstTensor(node_map[in1_pat_][0]);
      return node_map[in2_pat_][0];
    }
    if (IsExpectedScalar(node_map[in2_pat_][0], 1)) {
      FreeConstTensor(node_map[in2_pat_][0]);
      return node_map[in1_pat_][0];
    }

    // Fold *0.
    bool is_1st_zero = IsExpectedScalar(node_map[in1_pat_][0], 0);
    bool is_2nd_zero = IsExpectedScalar(node_map[in2_pat_][0], 0);
    if (is_1st_zero || is_2nd_zero) {
      static auto op = Op::Get("raf.op.zeros");
      auto call = Downcast<Call>(pre);
      auto non_zero_in = (is_1st_zero) ? call->args[1] : call->args[0];
      const TensorTypeNode* ttype_node = non_zero_in->checked_type().as<TensorTypeNode>();
      CHECK(ttype_node) << "Expected a TensorType, but got " << non_zero_in->GetTypeKey();
      try {
        auto shape = ArrayToInt(ttype_node->shape);
        CHECK_GT(shape.size(), 0U) << "Expected non-scalar tensor (ndim > 0)";

        // Free the useless 0 tensor(s).
        if (is_1st_zero) {
          FreeConstTensor(node_map[in1_pat_][0]);
        }
        if (is_2nd_zero) {
          FreeConstTensor(node_map[in2_pat_][0]);
        }
        return Call(op, {MakeConstant(ArrayToIntTuple(shape)),
                         MakeConstant(StringValue::make(DLDataType2String(ttype_node->dtype)))});
      } catch (const dmlc::Error& e) {
        // Shape is not static, don't concretize.
        return post;
      }
    }
    return post;
  }

 protected:
  DFPattern in1_pat_, in2_pat_;
};

/*!
 * \brief Remove +0 and -0.
 */
class ConcretizeAddSubRewrite : public DFPatternRewrite {
 public:
  ConcretizeAddSubRewrite() {
    in1_pat_ = IsWildcard();
    in2_pat_ = IsWildcard();
    out_pat_ = IsWildcard();
    op_pat_ = IsOp("raf.op.add") || IsOp("raf.op.subtract");
    pattern_ = op_pat_({in1_pat_, in2_pat_, out_pat_, IsWildcard()});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    // Do not rewrite if the output is shared.
    if (!node_map[out_pat_][0]->IsInstance<ConstantNode>()) {
      return post;
    }

    Op op = Downcast<Op>(node_map[op_pat_][0]);
    static auto add_op = Op::Get("raf.op.add");

    // Remove 0+x, x+0, and x-0. Note that 0-x cannot be simplified.
    if (op == add_op && IsExpectedScalar(node_map[in1_pat_][0], 0)) {
      FreeConstTensor(node_map[in1_pat_][0]);
      return node_map[in2_pat_][0];
    }
    if (IsExpectedScalar(node_map[in2_pat_][0], 0)) {
      FreeConstTensor(node_map[in2_pat_][0]);
      return node_map[in1_pat_][0];
    }
    return post;
  }

 protected:
  DFPattern in1_pat_, in2_pat_, out_pat_, op_pat_;
};

/*!
 * \brief Check whether the cast from lhs to rhs is reversible. A cast is reversible
 * if lhs can be casted to rhs and then casted back without significant truncation
 */
bool IsCastReversible(const DataType& lhs, const DataType& rhs) {
  /*! Reversible cast map:
   *        Bool  UInt  Int  Float (to)
   * Bool    Y     Y     Y     Y
   * UInt    N     Y     Y     Y
   * Int     N     N     Y     Y
   * Float   N     N     N     Y
   * (from)
   */
  auto cast_level = [](const DataType& type) {
    if (type.is_bool()) {
      return 4;
    }
    if (type.is_uint()) {
      return 3;
    }
    if (type.is_int()) {
      return 2;
    }
    if (type.is_float() || type.is_bfloat16()) {
      return 1;
    }
    return -1;
  };
  int lhs_level = cast_level(lhs);
  int rhs_level = cast_level(rhs);
  if (lhs_level == -1 || rhs_level == -1) {
    // handle or custom data type, don't simplify
    return false;
  }
  // type with higher cast level can be reversibly cast to type with lower level
  return lhs_level >= rhs_level;
}

/*! \brief Simplify useless cast ops. */
class SimplifyCast : public DFPatternRewrite {
 public:
  SimplifyCast() {
    data_pat_ = IsWildcard();
    pattern_ = IsOp("raf.op.cast")({data_pat_, IsWildcard()});
    pattern_ = IsOp("raf.op.cast")({pattern_, IsWildcard()}) || pattern_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    static auto cast_op = Op::Get("raf.op.cast");
    const TensorTypeNode* out_ty = pre->checked_type().as<TensorTypeNode>();

    // Find the data node to get its type, because node_map[data_pat_] does not have checked type.
    auto arg = Downcast<Call>(pre)->args[0];
    if (auto prev_node = arg.as<CallNode>()) {
      if (prev_node->op->IsInstance<OpNode>() && Downcast<Op>(prev_node->op) == cast_op) {
        // ignore the situation where the cast of arg to intermediate type is not reversible
        auto intermediate_dtype = arg->checked_type().as<TensorTypeNode>()->dtype;
        if (IsCastReversible(out_ty->dtype, intermediate_dtype)) {
          arg = prev_node->args[0];
        }
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

class SimplifyMatmulReshapeBiasAct : public DFPatternRewrite {
 public:
  SimplifyMatmulReshapeBiasAct() {
    data_pat_ = IsWildcard();
    weight_pat_ = IsWildcard();
    bias_pat_ = IsWildcard();
    matmul_op_ = IsOp("raf.op.dense") || IsOp("raf.op.matmul") || IsOp("raf.op.matmul_nt") ||
                 IsOp("raf.op.matmul_tn") || IsOp("raf.op.matmul_tt") ||
                 IsOp("raf.op.batch_matmul") || IsOp("raf.op.batch_matmul_nt") ||
                 IsOp("raf.op.batch_matmul_tn") || IsOp("raf.op.batch_matmul_tt");
    pattern_ = matmul_op_({data_pat_, weight_pat_});
    pattern_ = IsOp("raf.op.reshape")({pattern_, IsWildcard(), IsWildcard()});
    pattern_ = IsOp("raf.op.add")({pattern_, bias_pat_, IsWildcard(), IsWildcard()});
    act_op_ = IsOp("raf.op.relu") || IsOp("raf.op.gelu");
    pattern_ = pattern_ || act_op_({pattern_});
  }

  /*! \brief Traverse back to find the bias add call. */
  inline Call FindBiasAddCall(const Expr& pre) const {
    auto call_node = pre.as<CallNode>();
    ICHECK(call_node);

    if (call_node->args.size() < 2) {
      call_node = call_node->args[0].as<CallNode>();
    }
    static const Op& add_op = Op::Get("raf.op.add");
    CHECK_EQ(call_node->op, add_op)
        << "Expected an add call, but got " << raf::ir::AsText(GetRef<Call>(call_node));
    return GetRef<Call>(call_node);
  }

  /*! \brief Check whether moving reshape after bias_add is legal. */
  inline bool Check(const Expr& pre, const Expr& post,
                    const Map<DFPattern, Array<Expr>>& node_map) const {
    auto call = FindBiasAddCall(pre);

    auto ttype1 = call->args[0]->checked_type().as<TensorTypeNode>();
    auto ttype2 = call->args[1]->checked_type().as<TensorTypeNode>();
    if (ttype1 == nullptr || ttype2 == nullptr) {
      return false;
    }

    // We need to check if bias_add inputs are still broadcastable after moving reshape.
    static const Op& reshape_op = Op::Get("raf.op.reshape");
    auto reshape_call_node = call->args[0].as<CallNode>();
    auto other_arg_ttype = ttype2;
    if (reshape_call_node == nullptr || reshape_call_node->op != reshape_op) {
      // We may need to also check second argument since we have commutative matching
      reshape_call_node = call.as<CallNode>()->args[1].as<CallNode>();
      other_arg_ttype = ttype1;
      CHECK(reshape_call_node != nullptr && reshape_call_node->op == reshape_op)
          << "Expected an add call with a reshape call as argument, but got "
          << raf::ir::AsText(call);
    }

    if (auto reshape_arg_ttype = reshape_call_node->args[0]->checked_type().as<TensorTypeNode>()) {
      try {
        // try broadcast reshape_arg and other_arg
        BroadcastShape(GetRef<TensorType>(reshape_arg_ttype), GetRef<TensorType>(other_arg_ttype));
        return true;
      } catch (const dmlc::Error& e) {
        // shape not compatible, don't modify
      }
    }
    return false;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    static auto reshape_op = Op::Get("raf.op.reshape");
    static auto add_op = Op::Get("raf.op.add");

    if (!Check(pre, post, node_map)) {
      return post;
    }

    // Final output type.
    const TensorTypeNode* out_ty = pre->checked_type().as<TensorTypeNode>();
    Array<Integer> ret_shape;
    for (auto dim : out_ty->shape) {
      ICHECK(dim.as<IntImmNode>() != nullptr);
      ret_shape.push_back(Downcast<Integer>(dim));
    }

    // Create a new subgraph: matmul -> add -> (act) -> reshape.
    auto bias_call = FindBiasAddCall(pre);
    auto data = node_map[data_pat_][0];
    auto weight = node_map[weight_pat_][0];
    auto bias = node_map[bias_pat_][0];

    Op matmul_op = Downcast<Op>(node_map[matmul_op_][0]);
    auto ret = Call(matmul_op, {data, weight});
    ret = Call(add_op, {ret, bias, bias_call->args[2], bias_call->args[3]});
    if (node_map.count(act_op_) > 0) {
      Op act_op = Downcast<Op>(node_map[act_op_][0]);
      ret = Call(act_op, {ret});
    }
    ret = Call(reshape_op, {ret, MakeConstant(ArrayToIntTuple(ret_shape)),
                            MakeConstant(BoolValue::make(false))});
    return ret;
  }

 private:
  /*! \brief Pattern input. */
  DFPattern data_pat_, weight_pat_, bias_pat_, matmul_op_, act_op_;
};

class SimplifyReshape : public DFPatternRewrite {
 public:
  SimplifyReshape() {
    data_pat_ = IsWildcard();
    pattern_ = IsOp("raf.op.reshape")({data_pat_, IsWildcard(), IsWildcard()});
    pattern_ = IsOp("raf.op.reshape")({pattern_, IsWildcard(), IsWildcard()}) || pattern_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    static auto reshape_op = Op::Get("raf.op.reshape");
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
        static auto op = Op::Get("raf.op.reshape");
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
  composer.AddRewrite<ConcretizeMultiplyRewrite>();
  composer.AddRewrite<ConcretizeAddSubRewrite>();
  auto ret = raf::ir::RAFRewritePatterns(composer.MakeCallbacks(), expr, mod);

  // Phase 2: Sequence patterns that may need to be applied iteratively.
  composer.Clear();
  composer.AddRewrite<SimplifyMatmulReshapeBiasAct>();
  composer.AddRewrite<SimplifyCast>();
  composer.AddRewrite<SimplifyReshape>();
  return raf::ir::RAFRewritePatterns(composer.MakeCallbacks(), ret, mod);
}

}  // namespace simplify_expr

Pass SimplifyExpr() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(simplify_expr::SimplifyExpr(f, m));
  };
  return CreateRAFFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

RAF_REGISTER_GLOBAL("raf.pass_.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace pass
}  // namespace raf
