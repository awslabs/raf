/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/binary.cc
 * \brief Declaration of binary operators
 */
#include <tvm/arith/analyzer.h>
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/ufunc.h"
#include "../ty/utils.h"
#include "./declare_utils.h"
#include <cmath>

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

#define MNM_SWITCH_SCALAR(var, value, body)                     \
  do {                                                          \
    if (const auto* var = (value).as<IntValueObj>()) {          \
      body;                                                     \
    } else if (const auto* var = (value).as<FloatValueObj>()) { \
      body;                                                     \
    } else if (const auto* var = (value).as<BoolValueObj>()) {  \
      body;                                                     \
    }                                                           \
  } while (0);

#define MNM_BINARY_SCALAR(op, x1, x2)                                        \
  MNM_SWITCH_SCALAR(v1, x1, MNM_SWITCH_SCALAR(v2, x2, {                      \
                      call->callee = ir::NullValue<OpValue>();               \
                      call->out = ScalarValue::make(v1->value op v2->value); \
                      return;                                                \
                    }));

#define MNM_BINARY_TENSOR(x1, x2)                                             \
  if (x1->IsInstance<TensorValueObj>() && x2->IsInstance<TensorValueObj>()) { \
    const TensorValue& tv = MakeBinaryTensor(x1, x2);                         \
    call->out = tv;                                                           \
    call->device = tv->tensor->device;                                        \
    return;                                                                   \
  }

#define MNM_BINARY_INPLACE_TENSOR(x1, x2, out)                                                   \
  if (x1->IsInstance<TensorValueObj>() && x2->IsInstance<TensorValueObj>() && (out).defined()) { \
    TensorValue tv_out = ir::Downcast<TensorValue>(out);                                         \
    call->out = tv_out;                                                                          \
    call->device = tv_out->tensor->device;                                                       \
    return;                                                                                      \
  }

#define MNM_LOGICAL_BINARY_TENSOR(x1, x2)                                     \
  if (x1->IsInstance<TensorValueObj>() && x2->IsInstance<TensorValueObj>()) { \
    const TensorValue& tv = MakeBinaryTensor(x1, x2, true);                   \
    call->out = tv;                                                           \
    call->device = tv->tensor->device;                                        \
    return;                                                                   \
  }

#define MNM_REGISTER_BINARY_BCAST_OP(op_name, body) \
  MNM_OP_DECLARE(op_name, body).set_attr<TOpPattern>("TOpPattern", kBroadcast)

TensorValue MakeBinaryTensor(DLTensor* x1, DLTensor* x2, bool is_logical = false) {
  int ndim_1 = x1->ndim;
  int ndim_2 = x2->ndim;
  int ndim = std::max(ndim_1, ndim_2);
  std::vector<int64_t> oshape(ndim);
  for (int i = 0; i < ndim; ++i) {
    int64_t dim_1 = (i < ndim_1) ? x1->shape[ndim_1 - 1 - i] : 1;
    int64_t dim_2 = (i < ndim_2) ? x2->shape[ndim_2 - 1 - i] : 1;
    if (dim_1 == 1) {
      oshape[ndim - 1 - i] = dim_2;
    } else if (dim_2 == 1) {
      oshape[ndim - 1 - i] = dim_1;
    } else if (dim_1 == dim_2) {
      oshape[ndim - 1 - i] = dim_1;
    } else {
      LOG(FATAL) << "Cannot broadcast";
      throw;
    }
  }
  if (is_logical) {
    DLDataType dtype;
    dtype.code = DLDataTypeCode(1);
    dtype.bits = 1;
    dtype.lanes = 1;
    return TensorValue::Assemble(x1->device, dtype, oshape);
  }
  return TensorValue::Assemble(x1->device, x1->dtype, oshape);
}

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.add", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  const Value& out = args->out;
  if (!args->where.defined()) {
    MNM_BINARY_SCALAR(+, x1, x2);
    MNM_BINARY_TENSOR(x1, x2);
    MNM_BINARY_INPLACE_TENSOR(x1, x2, out);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.subtract", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  const Value& out = args->out;
  if (!args->where.defined()) {
    MNM_BINARY_SCALAR(-, x1, x2);
    MNM_BINARY_TENSOR(x1, x2);
    MNM_BINARY_INPLACE_TENSOR(x1, x2, out);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.multiply", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(*, x1, x2);
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.power", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(v1, x1, MNM_SWITCH_SCALAR(v2, x2, {
                      call->callee = ir::NullValue<OpValue>();
                      double a1 = v1->value;
                      double a2 = v2->value;
                      double result = std::pow(a1, a2);
                      call->out = ScalarValue::make(result);
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.divide", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                      if (s2->value == 0) {
                        LOG(FATAL) << "ZeroDivisionError: division by zero";
                        throw;
                      }
                      call->callee = ir::NullValue<OpValue>();
                      call->out = ScalarValue::make(s1->value / s2->value);
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.floor_divide", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                      if (s2->value == 0) {
                        LOG(FATAL) << "ZeroDivisionError: division by zero";
                        throw;
                      }
                      call->callee = ir::NullValue<OpValue>();
                      call->out = ScalarValue::make(floor(s1->value / s2->value));
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.mod", [](const CallValues& call) {
  // TODO(@junrushao1994): python-style Euclidean division modulo
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                      if (s2->value == 0) {
                        LOG(FATAL) << "ZeroDivisionError: division by zero";
                        throw;
                      }
                      call->callee = ir::NullValue<OpValue>();
                      if (s1->IsInstance<FloatValueObj>() || s2->IsInstance<FloatValueObj>()) {
                        double a1 = s1->value;
                        double a2 = s2->value;
                        double result = fmod(a1, a2);
                        call->out = ScalarValue::make(result);
                      } else {
                        int64_t a1 = s1->value;
                        int64_t a2 = s2->value;
                        int64_t result = a1 % a2;
                        call->out = ScalarValue::make(result);
                      }
                      return;
                    }));
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.less", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(<, x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.greater", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(>, x1, x2);
  MNM_LOGICAL_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.right_shift", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                      if (!s1->IsInstance<IntValueObj>() || !s2->IsInstance<IntValueObj>()) {
                        LOG(FATAL) << "func 'right_shift' not supported for the input types";
                        throw;
                      }
                      call->callee = ir::NullValue<OpValue>();
                      int64_t a1 = s1->value;
                      int64_t a2 = s2->value;
                      int64_t result = a1 >> a2;
                      call->out = ScalarValue::make(result);
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.less_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(<=, x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.greater_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(>=, x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(==, x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.not_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(!=, x1, x2);
  MNM_LOGICAL_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.maximum", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(v1, x1, MNM_SWITCH_SCALAR(v2, x2, {
                      call->callee = ir::NullValue<OpValue>();
                      call->out = ScalarValue::make(v1->value > v2->value ? v1->value : v2->value);
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.minimum", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(v1, x1, MNM_SWITCH_SCALAR(v2, x2, {
                      call->callee = ir::NullValue<OpValue>();
                      call->out = ScalarValue::make(v1->value < v2->value ? v1->value : v2->value);
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.logical_and", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_BINARY_SCALAR(&&, x1, x2);
  MNM_BINARY_TENSOR(x1, x2);
});

void CollapseAxis(const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);

  auto x1_type = op::type::GetType(args->x1);
  auto x2_type = op::type::GetType(args->x2);

  auto* t1_type = x1_type.as<ir::TensorTypeNode>();
  auto* t2_type = x2_type.as<ir::TensorTypeNode>();

  call->callee = ir::NullValue<OpValue>();

  if (t1_type && t2_type) {
    CHECK_LE(t2_type->shape.size(), t1_type->shape.size());
    int offset = t1_type->shape.size() - t2_type->shape.size();
    ir::Array<Value> res;
    for (int i = 0; i < offset; ++i) {
      res.push_back(ScalarValue::make(i));
    }
    tvm::arith::Analyzer analyzer;
    for (int i = 0; i < t2_type->shape.size(); ++i) {
      if (!analyzer.CanProve((t1_type->shape[i + offset] - t2_type->shape[i]) == 0)) {
        auto* si = t2_type->shape[i].as<ir::IntImmNode>();
        CHECK(si && si->value == 1) << "The collapsed dimension should be 1-sized!";
        res.push_back(ScalarValue::make(i + offset));
      }
    }
    call->out = TupleValue::make(res);
  } else {
    call->out = ir::NullValue<Value>();
  }
}

MNM_REGISTER_BINARY_BCAST_OP("mnm.op.left_shift", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                      if (s2->value < 0) {
                        LOG(FATAL) << "ValueError: Negative shift count";
                        throw;
                      }

                      call->callee = ir::NullValue<OpValue>();
                      if (s1->IsInstance<IntValueObj>() && s2->IsInstance<IntValueObj>()) {
                        int64_t a1 = s1->value;
                        int64_t a2 = s2->value;
                        int64_t result = a1 << a2;
                        call->out = ScalarValue::make(result);
                      } else {
                        LOG(FATAL) << "ValueError: Int value expected";
                        throw;
                      }
                      return;
                    }));
  MNM_BINARY_TENSOR(x1, x2);
});

// TODO(@icemelon9): Currently use opaque for shape related op.
MNM_OP_DECLARE("mnm.op.get_reduce_axis", CollapseAxis).set_attr<TOpPattern>("TOpPattern", kOpaque);

void CollapseKeep(const CallValues& call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  auto x1_type = op::type::GetType(args->x1);
  auto x2_type = op::type::GetType(args->x2);

  auto* t1_type = x1_type.as<ir::TensorTypeNode>();
  auto* t2_type = x2_type.as<ir::TensorTypeNode>();

  call->callee = ir::NullValue<OpValue>();

  if (t1_type && t2_type) {
    CHECK_LE(t2_type->shape.size(), t1_type->shape.size());
    int offset = t1_type->shape.size() - t2_type->shape.size();
    ir::Array<Value> res;
    for (int i = 0; i < offset; ++i) {
      res.push_back(ScalarValue::make(0));
    }
    tvm::arith::Analyzer analyzer;
    for (int i = 0; i < t2_type->shape.size(); ++i) {
      if (!analyzer.CanProve((t1_type->shape[i + offset] - t2_type->shape[i]) == 0)) {
        auto* si = t2_type->shape[i].as<ir::IntImmNode>();
        CHECK(si && si->value == 1) << "The collapsed dimension should be 1-sized!";
        res.push_back(ScalarValue::make(1));
      }
    }
    call->out = TupleValue::make(res);
  } else {
    call->out = ir::NullValue<Value>();
  }
}

// TODO(@icemelon9): Currently use opaque for shape related op.
MNM_OP_DECLARE("mnm.op.get_kept_dims", CollapseKeep).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace mnm
