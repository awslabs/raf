/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/binary.cc
 * \brief Declaration of binary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

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

#define MNM_BINARY_SCALAR(op, x1, x2)                                      \
  MNM_SWITCH_SCALAR(v1, x1, MNM_SWITCH_SCALAR(v2, x2, {                    \
                      call->callee = ir::NullValue<OpValue>();             \
                      call->out = ScalarValue::make(v1->data op v2->data); \
                      return;                                              \
                    }));

#define MNM_BINARY_TENSOR(x1, x2)                                             \
  if (x1->IsInstance<TensorValueObj>() && x2->IsInstance<TensorValueObj>()) { \
    const TensorValue& tv = MakeBinaryTensor(x1, x2);                         \
    call->out = tv;                                                           \
    call->ctx = tv->tensor->ctx;                                              \
    return;                                                                   \
  }

TensorValue MakeBinaryTensor(DLTensor* x1, DLTensor* x2) {
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
  return TensorValue::Assemble(x1->ctx, x1->dtype, oshape);
}

MNM_OP_DECLARE("mnm.op.add", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(+, x1, x2);
    MNM_BINARY_TENSOR(x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.subtract", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(-, x1, x2);
    MNM_BINARY_TENSOR(x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.multiply", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(*, x1, x2);
    MNM_BINARY_TENSOR(x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.divide", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                        if (s2->data == 0) {
                          LOG(FATAL) << "ZeroDivisionError: division by zero";
                          throw;
                        }
                        call->callee = ir::NullValue<OpValue>();
                        call->out = ScalarValue::make(s1->data / s2->data);
                        return;
                      }));
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.mod", [](const CallValues& call) {
  // TODO(@junrushao1994): python-style Euclidean division modulo
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_SWITCH_SCALAR(s1, x1, MNM_SWITCH_SCALAR(s2, x2, {
                        if (s2->data == 0) {
                          LOG(FATAL) << "ZeroDivisionError: division by zero";
                          throw;
                        }
                        call->callee = ir::NullValue<OpValue>();
                        if (s1->IsInstance<FloatValueObj>() || s2->IsInstance<FloatValueObj>()) {
                          double a1 = s1->data;
                          double a2 = s2->data;
                          double result = fmod(a1, a2);
                          call->out = ScalarValue::make(result);
                        } else {
                          int64_t a1 = s1->data;
                          int64_t a2 = s2->data;
                          int64_t result = a1 % a2;
                          call->out = ScalarValue::make(result);
                        }
                        return;
                      }));
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.less", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(<, x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.greater", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(>, x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.less_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(<=, x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.greater_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(>=, x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(==, x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.not_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  const Value& x1 = args->x1;
  const Value& x2 = args->x2;
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(!=, x1, x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

void CollapseAxis(const CallValues &call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  DLTensor *x1 = args->x1;
  DLTensor *x2 = args->x2;

  call->callee = ir::NullValue<OpValue>();
  call->ctx = x1->ctx;

  CHECK_LE(x2->ndim, x1->ndim);
  int offset = x1->ndim - x2->ndim;
  ir::Array<Value> res;
  for (int i = 0; i < offset; ++i) {
    res.push_back(ScalarValue::make(i));
  }
  for (int i = 0; i < x2->ndim; ++i) {
    if (x1->shape[i + offset] != x2->shape[i]) {
      CHECK_EQ(x2->shape[i], 1) << "The collapsed dimension should be 1-sized!";
      res.push_back(ScalarValue::make(i + offset));
    }
  }

  call->out = TupleValue::make(res);
}

MNM_OP_DECLARE("mnm.op.get_reduce_axis", CollapseAxis);

void CollapseKeep(const CallValues &call) {
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  DLTensor *x1 = args->x1;
  DLTensor *x2 = args->x2;

  call->callee = ir::NullValue<OpValue>();
  call->ctx = x1->ctx;

  CHECK_LE(x2->ndim, x1->ndim);
  int offset = x1->ndim - x2->ndim;
  ir::Array<Value> res;
  for (int i = 0; i < offset; ++i) {
    res.push_back(ScalarValue::make(0));
  }
  for (int i = 0; i < x2->ndim; ++i) {
    if (x1->shape[i + offset] != x2->shape[i]) {
      CHECK_EQ(x2->shape[i], 1) << "The collapsed dimension should be 1-sized!";
      res.push_back(ScalarValue::make(1));
    }
  }

  call->out = TupleValue::make(res);
}

MNM_OP_DECLARE("mnm.op.get_kept_dims", CollapseKeep);

}  // namespace declare
}  // namespace op
}  // namespace mnm
