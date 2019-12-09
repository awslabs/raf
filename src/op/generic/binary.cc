/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/generic/binary.cc
 * \brief Declaration of binary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/ufunc.h"
#include "./generic_utils.h"

namespace mnm {
namespace op {
namespace generic {

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

MNM_OP_DECLARE("mnm.op.add", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(+, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.subtract", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(-, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.multiply", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(*, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.divide", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_SWITCH_SCALAR(x1, args->x1, MNM_SWITCH_SCALAR(x2, args->x2, {
                        if (x2->data == 0) {
                          LOG(FATAL) << "ZeroDivisionError: division by zero";
                          throw;
                        }
                        call->callee = ir::NullValue<OpValue>();
                        call->out = ScalarValue::make(x1->data / x2->data);
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
  if (!args->out.defined() && !args->where.defined()) {
    MNM_SWITCH_SCALAR(x1, args->x1, MNM_SWITCH_SCALAR(x2, args->x2, {
                        if (x2->data == 0) {
                          LOG(FATAL) << "ZeroDivisionError: division by zero";
                          throw;
                        }
                        call->callee = ir::NullValue<OpValue>();
                        if (x1->IsInstance<FloatValueObj>() || x2->IsInstance<FloatValueObj>()) {
                          double a1 = x1->data;
                          double a2 = x2->data;
                          double result = fmod(a1, a2);
                          call->out = ScalarValue::make(result);
                        } else {
                          int64_t a1 = x1->data;
                          int64_t a2 = x2->data;
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
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(<, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.greater", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(>, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.less_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(<=, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.greater_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(>=, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(==, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.not_equal", [](const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(!=, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

}  // namespace generic
}  // namespace op
}  // namespace mnm
