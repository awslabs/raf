#include <mnm/op.h>
#include <mnm/tensor.h>
#include "../args/ufunc.h"

namespace mnm {
namespace op {
namespace generics {

using namespace mnm::op::args;
using namespace mnm::value;
using tensor::Tensor;

#define MNM_SWITCH_SCALAR(var, value, body)                      \
  do                                                             \
    if (const auto* var = (value).as<IntValueNode>()) {          \
      body;                                                      \
    } else if (const auto* var = (value).as<FloatValueNode>()) { \
      body;                                                      \
    } else if (const auto* var = (value).as<BoolValueNode>()) {  \
      body;                                                      \
    }                                                            \
  while (0);

#define MNM_BINARY_SCALAR(op, x1, x2)                                      \
  MNM_SWITCH_SCALAR(v1, x1, MNM_SWITCH_SCALAR(v2, x2, {                    \
                      call->callee = ir::NullValue<OpValue>();             \
                      call->out = ScalarValue::make(v1->data op v2->data); \
                      return;                                              \
                    }));

void Add(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(+, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.add", BinaryUfuncArgs)
    .describe(R"code(This is Add.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Add);

void Subtract(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(-, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.subtract", BinaryUfuncArgs)
    .describe(R"code(This is Subtract.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Subtract);

void Multiply(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(*, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.multiply", BinaryUfuncArgs)
    .describe(R"code(This is Multiply.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Multiply);

void Divide(const CallValues& call) {
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
}

MNM_REGISTER_OP("mnm.op.divide", BinaryUfuncArgs)
    .describe(R"code(This is Divide.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Divide);

void Mod(const CallValues& call) {
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
                        if (x1->is_type<FloatValueNode>() || x2->is_type<FloatValueNode>()) {
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
}

MNM_REGISTER_OP("mnm.op.mod", BinaryUfuncArgs)
    .describe(R"code(This is mod.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Mod);

void Less(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(<, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.less", BinaryUfuncArgs)
    .describe(R"code(This is Less.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Less);

void Greater(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(>, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.greater", BinaryUfuncArgs)
    .describe(R"code(This is Greater.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Greater);

void LessEqual(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(<=, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.less_equal", BinaryUfuncArgs)
    .describe(R"code(This is Add.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", LessEqual);

void GreaterEqual(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(>=, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.greater_equal", BinaryUfuncArgs)
    .describe(R"code(This is GreaterEqual.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", GreaterEqual);

void Equal(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(==, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.equal", BinaryUfuncArgs)
    .describe(R"code(This is Equal.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Equal);

void NotEqual(const CallValues& call) {
  const auto* args = call->args.as<BinaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_BINARY_SCALAR(!=, args->x1, args->x2);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.not_equal", BinaryUfuncArgs)
    .describe(R"code(This is NotEqual.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", NotEqual);

}  // namespace generics
}  // namespace op
}  // namespace mnm
