#include <mnm/op.h>
#include <mnm/tensor.h>
#include "../schema/ufunc.h"

namespace mnm {
namespace op {
namespace generic {

using namespace mnm::op::schema;
using namespace mnm::value;

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

#define MNM_UNARY_SCALAR(op, x)                \
  MNM_SWITCH_SCALAR(v, x, {                    \
    call->callee = ir::NullValue<OpValue>();   \
    call->out = ScalarValue::make(op v->data); \
    return;                                    \
  });

MNM_OP_DECLARE("mnm.op.negative", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_UNARY_SCALAR(-, args->x);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

MNM_OP_DECLARE("mnm.op.logical_not", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_UNARY_SCALAR(!, args->x);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
});

void Unary(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.relu", Unary);
MNM_OP_DECLARE("mnm.op.tanh", Unary);
MNM_OP_DECLARE("mnm.op.sigmoid", Unary);

void UnaryDx(const CallValues& call) {
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<UnaryDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.relu_dx", UnaryDx);
MNM_OP_DECLARE("mnm.op.tanh_dx", UnaryDx);
MNM_OP_DECLARE("mnm.op.sigmoid_dx", UnaryDx);

}  // namespace generic
}  // namespace op
}  // namespace mnm
