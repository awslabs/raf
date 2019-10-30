#include <mnm/op.h>
#include <mnm/tensor.h>
#include "../args/ufunc.h"

namespace mnm {
namespace op {
namespace generics {

using namespace mnm::op::args;
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

void Negative(const CallValues& call) {
  const auto* args = call->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_UNARY_SCALAR(-, args->x);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.negative", UnaryUfuncArgs)
    .describe(R"code(This is Negative.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Negative);

void LogicalNot(const CallValues& call) {
  const auto* args = call->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  if (!args->out.defined() && !args->where.defined()) {
    MNM_UNARY_SCALAR(!, args->x);
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

MNM_REGISTER_OP("mnm.op.logical_not", UnaryUfuncArgs)
    .describe(R"code(This is LogicalNot.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", LogicalNot);

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

MNM_REGISTER_OP("mnm.op.relu", UnaryArgs)
    .describe(R"code(This is ReLU.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Unary);

MNM_REGISTER_OP("mnm.op.tanh", UnaryArgs)
    .describe(R"code(This is tanh.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Unary);

MNM_REGISTER_OP("mnm.op.sigmoid", UnaryArgs)
    .describe(R"code(This is sigmoid.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Unary);

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

MNM_REGISTER_OP("mnm.op.relu_dx", UnaryDxArgs)
    .describe(R"code(This is relu dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", UnaryDx);

MNM_REGISTER_OP("mnm.op.tanh_dx", UnaryDxArgs)
    .describe(R"code(This is tanh dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", UnaryDx);

MNM_REGISTER_OP("mnm.op.sigmoid_dx", UnaryDxArgs)
    .describe(R"code(This is sigmoid dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", UnaryDx);

}  // namespace generics
}  // namespace op
}  // namespace mnm
