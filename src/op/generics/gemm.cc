#include <mnm/op.h>
#include <mnm/tensor.h>
#include "../args/ufunc.h"

namespace mnm {
namespace op {
namespace generics {

using namespace mnm::op::args;
using namespace mnm::value;

void Linear(const CallValues& call) {
  /*
   * This is essentially transposed matrix multiplication.
   * [..., a] * [b, a] => [..., b]
   */
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x1;
  const DLTensor* w = args->x2;
  // x is of shape [..., a]
  CHECK_GE(x->ndim, 1);
  // w is of shape [b, a]
  CHECK_EQ(w->ndim, 2);
  std::vector<int64_t> o_shape(x->shape, x->shape + x->ndim);
  int64_t b = w->shape[0];
  int64_t a = w->shape[1];
  CHECK_EQ(o_shape.back(), a);
  o_shape.back() = b;
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx, /*dtype=*/x->dtype, /*shape=*/o_shape);
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.linear", BinaryArgs)
    .describe(R"code(This is linear.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Linear);

}  // namespace generics
}  // namespace op
}  // namespace mnm
