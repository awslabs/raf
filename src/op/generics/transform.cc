#include <mnm/op.h>
#include <mnm/tensor.h>
#include "../args/transform.h"
#include "../args/ufunc.h"
#include "../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace generics {

using namespace mnm::op::args;
using namespace mnm::value;
using tensor::Tensor;

void BatchFlatten(const CallValues& call) {
  using common::shape_utils::IsCompact;
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const Tensor& data = args->x;
  const int ndim = data->ndim;
  CHECK_GE(ndim, 2) << "ValueError: batch_flatten only works with ndim >= 2";

  if (IsCompact(*data.operator->())) {
    const int64_t* dshape = data->shape;
    int64_t flat{1};
    for (int i = 1; i < ndim; ++i) {
      flat = flat * int64_t{dshape[i]};
    }
    call->callee = ir::NullValue<OpValue>();
    call->out = TensorValue::make(data.CreateView({dshape[0], flat}, {}, nullptr));
    return;
  }
  LOG(FATAL) << "NotImplementedError: for now we only support batch_flatten on contiguous tensor.";
  throw;
}

MNM_REGISTER_OP("mnm.op.batch_flatten", UnaryArgs)
    .describe(R"code(This is batch flatten.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", BatchFlatten);

}  // namespace generics
}  // namespace op
}  // namespace mnm
