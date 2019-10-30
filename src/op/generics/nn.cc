#include <mnm/op.h>
#include <mnm/tensor.h>
#include "../args/nn.h"

namespace mnm {
namespace op {
namespace generics {

using namespace mnm::op::args;
using namespace mnm::value;

template <int n>
static std::vector<int64_t> Pad(const std::vector<int64_t>& a) {
  int size = a.size();
  CHECK(size == 1 || size == n);
  return size == 1 ? std::vector<int64_t>(n, a[0]) : a;
}

static int NormalizeAxis(int axis, int ndim) {
  CHECK(-ndim >= axis && axis < ndim);
  return axis < 0 ? axis + ndim : axis;
}

void Conv2d(const CallValues& call) {
  // N.B.: NCHW + OIHW
  const auto* args = call->args.as<ConvArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* w = args->w;
  CHECK_EQ(x->ndim, 4);
  CHECK_EQ(w->ndim, 4);
  // TODO(@junrushao1994): deduce ctx here
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  int64_t n_in = x->shape[0];
  int64_t c_in = x->shape[1];
  int64_t h_in = x->shape[2];
  int64_t w_in = x->shape[3];
  int64_t out = w->shape[0];
  int64_t in = w->shape[1];
  int64_t kernel_h = w->shape[2];
  int64_t kernel_w = w->shape[3];
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h = padding[0];
  int64_t pad_w = padding[1];
  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  int64_t h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  int64_t w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  CHECK_EQ(c_in, in);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/{n_in, out, h_out, w_out});
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.conv2d", ConvArgs)
    .describe(R"code(This is conv2d.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Conv2d);

void Pool2D(const CallValues& call) {
  // NCHW
  const auto* args = call->args.as<PoolArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  CHECK_EQ(x->ndim, 4);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  int64_t n_in = x->shape[0];
  int64_t c_in = x->shape[1];
  int64_t h_in = x->shape[2];
  int64_t w_in = x->shape[3];
  int64_t kernel_h = kernel[0];
  int64_t kernel_w = kernel[1];
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h = padding[0];
  int64_t pad_w = padding[1];
  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  int64_t h_out, w_out;
  if (!args->ceil_mode) {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) + stride_h - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) + stride_w - 1) / stride_w + 1;
  }
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/{n_in, c_in, h_out, w_out});
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.max_pool2d", PoolArgs)
    .describe(R"code(This is max_pool2d.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Pool2D);

MNM_REGISTER_OP("mnm.op.avg_pool2d", PoolArgs)
    .describe(R"code(This is max_pool2d.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Pool2D);

void Softmax(const CallValues& call) {
  const auto* args = call->args.as<SoftmaxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  NormalizeAxis(args->axis, x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.softmax", SoftmaxArgs)
    .describe(R"code(This is softmax.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Softmax);

MNM_REGISTER_OP("mnm.op.log_softmax", SoftmaxArgs)
    .describe(R"code(This is log_softmax.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Softmax);

void BatchNorm(const CallValues& call) {
  const auto* args = call->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  // TODO(@junrushao1994): sanity check
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.batch_norm", BatchNormArgs)
    .describe(R"code(This is batch_norm.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", BatchNorm);

}  // namespace generics
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace generics {

void Conv2dDxw(const CallValues& call) {
  const auto* args = call->args.as<ConvDxwArgs>();
  CHECK(args != nullptr);
  const DLTensor* x_or_w = args->x_or_w;
  std::vector<int64_t> shape(x_or_w->shape, x_or_w->shape + x_or_w->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x_or_w->ctx,
                                    /*dtype=*/x_or_w->dtype,
                                    /*shape=*/shape);
  call->ctx = x_or_w->ctx;
}

MNM_REGISTER_OP("mnm.op.conv2d_dx", ConvDxwArgs)
    .describe(R"code(This is conv2d dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Conv2dDxw);

MNM_REGISTER_OP("mnm.op.conv2d_dw", ConvDxwArgs)
    .describe(R"code(This is conv2d dw.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Conv2dDxw);

void Pool2DDx(const CallValues& call) {
  const auto* args = call->args.as<PoolDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.max_pool2d_dx", PoolDxArgs)
    .describe(R"code(This is max_pool2d dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Pool2DDx);

MNM_REGISTER_OP("mnm.op.avg_pool2d_dx", PoolDxArgs)
    .describe(R"code(This is avg_pool2d dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", Pool2DDx);

void SoftmaxDx(const CallValues& call) {
  const auto* args = call->args.as<SoftmaxDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_REGISTER_OP("mnm.op.softmax_dx", SoftmaxDxArgs)
    .describe(R"code(This is softmax dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", SoftmaxDx);

MNM_REGISTER_OP("mnm.op.log_softmax_dx", SoftmaxDxArgs)
    .describe(R"code(This is log_softmax dx.
)code" MNM_ADD_FILELINE)
    .set_attr<FMNMDeclare>("FMNMDeclare", SoftmaxDx);

}  // namespace generics
}  // namespace op
}  // namespace mnm
