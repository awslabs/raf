#include <cudnn.h>
#include <mnm/op.h>
#include "../../../common/arg_utils.h"
#include "../../../common/cuda.h"
#include "../../../common/shape_utils.h"
#include "./util.h"

#include "../../attrs/conv.h"
#include "../../attrs/dropout.h"
#include "../../attrs/pool.h"
namespace mnm {
namespace op {
namespace backend {
namespace cudnn {
namespace generated {

class NanPropagationEnum : public EnumBase<NanPropagationEnum, 2, int32_t, cudnnNanPropagation_t> {
 public:
  ENUM_DEF_HEADER(NanPropagationEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(NanPropagationEnum, 0, NotPropagateNan, CUDNN_NOT_PROPAGATE_NAN,
                           "CUDNN_NOT_PROPAGATE_NAN");
  ENUM_DEF_ENTRY_WITH_NAME(NanPropagationEnum, 1, PropagateNan, CUDNN_PROPAGATE_NAN,
                           "CUDNN_PROPAGATE_NAN");
};

class ActivationModeEnum : public EnumBase<ActivationModeEnum, 6, int32_t, cudnnActivationMode_t> {
 public:
  ENUM_DEF_HEADER(ActivationModeEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(ActivationModeEnum, 0, ActivationSigmoid, CUDNN_ACTIVATION_SIGMOID,
                           "CUDNN_ACTIVATION_SIGMOID");
  ENUM_DEF_ENTRY_WITH_NAME(ActivationModeEnum, 1, ActivationRelu, CUDNN_ACTIVATION_RELU,
                           "CUDNN_ACTIVATION_RELU");
  ENUM_DEF_ENTRY_WITH_NAME(ActivationModeEnum, 2, ActivationTanh, CUDNN_ACTIVATION_TANH,
                           "CUDNN_ACTIVATION_TANH");
  ENUM_DEF_ENTRY_WITH_NAME(ActivationModeEnum, 3, ActivationClippedRelu,
                           CUDNN_ACTIVATION_CLIPPED_RELU, "CUDNN_ACTIVATION_CLIPPED_RELU");
  ENUM_DEF_ENTRY_WITH_NAME(ActivationModeEnum, 4, ActivationElu, CUDNN_ACTIVATION_ELU,
                           "CUDNN_ACTIVATION_ELU");
  ENUM_DEF_ENTRY_WITH_NAME(ActivationModeEnum, 5, ActivationIdentity, CUDNN_ACTIVATION_IDENTITY,
                           "CUDNN_ACTIVATION_IDENTITY");
};

class SoftmaxModeEnum : public EnumBase<SoftmaxModeEnum, 2, int32_t, cudnnSoftmaxMode_t> {
 public:
  ENUM_DEF_HEADER(SoftmaxModeEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(SoftmaxModeEnum, 0, SoftmaxModeInstance, CUDNN_SOFTMAX_MODE_INSTANCE,
                           "CUDNN_SOFTMAX_MODE_INSTANCE");
  ENUM_DEF_ENTRY_WITH_NAME(SoftmaxModeEnum, 1, SoftmaxModeChannel, CUDNN_SOFTMAX_MODE_CHANNEL,
                           "CUDNN_SOFTMAX_MODE_CHANNEL");
};

class TensorFormatEnum : public EnumBase<TensorFormatEnum, 3, int32_t, cudnnTensorFormat_t> {
 public:
  ENUM_DEF_HEADER(TensorFormatEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(TensorFormatEnum, 0, TensorNchw, CUDNN_TENSOR_NCHW, "CUDNN_TENSOR_NCHW");
  ENUM_DEF_ENTRY_WITH_NAME(TensorFormatEnum, 1, TensorNhwc, CUDNN_TENSOR_NHWC, "CUDNN_TENSOR_NHWC");
  ENUM_DEF_ENTRY_WITH_NAME(TensorFormatEnum, 2, TensorNchwVectC, CUDNN_TENSOR_NCHW_VECT_C,
                           "CUDNN_TENSOR_NCHW_VECT_C");
};

class ConvolutionFwdAlgoEnum
    : public EnumBase<ConvolutionFwdAlgoEnum, 9, int32_t, cudnnConvolutionFwdAlgo_t> {
 public:
  ENUM_DEF_HEADER(ConvolutionFwdAlgoEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 0, ConvolutionFwdAlgoImplicitGemm,
                           CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                           "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 1, ConvolutionFwdAlgoImplicitPrecompGemm,
                           CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                           "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 2, ConvolutionFwdAlgoGemm,
                           CUDNN_CONVOLUTION_FWD_ALGO_GEMM, "CUDNN_CONVOLUTION_FWD_ALGO_GEMM");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 3, ConvolutionFwdAlgoDirect,
                           CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 4, ConvolutionFwdAlgoFft,
                           CUDNN_CONVOLUTION_FWD_ALGO_FFT, "CUDNN_CONVOLUTION_FWD_ALGO_FFT");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 5, ConvolutionFwdAlgoFftTiling,
                           CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                           "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 6, ConvolutionFwdAlgoWinograd,
                           CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                           "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 7, ConvolutionFwdAlgoWinogradNonfused,
                           CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                           "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionFwdAlgoEnum, 8, ConvolutionFwdAlgoCount,
                           CUDNN_CONVOLUTION_FWD_ALGO_COUNT, "CUDNN_CONVOLUTION_FWD_ALGO_COUNT");
};

class PoolingModeEnum : public EnumBase<PoolingModeEnum, 4, int32_t, cudnnPoolingMode_t> {
 public:
  ENUM_DEF_HEADER(PoolingModeEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(PoolingModeEnum, 0, PoolingMax, CUDNN_POOLING_MAX, "CUDNN_POOLING_MAX");
  ENUM_DEF_ENTRY_WITH_NAME(PoolingModeEnum, 1, PoolingAverageCountIncludePadding,
                           CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                           "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING");
  ENUM_DEF_ENTRY_WITH_NAME(PoolingModeEnum, 2, PoolingAverageCountExcludePadding,
                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                           "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING");
  ENUM_DEF_ENTRY_WITH_NAME(PoolingModeEnum, 3, PoolingMaxDeterministic,
                           CUDNN_POOLING_MAX_DETERMINISTIC, "CUDNN_POOLING_MAX_DETERMINISTIC");
};

class ConvolutionModeEnum
    : public EnumBase<ConvolutionModeEnum, 2, int32_t, cudnnConvolutionMode_t> {
 public:
  ENUM_DEF_HEADER(ConvolutionModeEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionModeEnum, 0, Convolution, CUDNN_CONVOLUTION,
                           "CUDNN_CONVOLUTION");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionModeEnum, 1, CrossCorrelation, CUDNN_CROSS_CORRELATION,
                           "CUDNN_CROSS_CORRELATION");
};

class SoftmaxAlgorithmEnum
    : public EnumBase<SoftmaxAlgorithmEnum, 3, int32_t, cudnnSoftmaxAlgorithm_t> {
 public:
  ENUM_DEF_HEADER(SoftmaxAlgorithmEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(SoftmaxAlgorithmEnum, 0, SoftmaxFast, CUDNN_SOFTMAX_FAST,
                           "CUDNN_SOFTMAX_FAST");
  ENUM_DEF_ENTRY_WITH_NAME(SoftmaxAlgorithmEnum, 1, SoftmaxAccurate, CUDNN_SOFTMAX_ACCURATE,
                           "CUDNN_SOFTMAX_ACCURATE");
  ENUM_DEF_ENTRY_WITH_NAME(SoftmaxAlgorithmEnum, 2, SoftmaxLog, CUDNN_SOFTMAX_LOG,
                           "CUDNN_SOFTMAX_LOG");
};

AlgorithmCache<cudnnConvolutionFwdAlgo_t> _cache_cudnnConvolutionFwdAlgo_t;
cudnnConvolutionFwdAlgo_t FindConvolutionForwardAlgorithm(
    const std::vector<int>& key, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc) {
  if (_cache_cudnnConvolutionFwdAlgo_t.has(key)) {
    return _cache_cudnnConvolutionFwdAlgo_t.get(key);
  }
  int cnt;
  cudnnConvolutionFwdAlgoPerf_t res;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(CUDNNThreadEntry::ThreadLocal()->handle, xDesc,
                                                  wDesc, convDesc, yDesc, 1, &cnt, &res));
  if (res.status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm!\n";
    throw;
  }
  _cache_cudnnConvolutionFwdAlgo_t.set(key, res.algo);
  return res.algo;
}

// cudnnConvolutionForward(cudnnHandle_t handle,
//                         const void *alpha,
//                         const cudnnTensorDescriptor_t xDesc,
//                         const void *x,
//                         const cudnnFilterDescriptor_t wDesc,
//                         const void *w,
//                         const cudnnConvolutionDescriptor_t convDesc,
//                         cudnnConvolutionFwdAlgo_t algo,
//                         void *workSpace,
//                         size_t workSpaceSizeInBytes,
//                         const void *beta,
//                         const cudnnTensorDescriptor_t yDesc,
//                         void *y)
class ConvolutionForward_for_op_conv2d : public mnm::op::OpEnv {
 public:
  ConvolutionForward_for_op_conv2d() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t algo;
  size_t workSpaceSizeInBytes;
  cudnnTensorDescriptor_t yDesc;
  void* workSpace;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);
    static auto casted_ptr = attrs.as<attrs::Conv2DAttrs>();

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateFilterDescriptor(&wDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    CUDNN_CALL(cudnnSetFilterNdDescriptor(wDesc, CUDNNDType(dtype),
                                          TensorFormatEnum(TensorFormatEnum::TensorNchw()),
                                          shape_1.size(), dmlc::BeginPtr(shape_1)));

    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(
        convDesc, casted_ptr->stride.size(),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->padding)),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->stride)),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->dilation)),
        ConvolutionModeEnum(ConvolutionModeEnum::CrossCorrelation()), CUDNNDType(dtype)));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(convDesc, casted_ptr->groups));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_2, dlts[2]);
    FORM_STRIDE(stride_2, shape_2);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_2.size(),
                                          dmlc::BeginPtr(shape_2), dmlc::BeginPtr(stride_2)));

    int size_2 = ((shape_2[0] * stride_2[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[2]->data), ctx, size_2);

    auto key = ConcatVecs(shape_0, shape_1, shape_2, stride_0, stride_2, casted_ptr->padding,
                          casted_ptr->stride, casted_ptr->dilation);
    FindConvolutionForwardAlgorithm(key, xDesc, wDesc, convDesc, yDesc);
    size_t workSpaceSizeInBytes;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                       xDesc, wDesc, convDesc, yDesc, algo,
                                                       &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, ctx, workSpaceSizeInBytes);
    this->workSpaceSizeInBytes = workSpaceSizeInBytes;
  }

  ~ConvolutionForward_for_op_conv2d() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<ConvolutionForward_for_op_conv2d> res =
        std::make_unique<ConvolutionForward_for_op_conv2d>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnConvolutionForward(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(dtype).const_addr<1>(), xDesc,
        dlts[0]->data, wDesc, dlts[1]->data, convDesc, algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[2]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.conv2d", DevType::kCUDA(), "generated_cudnn",
                         ConvolutionForward_for_op_conv2d::make);

// cudnnAddTensor(cudnnHandle_t handle,
//                const void *alpha,
//                const cudnnTensorDescriptor_t aDesc,
//                const void *A,
//                const void *beta,
//                const cudnnTensorDescriptor_t cDesc,
//                void *C)
class AddTensor_for_op_add : public mnm::op::OpEnv {
 public:
  AddTensor_for_op_add() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t aDesc;
  cudnnTensorDescriptor_t cDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);

    CUDNN_CALL(cudnnCreateTensorDescriptor(&aDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(aDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&cDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);
  }

  ~AddTensor_for_op_add() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(aDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<AddTensor_for_op_add> res = std::make_unique<AddTensor_for_op_add>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnAddTensor(CUDNNThreadEntry::ThreadLocal()->handle,
                              CUDNNDType(dtype).const_addr<1>(), aDesc, dlts[0]->data,
                              CUDNNDType(dtype).const_addr<0>(), cDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.add", DevType::kCUDA(), "generated_cudnn",
                         AddTensor_for_op_add::make);

// cudnnActivationForward(cudnnHandle_t handle,
//                        cudnnActivationDescriptor_t activationDesc,
//                        const void *alpha,
//                        const cudnnTensorDescriptor_t xDesc,
//                        const void *x,
//                        const void *beta,
//                        const cudnnTensorDescriptor_t yDesc,
//                        void *y)
class ActivationForward_for_op_relu : public mnm::op::OpEnv {
 public:
  ActivationForward_for_op_relu() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnActivationDescriptor_t activationDesc;
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);

    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(
        activationDesc, ActivationModeEnum(ActivationModeEnum::ActivationRelu()),
        NanPropagationEnum(NanPropagationEnum::PropagateNan()), 0.0));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);
  }

  ~ActivationForward_for_op_relu() {
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<ActivationForward_for_op_relu> res =
        std::make_unique<ActivationForward_for_op_relu>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(dtype).const_addr<1>(), xDesc, dlts[0]->data,
                                      CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.relu", DevType::kCUDA(), "generated_cudnn",
                         ActivationForward_for_op_relu::make);

// cudnnActivationForward(cudnnHandle_t handle,
//                        cudnnActivationDescriptor_t activationDesc,
//                        const void *alpha,
//                        const cudnnTensorDescriptor_t xDesc,
//                        const void *x,
//                        const void *beta,
//                        const cudnnTensorDescriptor_t yDesc,
//                        void *y)
class ActivationForward_for_op_tanh : public mnm::op::OpEnv {
 public:
  ActivationForward_for_op_tanh() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnActivationDescriptor_t activationDesc;
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);

    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(
        activationDesc, ActivationModeEnum(ActivationModeEnum::ActivationTanh()),
        NanPropagationEnum(NanPropagationEnum::PropagateNan()), 0.0));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);
  }

  ~ActivationForward_for_op_tanh() {
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<ActivationForward_for_op_tanh> res =
        std::make_unique<ActivationForward_for_op_tanh>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(dtype).const_addr<1>(), xDesc, dlts[0]->data,
                                      CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.tanh", DevType::kCUDA(), "generated_cudnn",
                         ActivationForward_for_op_tanh::make);

// cudnnActivationForward(cudnnHandle_t handle,
//                        cudnnActivationDescriptor_t activationDesc,
//                        const void *alpha,
//                        const cudnnTensorDescriptor_t xDesc,
//                        const void *x,
//                        const void *beta,
//                        const cudnnTensorDescriptor_t yDesc,
//                        void *y)
class ActivationForward_for_op_sigmoid : public mnm::op::OpEnv {
 public:
  ActivationForward_for_op_sigmoid() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnActivationDescriptor_t activationDesc;
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);

    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(
        activationDesc, ActivationModeEnum(ActivationModeEnum::ActivationSigmoid()),
        NanPropagationEnum(NanPropagationEnum::PropagateNan()), 0.0));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);
  }

  ~ActivationForward_for_op_sigmoid() {
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<ActivationForward_for_op_sigmoid> res =
        std::make_unique<ActivationForward_for_op_sigmoid>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(dtype).const_addr<1>(), xDesc, dlts[0]->data,
                                      CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.sigmoid", DevType::kCUDA(), "generated_cudnn",
                         ActivationForward_for_op_sigmoid::make);

// cudnnSoftmaxForward(cudnnHandle_t handle,
//                     cudnnSoftmaxAlgorithm_t algo,
//                     cudnnSoftmaxMode_t mode,
//                     const void *alpha,
//                     const cudnnTensorDescriptor_t xDesc,
//                     const void *x,
//                     const void *beta,
//                     const cudnnTensorDescriptor_t yDesc,
//                     void *y)
class SoftmaxForward_for_op_softmax : public mnm::op::OpEnv {
 public:
  SoftmaxForward_for_op_softmax() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnSoftmaxAlgorithm_t algo;
  cudnnSoftmaxMode_t mode;
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);

    algo = SoftmaxAlgorithmEnum(SoftmaxAlgorithmEnum::SoftmaxAccurate());
    mode = SoftmaxModeEnum(SoftmaxModeEnum::SoftmaxModeInstance());
  }

  ~SoftmaxForward_for_op_softmax() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<SoftmaxForward_for_op_softmax> res =
        std::make_unique<SoftmaxForward_for_op_softmax>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, algo, mode,
                                   CUDNNDType(dtype).const_addr<1>(), xDesc, dlts[0]->data,
                                   CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.softmax", DevType::kCUDA(), "generated_cudnn",
                         SoftmaxForward_for_op_softmax::make);

// cudnnPoolingForward(cudnnHandle_t handle,
//                     const cudnnPoolingDescriptor_t poolingDesc,
//                     const void *alpha,
//                     const cudnnTensorDescriptor_t xDesc,
//                     const void *x,
//                     const void *beta,
//                     const cudnnTensorDescriptor_t yDesc,
//                     void *y)
class PoolingForward_for_op_max_pool2d : public mnm::op::OpEnv {
 public:
  PoolingForward_for_op_max_pool2d() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnPoolingDescriptor_t poolingDesc;
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);
    static auto casted_ptr = attrs.as<attrs::MaxPoolAttrs>();

    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(
        poolingDesc, PoolingModeEnum(PoolingModeEnum::PoolingMax()),
        NanPropagationEnum(NanPropagationEnum::PropagateNan()), casted_ptr->kernel_size.size(),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->kernel_size)),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->padding)),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->stride))));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);
  }

  ~PoolingForward_for_op_max_pool2d() {
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<PoolingForward_for_op_max_pool2d> res =
        std::make_unique<PoolingForward_for_op_max_pool2d>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(dtype).const_addr<1>(), xDesc, dlts[0]->data,
                                   CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.max_pool2d", DevType::kCUDA(), "generated_cudnn",
                         PoolingForward_for_op_max_pool2d::make);

// cudnnPoolingForward(cudnnHandle_t handle,
//                     const cudnnPoolingDescriptor_t poolingDesc,
//                     const void *alpha,
//                     const cudnnTensorDescriptor_t xDesc,
//                     const void *x,
//                     const void *beta,
//                     const cudnnTensorDescriptor_t yDesc,
//                     void *y)
class PoolingForward_for_op_avg_pool2d : public mnm::op::OpEnv {
 public:
  PoolingForward_for_op_avg_pool2d() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnPoolingDescriptor_t poolingDesc;
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);
    static auto casted_ptr = attrs.as<attrs::AvgPoolAttrs>();

    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        casted_ptr->include_pad
            ? PoolingModeEnum(PoolingModeEnum::PoolingAverageCountIncludePadding())
            : PoolingModeEnum(PoolingModeEnum::PoolingAverageCountExcludePadding()),
        NanPropagationEnum(NanPropagationEnum::PropagateNan()), casted_ptr->kernel_size.size(),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->kernel_size)),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->padding)),
        dmlc::BeginPtr(common::shape_utils::MakeShape<int>(casted_ptr->stride))));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);
  }

  ~PoolingForward_for_op_avg_pool2d() {
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<PoolingForward_for_op_avg_pool2d> res =
        std::make_unique<PoolingForward_for_op_avg_pool2d>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(dtype).const_addr<1>(), xDesc, dlts[0]->data,
                                   CUDNNDType(dtype).const_addr<0>(), yDesc, dlts[1]->data));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.avg_pool2d", DevType::kCUDA(), "generated_cudnn",
                         PoolingForward_for_op_avg_pool2d::make);

// cudnnDropoutForward(cudnnHandle_t handle,
//                     const cudnnDropoutDescriptor_t dropoutDesc,
//                     const cudnnTensorDescriptor_t xdesc,
//                     const void *x,
//                     const cudnnTensorDescriptor_t ydesc,
//                     void *y,
//                     void *reserveSpace,
//                     size_t reserveSpaceSizeInBytes)
class DropoutForward_for_op_dropout : public mnm::op::OpEnv {
 public:
  DropoutForward_for_op_dropout() {
  }

  DType dtype;
  cudnnHandle_t handle;
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t xdesc;
  cudnnTensorDescriptor_t ydesc;
  size_t reserveSpaceSizeInBytes;
  void* state;
  void* reserveSpace;

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);
    auto ctx = common::arg_utils::DeduceCtx(dlts);
    static auto casted_ptr = attrs.as<attrs::DropoutAttrs>();

    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
    size_t stateSizeInBytes;
    CUDNN_CALL(
        cudnnDropoutGetStatesSize(CUDNNThreadEntry::ThreadLocal()->handle, &stateSizeInBytes));
    RequestMemory(&state, ctx, stateSizeInBytes);
    CUDNN_CALL(cudnnSetDropoutDescriptor(dropoutDesc, CUDNNThreadEntry::ThreadLocal()->handle,
                                         casted_ptr->dropout, state, stateSizeInBytes, time(0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xdesc));
    FORM_SHAPE(shape_0, dlts[0]);
    FORM_STRIDE(stride_0, shape_0);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(xdesc, CUDNNDType(dtype), shape_0.size(),
                                          dmlc::BeginPtr(shape_0), dmlc::BeginPtr(stride_0)));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&ydesc));
    FORM_SHAPE(shape_1, dlts[1]);
    FORM_STRIDE(stride_1, shape_1);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(ydesc, CUDNNDType(dtype), shape_1.size(),
                                          dmlc::BeginPtr(shape_1), dmlc::BeginPtr(stride_1)));

    int size_1 = ((shape_1[0] * stride_1[0]) * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size_1);

    size_t reserveSpaceSizeInBytes;
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(xdesc, &reserveSpaceSizeInBytes));
    RequestMemory(&reserveSpace, ctx, reserveSpaceSizeInBytes);
    this->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
  }

  ~DropoutForward_for_op_dropout() {
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropoutDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xdesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(ydesc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<DropoutForward_for_op_dropout> res =
        std::make_unique<DropoutForward_for_op_dropout>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override final {
    auto dlts = common::arg_utils::AsVector(args);
    CUDNN_CALL(cudnnDropoutForward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, xdesc,
                                   dlts[0]->data, ydesc, dlts[1]->data, reserveSpace,
                                   reserveSpaceSizeInBytes));
  }

  // TODO(@were): After executor is done, remove these two!
  // TODO(@junrushao1994): Implement the executor!
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
  void RequestWorkspace(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.dropout", DevType::kCUDA(), "generated_cudnn",
                         DropoutForward_for_op_dropout::make);

}  // namespace generated
}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
