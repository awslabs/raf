/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/cudnn/dropout.cc
 * \brief Manually-written cuDNN binding for dropout
 */
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/op_utils.h"
#include "../../../common/cuda_utils.h"
#include "../../schema/nn.h"
#include "./cudnn_utils.h"

namespace mnm {
namespace op {
namespace cudnn {
namespace manual {

using namespace mnm::ir;
using ir::Array;
using ir::Attrs;
using value::TensorValue;
using value::TupleValue;
using value::Value;

Integer GetStateSizeInBytes() {
  size_t stateSizeInBytes;
  CUDNN_CALL(cudnnDropoutGetStatesSize(CUDNNThreadEntry::ThreadLocal()->handle, &stateSizeInBytes));
  LOG(INFO) << "The init inputs states should be a 1 dim uint8 tensor";
  return tvm::Integer(stateSizeInBytes);
}

MNM_REGISTER_GLOBAL("mnm.op.cudnn.manual.GetStateSizeInBytes").set_body_typed(GetStateSizeInBytes);

static auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");

class DropoutImplementedByCUDNNDropoutForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xdesc;
  cudnnTensorDescriptor_t ydesc;
  float dropout;
  cudnnDropoutDescriptor_t dropoutDesc;
  size_t stateSizeInBytes;
  // void* states;
  size_t reserveSpaceSizeInBytes;
  void* reserveSpace;
  explicit DropoutImplementedByCUDNNDropoutForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op._contrib_dropout");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("in_states"),
    };
    auto args = cv->args.as<mnm::op::schema::DropoutArgs>();
    (void)args;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = tv->fields[0];
    (void)out;
    auto xdesc_tt = SquashTensorShape(x, {});
    xdesc = NormalizeTensorType(xdesc_tt);
    auto ydesc_tt = SquashTensorShape(out, {});
    ydesc = NormalizeTensorType(ydesc_tt);
    dropout = args->p;
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
    CUDNN_CALL(
        cudnnDropoutGetStatesSize(CUDNNThreadEntry::ThreadLocal()->handle, &stateSizeInBytes));
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(xdesc, &reserveSpaceSizeInBytes));
    RequestWorkspace(&reserveSpace, cv->device, reserveSpaceSizeInBytes);
    DLTensor* in_states = args->in_states.value();
    (void)in_states;
    void* out_states = tv->fields[2];
    out_states = in_states->data;
    CUDNN_CALL(cudnnRestoreDropoutDescriptor(dropoutDesc, CUDNNThreadEntry::ThreadLocal()->handle,
                                             dropout, out_states, stateSizeInBytes, time(0)));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op._contrib_dropout"));
  }

 public:
  ~DropoutImplementedByCUDNNDropoutForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xdesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(ydesc));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropoutDesc));
  }
  void Execute(const CallValues& cv) {
    LOG(INFO) << "CUDNN implementation of _contrib_dropout not support mask return";
    auto args = cv->args.as<mnm::op::schema::DropoutArgs>();
    (void)args;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = tv->fields[0];
    (void)out;

    CUDNN_CALL(cudnnDropoutForward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, xdesc,
                                   x->data, ydesc, out->data, reserveSpace,
                                   reserveSpaceSizeInBytes));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    LOG(INFO) << "CUDNN implementation of _contrib_dropout not support mask return";
    CHECK_EQ(inputs.size(), 1);
    TupleValue tv = Downcast<TupleValue>(output);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(tv->fields[0]);

    CUDNN_CALL(cudnnDropoutForward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, xdesc,
                                   x->data, ydesc, out->data, reserveSpace,
                                   reserveSpaceSizeInBytes));
  }
  static OpEnv* make(const CallValues& cv) {
    return new DropoutImplementedByCUDNNDropoutForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op._contrib_dropout", DropoutImplementedByCUDNNDropoutForward::make,
                DevType::kCUDA(), "generated_cudnn");

}  // namespace manual
}  // namespace cudnn
}  // namespace op
}  // namespace mnm
