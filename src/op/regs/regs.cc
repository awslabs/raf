/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/regs/regs.cc
 * \brief Register op schemas.
 */
#include <algorithm>
#include <array>
#include "./regs_utils.h"
#include "./ffi2expr.h"
#include "./ffi2schema.h"
#include "./value2schema.h"
#include "./schema2value.h"
#include "../schema/list_args.h"
#include "../schema/algorithm.h"
#include "../schema/communication.h"
#include "../schema/init.h"
#include "../schema/likes.h"
#include "../schema/loss.h"
#include "../schema/memory.h"
#include "../schema/nn.h"
#include "../schema/optimizer.h"
#include "../schema/random.h"
#include "../schema/reduce.h"
#include "../schema/stream.h"
#include "../schema/transform.h"
#include "../schema/ufunc.h"
#include "../schema/vision.h"
#include "../schema/vm.h"

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::registry;
using namespace mnm::binding;
using mnm::executor::interpreter::InvokePrimitive;
using mnm::op::FMNMSchema;
using mnm::op::FMNMSchemaFieldIndex;

// Part 0. Op names
namespace mnm {
namespace op {
namespace regs {
namespace names {
static const char _allgather[] = "mnm.op._allgather";
static const char _allreduce[] = "mnm.op._allreduce";
static const char _broadcast[] = "mnm.op._broadcast";
static const char _contrib_dropout[] = "mnm.op._contrib_dropout";
static const char _contrib_dropout_dx[] = "mnm.op._contrib_dropout_dx";
static const char _recv[] = "mnm.op._recv";
static const char _reduce[] = "mnm.op._reduce";
static const char _reduce_scatter[] = "mnm.op._reduce_scatter";
static const char _send[] = "mnm.op._send";
static const char abs[] = "mnm.op.abs";
static const char adaptive_avg_pool2d[] = "mnm.op.adaptive_avg_pool2d";
static const char adaptive_avg_pool2d_dx[] = "mnm.op.adaptive_avg_pool2d_dx";
static const char adaptive_max_pool2d[] = "mnm.op.adaptive_max_pool2d";
static const char adaptive_max_pool2d_dx[] = "mnm.op.adaptive_max_pool2d_dx";
static const char add[] = "mnm.op.add";
static const char add_event[] = "mnm.op.add_event";
static const char adv_index[] = "mnm.op.adv_index";
static const char adv_index_dx[] = "mnm.op.adv_index_dx";
static const char all[] = "mnm.op.all";
static const char any[] = "mnm.op.any";
static const char arange[] = "mnm.op.arange";
static const char argmax[] = "mnm.op.argmax";
static const char argmin[] = "mnm.op.argmin";
static const char argsort[] = "mnm.op.argsort";
static const char argwhere[] = "mnm.op.argwhere";
static const char atan[] = "mnm.op.atan";
static const char avg_pool2d[] = "mnm.op.avg_pool2d";
static const char avg_pool2d_dx[] = "mnm.op.avg_pool2d_dx";
static const char batch_flatten[] = "mnm.op.batch_flatten";
static const char batch_matmul[] = "mnm.op.batch_matmul";
static const char batch_matmul_nt[] = "mnm.op.batch_matmul_nt";
static const char batch_matmul_tn[] = "mnm.op.batch_matmul_tn";
static const char batch_matmul_tt[] = "mnm.op.batch_matmul_tt";
static const char batch_norm_infer[] = "mnm.op.batch_norm_infer";
static const char batch_norm_train[] = "mnm.op.batch_norm_train";
static const char batch_norm_train_dxwb[] = "mnm.op.batch_norm_train_dxwb";
static const char bias_add[] = "mnm.op.bias_add";
static const char broadcast_to[] = "mnm.op.broadcast_to";
static const char broadcast_to_like[] = "mnm.op.broadcast_to_like";
static const char cast[] = "mnm.op.cast";
static const char cast_like[] = "mnm.op.cast_like";
static const char ceil[] = "mnm.op.ceil";
static const char clip[] = "mnm.op.clip";
static const char clip_dx[] = "mnm.op.clip_dx";
static const char collapse_sum_like[] = "mnm.op.collapse_sum_like";
static const char compiler_begin[] = "mnm.op.compiler_begin";
static const char compiler_end[] = "mnm.op.compiler_end";
static const char concatenate[] = "mnm.op.concatenate";
static const char concatenate_dx[] = "mnm.op.concatenate_dx";
static const char conv2d[] = "mnm.op.conv2d";
static const char conv2d_dw[] = "mnm.op.conv2d_dw";
static const char conv2d_dx[] = "mnm.op.conv2d_dx";
static const char conv2d_transpose[] = "mnm.op.conv2d_transpose";
static const char conv2d_transpose_dw[] = "mnm.op.conv2d_transpose_dw";
static const char conv2d_transpose_dx[] = "mnm.op.conv2d_transpose_dx";
static const char copy[] = "mnm.op.copy";
static const char cos[] = "mnm.op.cos";
static const char cross_entropy[] = "mnm.op.cross_entropy";
static const char cross_entropy_dpred[] = "mnm.op.cross_entropy_dpred";
static const char cross_entropy_dtrue[] = "mnm.op.cross_entropy_dtrue";
static const char cumsum[] = "mnm.op.cumsum";
static const char dense[] = "mnm.op.dense";
static const char device_copy[] = "mnm.op.device_copy";
static const char divide[] = "mnm.op.divide";
static const char embedding[] = "mnm.op.embedding";
static const char embedding_dx[] = "mnm.op.embedding_dx";
static const char equal[] = "mnm.op.equal";
static const char erf[] = "mnm.op.erf";
static const char erf_dx[] = "mnm.op.erf_dx";
static const char exp[] = "mnm.op.exp";
static const char expand_dims[] = "mnm.op.expand_dims";
static const char floor[] = "mnm.op.floor";
static const char floor_divide[] = "mnm.op.floor_divide";
static const char full[] = "mnm.op.full";
static const char full_like[] = "mnm.op.full_like";
static const char gather[] = "mnm.op.gather";
static const char gather_dx[] = "mnm.op.gather_dx";
static const char gather_nd[] = "mnm.op.gather_nd";
static const char gather_nd_dx[] = "mnm.op.gather_nd_dx";
static const char gelu[] = "mnm.op.gelu";
static const char gelu_dx[] = "mnm.op.gelu_dx";
static const char get_kept_dims[] = "mnm.op.get_kept_dims";
static const char get_reduce_axis[] = "mnm.op.get_reduce_axis";
static const char get_valid_counts[] = "mnm.op.get_valid_counts";
static const char greater[] = "mnm.op.greater";
static const char greater_equal[] = "mnm.op.greater_equal";
static const char layer_norm[] = "mnm.op.layer_norm";
static const char layer_norm_dx[] = "mnm.op.layer_norm_dx";
static const char left_shift[] = "mnm.op.left_shift";
static const char less[] = "mnm.op.less";
static const char less_equal[] = "mnm.op.less_equal";
static const char log[] = "mnm.op.log";
static const char log2[] = "mnm.op.log2";
static const char log_softmax[] = "mnm.op.log_softmax";
static const char log_softmax_dx[] = "mnm.op.log_softmax_dx";
static const char logical_and[] = "mnm.op.logical_and";
static const char logical_not[] = "mnm.op.logical_not";
static const char matmul[] = "mnm.op.matmul";
static const char matmul_nt[] = "mnm.op.matmul_nt";
static const char matmul_tn[] = "mnm.op.matmul_tn";
static const char matmul_tt[] = "mnm.op.matmul_tt";
static const char max[] = "mnm.op.max";
static const char max_pool2d[] = "mnm.op.max_pool2d";
static const char max_pool2d_dx[] = "mnm.op.max_pool2d_dx";
static const char maximum[] = "mnm.op.maximum";
static const char mean[] = "mnm.op.mean";
static const char mean_dx[] = "mnm.op.mean_dx";
static const char mesh_grid[] = "mnm.op.mesh_grid";
static const char min[] = "mnm.op.min";
static const char minimum[] = "mnm.op.minimum";
static const char mod[] = "mnm.op.mod";
static const char multiply[] = "mnm.op.multiply";
static const char ndarray_size[] = "mnm.op.ndarray_size";
static const char negative[] = "mnm.op.negative";
static const char nll_loss[] = "mnm.op.nll_loss";
static const char nll_loss_dpred[] = "mnm.op.nll_loss_dpred";
static const char nll_loss_dtrue[] = "mnm.op.nll_loss_dtrue";
static const char non_max_suppression[] = "mnm.op.non_max_suppression";
static const char not_equal[] = "mnm.op.not_equal";
static const char one_hot[] = "mnm.op.one_hot";
static const char ones[] = "mnm.op.ones";
static const char ones_like[] = "mnm.op.ones_like";
static const char pad[] = "mnm.op.pad";
static const char power[] = "mnm.op.power";
static const char prod[] = "mnm.op.prod";
static const char prod_dx[] = "mnm.op.prod_dx";
static const char relu[] = "mnm.op.relu";
static const char relu_dx[] = "mnm.op.relu_dx";
static const char repeat[] = "mnm.op.repeat";
static const char repeat_dx[] = "mnm.op.repeat_dx";
static const char reshape[] = "mnm.op.reshape";
static const char resize2d[] = "mnm.op.resize2d";
static const char resize2d_dx[] = "mnm.op.resize2d_dx";
static const char reverse[] = "mnm.op.reverse";
static const char reverse_sequence[] = "mnm.op.reverse_sequence";
static const char right_shift[] = "mnm.op.right_shift";
static const char roi_align[] = "mnm.op.roi_align";
static const char roi_align_dx[] = "mnm.op.roi_align_dx";
static const char round[] = "mnm.op.round";
static const char rsqrt[] = "mnm.op.rsqrt";
static const char scatter[] = "mnm.op.scatter";
static const char scatter_dx[] = "mnm.op.scatter_dx";
static const char sequence_mask[] = "mnm.op.sequence_mask";
static const char set_stream[] = "mnm.op.set_stream";
static const char sgd[] = "mnm.op.sgd";
static const char shape[] = "mnm.op.shape";
static const char sigmoid[] = "mnm.op.sigmoid";
static const char sigmoid_dx[] = "mnm.op.sigmoid_dx";
static const char sign[] = "mnm.op.sign";
static const char sin[] = "mnm.op.sin";
static const char smooth_l1_loss[] = "mnm.op.smooth_l1_loss";
static const char smooth_l1_loss_dpred[] = "mnm.op.smooth_l1_loss_dpred";
static const char smooth_l1_loss_dtrue[] = "mnm.op.smooth_l1_loss_dtrue";
static const char softmax[] = "mnm.op.softmax";
static const char softmax_dx[] = "mnm.op.softmax_dx";
static const char sort[] = "mnm.op.sort";
static const char split[] = "mnm.op.split";
static const char sqrt[] = "mnm.op.sqrt";
static const char sqrt_dx[] = "mnm.op.sqrt_dx";
static const char squeeze[] = "mnm.op.squeeze";
static const char stack[] = "mnm.op.stack";
static const char stream_barrier[] = "mnm.op.stream_barrier";
static const char stream_sync[] = "mnm.op.stream_sync";
static const char strided_slice[] = "mnm.op.strided_slice";
static const char strided_slice_dx[] = "mnm.op.strided_slice_dx";
static const char subtract[] = "mnm.op.subtract";
static const char sum[] = "mnm.op.sum";
static const char sum_dx[] = "mnm.op.sum_dx";
static const char swap_axis[] = "mnm.op.swap_axis";
static const char take[] = "mnm.op.take";
static const char take_dx[] = "mnm.op.take_dx";
static const char tanh[] = "mnm.op.tanh";
static const char tanh_dx[] = "mnm.op.tanh_dx";
static const char threefry_generate[] = "mnm.op.threefry_generate";
static const char threefry_split[] = "mnm.op.threefry_split";
static const char threshold[] = "mnm.op.threshold";
static const char threshold_dx[] = "mnm.op.threshold_dx";
static const char topk[] = "mnm.op.topk";
static const char transpose[] = "mnm.op.transpose";
static const char transpose_dx[] = "mnm.op.transpose_dx";
static const char trunc[] = "mnm.op.trunc";
static const char upper_bound_argwhere[] = "mnm.op.upper_bound.argwhere";
static const char vm_alloc_storage[] = "mnm.op.vm.alloc_storage";
static const char vm_alloc_tensor[] = "mnm.op.vm.alloc_tensor";
static const char vm_free[] = "mnm.op.vm.free";
static const char vm_infer_type[] = "mnm.op.vm.infer_type";
static const char vm_invoke_op[] = "mnm.op.vm.invoke_op";
static const char vm_set_shape[] = "mnm.op.vm.set_shape";
static const char wait_event[] = "mnm.op.wait_event";
static const char where[] = "mnm.op.where";
static const char zeros[] = "mnm.op.zeros";
static const char zeros_like[] = "mnm.op.zeros_like";
}  // namespace names
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 1.1. FFI to schema (for each schema)
namespace mnm {
namespace op {
namespace regs {
namespace ffi2schema {

#define MNM_TAPE(i, norm, name)               \
  try {                                       \
    attrs->name = norm(values[i], tapes + i); \
  } catch (const dmlc::Error& e) {            \
    FillError(e, "{arg}", #name);             \
  }

#define MNM_POD(i, norm, name)     \
  try {                            \
    attrs->name = norm(values[i]); \
  } catch (const dmlc::Error& e) { \
    FillError(e, "{arg}", #name);  \
  }

#define MNM_PRELUDE(obj, n)                                                                \
  const int size = values.size();                                                          \
  CHECK_EQ(size, n) << "TypeError: Mismatched number of arguments for operator \"{op}\": " \
                    << "Expected " << n << ", but get " << size;                           \
  auto attrs = make_object<obj>();

Attrs AdaptivePool(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AdaptivePoolArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  MNM_POD(2, ffi2schema::String, layout);
  return Attrs(attrs);
}

Attrs AdaptivePoolDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AdaptivePoolDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

Attrs AdvIndex(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AdvIndexArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, inputs);
  return Attrs(attrs);
}

Attrs AdvIndexDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AdvIndexDxArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_POD(1, ffi2schema::TupleTensor, inputs);
  return Attrs(attrs);
}

Attrs Allgather(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AllgatherArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs AllocStorage(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AllocStorageArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, size);
  MNM_TAPE(1, ffi2schema::ArrayLike, alignment);
  MNM_POD(2, ffi2schema::Int, device_type);
  MNM_POD(3, ffi2schema::Int, device_id);
  MNM_POD(4, ffi2schema::String, dtype);
  return Attrs(attrs);
}

Attrs AllocTensor(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AllocTensorArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, storage);
  MNM_TAPE(1, ffi2schema::ArrayLike, shape);
  MNM_POD(2, ffi2schema::String, dtype);
  MNM_POD(3, ffi2schema::IntOrTupleInt, assert_shape);
  MNM_POD(4, ffi2schema::Bool, own);
  return Attrs(attrs);
}

Attrs Allreduce(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::AllreduceArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  MNM_POD(1, ffi2schema::String, computation);
  return Attrs(attrs);
}

Attrs Arange(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ArangeArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, start);
  MNM_TAPE(1, ffi2schema::Tensor, stop);
  MNM_TAPE(2, ffi2schema::Tensor, step);
  MNM_POD(3, ffi2schema::String, dtype);
  MNM_POD(4, ffi2schema::String, device);
  return Attrs(attrs);
}

Attrs Argsort(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ArgsortArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Int, axis);
  MNM_POD(2, ffi2schema::Bool, is_ascend);
  MNM_POD(3, ffi2schema::String, dtype);
  return Attrs(attrs);
}

Attrs Argwhere(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ArgwhereArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, condition);
  return Attrs(attrs);
}

Attrs BatchNorm(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BatchNormArgs, 7);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, running_mean);
  MNM_TAPE(2, ffi2schema::Tensor, running_var);
  MNM_TAPE(3, ffi2schema::Tensor, w);
  MNM_TAPE(4, ffi2schema::Tensor, b);
  MNM_POD(5, ffi2schema::Double, momentum);
  MNM_POD(6, ffi2schema::Double, eps);
  return Attrs(attrs);
}

Attrs BatchNormTrainDxwb(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BatchNormTrainDxwbArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_TAPE(1, ffi2schema::Tensor, x);
  MNM_TAPE(2, ffi2schema::Tensor, w);
  MNM_TAPE(3, ffi2schema::Tensor, b);
  MNM_POD(4, ffi2schema::Double, eps);
  return Attrs(attrs);
}

Attrs BiasAdd(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BiasAddArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, bias);
  MNM_POD(2, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs Binary(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BinaryArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x1);
  MNM_TAPE(1, ffi2schema::ArrayLike, x2);
  return Attrs(attrs);
}

Attrs BinaryDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BinaryDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x1);
  MNM_TAPE(1, ffi2schema::ArrayLike, x2);
  MNM_TAPE(2, ffi2schema::Tensor, y);
  MNM_TAPE(3, ffi2schema::Tensor, dy);
  return Attrs(attrs);
}

Attrs BinaryUfunc(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BinaryUfuncArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x1);
  MNM_TAPE(1, ffi2schema::ArrayLike, x2);
  MNM_TAPE(2, ffi2schema::ArrayLike, out);
  MNM_TAPE(3, ffi2schema::ArrayLike, where);
  return Attrs(attrs);
}

Attrs Broadcast(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BroadcastArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  MNM_POD(1, ffi2schema::Int, root);
  return Attrs(attrs);
}

Attrs BroadcastTo(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BroadcastToArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

Attrs BroadcastToLike(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::BroadcastToLikeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, broadcast_type);
  return Attrs(attrs);
}

Attrs Cast(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::CastArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::String, dtype);
  return Attrs(attrs);
}

Attrs CastLike(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::CastLikeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, dtype_like);
  return Attrs(attrs);
}

Attrs Clip(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ClipArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Double, a_min);
  MNM_POD(2, ffi2schema::Double, a_max);
  return Attrs(attrs);
}

Attrs ClipDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ClipDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_POD(2, ffi2schema::Double, a_min);
  MNM_POD(3, ffi2schema::Double, a_max);
  return Attrs(attrs);
}

Attrs CollapseLike(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::CollapseLikeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

Attrs CommReduce(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::CommReduceArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  MNM_POD(1, ffi2schema::Int, root);
  MNM_POD(2, ffi2schema::String, computation);
  return Attrs(attrs);
}

Attrs Concatenate(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ConcatenateArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs Conv(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ConvArgs, 9);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, w);
  MNM_POD(2, ffi2schema::IntOrTupleInt, stride);
  MNM_POD(3, ffi2schema::IntOrTupleInt, padding);
  MNM_POD(4, ffi2schema::IntOrTupleInt, dilation);
  MNM_POD(5, ffi2schema::Int, groups);
  MNM_POD(6, ffi2schema::String, layout);
  MNM_POD(7, ffi2schema::String, kernel_layout);
  MNM_POD(8, ffi2schema::String, out_layout);
  return Attrs(attrs);
}

Attrs ConvDxw(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ConvDxwArgs, 8);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x_or_w);
  MNM_TAPE(1, ffi2schema::OptionalTensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::IntArray, shape);
  MNM_POD(4, ffi2schema::IntOrTupleInt, stride);
  MNM_POD(5, ffi2schema::IntOrTupleInt, padding);
  MNM_POD(6, ffi2schema::IntOrTupleInt, dilation);
  MNM_POD(7, ffi2schema::Int, groups);
  return Attrs(attrs);
}

Attrs ConvTrans(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ConvTransArgs, 10);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, w);
  MNM_POD(2, ffi2schema::IntOrTupleInt, stride);
  MNM_POD(3, ffi2schema::IntOrTupleInt, padding);
  MNM_POD(4, ffi2schema::IntOrTupleInt, output_padding);
  MNM_POD(5, ffi2schema::IntOrTupleInt, dilation);
  MNM_POD(6, ffi2schema::Int, groups);
  MNM_POD(7, ffi2schema::String, layout);
  MNM_POD(8, ffi2schema::String, kernel_layout);
  MNM_POD(9, ffi2schema::String, out_layout);
  return Attrs(attrs);
}

Attrs ConvTransposeDxw(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ConvTransposeDxwArgs, 9);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x_or_w);
  MNM_TAPE(1, ffi2schema::OptionalTensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::IntArray, shape);
  MNM_POD(4, ffi2schema::IntOrTupleInt, stride);
  MNM_POD(5, ffi2schema::IntOrTupleInt, padding);
  MNM_POD(6, ffi2schema::IntOrTupleInt, output_padding);
  MNM_POD(7, ffi2schema::IntOrTupleInt, dilation);
  MNM_POD(8, ffi2schema::Int, groups);
  return Attrs(attrs);
}

Attrs Cumsum(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::CumsumArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  MNM_POD(2, ffi2schema::String, dtype);
  MNM_POD(3, ffi2schema::Bool, exclusive);
  return Attrs(attrs);
}

Attrs DeviceCopy(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::DeviceCopyArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Int, src_dev_type);
  MNM_POD(2, ffi2schema::Int, dst_dev_type);
  return Attrs(attrs);
}

Attrs Dropout(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::DropoutArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Double, p);
  MNM_TAPE(2, ffi2schema::OptionalTensor, in_states);
  return Attrs(attrs);
}

Attrs DropoutDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::DropoutDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_TAPE(1, ffi2schema::Tensor, mask);
  MNM_TAPE(2, ffi2schema::Tensor, reserve_space);
  MNM_POD(3, ffi2schema::Double, p);
  return Attrs(attrs);
}

Attrs Embedding(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::EmbeddingArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, indices);
  return Attrs(attrs);
}

Attrs EmbeddingDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::EmbeddingDxArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_TAPE(1, ffi2schema::Tensor, indices);
  MNM_POD(2, ffi2schema::IntOrTupleInt, num_weight);
  return Attrs(attrs);
}

Attrs Event(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::EventArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::Int, event_id);
  MNM_POD(1, ffi2schema::Int, stream_id);
  return Attrs(attrs);
}

Attrs ExpandDims(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ExpandDimsArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  MNM_POD(2, ffi2schema::Int, num_newaxis);
  return Attrs(attrs);
}

Attrs Free(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::FreeArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, memory);
  return Attrs(attrs);
}

Attrs Full(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::FullArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::Double, fill_value);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  MNM_POD(2, ffi2schema::String, dtype);
  MNM_POD(3, ffi2schema::String, device);
  return Attrs(attrs);
}

Attrs FullLike(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::FullLikeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Double, fill_value);
  return Attrs(attrs);
}

Attrs Gather(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::GatherArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Int, axis);
  MNM_TAPE(2, ffi2schema::Tensor, indices);
  return Attrs(attrs);
}

Attrs GatherDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::GatherDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Int, axis);
  MNM_TAPE(2, ffi2schema::Tensor, indices);
  MNM_TAPE(3, ffi2schema::Tensor, dy);
  return Attrs(attrs);
}

Attrs GatherNd(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::GatherNdArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, indices);
  return Attrs(attrs);
}

Attrs GatherNdDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::GatherNdDxArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, indices);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  return Attrs(attrs);
}

Attrs GetValidCounts(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::GetValidCountsArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, score_threshold);
  MNM_POD(2, ffi2schema::Int, id_index);
  MNM_POD(3, ffi2schema::Int, score_index);
  return Attrs(attrs);
}

Attrs InferType(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::InferTypeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, func);
  MNM_TAPE(1, ffi2schema::ArrayLike, inputs);
  return Attrs(attrs);
}

Attrs InitOp(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::InitOpArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::IntOrTupleInt, shape);
  MNM_POD(1, ffi2schema::String, dtype);
  MNM_POD(2, ffi2schema::String, device);
  return Attrs(attrs);
}

Attrs InvokeOp(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::InvokeOpArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, func);
  MNM_TAPE(1, ffi2schema::ArrayLike, inputs);
  MNM_TAPE(2, ffi2schema::ArrayLike, outputs);
  return Attrs(attrs);
}

Attrs LayerNorm(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::LayerNormArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::OptionalTensor, scale);
  MNM_TAPE(2, ffi2schema::OptionalTensor, bias);
  MNM_POD(3, ffi2schema::Int, axis);
  MNM_POD(4, ffi2schema::Double, eps);
  return Attrs(attrs);
}

Attrs LayerNormDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::LayerNormDxArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::OptionalTensor, scale);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::Int, axis);
  MNM_POD(4, ffi2schema::Double, eps);
  return Attrs(attrs);
}

Attrs LocalResponseNorm(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::LocalResponseNormArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, size);
  MNM_POD(2, ffi2schema::Double, alpha);
  MNM_POD(3, ffi2schema::Double, beta);
  MNM_POD(4, ffi2schema::Double, k);
  return Attrs(attrs);
}

Attrs Loss(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::LossArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, y_true);
  MNM_TAPE(1, ffi2schema::Tensor, y_pred);
  return Attrs(attrs);
}

Attrs LossDtp(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::LossDtpArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_TAPE(1, ffi2schema::Tensor, y_true);
  MNM_TAPE(2, ffi2schema::Tensor, y_pred);
  return Attrs(attrs);
}

Attrs MeanDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::MeanDxArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_POD(1, ffi2schema::IntOrTupleInt, axis);
  MNM_POD(2, ffi2schema::IntOrTupleInt, x_shape);
  MNM_POD(3, ffi2schema::Bool, keepdims);
  MNM_POD(4, ffi2schema::Bool, exclude);
  return Attrs(attrs);
}

Attrs MeshGrid(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::MeshGridArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  return Attrs(attrs);
}

Attrs NonMaxSuppression(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::NonMaxSuppressionArgs, 12);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, valid_count);
  MNM_TAPE(2, ffi2schema::Tensor, indices);
  MNM_TAPE(3, ffi2schema::Tensor, max_output_size);
  MNM_TAPE(4, ffi2schema::Tensor, iou_threshold);
  MNM_POD(5, ffi2schema::Bool, force_suppress);
  MNM_POD(6, ffi2schema::Int, top_k);
  MNM_POD(7, ffi2schema::Int, coord_start);
  MNM_POD(8, ffi2schema::Int, score_index);
  MNM_POD(9, ffi2schema::Int, id_index);
  MNM_POD(10, ffi2schema::Bool, return_indices);
  MNM_POD(11, ffi2schema::Bool, invalid_to_bottom);
  return Attrs(attrs);
}

Attrs OneHot(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::OneHotArgs, 7);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, indices);
  MNM_TAPE(1, ffi2schema::Tensor, on_value);
  MNM_TAPE(2, ffi2schema::Tensor, off_value);
  MNM_POD(3, ffi2schema::Int, depth);
  MNM_POD(4, ffi2schema::Int, axis);
  MNM_POD(5, ffi2schema::String, dtype);
  MNM_POD(6, ffi2schema::String, device);
  return Attrs(attrs);
}

Attrs Pad(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::PadArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, pad_width);
  MNM_POD(2, ffi2schema::Double, pad_value);
  MNM_POD(3, ffi2schema::String, pad_mode);
  return Attrs(attrs);
}

Attrs Pool(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::PoolArgs, 8);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, kernel);
  MNM_POD(2, ffi2schema::IntOrTupleInt, stride);
  MNM_POD(3, ffi2schema::IntOrTupleInt, padding);
  MNM_POD(4, ffi2schema::IntOrTupleInt, dilation);
  MNM_POD(5, ffi2schema::Bool, ceil_mode);
  MNM_POD(6, ffi2schema::Bool, include_pad);
  MNM_POD(7, ffi2schema::String, layout);
  return Attrs(attrs);
}

Attrs PoolDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::PoolDxArgs, 9);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::IntOrTupleInt, kernel);
  MNM_POD(4, ffi2schema::IntOrTupleInt, stride);
  MNM_POD(5, ffi2schema::IntOrTupleInt, padding);
  MNM_POD(6, ffi2schema::IntOrTupleInt, dilation);
  MNM_POD(7, ffi2schema::Bool, ceil_mode);
  MNM_POD(8, ffi2schema::Bool, include_pad);
  return Attrs(attrs);
}

Attrs ProdDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ProdDxArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_POD(2, ffi2schema::IntOrTupleInt, axis);
  MNM_POD(3, ffi2schema::Bool, keepdims);
  MNM_POD(4, ffi2schema::Bool, exclude);
  return Attrs(attrs);
}

Attrs Recv(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::RecvArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::Int, peer);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  MNM_POD(2, ffi2schema::String, dtype);
  MNM_TAPE(3, ffi2schema::OptionalTensor, token);
  return Attrs(attrs);
}

Attrs Reduce(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ReduceArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, axis);
  MNM_POD(2, ffi2schema::Bool, keepdims);
  MNM_POD(3, ffi2schema::Bool, exclude);
  return Attrs(attrs);
}

Attrs ReduceScatter(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ReduceScatterArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  return Attrs(attrs);
}

Attrs Repeat(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::RepeatArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, repeats);
  MNM_TAPE(2, ffi2schema::ArrayLike, axis);
  return Attrs(attrs);
}

Attrs RepeatDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::RepeatDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_POD(2, ffi2schema::Int, repeats);
  MNM_TAPE(3, ffi2schema::ArrayLike, axis);
  return Attrs(attrs);
}

Attrs Reshape(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ReshapeArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  MNM_POD(2, ffi2schema::Bool, reverse);
  return Attrs(attrs);
}

Attrs Resize2D(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::Resize2DArgs, 9);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, size);
  MNM_POD(2, ffi2schema::String, layout);
  MNM_POD(3, ffi2schema::String, method);
  MNM_POD(4, ffi2schema::String, coordinate_transformation_mode);
  MNM_POD(5, ffi2schema::String, rounding_method);
  MNM_POD(6, ffi2schema::Double, cubic_alpha);
  MNM_POD(7, ffi2schema::Int, cubic_exclude);
  MNM_POD(8, ffi2schema::String, out_dtype);
  return Attrs(attrs);
}

Attrs Resize2DDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::Resize2DDxArgs, 10);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_POD(2, ffi2schema::IntOrTupleInt, size);
  MNM_POD(3, ffi2schema::String, layout);
  MNM_POD(4, ffi2schema::String, method);
  MNM_POD(5, ffi2schema::String, coordinate_transformation_mode);
  MNM_POD(6, ffi2schema::String, rounding_method);
  MNM_POD(7, ffi2schema::Double, cubic_alpha);
  MNM_POD(8, ffi2schema::Int, cubic_exclude);
  MNM_POD(9, ffi2schema::String, out_dtype);
  return Attrs(attrs);
}

Attrs Reverse(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ReverseArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs ReverseSequence(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ReverseSequenceArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, sequence_length);
  MNM_POD(2, ffi2schema::Int, seq_axis);
  MNM_POD(3, ffi2schema::Int, batch_axis);
  return Attrs(attrs);
}

Attrs RoiAlign(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::RoiAlignArgs, 7);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, rois);
  MNM_POD(2, ffi2schema::IntOrTupleInt, pooled_size);
  MNM_POD(3, ffi2schema::Double, spatial_scale);
  MNM_POD(4, ffi2schema::Int, sample_ratio);
  MNM_POD(5, ffi2schema::String, layout);
  MNM_POD(6, ffi2schema::String, mode);
  return Attrs(attrs);
}

Attrs RoiAlignDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::RoiAlignDxArgs, 8);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::Tensor, rois);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::IntOrTupleInt, pooled_size);
  MNM_POD(4, ffi2schema::Double, spatial_scale);
  MNM_POD(5, ffi2schema::Int, sample_ratio);
  MNM_POD(6, ffi2schema::String, layout);
  MNM_POD(7, ffi2schema::String, mode);
  return Attrs(attrs);
}

Attrs Scatter(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ScatterArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, index);
  MNM_TAPE(2, ffi2schema::Tensor, src);
  MNM_TAPE(3, ffi2schema::ArrayLike, axis);
  return Attrs(attrs);
}

Attrs ScatterDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ScatterDxArgs, 6);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_TAPE(3, ffi2schema::Tensor, index);
  MNM_TAPE(4, ffi2schema::Tensor, src);
  MNM_TAPE(5, ffi2schema::ArrayLike, axis);
  return Attrs(attrs);
}

Attrs Send(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SendArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, peer);
  MNM_TAPE(2, ffi2schema::OptionalTensor, token);
  return Attrs(attrs);
}

Attrs SequenceMask(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SequenceMaskArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, sequence_length);
  MNM_POD(2, ffi2schema::Double, mask_value);
  MNM_POD(3, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs SetShape(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SetShapeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_TAPE(1, ffi2schema::ArrayLike, shape);
  return Attrs(attrs);
}

Attrs SetStream(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SetStreamArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::Int, device_id);
  MNM_POD(1, ffi2schema::Int, stream_id);
  return Attrs(attrs);
}

Attrs Sgd(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SgdArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dx);
  MNM_TAPE(2, ffi2schema::Tensor, v);
  MNM_POD(3, ffi2schema::Double, learning_rate);
  MNM_POD(4, ffi2schema::Double, mu);
  return Attrs(attrs);
}

Attrs Softmax(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SoftmaxArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs SoftmaxDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SoftmaxDxArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  MNM_POD(3, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs Sort(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SortArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Int, axis);
  MNM_POD(2, ffi2schema::Bool, is_ascend);
  return Attrs(attrs);
}

Attrs Split(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SplitArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::ArrayLike, indices_or_sections);
  MNM_POD(2, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs Squeeze(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SqueezeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, axis);
  return Attrs(attrs);
}

Attrs Stack(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::StackArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_POD(0, ffi2schema::TupleTensor, x);
  MNM_POD(1, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Attrs Stream(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::StreamArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, stream_tag);
  return Attrs(attrs);
}

Attrs StreamBarrier(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::StreamBarrierArgs, 0);  // NOLINT(whitespace/line_length)

  return Attrs(attrs);
}

Attrs StridedSlice(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::StridedSliceArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, begin);
  MNM_POD(2, ffi2schema::IntOrTupleInt, end);
  MNM_POD(3, ffi2schema::IntOrTupleInt, strides);
  MNM_POD(4, ffi2schema::String, slice_mode);
  return Attrs(attrs);
}

Attrs StridedSliceDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::StridedSliceDxArgs, 6);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_POD(1, ffi2schema::IntOrTupleInt, primal_shape);
  MNM_POD(2, ffi2schema::IntOrTupleInt, begin);
  MNM_POD(3, ffi2schema::IntOrTupleInt, end);
  MNM_POD(4, ffi2schema::IntOrTupleInt, strides);
  MNM_POD(5, ffi2schema::String, slice_mode);
  return Attrs(attrs);
}

Attrs Sum(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SumArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, axis);
  MNM_POD(2, ffi2schema::IntOrTupleInt, keepdims);
  MNM_POD(3, ffi2schema::Bool, exclude);
  return Attrs(attrs);
}

Attrs SumDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SumDxArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_POD(2, ffi2schema::IntOrTupleInt, axis);
  MNM_POD(3, ffi2schema::IntOrTupleInt, keepdims);
  MNM_POD(4, ffi2schema::Bool, exclude);
  return Attrs(attrs);
}

Attrs SwapAxis(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::SwapAxisArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::Int, axis1);
  MNM_POD(2, ffi2schema::Int, axis2);
  return Attrs(attrs);
}

Attrs Take(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TakeArgs, 4);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, indices);
  MNM_TAPE(2, ffi2schema::ArrayLike, axis);
  MNM_POD(3, ffi2schema::String, mode);
  return Attrs(attrs);
}

Attrs TakeDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TakeDxArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_TAPE(2, ffi2schema::Tensor, indices);
  MNM_TAPE(3, ffi2schema::ArrayLike, axis);
  MNM_POD(4, ffi2schema::String, mode);
  return Attrs(attrs);
}

Attrs Ternary(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TernaryArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x1);
  MNM_TAPE(1, ffi2schema::ArrayLike, x2);
  MNM_TAPE(2, ffi2schema::ArrayLike, x3);
  return Attrs(attrs);
}

Attrs TernaryDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TernaryDxArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x1);
  MNM_TAPE(1, ffi2schema::ArrayLike, x2);
  MNM_TAPE(2, ffi2schema::ArrayLike, x3);
  MNM_TAPE(3, ffi2schema::Tensor, y);
  MNM_TAPE(4, ffi2schema::Tensor, dy);
  return Attrs(attrs);
}

Attrs TernaryUfunc(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TernaryUfuncArgs, 5);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x1);
  MNM_TAPE(1, ffi2schema::ArrayLike, x2);
  MNM_TAPE(2, ffi2schema::ArrayLike, x3);
  MNM_TAPE(3, ffi2schema::ArrayLike, out);
  MNM_TAPE(4, ffi2schema::ArrayLike, where);
  return Attrs(attrs);
}

Attrs ThreefryGenerate(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ThreefryGenerateArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, key);
  MNM_POD(1, ffi2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

Attrs ThreefrySplit(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ThreefrySplitArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, key);
  return Attrs(attrs);
}

Attrs Threshold(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ThresholdArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x);
  MNM_POD(1, ffi2schema::Double, threshold);
  MNM_POD(2, ffi2schema::Double, value);
  return Attrs(attrs);
}

Attrs ThresholdDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::ThresholdDxArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x);
  MNM_TAPE(1, ffi2schema::Tensor, dy);
  MNM_POD(2, ffi2schema::Double, threshold);
  return Attrs(attrs);
}

Attrs Topk(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TopkArgs, 6);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, data);
  MNM_POD(1, ffi2schema::Int, k);
  MNM_POD(2, ffi2schema::Int, axis);
  MNM_POD(3, ffi2schema::String, ret_type);
  MNM_POD(4, ffi2schema::Bool, is_ascend);
  MNM_POD(5, ffi2schema::String, dtype);
  return Attrs(attrs);
}

Attrs Transpose(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TransposeArgs, 2);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, x);
  MNM_POD(1, ffi2schema::IntOrTupleInt, axes);
  return Attrs(attrs);
}

Attrs TransposeDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::TransposeDxArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, dy);
  MNM_POD(1, ffi2schema::IntOrTupleInt, axes);
  MNM_POD(2, ffi2schema::IntOrTupleInt, primal_shape);
  return Attrs(attrs);
}

Attrs Unary(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::UnaryArgs, 1);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x);
  return Attrs(attrs);
}

Attrs UnaryDx(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::UnaryDxArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::OptionalArrayLike, x);
  MNM_TAPE(1, ffi2schema::OptionalTensor, y);
  MNM_TAPE(2, ffi2schema::Tensor, dy);
  return Attrs(attrs);
}

Attrs UnaryUfunc(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::UnaryUfuncArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::ArrayLike, x);
  MNM_TAPE(1, ffi2schema::ArrayLike, out);
  MNM_TAPE(2, ffi2schema::ArrayLike, where);
  return Attrs(attrs);
}

Attrs Where(const TVMArgs& values, GradTape* tapes) {
  MNM_PRELUDE(schema::WhereArgs, 3);  // NOLINT(whitespace/line_length)
  MNM_TAPE(0, ffi2schema::Tensor, condition);
  MNM_TAPE(1, ffi2schema::Tensor, x);
  MNM_TAPE(2, ffi2schema::Tensor, y);
  return Attrs(attrs);
}

#undef MNM_PRELUDE
#undef MNM_POD
#undef MNM_TAPE
}  // namespace ffi2schema
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 1.2. Imperative API, uses "Part 1.1. FFI to schema"
namespace mnm {
namespace op {
namespace regs {
namespace imperative {

#define MNM_PRELUDE(op, n_args, func, obj)                                   \
  const auto* opack = OpPack<names::op, n_args>::Get();                      \
  const auto* vpack = VarPack::Get();                                        \
  std::array<GradTape, n_args> prev_tapes;                                   \
  std::vector<Expr> grads(opack->grads.begin(), opack->grads.end());         \
  Attrs _schema;                                                             \
  try {                                                                      \
    _schema = func(args, prev_tapes.data());                                 \
  } catch (const dmlc::Error& e) {                                           \
    FillError(e, "{op}", names::op);                                         \
  }                                                                          \
  Value value = InvokePrimitive(CallValues::make(opack->opv, _schema));      \
  int n_tapes = grads.size();                                                \
  bool full_grads = RemoveNoGrad(prev_tapes.data(), grads.data(), &n_tapes); \
  /* case 1: no grad required */                                             \
  if (n_tapes == 0) {                                                        \
    *ret = DeTuple(value);                                                   \
    return;                                                                  \
  }                                                                          \
  Expr body = Tuple({grads.begin(), grads.begin() + n_tapes});               \
  std::vector<const ExprNode*> used_vars;                                    \
  if (full_grads) {                                                          \
    /* case 2: full grad required, use pre-computed results */               \
    used_vars = opack->grad_used_vars;                                       \
  } else {                                                                   \
    /* case 3: partial grad required, have to collect vars */                \
    CollectVars(body, &used_vars);                                           \
  }                                                                          \
  const auto* schema = _schema.as<obj>();                                    \
  Map<Var, Value> env;

#define MNM_SET_ENV(var, value)                                                    \
  {                                                                                \
    const auto& _v = (var);                                                        \
    if (std::binary_search(used_vars.begin(), used_vars.end(), _v.operator->())) { \
      env.Set(_v, value);                                                          \
    }                                                                              \
  }

#define MNM_RET()                                                                               \
  DeStruct(                                                                                     \
      std::move(value),                                                                         \
      ClosureValue::make(/*env=*/std::move(env), /*func=*/Function({vpack->dy}, body, {}, {})), \
      {prev_tapes.begin(), prev_tapes.begin() + n_tapes});

MNM_REGISTER_GLOBAL("mnm.op.imp._allgather").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_allgather, 2, ffi2schema::Allgather,
              schema::AllgatherArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._allreduce").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_allreduce, 2, ffi2schema::Allreduce,
              schema::AllreduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::String(schema->computation));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._broadcast").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_broadcast, 2, ffi2schema::Broadcast,
              schema::BroadcastArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->root));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._contrib_dropout").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_contrib_dropout, 3, ffi2schema::Dropout,
              schema::DropoutArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Double(schema->p));
  MNM_SET_ENV(vpack->x[2], schema2value::OptionalTensor(schema->in_states));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._contrib_dropout_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_contrib_dropout_dx, 4, ffi2schema::DropoutDx,
              schema::DropoutDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->mask));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->reserve_space));
  MNM_SET_ENV(vpack->x[3], schema2value::Double(schema->p));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._recv").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_recv, 4, ffi2schema::Recv, schema::RecvArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Int(schema->peer));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[3], schema2value::OptionalTensor(schema->token));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._reduce").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_reduce, 3, ffi2schema::CommReduce,
              schema::CommReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->root));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->computation));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._reduce_scatter").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_reduce_scatter, 1, ffi2schema::ReduceScatter,
              schema::ReduceScatterArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp._send").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(_send, 3, ffi2schema::Send, schema::SendArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->peer));
  MNM_SET_ENV(vpack->x[2], schema2value::OptionalTensor(schema->token));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.abs").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(abs, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.adaptive_avg_pool2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(adaptive_avg_pool2d, 3, ffi2schema::AdaptivePool,
              schema::AdaptivePoolArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.adaptive_avg_pool2d_dx")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      MNM_PRELUDE(adaptive_avg_pool2d_dx, 4, ffi2schema::AdaptivePoolDx,
                  schema::AdaptivePoolDxArgs);  // NOLINT(whitespace/line_length)
      MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
      MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
      MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
      MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->shape));
      MNM_SET_ENV(vpack->y, value);
      *ret = MNM_RET();
    });

MNM_REGISTER_GLOBAL("mnm.op.imp.adaptive_max_pool2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(adaptive_max_pool2d, 3, ffi2schema::AdaptivePool,
              schema::AdaptivePoolArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.adaptive_max_pool2d_dx")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      MNM_PRELUDE(adaptive_max_pool2d_dx, 4, ffi2schema::AdaptivePoolDx,
                  schema::AdaptivePoolDxArgs);  // NOLINT(whitespace/line_length)
      MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
      MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
      MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
      MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->shape));
      MNM_SET_ENV(vpack->y, value);
      *ret = MNM_RET();
    });

MNM_REGISTER_GLOBAL("mnm.op.imp.add").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(add, 4, ffi2schema::BinaryUfunc,
              schema::BinaryUfuncArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->x[2], schema2value::ArrayLike(schema->out));
  MNM_SET_ENV(vpack->x[3], schema2value::ArrayLike(schema->where));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.add_event").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(add_event, 2, ffi2schema::Event,
              schema::EventArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Int(schema->event_id));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->stream_id));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.adv_index").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(adv_index, 1, ffi2schema::AdvIndex,
              schema::AdvIndexArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->inputs));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.adv_index_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(adv_index_dx, 2, ffi2schema::AdvIndexDx,
              schema::AdvIndexDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::TupleTensor(schema->inputs));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.all").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(all, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.any").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(any, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.arange").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(arange, 5, ffi2schema::Arange, schema::ArangeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->start));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->stop));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->step));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[4], schema2value::String(schema->device));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.argmax").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(argmax, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.argmin").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(argmin, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.argsort").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(argsort, 4, ffi2schema::Argsort,
              schema::ArgsortArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->is_ascend));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.argwhere").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(argwhere, 1, ffi2schema::Argwhere,
              schema::ArgwhereArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->condition));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.atan").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(atan, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.avg_pool2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(avg_pool2d, 8, ffi2schema::Pool, schema::PoolArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->kernel));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[5], schema2value::Bool(schema->ceil_mode));
  MNM_SET_ENV(vpack->x[6], schema2value::Bool(schema->include_pad));
  MNM_SET_ENV(vpack->x[7], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.avg_pool2d_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(avg_pool2d_dx, 9, ffi2schema::PoolDx,
              schema::PoolDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->kernel));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[6], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[7], schema2value::Bool(schema->ceil_mode));
  MNM_SET_ENV(vpack->x[8], schema2value::Bool(schema->include_pad));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_flatten").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_flatten, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_matmul, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_matmul_nt").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_matmul_nt, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_matmul_tn").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_matmul_tn, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_matmul_tt").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_matmul_tt, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_norm_infer").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_norm_infer, 7, ffi2schema::BatchNorm,
              schema::BatchNormArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->running_mean));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->running_var));
  MNM_SET_ENV(vpack->x[3], schema2value::Tensor(schema->w));
  MNM_SET_ENV(vpack->x[4], schema2value::Tensor(schema->b));
  MNM_SET_ENV(vpack->x[5], schema2value::Double(schema->momentum));
  MNM_SET_ENV(vpack->x[6], schema2value::Double(schema->eps));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_norm_train").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(batch_norm_train, 7, ffi2schema::BatchNorm,
              schema::BatchNormArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->running_mean));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->running_var));
  MNM_SET_ENV(vpack->x[3], schema2value::Tensor(schema->w));
  MNM_SET_ENV(vpack->x[4], schema2value::Tensor(schema->b));
  MNM_SET_ENV(vpack->x[5], schema2value::Double(schema->momentum));
  MNM_SET_ENV(vpack->x[6], schema2value::Double(schema->eps));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.batch_norm_train_dxwb")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      MNM_PRELUDE(batch_norm_train_dxwb, 5, ffi2schema::BatchNormTrainDxwb,
                  schema::BatchNormTrainDxwbArgs);  // NOLINT(whitespace/line_length)
      MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
      MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->x));
      MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->w));
      MNM_SET_ENV(vpack->x[3], schema2value::Tensor(schema->b));
      MNM_SET_ENV(vpack->x[4], schema2value::Double(schema->eps));
      MNM_SET_ENV(vpack->y, value);
      *ret = MNM_RET();
    });

MNM_REGISTER_GLOBAL("mnm.op.imp.bias_add").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(bias_add, 3, ffi2schema::BiasAdd,
              schema::BiasAddArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->bias));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.broadcast_to").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(broadcast_to, 2, ffi2schema::BroadcastTo,
              schema::BroadcastToArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.broadcast_to_like").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(broadcast_to_like, 2, ffi2schema::BroadcastToLike,
              schema::BroadcastToLikeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->broadcast_type));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cast").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cast, 2, ffi2schema::Cast, schema::CastArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cast_like").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cast_like, 2, ffi2schema::CastLike,
              schema::CastLikeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dtype_like));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.ceil").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(ceil, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.clip").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(clip, 3, ffi2schema::Clip, schema::ClipArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Double(schema->a_min));
  MNM_SET_ENV(vpack->x[2], schema2value::Double(schema->a_max));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.clip_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(clip_dx, 4, ffi2schema::ClipDx,
              schema::ClipDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::Double(schema->a_min));
  MNM_SET_ENV(vpack->x[3], schema2value::Double(schema->a_max));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.collapse_sum_like").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(collapse_sum_like, 2, ffi2schema::CollapseLike,
              schema::CollapseLikeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.compiler_begin").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(compiler_begin, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.compiler_end").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(compiler_end, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.concatenate").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(concatenate, 2, ffi2schema::Concatenate,
              schema::ConcatenateArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.concatenate_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(concatenate_dx, 2, ffi2schema::Concatenate,
              schema::ConcatenateArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(conv2d, 9, ffi2schema::Conv, schema::ConvArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->w));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[5], schema2value::Int(schema->groups));
  MNM_SET_ENV(vpack->x[6], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->x[7], schema2value::String(schema->kernel_layout));
  MNM_SET_ENV(vpack->x[8], schema2value::String(schema->out_layout));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_dw").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(conv2d_dw, 8, ffi2schema::ConvDxw,
              schema::ConvDxwArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x_or_w));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntArray(schema->shape));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[6], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[7], schema2value::Int(schema->groups));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(conv2d_dx, 8, ffi2schema::ConvDxw,
              schema::ConvDxwArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x_or_w));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntArray(schema->shape));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[6], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[7], schema2value::Int(schema->groups));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_transpose").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(conv2d_transpose, 10, ffi2schema::ConvTrans,
              schema::ConvTransArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->w));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->output_padding));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[6], schema2value::Int(schema->groups));
  MNM_SET_ENV(vpack->x[7], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->x[8], schema2value::String(schema->kernel_layout));
  MNM_SET_ENV(vpack->x[9], schema2value::String(schema->out_layout));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_transpose_dw").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(conv2d_transpose_dw, 9, ffi2schema::ConvTransposeDxw,
              schema::ConvTransposeDxwArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x_or_w));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntArray(schema->shape));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[6], schema2value::IntOrTupleInt(schema->output_padding));
  MNM_SET_ENV(vpack->x[7], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[8], schema2value::Int(schema->groups));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_transpose_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(conv2d_transpose_dx, 9, ffi2schema::ConvTransposeDxw,
              schema::ConvTransposeDxwArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x_or_w));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntArray(schema->shape));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[6], schema2value::IntOrTupleInt(schema->output_padding));
  MNM_SET_ENV(vpack->x[7], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[8], schema2value::Int(schema->groups));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.copy").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(copy, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cos").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cos, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cross_entropy").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cross_entropy, 2, ffi2schema::Loss,
              schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cross_entropy_dpred").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cross_entropy_dpred, 2, ffi2schema::Loss,
              schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cross_entropy_dtrue").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cross_entropy_dtrue, 2, ffi2schema::Loss,
              schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.cumsum").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(cumsum, 4, ffi2schema::Cumsum, schema::CumsumArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclusive));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.dense").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(dense, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.device_copy").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(device_copy, 3, ffi2schema::DeviceCopy,
              schema::DeviceCopyArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->src_dev_type));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->dst_dev_type));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.divide").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(divide, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.embedding").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(embedding, 2, ffi2schema::Embedding,
              schema::EmbeddingArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.embedding_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(embedding_dx, 3, ffi2schema::EmbeddingDx,
              schema::EmbeddingDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->num_weight));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.equal").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(equal, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.erf").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(erf, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.erf_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(erf_dx, 3, ffi2schema::UnaryDx,
              schema::UnaryDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::OptionalArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.exp").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(exp, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.expand_dims").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(expand_dims, 3, ffi2schema::ExpandDims,
              schema::ExpandDimsArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->num_newaxis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.floor").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(floor, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.floor_divide").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(floor_divide, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.full").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(full, 4, ffi2schema::Full, schema::FullArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Double(schema->fill_value));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->device));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.full_like").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(full_like, 2, ffi2schema::FullLike,
              schema::FullLikeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Double(schema->fill_value));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.gather").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(gather, 3, ffi2schema::Gather, schema::GatherArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.gather_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(gather_dx, 4, ffi2schema::GatherDx,
              schema::GatherDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[3], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.gather_nd").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(gather_nd, 2, ffi2schema::GatherNd,
              schema::GatherNdArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.gather_nd_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(gather_nd_dx, 3, ffi2schema::GatherNdDx,
              schema::GatherNdDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.gelu").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(gelu, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.gelu_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(gelu_dx, 3, ffi2schema::UnaryDx,
              schema::UnaryDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::OptionalArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.get_kept_dims").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(get_kept_dims, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.get_reduce_axis").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(get_reduce_axis, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.get_valid_counts").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(get_valid_counts, 4, ffi2schema::GetValidCounts,
              schema::GetValidCountsArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->score_threshold));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->id_index));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->score_index));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.greater").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(greater, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.greater_equal").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(greater_equal, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.layer_norm").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(layer_norm, 5, ffi2schema::LayerNorm,
              schema::LayerNormArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->scale));
  MNM_SET_ENV(vpack->x[2], schema2value::OptionalTensor(schema->bias));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[4], schema2value::Double(schema->eps));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.layer_norm_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(layer_norm_dx, 5, ffi2schema::LayerNormDx,
              schema::LayerNormDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->scale));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[4], schema2value::Double(schema->eps));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.left_shift").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(left_shift, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.less").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(less, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.less_equal").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(less_equal, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.log").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(log, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.log2").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(log2, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.log_softmax").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(log_softmax, 2, ffi2schema::Softmax,
              schema::SoftmaxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.log_softmax_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(log_softmax_dx, 4, ffi2schema::SoftmaxDx,
              schema::SoftmaxDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.logical_and").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(logical_and, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.logical_not").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(logical_not, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(matmul, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.matmul_nt").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(matmul_nt, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.matmul_tn").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(matmul_tn, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.matmul_tt").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(matmul_tt, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.max").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(max, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.max_pool2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(max_pool2d, 8, ffi2schema::Pool, schema::PoolArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->kernel));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[5], schema2value::Bool(schema->ceil_mode));
  MNM_SET_ENV(vpack->x[6], schema2value::Bool(schema->include_pad));
  MNM_SET_ENV(vpack->x[7], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.max_pool2d_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(max_pool2d_dx, 9, ffi2schema::PoolDx,
              schema::PoolDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->kernel));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->stride));
  MNM_SET_ENV(vpack->x[5], schema2value::IntOrTupleInt(schema->padding));
  MNM_SET_ENV(vpack->x[6], schema2value::IntOrTupleInt(schema->dilation));
  MNM_SET_ENV(vpack->x[7], schema2value::Bool(schema->ceil_mode));
  MNM_SET_ENV(vpack->x[8], schema2value::Bool(schema->include_pad));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.maximum").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(maximum, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.mean").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(mean, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.mean_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(mean_dx, 5, ffi2schema::MeanDx,
              schema::MeanDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->x_shape));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[4], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.mesh_grid").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(mesh_grid, 1, ffi2schema::MeshGrid,
              schema::MeshGridArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.min").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(min, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.minimum").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(minimum, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.mod").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(mod, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.multiply").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(multiply, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.ndarray_size").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(ndarray_size, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.negative").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(negative, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.nll_loss").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(nll_loss, 2, ffi2schema::Loss, schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.nll_loss_dpred").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(nll_loss_dpred, 3, ffi2schema::LossDtp,
              schema::LossDtpArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.nll_loss_dtrue").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(nll_loss_dtrue, 3, ffi2schema::LossDtp,
              schema::LossDtpArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.non_max_suppression").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(non_max_suppression, 12, ffi2schema::NonMaxSuppression,
              schema::NonMaxSuppressionArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->valid_count));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[3], schema2value::Tensor(schema->max_output_size));
  MNM_SET_ENV(vpack->x[4], schema2value::Tensor(schema->iou_threshold));
  MNM_SET_ENV(vpack->x[5], schema2value::Bool(schema->force_suppress));
  MNM_SET_ENV(vpack->x[6], schema2value::Int(schema->top_k));
  MNM_SET_ENV(vpack->x[7], schema2value::Int(schema->coord_start));
  MNM_SET_ENV(vpack->x[8], schema2value::Int(schema->score_index));
  MNM_SET_ENV(vpack->x[9], schema2value::Int(schema->id_index));
  MNM_SET_ENV(vpack->x[10], schema2value::Bool(schema->return_indices));
  MNM_SET_ENV(vpack->x[11], schema2value::Bool(schema->invalid_to_bottom));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.not_equal").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(not_equal, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.one_hot").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(one_hot, 7, ffi2schema::OneHot,
              schema::OneHotArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->on_value));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->off_value));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->depth));
  MNM_SET_ENV(vpack->x[4], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[5], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[6], schema2value::String(schema->device));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.ones").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(ones, 3, ffi2schema::InitOp, schema::InitOpArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[1], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->device));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.ones_like").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(ones_like, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.pad").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(pad, 4, ffi2schema::Pad, schema::PadArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->pad_width));
  MNM_SET_ENV(vpack->x[2], schema2value::Double(schema->pad_value));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->pad_mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.power").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(power, 2, ffi2schema::Binary, schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.prod").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(prod, 4, ffi2schema::Reduce, schema::ReduceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.prod_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(prod_dx, 5, ffi2schema::ProdDx,
              schema::ProdDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->keepdims));
  MNM_SET_ENV(vpack->x[4], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.relu").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(relu, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.relu_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(relu_dx, 3, ffi2schema::UnaryDx,
              schema::UnaryDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::OptionalArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.repeat").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(repeat, 3, ffi2schema::Repeat, schema::RepeatArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->repeats));
  MNM_SET_ENV(vpack->x[2], schema2value::ArrayLike(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.repeat_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(repeat_dx, 4, ffi2schema::RepeatDx,
              schema::RepeatDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->repeats));
  MNM_SET_ENV(vpack->x[3], schema2value::ArrayLike(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.reshape").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(reshape, 3, ffi2schema::Reshape,
              schema::ReshapeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->reverse));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.resize2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(resize2d, 9, ffi2schema::Resize2D,
              schema::Resize2DArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->size));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->method));
  MNM_SET_ENV(vpack->x[4], schema2value::String(schema->coordinate_transformation_mode));
  MNM_SET_ENV(vpack->x[5], schema2value::String(schema->rounding_method));
  MNM_SET_ENV(vpack->x[6], schema2value::Double(schema->cubic_alpha));
  MNM_SET_ENV(vpack->x[7], schema2value::Int(schema->cubic_exclude));
  MNM_SET_ENV(vpack->x[8], schema2value::String(schema->out_dtype));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.resize2d_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(resize2d_dx, 10, ffi2schema::Resize2DDx,
              schema::Resize2DDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->size));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->x[4], schema2value::String(schema->method));
  MNM_SET_ENV(vpack->x[5], schema2value::String(schema->coordinate_transformation_mode));
  MNM_SET_ENV(vpack->x[6], schema2value::String(schema->rounding_method));
  MNM_SET_ENV(vpack->x[7], schema2value::Double(schema->cubic_alpha));
  MNM_SET_ENV(vpack->x[8], schema2value::Int(schema->cubic_exclude));
  MNM_SET_ENV(vpack->x[9], schema2value::String(schema->out_dtype));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.reverse").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(reverse, 2, ffi2schema::Reverse,
              schema::ReverseArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.reverse_sequence").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(reverse_sequence, 4, ffi2schema::ReverseSequence,
              schema::ReverseSequenceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->sequence_length));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->seq_axis));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->batch_axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.right_shift").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(right_shift, 2, ffi2schema::Binary,
              schema::BinaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.roi_align").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(roi_align, 7, ffi2schema::RoiAlign,
              schema::RoiAlignArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->rois));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->pooled_size));
  MNM_SET_ENV(vpack->x[3], schema2value::Double(schema->spatial_scale));
  MNM_SET_ENV(vpack->x[4], schema2value::Int(schema->sample_ratio));
  MNM_SET_ENV(vpack->x[5], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->x[6], schema2value::String(schema->mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.roi_align_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(roi_align_dx, 8, ffi2schema::RoiAlignDx,
              schema::RoiAlignDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->rois));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->pooled_size));
  MNM_SET_ENV(vpack->x[4], schema2value::Double(schema->spatial_scale));
  MNM_SET_ENV(vpack->x[5], schema2value::Int(schema->sample_ratio));
  MNM_SET_ENV(vpack->x[6], schema2value::String(schema->layout));
  MNM_SET_ENV(vpack->x[7], schema2value::String(schema->mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.round").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(round, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.rsqrt").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(rsqrt, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.scatter").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(scatter, 4, ffi2schema::Scatter,
              schema::ScatterArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->index));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->src));
  MNM_SET_ENV(vpack->x[3], schema2value::ArrayLike(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.scatter_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(scatter_dx, 6, ffi2schema::ScatterDx,
              schema::ScatterDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::Tensor(schema->index));
  MNM_SET_ENV(vpack->x[4], schema2value::Tensor(schema->src));
  MNM_SET_ENV(vpack->x[5], schema2value::ArrayLike(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sequence_mask").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sequence_mask, 4, ffi2schema::SequenceMask,
              schema::SequenceMaskArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->sequence_length));
  MNM_SET_ENV(vpack->x[2], schema2value::Double(schema->mask_value));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.set_stream").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(set_stream, 2, ffi2schema::SetStream,
              schema::SetStreamArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Int(schema->device_id));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->stream_id));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sgd").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sgd, 5, ffi2schema::Sgd, schema::SgdArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dx));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->v));
  MNM_SET_ENV(vpack->x[3], schema2value::Double(schema->learning_rate));
  MNM_SET_ENV(vpack->x[4], schema2value::Double(schema->mu));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.shape").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(shape, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sigmoid").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sigmoid, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sigmoid_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sigmoid_dx, 3, ffi2schema::UnaryDx,
              schema::UnaryDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::OptionalArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sign").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sign, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sin").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sin, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.smooth_l1_loss").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(smooth_l1_loss, 2, ffi2schema::Loss,
              schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.smooth_l1_loss_dpred").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(smooth_l1_loss_dpred, 2, ffi2schema::Loss,
              schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.smooth_l1_loss_dtrue").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(smooth_l1_loss_dtrue, 2, ffi2schema::Loss,
              schema::LossArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->y_true));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y_pred));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.softmax").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(softmax, 2, ffi2schema::Softmax,
              schema::SoftmaxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.softmax_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(softmax_dx, 4, ffi2schema::SoftmaxDx,
              schema::SoftmaxDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sort").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sort, 3, ffi2schema::Sort, schema::SortArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::Bool(schema->is_ascend));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.split").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(split, 3, ffi2schema::Split, schema::SplitArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->indices_or_sections));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sqrt").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sqrt, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sqrt_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sqrt_dx, 3, ffi2schema::UnaryDx,
              schema::UnaryDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::OptionalArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.squeeze").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(squeeze, 2, ffi2schema::Squeeze,
              schema::SqueezeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.stack").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(stack, 2, ffi2schema::Stack, schema::StackArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::TupleTensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.stream_barrier").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(stream_barrier, 0, ffi2schema::StreamBarrier,
              schema::StreamBarrierArgs);  // NOLINT(whitespace/line_length)

  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.stream_sync").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(stream_sync, 2, ffi2schema::Stream,
              schema::StreamArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->stream_tag));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.strided_slice").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(strided_slice, 5, ffi2schema::StridedSlice,
              schema::StridedSliceArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->begin));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->end));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->strides));
  MNM_SET_ENV(vpack->x[4], schema2value::String(schema->slice_mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.strided_slice_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(strided_slice_dx, 6, ffi2schema::StridedSliceDx,
              schema::StridedSliceDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->primal_shape));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->begin));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->end));
  MNM_SET_ENV(vpack->x[4], schema2value::IntOrTupleInt(schema->strides));
  MNM_SET_ENV(vpack->x[5], schema2value::String(schema->slice_mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.subtract").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(subtract, 4, ffi2schema::BinaryUfunc,
              schema::BinaryUfuncArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x1));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->x2));
  MNM_SET_ENV(vpack->x[2], schema2value::ArrayLike(schema->out));
  MNM_SET_ENV(vpack->x[3], schema2value::ArrayLike(schema->where));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sum").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sum, 4, ffi2schema::Sum, schema::SumArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->keepdims));
  MNM_SET_ENV(vpack->x[3], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.sum_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(sum_dx, 5, ffi2schema::SumDx, schema::SumDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->axis));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->keepdims));
  MNM_SET_ENV(vpack->x[4], schema2value::Bool(schema->exclude));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.swap_axis").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(swap_axis, 3, ffi2schema::SwapAxis,
              schema::SwapAxisArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->axis1));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->axis2));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.take").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(take, 4, ffi2schema::Take, schema::TakeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[2], schema2value::ArrayLike(schema->axis));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.take_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(take_dx, 5, ffi2schema::TakeDx,
              schema::TakeDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->indices));
  MNM_SET_ENV(vpack->x[3], schema2value::ArrayLike(schema->axis));
  MNM_SET_ENV(vpack->x[4], schema2value::String(schema->mode));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.tanh").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(tanh, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.tanh_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(tanh_dx, 3, ffi2schema::UnaryDx,
              schema::UnaryDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::OptionalArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::OptionalTensor(schema->y));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.threefry_generate").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(threefry_generate, 2, ffi2schema::ThreefryGenerate,
              schema::ThreefryGenerateArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->key));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.threefry_split").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(threefry_split, 1, ffi2schema::ThreefrySplit,
              schema::ThreefrySplitArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->key));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.threshold").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(threshold, 3, ffi2schema::Threshold,
              schema::ThresholdArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Double(schema->threshold));
  MNM_SET_ENV(vpack->x[2], schema2value::Double(schema->value));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.threshold_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(threshold_dx, 3, ffi2schema::ThresholdDx,
              schema::ThresholdDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[2], schema2value::Double(schema->threshold));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.topk").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(topk, 6, ffi2schema::Topk, schema::TopkArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->k));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->axis));
  MNM_SET_ENV(vpack->x[3], schema2value::String(schema->ret_type));
  MNM_SET_ENV(vpack->x[4], schema2value::Bool(schema->is_ascend));
  MNM_SET_ENV(vpack->x[5], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.transpose").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(transpose, 2, ffi2schema::Transpose,
              schema::TransposeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axes));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.transpose_dx").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(transpose_dx, 3, ffi2schema::TransposeDx,
              schema::TransposeDxArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->dy));
  MNM_SET_ENV(vpack->x[1], schema2value::IntOrTupleInt(schema->axes));
  MNM_SET_ENV(vpack->x[2], schema2value::IntOrTupleInt(schema->primal_shape));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.trunc").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(trunc, 1, ffi2schema::Unary, schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.upper_bound.argwhere").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(upper_bound_argwhere, 1, ffi2schema::Argwhere,
              schema::ArgwhereArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->condition));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.vm.alloc_storage").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(vm_alloc_storage, 5, ffi2schema::AllocStorage,
              schema::AllocStorageArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->size));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->alignment));
  MNM_SET_ENV(vpack->x[2], schema2value::Int(schema->device_type));
  MNM_SET_ENV(vpack->x[3], schema2value::Int(schema->device_id));
  MNM_SET_ENV(vpack->x[4], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.vm.alloc_tensor").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(vm_alloc_tensor, 5, ffi2schema::AllocTensor,
              schema::AllocTensorArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->storage));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->shape));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[3], schema2value::IntOrTupleInt(schema->assert_shape));
  MNM_SET_ENV(vpack->x[4], schema2value::Bool(schema->own));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.vm.free").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(vm_free, 1, ffi2schema::Free, schema::FreeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->memory));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.vm.infer_type").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(vm_infer_type, 2, ffi2schema::InferType,
              schema::InferTypeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->func));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->inputs));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.vm.invoke_op").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(vm_invoke_op, 3, ffi2schema::InvokeOp,
              schema::InvokeOpArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->func));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->inputs));
  MNM_SET_ENV(vpack->x[2], schema2value::ArrayLike(schema->outputs));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.vm.set_shape").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(vm_set_shape, 2, ffi2schema::SetShape,
              schema::SetShapeArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->data));
  MNM_SET_ENV(vpack->x[1], schema2value::ArrayLike(schema->shape));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.wait_event").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(wait_event, 2, ffi2schema::Event,
              schema::EventArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Int(schema->event_id));
  MNM_SET_ENV(vpack->x[1], schema2value::Int(schema->stream_id));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.where").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(where, 3, ffi2schema::Where, schema::WhereArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::Tensor(schema->condition));
  MNM_SET_ENV(vpack->x[1], schema2value::Tensor(schema->x));
  MNM_SET_ENV(vpack->x[2], schema2value::Tensor(schema->y));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.zeros").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(zeros, 3, ffi2schema::InitOp, schema::InitOpArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::IntOrTupleInt(schema->shape));
  MNM_SET_ENV(vpack->x[1], schema2value::String(schema->dtype));
  MNM_SET_ENV(vpack->x[2], schema2value::String(schema->device));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

MNM_REGISTER_GLOBAL("mnm.op.imp.zeros_like").set_body([](TVMArgs args, TVMRetValue* ret) {
  MNM_PRELUDE(zeros_like, 1, ffi2schema::Unary,
              schema::UnaryArgs);  // NOLINT(whitespace/line_length)
  MNM_SET_ENV(vpack->x[0], schema2value::ArrayLike(schema->x));
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
});

#undef MNM_RET
#undef MNM_SET_ENV
#undef MNM_PRELUDE

}  // namespace imperative
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 2.1. FFI to Array<Expr> (for each schema)
namespace mnm {
namespace op {
namespace regs {
namespace ffi2expr {

#define MNM_PRELUDE(n)                                                                     \
  const int size = values.size();                                                          \
  CHECK_EQ(size, n) << "TypeError: Mismatched number of arguments for operator \"{op}\": " \
                    << "Expected " << n << ", but get " << size;                           \
  std::vector<Expr> result;

#define MNM_ARG(i, norm, name)         \
  try {                                \
    result.push_back(norm(values[i])); \
  } catch (const dmlc::Error& e) {     \
    FillError(e, "{arg}", #name);      \
  }

#define MNM_RET() return Array<Expr>(result);

Array<Expr> AdaptivePool(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_ARG(2, ffi2expr::String, layout);
  MNM_RET();
}

Array<Expr> AdaptivePoolDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, shape);
  MNM_RET();
}

Array<Expr> AdvIndex(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::TupleTensor, inputs);
  MNM_RET();
}

Array<Expr> AdvIndexDx(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::TupleTensor, inputs);
  MNM_RET();
}

Array<Expr> Allgather(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> AllocStorage(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::ArrayLike, size);
  MNM_ARG(1, ffi2expr::ArrayLike, alignment);
  MNM_ARG(2, ffi2expr::Int, device_type);
  MNM_ARG(3, ffi2expr::Int, device_id);
  MNM_ARG(4, ffi2expr::String, dtype);
  MNM_RET();
}

Array<Expr> AllocTensor(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, storage);
  MNM_ARG(1, ffi2expr::ArrayLike, shape);
  MNM_ARG(2, ffi2expr::String, dtype);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, assert_shape);
  MNM_ARG(4, ffi2expr::Bool, own);
  MNM_RET();
}

Array<Expr> Allreduce(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_ARG(1, ffi2expr::String, computation);
  MNM_RET();
}

Array<Expr> Arange(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, start);
  MNM_ARG(1, ffi2expr::Tensor, stop);
  MNM_ARG(2, ffi2expr::Tensor, step);
  MNM_ARG(3, ffi2expr::String, dtype);
  MNM_ARG(4, ffi2expr::String, device);
  MNM_RET();
}

Array<Expr> Argsort(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_ARG(2, ffi2expr::Bool, is_ascend);
  MNM_ARG(3, ffi2expr::String, dtype);
  MNM_RET();
}

Array<Expr> Argwhere(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::Tensor, condition);
  MNM_RET();
}

Array<Expr> BatchNorm(const TVMArgs& values) {
  MNM_PRELUDE(7);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, running_mean);
  MNM_ARG(2, ffi2expr::Tensor, running_var);
  MNM_ARG(3, ffi2expr::Tensor, w);
  MNM_ARG(4, ffi2expr::Tensor, b);
  MNM_ARG(5, ffi2expr::Double, momentum);
  MNM_ARG(6, ffi2expr::Double, eps);
  MNM_RET();
}

Array<Expr> BatchNormTrainDxwb(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::Tensor, x);
  MNM_ARG(2, ffi2expr::Tensor, w);
  MNM_ARG(3, ffi2expr::Tensor, b);
  MNM_ARG(4, ffi2expr::Double, eps);
  MNM_RET();
}

Array<Expr> BiasAdd(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, bias);
  MNM_ARG(2, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> Binary(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::ArrayLike, x1);
  MNM_ARG(1, ffi2expr::ArrayLike, x2);
  MNM_RET();
}

Array<Expr> BinaryDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::ArrayLike, x1);
  MNM_ARG(1, ffi2expr::ArrayLike, x2);
  MNM_ARG(2, ffi2expr::Tensor, y);
  MNM_ARG(3, ffi2expr::Tensor, dy);
  MNM_RET();
}

Array<Expr> BinaryUfunc(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::ArrayLike, x1);
  MNM_ARG(1, ffi2expr::ArrayLike, x2);
  MNM_ARG(2, ffi2expr::ArrayLike, out);
  MNM_ARG(3, ffi2expr::ArrayLike, where);
  MNM_RET();
}

Array<Expr> Broadcast(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_ARG(1, ffi2expr::Int, root);
  MNM_RET();
}

Array<Expr> BroadcastTo(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_RET();
}

Array<Expr> BroadcastToLike(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, broadcast_type);
  MNM_RET();
}

Array<Expr> Cast(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::String, dtype);
  MNM_RET();
}

Array<Expr> CastLike(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, dtype_like);
  MNM_RET();
}

Array<Expr> Clip(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Double, a_min);
  MNM_ARG(2, ffi2expr::Double, a_max);
  MNM_RET();
}

Array<Expr> ClipDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::Double, a_min);
  MNM_ARG(3, ffi2expr::Double, a_max);
  MNM_RET();
}

Array<Expr> CollapseLike(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_RET();
}

Array<Expr> CommReduce(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_ARG(1, ffi2expr::Int, root);
  MNM_ARG(2, ffi2expr::String, computation);
  MNM_RET();
}

Array<Expr> Concatenate(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> Conv(const TVMArgs& values) {
  MNM_PRELUDE(9);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, w);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, stride);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, padding);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, dilation);
  MNM_ARG(5, ffi2expr::Int, groups);
  MNM_ARG(6, ffi2expr::String, layout);
  MNM_ARG(7, ffi2expr::String, kernel_layout);
  MNM_ARG(8, ffi2expr::String, out_layout);
  MNM_RET();
}

Array<Expr> ConvDxw(const TVMArgs& values) {
  MNM_PRELUDE(8);
  MNM_ARG(0, ffi2expr::Tensor, x_or_w);
  MNM_ARG(1, ffi2expr::OptionalTensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::IntArray, shape);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, stride);
  MNM_ARG(5, ffi2expr::IntOrTupleInt, padding);
  MNM_ARG(6, ffi2expr::IntOrTupleInt, dilation);
  MNM_ARG(7, ffi2expr::Int, groups);
  MNM_RET();
}

Array<Expr> ConvTrans(const TVMArgs& values) {
  MNM_PRELUDE(10);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, w);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, stride);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, padding);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, output_padding);
  MNM_ARG(5, ffi2expr::IntOrTupleInt, dilation);
  MNM_ARG(6, ffi2expr::Int, groups);
  MNM_ARG(7, ffi2expr::String, layout);
  MNM_ARG(8, ffi2expr::String, kernel_layout);
  MNM_ARG(9, ffi2expr::String, out_layout);
  MNM_RET();
}

Array<Expr> ConvTransposeDxw(const TVMArgs& values) {
  MNM_PRELUDE(9);
  MNM_ARG(0, ffi2expr::Tensor, x_or_w);
  MNM_ARG(1, ffi2expr::OptionalTensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::IntArray, shape);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, stride);
  MNM_ARG(5, ffi2expr::IntOrTupleInt, padding);
  MNM_ARG(6, ffi2expr::IntOrTupleInt, output_padding);
  MNM_ARG(7, ffi2expr::IntOrTupleInt, dilation);
  MNM_ARG(8, ffi2expr::Int, groups);
  MNM_RET();
}

Array<Expr> Cumsum(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_ARG(2, ffi2expr::String, dtype);
  MNM_ARG(3, ffi2expr::Bool, exclusive);
  MNM_RET();
}

Array<Expr> DeviceCopy(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Int, src_dev_type);
  MNM_ARG(2, ffi2expr::Int, dst_dev_type);
  MNM_RET();
}

Array<Expr> Dropout(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Double, p);
  MNM_ARG(2, ffi2expr::OptionalTensor, in_states);
  MNM_RET();
}

Array<Expr> DropoutDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::Tensor, mask);
  MNM_ARG(2, ffi2expr::Tensor, reserve_space);
  MNM_ARG(3, ffi2expr::Double, p);
  MNM_RET();
}

Array<Expr> Embedding(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, indices);
  MNM_RET();
}

Array<Expr> EmbeddingDx(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::Tensor, indices);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, num_weight);
  MNM_RET();
}

Array<Expr> Event(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Int, event_id);
  MNM_ARG(1, ffi2expr::Int, stream_id);
  MNM_RET();
}

Array<Expr> ExpandDims(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_ARG(2, ffi2expr::Int, num_newaxis);
  MNM_RET();
}

Array<Expr> Free(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::Tensor, memory);
  MNM_RET();
}

Array<Expr> Full(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Double, fill_value);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_ARG(2, ffi2expr::String, dtype);
  MNM_ARG(3, ffi2expr::String, device);
  MNM_RET();
}

Array<Expr> FullLike(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Double, fill_value);
  MNM_RET();
}

Array<Expr> Gather(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_ARG(2, ffi2expr::Tensor, indices);
  MNM_RET();
}

Array<Expr> GatherDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_ARG(2, ffi2expr::Tensor, indices);
  MNM_ARG(3, ffi2expr::Tensor, dy);
  MNM_RET();
}

Array<Expr> GatherNd(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, indices);
  MNM_RET();
}

Array<Expr> GatherNdDx(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, indices);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_RET();
}

Array<Expr> GetValidCounts(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, score_threshold);
  MNM_ARG(2, ffi2expr::Int, id_index);
  MNM_ARG(3, ffi2expr::Int, score_index);
  MNM_RET();
}

Array<Expr> InferType(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::ArrayLike, func);
  MNM_ARG(1, ffi2expr::ArrayLike, inputs);
  MNM_RET();
}

Array<Expr> InitOp(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::IntOrTupleInt, shape);
  MNM_ARG(1, ffi2expr::String, dtype);
  MNM_ARG(2, ffi2expr::String, device);
  MNM_RET();
}

Array<Expr> InvokeOp(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::ArrayLike, func);
  MNM_ARG(1, ffi2expr::ArrayLike, inputs);
  MNM_ARG(2, ffi2expr::ArrayLike, outputs);
  MNM_RET();
}

Array<Expr> LayerNorm(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::OptionalTensor, scale);
  MNM_ARG(2, ffi2expr::OptionalTensor, bias);
  MNM_ARG(3, ffi2expr::Int, axis);
  MNM_ARG(4, ffi2expr::Double, eps);
  MNM_RET();
}

Array<Expr> LayerNormDx(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::OptionalTensor, scale);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::Int, axis);
  MNM_ARG(4, ffi2expr::Double, eps);
  MNM_RET();
}

Array<Expr> LocalResponseNorm(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, size);
  MNM_ARG(2, ffi2expr::Double, alpha);
  MNM_ARG(3, ffi2expr::Double, beta);
  MNM_ARG(4, ffi2expr::Double, k);
  MNM_RET();
}

Array<Expr> Loss(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, y_true);
  MNM_ARG(1, ffi2expr::Tensor, y_pred);
  MNM_RET();
}

Array<Expr> LossDtp(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::Tensor, y_true);
  MNM_ARG(2, ffi2expr::Tensor, y_pred);
  MNM_RET();
}

Array<Expr> MeanDx(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, axis);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, x_shape);
  MNM_ARG(3, ffi2expr::Bool, keepdims);
  MNM_ARG(4, ffi2expr::Bool, exclude);
  MNM_RET();
}

Array<Expr> MeshGrid(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_RET();
}

Array<Expr> NonMaxSuppression(const TVMArgs& values) {
  MNM_PRELUDE(12);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, valid_count);
  MNM_ARG(2, ffi2expr::Tensor, indices);
  MNM_ARG(3, ffi2expr::Tensor, max_output_size);
  MNM_ARG(4, ffi2expr::Tensor, iou_threshold);
  MNM_ARG(5, ffi2expr::Bool, force_suppress);
  MNM_ARG(6, ffi2expr::Int, top_k);
  MNM_ARG(7, ffi2expr::Int, coord_start);
  MNM_ARG(8, ffi2expr::Int, score_index);
  MNM_ARG(9, ffi2expr::Int, id_index);
  MNM_ARG(10, ffi2expr::Bool, return_indices);
  MNM_ARG(11, ffi2expr::Bool, invalid_to_bottom);
  MNM_RET();
}

Array<Expr> OneHot(const TVMArgs& values) {
  MNM_PRELUDE(7);
  MNM_ARG(0, ffi2expr::Tensor, indices);
  MNM_ARG(1, ffi2expr::Tensor, on_value);
  MNM_ARG(2, ffi2expr::Tensor, off_value);
  MNM_ARG(3, ffi2expr::Int, depth);
  MNM_ARG(4, ffi2expr::Int, axis);
  MNM_ARG(5, ffi2expr::String, dtype);
  MNM_ARG(6, ffi2expr::String, device);
  MNM_RET();
}

Array<Expr> Pad(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, pad_width);
  MNM_ARG(2, ffi2expr::Double, pad_value);
  MNM_ARG(3, ffi2expr::String, pad_mode);
  MNM_RET();
}

Array<Expr> Pool(const TVMArgs& values) {
  MNM_PRELUDE(8);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, kernel);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, stride);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, padding);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, dilation);
  MNM_ARG(5, ffi2expr::Bool, ceil_mode);
  MNM_ARG(6, ffi2expr::Bool, include_pad);
  MNM_ARG(7, ffi2expr::String, layout);
  MNM_RET();
}

Array<Expr> PoolDx(const TVMArgs& values) {
  MNM_PRELUDE(9);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, kernel);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, stride);
  MNM_ARG(5, ffi2expr::IntOrTupleInt, padding);
  MNM_ARG(6, ffi2expr::IntOrTupleInt, dilation);
  MNM_ARG(7, ffi2expr::Bool, ceil_mode);
  MNM_ARG(8, ffi2expr::Bool, include_pad);
  MNM_RET();
}

Array<Expr> ProdDx(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, axis);
  MNM_ARG(3, ffi2expr::Bool, keepdims);
  MNM_ARG(4, ffi2expr::Bool, exclude);
  MNM_RET();
}

Array<Expr> Recv(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Int, peer);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_ARG(2, ffi2expr::String, dtype);
  MNM_ARG(3, ffi2expr::OptionalTensor, token);
  MNM_RET();
}

Array<Expr> Reduce(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, axis);
  MNM_ARG(2, ffi2expr::Bool, keepdims);
  MNM_ARG(3, ffi2expr::Bool, exclude);
  MNM_RET();
}

Array<Expr> ReduceScatter(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_RET();
}

Array<Expr> Repeat(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, repeats);
  MNM_ARG(2, ffi2expr::ArrayLike, axis);
  MNM_RET();
}

Array<Expr> RepeatDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::Int, repeats);
  MNM_ARG(3, ffi2expr::ArrayLike, axis);
  MNM_RET();
}

Array<Expr> Reshape(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_ARG(2, ffi2expr::Bool, reverse);
  MNM_RET();
}

Array<Expr> Resize2D(const TVMArgs& values) {
  MNM_PRELUDE(9);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, size);
  MNM_ARG(2, ffi2expr::String, layout);
  MNM_ARG(3, ffi2expr::String, method);
  MNM_ARG(4, ffi2expr::String, coordinate_transformation_mode);
  MNM_ARG(5, ffi2expr::String, rounding_method);
  MNM_ARG(6, ffi2expr::Double, cubic_alpha);
  MNM_ARG(7, ffi2expr::Int, cubic_exclude);
  MNM_ARG(8, ffi2expr::String, out_dtype);
  MNM_RET();
}

Array<Expr> Resize2DDx(const TVMArgs& values) {
  MNM_PRELUDE(10);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, size);
  MNM_ARG(3, ffi2expr::String, layout);
  MNM_ARG(4, ffi2expr::String, method);
  MNM_ARG(5, ffi2expr::String, coordinate_transformation_mode);
  MNM_ARG(6, ffi2expr::String, rounding_method);
  MNM_ARG(7, ffi2expr::Double, cubic_alpha);
  MNM_ARG(8, ffi2expr::Int, cubic_exclude);
  MNM_ARG(9, ffi2expr::String, out_dtype);
  MNM_RET();
}

Array<Expr> Reverse(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> ReverseSequence(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, sequence_length);
  MNM_ARG(2, ffi2expr::Int, seq_axis);
  MNM_ARG(3, ffi2expr::Int, batch_axis);
  MNM_RET();
}

Array<Expr> RoiAlign(const TVMArgs& values) {
  MNM_PRELUDE(7);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, rois);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, pooled_size);
  MNM_ARG(3, ffi2expr::Double, spatial_scale);
  MNM_ARG(4, ffi2expr::Int, sample_ratio);
  MNM_ARG(5, ffi2expr::String, layout);
  MNM_ARG(6, ffi2expr::String, mode);
  MNM_RET();
}

Array<Expr> RoiAlignDx(const TVMArgs& values) {
  MNM_PRELUDE(8);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Tensor, rois);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, pooled_size);
  MNM_ARG(4, ffi2expr::Double, spatial_scale);
  MNM_ARG(5, ffi2expr::Int, sample_ratio);
  MNM_ARG(6, ffi2expr::String, layout);
  MNM_ARG(7, ffi2expr::String, mode);
  MNM_RET();
}

Array<Expr> Scatter(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, index);
  MNM_ARG(2, ffi2expr::Tensor, src);
  MNM_ARG(3, ffi2expr::ArrayLike, axis);
  MNM_RET();
}

Array<Expr> ScatterDx(const TVMArgs& values) {
  MNM_PRELUDE(6);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::Tensor, index);
  MNM_ARG(4, ffi2expr::Tensor, src);
  MNM_ARG(5, ffi2expr::ArrayLike, axis);
  MNM_RET();
}

Array<Expr> Send(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, peer);
  MNM_ARG(2, ffi2expr::OptionalTensor, token);
  MNM_RET();
}

Array<Expr> SequenceMask(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, sequence_length);
  MNM_ARG(2, ffi2expr::Double, mask_value);
  MNM_ARG(3, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> SetShape(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::ArrayLike, shape);
  MNM_RET();
}

Array<Expr> SetStream(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Int, device_id);
  MNM_ARG(1, ffi2expr::Int, stream_id);
  MNM_RET();
}

Array<Expr> Sgd(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dx);
  MNM_ARG(2, ffi2expr::Tensor, v);
  MNM_ARG(3, ffi2expr::Double, learning_rate);
  MNM_ARG(4, ffi2expr::Double, mu);
  MNM_RET();
}

Array<Expr> Softmax(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> SoftmaxDx(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_ARG(3, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> Sort(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_ARG(2, ffi2expr::Bool, is_ascend);
  MNM_RET();
}

Array<Expr> Split(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::ArrayLike, indices_or_sections);
  MNM_ARG(2, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> Squeeze(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, axis);
  MNM_RET();
}

Array<Expr> Stack(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::TupleTensor, x);
  MNM_ARG(1, ffi2expr::Int, axis);
  MNM_RET();
}

Array<Expr> Stream(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, stream_tag);
  MNM_RET();
}

Array<Expr> StreamBarrier(const TVMArgs& values) {
  MNM_PRELUDE(0);

  MNM_RET();
}

Array<Expr> StridedSlice(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, begin);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, end);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, strides);
  MNM_ARG(4, ffi2expr::String, slice_mode);
  MNM_RET();
}

Array<Expr> StridedSliceDx(const TVMArgs& values) {
  MNM_PRELUDE(6);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, primal_shape);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, begin);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, end);
  MNM_ARG(4, ffi2expr::IntOrTupleInt, strides);
  MNM_ARG(5, ffi2expr::String, slice_mode);
  MNM_RET();
}

Array<Expr> Sum(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, axis);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, keepdims);
  MNM_ARG(3, ffi2expr::Bool, exclude);
  MNM_RET();
}

Array<Expr> SumDx(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, axis);
  MNM_ARG(3, ffi2expr::IntOrTupleInt, keepdims);
  MNM_ARG(4, ffi2expr::Bool, exclude);
  MNM_RET();
}

Array<Expr> SwapAxis(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Int, axis1);
  MNM_ARG(2, ffi2expr::Int, axis2);
  MNM_RET();
}

Array<Expr> Take(const TVMArgs& values) {
  MNM_PRELUDE(4);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, indices);
  MNM_ARG(2, ffi2expr::ArrayLike, axis);
  MNM_ARG(3, ffi2expr::String, mode);
  MNM_RET();
}

Array<Expr> TakeDx(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::Tensor, indices);
  MNM_ARG(3, ffi2expr::ArrayLike, axis);
  MNM_ARG(4, ffi2expr::String, mode);
  MNM_RET();
}

Array<Expr> Ternary(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::ArrayLike, x1);
  MNM_ARG(1, ffi2expr::ArrayLike, x2);
  MNM_ARG(2, ffi2expr::ArrayLike, x3);
  MNM_RET();
}

Array<Expr> TernaryDx(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::ArrayLike, x1);
  MNM_ARG(1, ffi2expr::ArrayLike, x2);
  MNM_ARG(2, ffi2expr::ArrayLike, x3);
  MNM_ARG(3, ffi2expr::Tensor, y);
  MNM_ARG(4, ffi2expr::Tensor, dy);
  MNM_RET();
}

Array<Expr> TernaryUfunc(const TVMArgs& values) {
  MNM_PRELUDE(5);
  MNM_ARG(0, ffi2expr::ArrayLike, x1);
  MNM_ARG(1, ffi2expr::ArrayLike, x2);
  MNM_ARG(2, ffi2expr::ArrayLike, x3);
  MNM_ARG(3, ffi2expr::ArrayLike, out);
  MNM_ARG(4, ffi2expr::ArrayLike, where);
  MNM_RET();
}

Array<Expr> ThreefryGenerate(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, key);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, shape);
  MNM_RET();
}

Array<Expr> ThreefrySplit(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::Tensor, key);
  MNM_RET();
}

Array<Expr> Threshold(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::ArrayLike, x);
  MNM_ARG(1, ffi2expr::Double, threshold);
  MNM_ARG(2, ffi2expr::Double, value);
  MNM_RET();
}

Array<Expr> ThresholdDx(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::ArrayLike, x);
  MNM_ARG(1, ffi2expr::Tensor, dy);
  MNM_ARG(2, ffi2expr::Double, threshold);
  MNM_RET();
}

Array<Expr> Topk(const TVMArgs& values) {
  MNM_PRELUDE(6);
  MNM_ARG(0, ffi2expr::Tensor, data);
  MNM_ARG(1, ffi2expr::Int, k);
  MNM_ARG(2, ffi2expr::Int, axis);
  MNM_ARG(3, ffi2expr::String, ret_type);
  MNM_ARG(4, ffi2expr::Bool, is_ascend);
  MNM_ARG(5, ffi2expr::String, dtype);
  MNM_RET();
}

Array<Expr> Transpose(const TVMArgs& values) {
  MNM_PRELUDE(2);
  MNM_ARG(0, ffi2expr::Tensor, x);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, axes);
  MNM_RET();
}

Array<Expr> TransposeDx(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, dy);
  MNM_ARG(1, ffi2expr::IntOrTupleInt, axes);
  MNM_ARG(2, ffi2expr::IntOrTupleInt, primal_shape);
  MNM_RET();
}

Array<Expr> Unary(const TVMArgs& values) {
  MNM_PRELUDE(1);
  MNM_ARG(0, ffi2expr::ArrayLike, x);
  MNM_RET();
}

Array<Expr> UnaryDx(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::OptionalArrayLike, x);
  MNM_ARG(1, ffi2expr::OptionalTensor, y);
  MNM_ARG(2, ffi2expr::Tensor, dy);
  MNM_RET();
}

Array<Expr> UnaryUfunc(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::ArrayLike, x);
  MNM_ARG(1, ffi2expr::ArrayLike, out);
  MNM_ARG(2, ffi2expr::ArrayLike, where);
  MNM_RET();
}

Array<Expr> Where(const TVMArgs& values) {
  MNM_PRELUDE(3);
  MNM_ARG(0, ffi2expr::Tensor, condition);
  MNM_ARG(1, ffi2expr::Tensor, x);
  MNM_ARG(2, ffi2expr::Tensor, y);
  MNM_RET();
}

#undef MNM_RET
#undef MNM_ARG
#undef MNM_PRELUDE

}  // namespace ffi2expr
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 2.2. Symbolic API, uses "Part 2.1. FFI to Array<Expr>"
namespace mnm {
namespace op {
namespace regs {
namespace symbolic {

#define MNM_SYMBOLIC_API(op_name, n_args, schema)                \
  [](TVMArgs args, TVMRetValue* ret) {                           \
    auto* pack = regs::OpPack<names::op_name, n_args>::Get();    \
    try {                                                        \
      *ret = BindSymbol(Call(pack->op, ffi2expr::schema(args))); \
    } catch (const dmlc::Error& e) {                             \
      FillError(e, "{op}", names::op_name);                      \
    }                                                            \
  }

MNM_REGISTER_GLOBAL("mnm.op.sym._allgather").set_body(MNM_SYMBOLIC_API(_allgather, 2, Allgather));
MNM_REGISTER_GLOBAL("mnm.op.sym._allreduce").set_body(MNM_SYMBOLIC_API(_allreduce, 2, Allreduce));
MNM_REGISTER_GLOBAL("mnm.op.sym._broadcast").set_body(MNM_SYMBOLIC_API(_broadcast, 2, Broadcast));
MNM_REGISTER_GLOBAL("mnm.op.sym._contrib_dropout")
    .set_body(MNM_SYMBOLIC_API(_contrib_dropout, 3, Dropout));
MNM_REGISTER_GLOBAL("mnm.op.sym._contrib_dropout_dx")
    .set_body(MNM_SYMBOLIC_API(_contrib_dropout_dx, 4, DropoutDx));
MNM_REGISTER_GLOBAL("mnm.op.sym._recv").set_body(MNM_SYMBOLIC_API(_recv, 4, Recv));
MNM_REGISTER_GLOBAL("mnm.op.sym._reduce").set_body(MNM_SYMBOLIC_API(_reduce, 3, CommReduce));
MNM_REGISTER_GLOBAL("mnm.op.sym._reduce_scatter")
    .set_body(MNM_SYMBOLIC_API(_reduce_scatter, 1, ReduceScatter));
MNM_REGISTER_GLOBAL("mnm.op.sym._send").set_body(MNM_SYMBOLIC_API(_send, 3, Send));
MNM_REGISTER_GLOBAL("mnm.op.sym.abs").set_body(MNM_SYMBOLIC_API(abs, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.adaptive_avg_pool2d")
    .set_body(MNM_SYMBOLIC_API(adaptive_avg_pool2d, 3, AdaptivePool));
MNM_REGISTER_GLOBAL("mnm.op.sym.adaptive_avg_pool2d_dx")
    .set_body(MNM_SYMBOLIC_API(adaptive_avg_pool2d_dx, 4, AdaptivePoolDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.adaptive_max_pool2d")
    .set_body(MNM_SYMBOLIC_API(adaptive_max_pool2d, 3, AdaptivePool));
MNM_REGISTER_GLOBAL("mnm.op.sym.adaptive_max_pool2d_dx")
    .set_body(MNM_SYMBOLIC_API(adaptive_max_pool2d_dx, 4, AdaptivePoolDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.add").set_body(MNM_SYMBOLIC_API(add, 4, BinaryUfunc));
MNM_REGISTER_GLOBAL("mnm.op.sym.add_event").set_body(MNM_SYMBOLIC_API(add_event, 2, Event));
MNM_REGISTER_GLOBAL("mnm.op.sym.adv_index").set_body(MNM_SYMBOLIC_API(adv_index, 1, AdvIndex));
MNM_REGISTER_GLOBAL("mnm.op.sym.adv_index_dx")
    .set_body(MNM_SYMBOLIC_API(adv_index_dx, 2, AdvIndexDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.all").set_body(MNM_SYMBOLIC_API(all, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.any").set_body(MNM_SYMBOLIC_API(any, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.arange").set_body(MNM_SYMBOLIC_API(arange, 5, Arange));
MNM_REGISTER_GLOBAL("mnm.op.sym.argmax").set_body(MNM_SYMBOLIC_API(argmax, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.argmin").set_body(MNM_SYMBOLIC_API(argmin, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.argsort").set_body(MNM_SYMBOLIC_API(argsort, 4, Argsort));
MNM_REGISTER_GLOBAL("mnm.op.sym.argwhere").set_body(MNM_SYMBOLIC_API(argwhere, 1, Argwhere));
MNM_REGISTER_GLOBAL("mnm.op.sym.atan").set_body(MNM_SYMBOLIC_API(atan, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.avg_pool2d").set_body(MNM_SYMBOLIC_API(avg_pool2d, 8, Pool));
MNM_REGISTER_GLOBAL("mnm.op.sym.avg_pool2d_dx")
    .set_body(MNM_SYMBOLIC_API(avg_pool2d_dx, 9, PoolDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_flatten").set_body(MNM_SYMBOLIC_API(batch_flatten, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_matmul").set_body(MNM_SYMBOLIC_API(batch_matmul, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_matmul_nt")
    .set_body(MNM_SYMBOLIC_API(batch_matmul_nt, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_matmul_tn")
    .set_body(MNM_SYMBOLIC_API(batch_matmul_tn, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_matmul_tt")
    .set_body(MNM_SYMBOLIC_API(batch_matmul_tt, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_norm_infer")
    .set_body(MNM_SYMBOLIC_API(batch_norm_infer, 7, BatchNorm));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_norm_train")
    .set_body(MNM_SYMBOLIC_API(batch_norm_train, 7, BatchNorm));
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_norm_train_dxwb")
    .set_body(MNM_SYMBOLIC_API(batch_norm_train_dxwb, 5, BatchNormTrainDxwb));
MNM_REGISTER_GLOBAL("mnm.op.sym.bias_add").set_body(MNM_SYMBOLIC_API(bias_add, 3, BiasAdd));
MNM_REGISTER_GLOBAL("mnm.op.sym.broadcast_to")
    .set_body(MNM_SYMBOLIC_API(broadcast_to, 2, BroadcastTo));
MNM_REGISTER_GLOBAL("mnm.op.sym.broadcast_to_like")
    .set_body(MNM_SYMBOLIC_API(broadcast_to_like, 2, BroadcastToLike));
MNM_REGISTER_GLOBAL("mnm.op.sym.cast").set_body(MNM_SYMBOLIC_API(cast, 2, Cast));
MNM_REGISTER_GLOBAL("mnm.op.sym.cast_like").set_body(MNM_SYMBOLIC_API(cast_like, 2, CastLike));
MNM_REGISTER_GLOBAL("mnm.op.sym.ceil").set_body(MNM_SYMBOLIC_API(ceil, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.clip").set_body(MNM_SYMBOLIC_API(clip, 3, Clip));
MNM_REGISTER_GLOBAL("mnm.op.sym.clip_dx").set_body(MNM_SYMBOLIC_API(clip_dx, 4, ClipDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.collapse_sum_like")
    .set_body(MNM_SYMBOLIC_API(collapse_sum_like, 2, CollapseLike));
MNM_REGISTER_GLOBAL("mnm.op.sym.compiler_begin")
    .set_body(MNM_SYMBOLIC_API(compiler_begin, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.compiler_end").set_body(MNM_SYMBOLIC_API(compiler_end, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.concatenate")
    .set_body(MNM_SYMBOLIC_API(concatenate, 2, Concatenate));
MNM_REGISTER_GLOBAL("mnm.op.sym.concatenate_dx")
    .set_body(MNM_SYMBOLIC_API(concatenate_dx, 2, Concatenate));
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d").set_body(MNM_SYMBOLIC_API(conv2d, 9, Conv));
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_dw").set_body(MNM_SYMBOLIC_API(conv2d_dw, 8, ConvDxw));
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_dx").set_body(MNM_SYMBOLIC_API(conv2d_dx, 8, ConvDxw));
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_transpose")
    .set_body(MNM_SYMBOLIC_API(conv2d_transpose, 10, ConvTrans));
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_transpose_dw")
    .set_body(MNM_SYMBOLIC_API(conv2d_transpose_dw, 9, ConvTransposeDxw));
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_transpose_dx")
    .set_body(MNM_SYMBOLIC_API(conv2d_transpose_dx, 9, ConvTransposeDxw));
MNM_REGISTER_GLOBAL("mnm.op.sym.copy").set_body(MNM_SYMBOLIC_API(copy, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.cos").set_body(MNM_SYMBOLIC_API(cos, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.cross_entropy").set_body(MNM_SYMBOLIC_API(cross_entropy, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.cross_entropy_dpred")
    .set_body(MNM_SYMBOLIC_API(cross_entropy_dpred, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.cross_entropy_dtrue")
    .set_body(MNM_SYMBOLIC_API(cross_entropy_dtrue, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.cumsum").set_body(MNM_SYMBOLIC_API(cumsum, 4, Cumsum));
MNM_REGISTER_GLOBAL("mnm.op.sym.dense").set_body(MNM_SYMBOLIC_API(dense, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.device_copy")
    .set_body(MNM_SYMBOLIC_API(device_copy, 3, DeviceCopy));
MNM_REGISTER_GLOBAL("mnm.op.sym.divide").set_body(MNM_SYMBOLIC_API(divide, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.embedding").set_body(MNM_SYMBOLIC_API(embedding, 2, Embedding));
MNM_REGISTER_GLOBAL("mnm.op.sym.embedding_dx")
    .set_body(MNM_SYMBOLIC_API(embedding_dx, 3, EmbeddingDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.equal").set_body(MNM_SYMBOLIC_API(equal, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.erf").set_body(MNM_SYMBOLIC_API(erf, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.erf_dx").set_body(MNM_SYMBOLIC_API(erf_dx, 3, UnaryDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.exp").set_body(MNM_SYMBOLIC_API(exp, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.expand_dims")
    .set_body(MNM_SYMBOLIC_API(expand_dims, 3, ExpandDims));
MNM_REGISTER_GLOBAL("mnm.op.sym.floor").set_body(MNM_SYMBOLIC_API(floor, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.floor_divide").set_body(MNM_SYMBOLIC_API(floor_divide, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.full").set_body(MNM_SYMBOLIC_API(full, 4, Full));
MNM_REGISTER_GLOBAL("mnm.op.sym.full_like").set_body(MNM_SYMBOLIC_API(full_like, 2, FullLike));
MNM_REGISTER_GLOBAL("mnm.op.sym.gather").set_body(MNM_SYMBOLIC_API(gather, 3, Gather));
MNM_REGISTER_GLOBAL("mnm.op.sym.gather_dx").set_body(MNM_SYMBOLIC_API(gather_dx, 4, GatherDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.gather_nd").set_body(MNM_SYMBOLIC_API(gather_nd, 2, GatherNd));
MNM_REGISTER_GLOBAL("mnm.op.sym.gather_nd_dx")
    .set_body(MNM_SYMBOLIC_API(gather_nd_dx, 3, GatherNdDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.gelu").set_body(MNM_SYMBOLIC_API(gelu, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.gelu_dx").set_body(MNM_SYMBOLIC_API(gelu_dx, 3, UnaryDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.get_kept_dims")
    .set_body(MNM_SYMBOLIC_API(get_kept_dims, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.get_reduce_axis")
    .set_body(MNM_SYMBOLIC_API(get_reduce_axis, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.get_valid_counts")
    .set_body(MNM_SYMBOLIC_API(get_valid_counts, 4, GetValidCounts));
MNM_REGISTER_GLOBAL("mnm.op.sym.greater").set_body(MNM_SYMBOLIC_API(greater, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.greater_equal")
    .set_body(MNM_SYMBOLIC_API(greater_equal, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.layer_norm").set_body(MNM_SYMBOLIC_API(layer_norm, 5, LayerNorm));
MNM_REGISTER_GLOBAL("mnm.op.sym.layer_norm_dx")
    .set_body(MNM_SYMBOLIC_API(layer_norm_dx, 5, LayerNormDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.left_shift").set_body(MNM_SYMBOLIC_API(left_shift, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.less").set_body(MNM_SYMBOLIC_API(less, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.less_equal").set_body(MNM_SYMBOLIC_API(less_equal, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.log").set_body(MNM_SYMBOLIC_API(log, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.log2").set_body(MNM_SYMBOLIC_API(log2, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.log_softmax").set_body(MNM_SYMBOLIC_API(log_softmax, 2, Softmax));
MNM_REGISTER_GLOBAL("mnm.op.sym.log_softmax_dx")
    .set_body(MNM_SYMBOLIC_API(log_softmax_dx, 4, SoftmaxDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.logical_and").set_body(MNM_SYMBOLIC_API(logical_and, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.logical_not").set_body(MNM_SYMBOLIC_API(logical_not, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.matmul").set_body(MNM_SYMBOLIC_API(matmul, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.matmul_nt").set_body(MNM_SYMBOLIC_API(matmul_nt, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.matmul_tn").set_body(MNM_SYMBOLIC_API(matmul_tn, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.matmul_tt").set_body(MNM_SYMBOLIC_API(matmul_tt, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.max").set_body(MNM_SYMBOLIC_API(max, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.max_pool2d").set_body(MNM_SYMBOLIC_API(max_pool2d, 8, Pool));
MNM_REGISTER_GLOBAL("mnm.op.sym.max_pool2d_dx")
    .set_body(MNM_SYMBOLIC_API(max_pool2d_dx, 9, PoolDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.maximum").set_body(MNM_SYMBOLIC_API(maximum, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.mean").set_body(MNM_SYMBOLIC_API(mean, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.mean_dx").set_body(MNM_SYMBOLIC_API(mean_dx, 5, MeanDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.mesh_grid").set_body(MNM_SYMBOLIC_API(mesh_grid, 1, MeshGrid));
MNM_REGISTER_GLOBAL("mnm.op.sym.min").set_body(MNM_SYMBOLIC_API(min, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.minimum").set_body(MNM_SYMBOLIC_API(minimum, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.mod").set_body(MNM_SYMBOLIC_API(mod, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.multiply").set_body(MNM_SYMBOLIC_API(multiply, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.ndarray_size").set_body(MNM_SYMBOLIC_API(ndarray_size, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.negative").set_body(MNM_SYMBOLIC_API(negative, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.nll_loss").set_body(MNM_SYMBOLIC_API(nll_loss, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.nll_loss_dpred")
    .set_body(MNM_SYMBOLIC_API(nll_loss_dpred, 3, LossDtp));
MNM_REGISTER_GLOBAL("mnm.op.sym.nll_loss_dtrue")
    .set_body(MNM_SYMBOLIC_API(nll_loss_dtrue, 3, LossDtp));
MNM_REGISTER_GLOBAL("mnm.op.sym.non_max_suppression")
    .set_body(MNM_SYMBOLIC_API(non_max_suppression, 12, NonMaxSuppression));
MNM_REGISTER_GLOBAL("mnm.op.sym.not_equal").set_body(MNM_SYMBOLIC_API(not_equal, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.one_hot").set_body(MNM_SYMBOLIC_API(one_hot, 7, OneHot));
MNM_REGISTER_GLOBAL("mnm.op.sym.ones").set_body(MNM_SYMBOLIC_API(ones, 3, InitOp));
MNM_REGISTER_GLOBAL("mnm.op.sym.ones_like").set_body(MNM_SYMBOLIC_API(ones_like, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.pad").set_body(MNM_SYMBOLIC_API(pad, 4, Pad));
MNM_REGISTER_GLOBAL("mnm.op.sym.power").set_body(MNM_SYMBOLIC_API(power, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.prod").set_body(MNM_SYMBOLIC_API(prod, 4, Reduce));
MNM_REGISTER_GLOBAL("mnm.op.sym.prod_dx").set_body(MNM_SYMBOLIC_API(prod_dx, 5, ProdDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.relu").set_body(MNM_SYMBOLIC_API(relu, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.relu_dx").set_body(MNM_SYMBOLIC_API(relu_dx, 3, UnaryDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.repeat").set_body(MNM_SYMBOLIC_API(repeat, 3, Repeat));
MNM_REGISTER_GLOBAL("mnm.op.sym.repeat_dx").set_body(MNM_SYMBOLIC_API(repeat_dx, 4, RepeatDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.reshape").set_body(MNM_SYMBOLIC_API(reshape, 3, Reshape));
MNM_REGISTER_GLOBAL("mnm.op.sym.resize2d").set_body(MNM_SYMBOLIC_API(resize2d, 9, Resize2D));
MNM_REGISTER_GLOBAL("mnm.op.sym.resize2d_dx")
    .set_body(MNM_SYMBOLIC_API(resize2d_dx, 10, Resize2DDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.reverse").set_body(MNM_SYMBOLIC_API(reverse, 2, Reverse));
MNM_REGISTER_GLOBAL("mnm.op.sym.reverse_sequence")
    .set_body(MNM_SYMBOLIC_API(reverse_sequence, 4, ReverseSequence));
MNM_REGISTER_GLOBAL("mnm.op.sym.right_shift").set_body(MNM_SYMBOLIC_API(right_shift, 2, Binary));
MNM_REGISTER_GLOBAL("mnm.op.sym.roi_align").set_body(MNM_SYMBOLIC_API(roi_align, 7, RoiAlign));
MNM_REGISTER_GLOBAL("mnm.op.sym.roi_align_dx")
    .set_body(MNM_SYMBOLIC_API(roi_align_dx, 8, RoiAlignDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.round").set_body(MNM_SYMBOLIC_API(round, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.rsqrt").set_body(MNM_SYMBOLIC_API(rsqrt, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.scatter").set_body(MNM_SYMBOLIC_API(scatter, 4, Scatter));
MNM_REGISTER_GLOBAL("mnm.op.sym.scatter_dx").set_body(MNM_SYMBOLIC_API(scatter_dx, 6, ScatterDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.sequence_mask")
    .set_body(MNM_SYMBOLIC_API(sequence_mask, 4, SequenceMask));
MNM_REGISTER_GLOBAL("mnm.op.sym.set_stream").set_body(MNM_SYMBOLIC_API(set_stream, 2, SetStream));
MNM_REGISTER_GLOBAL("mnm.op.sym.sgd").set_body(MNM_SYMBOLIC_API(sgd, 5, Sgd));
MNM_REGISTER_GLOBAL("mnm.op.sym.shape").set_body(MNM_SYMBOLIC_API(shape, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.sigmoid").set_body(MNM_SYMBOLIC_API(sigmoid, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.sigmoid_dx").set_body(MNM_SYMBOLIC_API(sigmoid_dx, 3, UnaryDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.sign").set_body(MNM_SYMBOLIC_API(sign, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.sin").set_body(MNM_SYMBOLIC_API(sin, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.smooth_l1_loss")
    .set_body(MNM_SYMBOLIC_API(smooth_l1_loss, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.smooth_l1_loss_dpred")
    .set_body(MNM_SYMBOLIC_API(smooth_l1_loss_dpred, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.smooth_l1_loss_dtrue")
    .set_body(MNM_SYMBOLIC_API(smooth_l1_loss_dtrue, 2, Loss));
MNM_REGISTER_GLOBAL("mnm.op.sym.softmax").set_body(MNM_SYMBOLIC_API(softmax, 2, Softmax));
MNM_REGISTER_GLOBAL("mnm.op.sym.softmax_dx").set_body(MNM_SYMBOLIC_API(softmax_dx, 4, SoftmaxDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.sort").set_body(MNM_SYMBOLIC_API(sort, 3, Sort));
MNM_REGISTER_GLOBAL("mnm.op.sym.split").set_body(MNM_SYMBOLIC_API(split, 3, Split));
MNM_REGISTER_GLOBAL("mnm.op.sym.sqrt").set_body(MNM_SYMBOLIC_API(sqrt, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.sqrt_dx").set_body(MNM_SYMBOLIC_API(sqrt_dx, 3, UnaryDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.squeeze").set_body(MNM_SYMBOLIC_API(squeeze, 2, Squeeze));
MNM_REGISTER_GLOBAL("mnm.op.sym.stack").set_body(MNM_SYMBOLIC_API(stack, 2, Stack));
MNM_REGISTER_GLOBAL("mnm.op.sym.stream_barrier")
    .set_body(MNM_SYMBOLIC_API(stream_barrier, 0, StreamBarrier));
MNM_REGISTER_GLOBAL("mnm.op.sym.stream_sync").set_body(MNM_SYMBOLIC_API(stream_sync, 2, Stream));
MNM_REGISTER_GLOBAL("mnm.op.sym.strided_slice")
    .set_body(MNM_SYMBOLIC_API(strided_slice, 5, StridedSlice));
MNM_REGISTER_GLOBAL("mnm.op.sym.strided_slice_dx")
    .set_body(MNM_SYMBOLIC_API(strided_slice_dx, 6, StridedSliceDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.subtract").set_body(MNM_SYMBOLIC_API(subtract, 4, BinaryUfunc));
MNM_REGISTER_GLOBAL("mnm.op.sym.sum").set_body(MNM_SYMBOLIC_API(sum, 4, Sum));
MNM_REGISTER_GLOBAL("mnm.op.sym.sum_dx").set_body(MNM_SYMBOLIC_API(sum_dx, 5, SumDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.swap_axis").set_body(MNM_SYMBOLIC_API(swap_axis, 3, SwapAxis));
MNM_REGISTER_GLOBAL("mnm.op.sym.take").set_body(MNM_SYMBOLIC_API(take, 4, Take));
MNM_REGISTER_GLOBAL("mnm.op.sym.take_dx").set_body(MNM_SYMBOLIC_API(take_dx, 5, TakeDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.tanh").set_body(MNM_SYMBOLIC_API(tanh, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.tanh_dx").set_body(MNM_SYMBOLIC_API(tanh_dx, 3, UnaryDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.threefry_generate")
    .set_body(MNM_SYMBOLIC_API(threefry_generate, 2, ThreefryGenerate));
MNM_REGISTER_GLOBAL("mnm.op.sym.threefry_split")
    .set_body(MNM_SYMBOLIC_API(threefry_split, 1, ThreefrySplit));
MNM_REGISTER_GLOBAL("mnm.op.sym.threshold").set_body(MNM_SYMBOLIC_API(threshold, 3, Threshold));
MNM_REGISTER_GLOBAL("mnm.op.sym.threshold_dx")
    .set_body(MNM_SYMBOLIC_API(threshold_dx, 3, ThresholdDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.topk").set_body(MNM_SYMBOLIC_API(topk, 6, Topk));
MNM_REGISTER_GLOBAL("mnm.op.sym.transpose").set_body(MNM_SYMBOLIC_API(transpose, 2, Transpose));
MNM_REGISTER_GLOBAL("mnm.op.sym.transpose_dx")
    .set_body(MNM_SYMBOLIC_API(transpose_dx, 3, TransposeDx));
MNM_REGISTER_GLOBAL("mnm.op.sym.trunc").set_body(MNM_SYMBOLIC_API(trunc, 1, Unary));
MNM_REGISTER_GLOBAL("mnm.op.sym.upper_bound.argwhere")
    .set_body(MNM_SYMBOLIC_API(upper_bound_argwhere, 1, Argwhere));
MNM_REGISTER_GLOBAL("mnm.op.sym.vm.alloc_storage")
    .set_body(MNM_SYMBOLIC_API(vm_alloc_storage, 5, AllocStorage));
MNM_REGISTER_GLOBAL("mnm.op.sym.vm.alloc_tensor")
    .set_body(MNM_SYMBOLIC_API(vm_alloc_tensor, 5, AllocTensor));
MNM_REGISTER_GLOBAL("mnm.op.sym.vm.free").set_body(MNM_SYMBOLIC_API(vm_free, 1, Free));
MNM_REGISTER_GLOBAL("mnm.op.sym.vm.infer_type")
    .set_body(MNM_SYMBOLIC_API(vm_infer_type, 2, InferType));
MNM_REGISTER_GLOBAL("mnm.op.sym.vm.invoke_op")
    .set_body(MNM_SYMBOLIC_API(vm_invoke_op, 3, InvokeOp));
MNM_REGISTER_GLOBAL("mnm.op.sym.vm.set_shape")
    .set_body(MNM_SYMBOLIC_API(vm_set_shape, 2, SetShape));
MNM_REGISTER_GLOBAL("mnm.op.sym.wait_event").set_body(MNM_SYMBOLIC_API(wait_event, 2, Event));
MNM_REGISTER_GLOBAL("mnm.op.sym.where").set_body(MNM_SYMBOLIC_API(where, 3, Where));
MNM_REGISTER_GLOBAL("mnm.op.sym.zeros").set_body(MNM_SYMBOLIC_API(zeros, 3, InitOp));
MNM_REGISTER_GLOBAL("mnm.op.sym.zeros_like").set_body(MNM_SYMBOLIC_API(zeros_like, 1, Unary));

#undef MNM_SYMBOLIC_API

}  // namespace symbolic
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 3.1. Array<Value> to schema (for each schema)
namespace mnm {
namespace op {
namespace regs {
namespace value2schema {

#define MNM_PRELUDE(lb, ub, schema)                                            \
  const int size = values.size();                                              \
  CHECK(size >= lb) << "TypeError: Too few arguments for operator \"{op}\". "  \
                    << "Expected at least " << lb << ", but get " << size;     \
  CHECK(size <= ub) << "TypeError: Too many arguments for operator \"{op}\". " \
                    << "Expected at most " << ub << ", but get " << size;      \
  auto attrs = make_object<schema>();

#define MNM_REQUIRED(i, norm, name)   \
  try {                               \
    attrs->name = norm(values[i]);    \
  } catch (const dmlc::Error& e) {    \
    try {                             \
      FillError(e, "{arg}", #name);   \
    } catch (const dmlc::Error& ee) { \
      FillError(ee, "{op}", op_name); \
    }                                 \
  }

#define MNM_OPTIONAL(i, norm, name)     \
  if (size > i) {                       \
    try {                               \
      attrs->name = norm(values[i]);    \
    } catch (const dmlc::Error& e) {    \
      try {                             \
        FillError(e, "{arg}", #name);   \
      } catch (const dmlc::Error& ee) { \
        FillError(ee, "{op}", op_name); \
      }                                 \
    }                                   \
  }

template <const char* op_name>
Attrs AdaptivePool(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::AdaptivePoolArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  MNM_OPTIONAL(2, value2schema::String, layout);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs AdaptivePoolDx(const Array<Value>& values) {
  MNM_PRELUDE(4, 4, schema::AdaptivePoolDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_REQUIRED(3, value2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs AdvIndex(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::AdvIndexArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, inputs);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs AdvIndexDx(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::AdvIndexDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_REQUIRED(1, value2schema::TupleTensor, inputs);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Allgather(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::AllgatherArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs AllocStorage(const Array<Value>& values) {
  MNM_PRELUDE(4, 5, schema::AllocStorageArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, size);
  MNM_REQUIRED(1, value2schema::ArrayLike, alignment);
  MNM_REQUIRED(2, value2schema::Int, device_type);
  MNM_REQUIRED(3, value2schema::Int, device_id);
  MNM_OPTIONAL(4, value2schema::String, dtype);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs AllocTensor(const Array<Value>& values) {
  MNM_PRELUDE(2, 5, schema::AllocTensorArgs);
  MNM_REQUIRED(0, value2schema::Tensor, storage);
  MNM_REQUIRED(1, value2schema::ArrayLike, shape);
  MNM_OPTIONAL(2, value2schema::String, dtype);
  MNM_OPTIONAL(3, value2schema::IntOrTupleInt, assert_shape);
  MNM_OPTIONAL(4, value2schema::Bool, own);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Allreduce(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::AllreduceArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  MNM_OPTIONAL(1, value2schema::String, computation);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Arange(const Array<Value>& values) {
  MNM_PRELUDE(3, 5, schema::ArangeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, start);
  MNM_REQUIRED(1, value2schema::Tensor, stop);
  MNM_REQUIRED(2, value2schema::Tensor, step);
  MNM_OPTIONAL(3, value2schema::String, dtype);
  MNM_OPTIONAL(4, value2schema::String, device);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Argsort(const Array<Value>& values) {
  MNM_PRELUDE(1, 4, schema::ArgsortArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_OPTIONAL(1, value2schema::Int, axis);
  MNM_OPTIONAL(2, value2schema::Bool, is_ascend);
  MNM_OPTIONAL(3, value2schema::String, dtype);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Argwhere(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::ArgwhereArgs);
  MNM_REQUIRED(0, value2schema::Tensor, condition);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BatchNorm(const Array<Value>& values) {
  MNM_PRELUDE(3, 7, schema::BatchNormArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, running_mean);
  MNM_REQUIRED(2, value2schema::Tensor, running_var);
  MNM_OPTIONAL(3, value2schema::Tensor, w);
  MNM_OPTIONAL(4, value2schema::Tensor, b);
  MNM_OPTIONAL(5, value2schema::Double, momentum);
  MNM_OPTIONAL(6, value2schema::Double, eps);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BatchNormTrainDxwb(const Array<Value>& values) {
  MNM_PRELUDE(5, 5, schema::BatchNormTrainDxwbArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_REQUIRED(1, value2schema::Tensor, x);
  MNM_REQUIRED(2, value2schema::Tensor, w);
  MNM_REQUIRED(3, value2schema::Tensor, b);
  MNM_REQUIRED(4, value2schema::Double, eps);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BiasAdd(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::BiasAddArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, bias);
  MNM_OPTIONAL(2, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Binary(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::BinaryArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x1);
  MNM_REQUIRED(1, value2schema::ArrayLike, x2);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BinaryDx(const Array<Value>& values) {
  MNM_PRELUDE(4, 4, schema::BinaryDxArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x1);
  MNM_REQUIRED(1, value2schema::ArrayLike, x2);
  MNM_REQUIRED(2, value2schema::Tensor, y);
  MNM_REQUIRED(3, value2schema::Tensor, dy);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BinaryUfunc(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::BinaryUfuncArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x1);
  MNM_REQUIRED(1, value2schema::ArrayLike, x2);
  MNM_OPTIONAL(2, value2schema::ArrayLike, out);
  MNM_OPTIONAL(3, value2schema::ArrayLike, where);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Broadcast(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::BroadcastArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  MNM_REQUIRED(1, value2schema::Int, root);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BroadcastTo(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::BroadcastToArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs BroadcastToLike(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::BroadcastToLikeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, broadcast_type);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Cast(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::CastArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::String, dtype);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs CastLike(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::CastLikeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, dtype_like);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Clip(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::ClipArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Double, a_min);
  MNM_REQUIRED(2, value2schema::Double, a_max);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ClipDx(const Array<Value>& values) {
  MNM_PRELUDE(4, 4, schema::ClipDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_REQUIRED(2, value2schema::Double, a_min);
  MNM_REQUIRED(3, value2schema::Double, a_max);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs CollapseLike(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::CollapseLikeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs CommReduce(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::CommReduceArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  MNM_REQUIRED(1, value2schema::Int, root);
  MNM_OPTIONAL(2, value2schema::String, computation);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Concatenate(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::ConcatenateArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  MNM_OPTIONAL(1, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Conv(const Array<Value>& values) {
  MNM_PRELUDE(2, 9, schema::ConvArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, w);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, stride);
  MNM_OPTIONAL(3, value2schema::IntOrTupleInt, padding);
  MNM_OPTIONAL(4, value2schema::IntOrTupleInt, dilation);
  MNM_OPTIONAL(5, value2schema::Int, groups);
  MNM_OPTIONAL(6, value2schema::String, layout);
  MNM_OPTIONAL(7, value2schema::String, kernel_layout);
  MNM_OPTIONAL(8, value2schema::String, out_layout);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ConvDxw(const Array<Value>& values) {
  MNM_PRELUDE(8, 8, schema::ConvDxwArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x_or_w);
  MNM_REQUIRED(1, value2schema::OptionalTensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_REQUIRED(3, value2schema::IntArray, shape);
  MNM_REQUIRED(4, value2schema::IntOrTupleInt, stride);
  MNM_REQUIRED(5, value2schema::IntOrTupleInt, padding);
  MNM_REQUIRED(6, value2schema::IntOrTupleInt, dilation);
  MNM_REQUIRED(7, value2schema::Int, groups);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ConvTrans(const Array<Value>& values) {
  MNM_PRELUDE(2, 10, schema::ConvTransArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, w);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, stride);
  MNM_OPTIONAL(3, value2schema::IntOrTupleInt, padding);
  MNM_OPTIONAL(4, value2schema::IntOrTupleInt, output_padding);
  MNM_OPTIONAL(5, value2schema::IntOrTupleInt, dilation);
  MNM_OPTIONAL(6, value2schema::Int, groups);
  MNM_OPTIONAL(7, value2schema::String, layout);
  MNM_OPTIONAL(8, value2schema::String, kernel_layout);
  MNM_OPTIONAL(9, value2schema::String, out_layout);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ConvTransposeDxw(const Array<Value>& values) {
  MNM_PRELUDE(9, 9, schema::ConvTransposeDxwArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x_or_w);
  MNM_REQUIRED(1, value2schema::OptionalTensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_REQUIRED(3, value2schema::IntArray, shape);
  MNM_REQUIRED(4, value2schema::IntOrTupleInt, stride);
  MNM_REQUIRED(5, value2schema::IntOrTupleInt, padding);
  MNM_REQUIRED(6, value2schema::IntOrTupleInt, output_padding);
  MNM_REQUIRED(7, value2schema::IntOrTupleInt, dilation);
  MNM_REQUIRED(8, value2schema::Int, groups);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Cumsum(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::CumsumArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, axis);
  MNM_OPTIONAL(2, value2schema::String, dtype);
  MNM_OPTIONAL(3, value2schema::Bool, exclusive);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs DeviceCopy(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::DeviceCopyArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_OPTIONAL(1, value2schema::Int, src_dev_type);
  MNM_OPTIONAL(2, value2schema::Int, dst_dev_type);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Dropout(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::DropoutArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::Double, p);
  MNM_OPTIONAL(2, value2schema::OptionalTensor, in_states);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs DropoutDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 4, schema::DropoutDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_REQUIRED(1, value2schema::Tensor, mask);
  MNM_REQUIRED(2, value2schema::Tensor, reserve_space);
  MNM_OPTIONAL(3, value2schema::Double, p);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Embedding(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::EmbeddingArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, indices);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs EmbeddingDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::EmbeddingDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_REQUIRED(1, value2schema::Tensor, indices);
  MNM_REQUIRED(2, value2schema::IntOrTupleInt, num_weight);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Event(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::EventArgs);
  MNM_REQUIRED(0, value2schema::Int, event_id);
  MNM_OPTIONAL(1, value2schema::Int, stream_id);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ExpandDims(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::ExpandDimsArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, axis);
  MNM_OPTIONAL(2, value2schema::Int, num_newaxis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Free(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::FreeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, memory);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Full(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::FullArgs);
  MNM_REQUIRED(0, value2schema::Double, fill_value);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  MNM_OPTIONAL(2, value2schema::String, dtype);
  MNM_OPTIONAL(3, value2schema::String, device);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs FullLike(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::FullLikeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Double, fill_value);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Gather(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::GatherArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Int, axis);
  MNM_REQUIRED(2, value2schema::Tensor, indices);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs GatherDx(const Array<Value>& values) {
  MNM_PRELUDE(4, 4, schema::GatherDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Int, axis);
  MNM_REQUIRED(2, value2schema::Tensor, indices);
  MNM_REQUIRED(3, value2schema::Tensor, dy);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs GatherNd(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::GatherNdArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, indices);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs GatherNdDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::GatherNdDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, indices);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs GetValidCounts(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::GetValidCountsArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, score_threshold);
  MNM_OPTIONAL(2, value2schema::Int, id_index);
  MNM_OPTIONAL(3, value2schema::Int, score_index);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs InferType(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::InferTypeArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, func);
  MNM_REQUIRED(1, value2schema::ArrayLike, inputs);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs InitOp(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::InitOpArgs);
  MNM_REQUIRED(0, value2schema::IntOrTupleInt, shape);
  MNM_OPTIONAL(1, value2schema::String, dtype);
  MNM_OPTIONAL(2, value2schema::String, device);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs InvokeOp(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::InvokeOpArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, func);
  MNM_REQUIRED(1, value2schema::ArrayLike, inputs);
  MNM_REQUIRED(2, value2schema::ArrayLike, outputs);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs LayerNorm(const Array<Value>& values) {
  MNM_PRELUDE(1, 5, schema::LayerNormArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::OptionalTensor, scale);
  MNM_OPTIONAL(2, value2schema::OptionalTensor, bias);
  MNM_OPTIONAL(3, value2schema::Int, axis);
  MNM_OPTIONAL(4, value2schema::Double, eps);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs LayerNormDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 5, schema::LayerNormDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::OptionalTensor, scale);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_OPTIONAL(3, value2schema::Int, axis);
  MNM_OPTIONAL(4, value2schema::Double, eps);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs LocalResponseNorm(const Array<Value>& values) {
  MNM_PRELUDE(2, 5, schema::LocalResponseNormArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, size);
  MNM_OPTIONAL(2, value2schema::Double, alpha);
  MNM_OPTIONAL(3, value2schema::Double, beta);
  MNM_OPTIONAL(4, value2schema::Double, k);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Loss(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::LossArgs);
  MNM_REQUIRED(0, value2schema::Tensor, y_true);
  MNM_REQUIRED(1, value2schema::Tensor, y_pred);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs LossDtp(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::LossDtpArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_REQUIRED(1, value2schema::Tensor, y_true);
  MNM_REQUIRED(2, value2schema::Tensor, y_pred);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs MeanDx(const Array<Value>& values) {
  MNM_PRELUDE(1, 5, schema::MeanDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_OPTIONAL(1, value2schema::IntOrTupleInt, axis);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, x_shape);
  MNM_OPTIONAL(3, value2schema::Bool, keepdims);
  MNM_OPTIONAL(4, value2schema::Bool, exclude);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs MeshGrid(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::MeshGridArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs NonMaxSuppression(const Array<Value>& values) {
  MNM_PRELUDE(5, 12, schema::NonMaxSuppressionArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, valid_count);
  MNM_REQUIRED(2, value2schema::Tensor, indices);
  MNM_REQUIRED(3, value2schema::Tensor, max_output_size);
  MNM_REQUIRED(4, value2schema::Tensor, iou_threshold);
  MNM_OPTIONAL(5, value2schema::Bool, force_suppress);
  MNM_OPTIONAL(6, value2schema::Int, top_k);
  MNM_OPTIONAL(7, value2schema::Int, coord_start);
  MNM_OPTIONAL(8, value2schema::Int, score_index);
  MNM_OPTIONAL(9, value2schema::Int, id_index);
  MNM_OPTIONAL(10, value2schema::Bool, return_indices);
  MNM_OPTIONAL(11, value2schema::Bool, invalid_to_bottom);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs OneHot(const Array<Value>& values) {
  MNM_PRELUDE(4, 7, schema::OneHotArgs);
  MNM_REQUIRED(0, value2schema::Tensor, indices);
  MNM_REQUIRED(1, value2schema::Tensor, on_value);
  MNM_REQUIRED(2, value2schema::Tensor, off_value);
  MNM_REQUIRED(3, value2schema::Int, depth);
  MNM_OPTIONAL(4, value2schema::Int, axis);
  MNM_OPTIONAL(5, value2schema::String, dtype);
  MNM_OPTIONAL(6, value2schema::String, device);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Pad(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::PadArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, pad_width);
  MNM_OPTIONAL(2, value2schema::Double, pad_value);
  MNM_OPTIONAL(3, value2schema::String, pad_mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Pool(const Array<Value>& values) {
  MNM_PRELUDE(3, 8, schema::PoolArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, kernel);
  MNM_REQUIRED(2, value2schema::IntOrTupleInt, stride);
  MNM_OPTIONAL(3, value2schema::IntOrTupleInt, padding);
  MNM_OPTIONAL(4, value2schema::IntOrTupleInt, dilation);
  MNM_OPTIONAL(5, value2schema::Bool, ceil_mode);
  MNM_OPTIONAL(6, value2schema::Bool, include_pad);
  MNM_OPTIONAL(7, value2schema::String, layout);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs PoolDx(const Array<Value>& values) {
  MNM_PRELUDE(9, 9, schema::PoolDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_REQUIRED(3, value2schema::IntOrTupleInt, kernel);
  MNM_REQUIRED(4, value2schema::IntOrTupleInt, stride);
  MNM_REQUIRED(5, value2schema::IntOrTupleInt, padding);
  MNM_REQUIRED(6, value2schema::IntOrTupleInt, dilation);
  MNM_REQUIRED(7, value2schema::Bool, ceil_mode);
  MNM_REQUIRED(8, value2schema::Bool, include_pad);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ProdDx(const Array<Value>& values) {
  MNM_PRELUDE(2, 5, schema::ProdDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, axis);
  MNM_OPTIONAL(3, value2schema::Bool, keepdims);
  MNM_OPTIONAL(4, value2schema::Bool, exclude);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Recv(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::RecvArgs);
  MNM_REQUIRED(0, value2schema::Int, peer);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  MNM_OPTIONAL(2, value2schema::String, dtype);
  MNM_OPTIONAL(3, value2schema::OptionalTensor, token);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Reduce(const Array<Value>& values) {
  MNM_PRELUDE(1, 4, schema::ReduceArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::IntOrTupleInt, axis);
  MNM_OPTIONAL(2, value2schema::Bool, keepdims);
  MNM_OPTIONAL(3, value2schema::Bool, exclude);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ReduceScatter(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::ReduceScatterArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Repeat(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::RepeatArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, repeats);
  MNM_OPTIONAL(2, value2schema::ArrayLike, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs RepeatDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 4, schema::RepeatDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_REQUIRED(2, value2schema::Int, repeats);
  MNM_OPTIONAL(3, value2schema::ArrayLike, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Reshape(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::ReshapeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  MNM_OPTIONAL(2, value2schema::Bool, reverse);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Resize2D(const Array<Value>& values) {
  MNM_PRELUDE(2, 9, schema::Resize2DArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, size);
  MNM_OPTIONAL(2, value2schema::String, layout);
  MNM_OPTIONAL(3, value2schema::String, method);
  MNM_OPTIONAL(4, value2schema::String, coordinate_transformation_mode);
  MNM_OPTIONAL(5, value2schema::String, rounding_method);
  MNM_OPTIONAL(6, value2schema::Double, cubic_alpha);
  MNM_OPTIONAL(7, value2schema::Int, cubic_exclude);
  MNM_OPTIONAL(8, value2schema::String, out_dtype);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Resize2DDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 10, schema::Resize2DDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_REQUIRED(2, value2schema::IntOrTupleInt, size);
  MNM_OPTIONAL(3, value2schema::String, layout);
  MNM_OPTIONAL(4, value2schema::String, method);
  MNM_OPTIONAL(5, value2schema::String, coordinate_transformation_mode);
  MNM_OPTIONAL(6, value2schema::String, rounding_method);
  MNM_OPTIONAL(7, value2schema::Double, cubic_alpha);
  MNM_OPTIONAL(8, value2schema::Int, cubic_exclude);
  MNM_OPTIONAL(9, value2schema::String, out_dtype);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Reverse(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::ReverseArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ReverseSequence(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::ReverseSequenceArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, sequence_length);
  MNM_OPTIONAL(2, value2schema::Int, seq_axis);
  MNM_OPTIONAL(3, value2schema::Int, batch_axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs RoiAlign(const Array<Value>& values) {
  MNM_PRELUDE(4, 7, schema::RoiAlignArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, rois);
  MNM_REQUIRED(2, value2schema::IntOrTupleInt, pooled_size);
  MNM_REQUIRED(3, value2schema::Double, spatial_scale);
  MNM_OPTIONAL(4, value2schema::Int, sample_ratio);
  MNM_OPTIONAL(5, value2schema::String, layout);
  MNM_OPTIONAL(6, value2schema::String, mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs RoiAlignDx(const Array<Value>& values) {
  MNM_PRELUDE(5, 8, schema::RoiAlignDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::Tensor, rois);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_REQUIRED(3, value2schema::IntOrTupleInt, pooled_size);
  MNM_REQUIRED(4, value2schema::Double, spatial_scale);
  MNM_OPTIONAL(5, value2schema::Int, sample_ratio);
  MNM_OPTIONAL(6, value2schema::String, layout);
  MNM_OPTIONAL(7, value2schema::String, mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Scatter(const Array<Value>& values) {
  MNM_PRELUDE(4, 4, schema::ScatterArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, index);
  MNM_REQUIRED(2, value2schema::Tensor, src);
  MNM_REQUIRED(3, value2schema::ArrayLike, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ScatterDx(const Array<Value>& values) {
  MNM_PRELUDE(6, 6, schema::ScatterDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_REQUIRED(3, value2schema::Tensor, index);
  MNM_REQUIRED(4, value2schema::Tensor, src);
  MNM_REQUIRED(5, value2schema::ArrayLike, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Send(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::SendArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, peer);
  MNM_OPTIONAL(2, value2schema::OptionalTensor, token);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs SequenceMask(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::SequenceMaskArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, sequence_length);
  MNM_OPTIONAL(2, value2schema::Double, mask_value);
  MNM_OPTIONAL(3, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs SetShape(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::SetShapeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_REQUIRED(1, value2schema::ArrayLike, shape);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs SetStream(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::SetStreamArgs);
  MNM_REQUIRED(0, value2schema::Int, device_id);
  MNM_REQUIRED(1, value2schema::Int, stream_id);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Sgd(const Array<Value>& values) {
  MNM_PRELUDE(5, 5, schema::SgdArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dx);
  MNM_REQUIRED(2, value2schema::Tensor, v);
  MNM_REQUIRED(3, value2schema::Double, learning_rate);
  MNM_REQUIRED(4, value2schema::Double, mu);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Softmax(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::SoftmaxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs SoftmaxDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 4, schema::SoftmaxDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  MNM_OPTIONAL(3, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Sort(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::SortArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_OPTIONAL(1, value2schema::Int, axis);
  MNM_OPTIONAL(2, value2schema::Bool, is_ascend);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Split(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::SplitArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::ArrayLike, indices_or_sections);
  MNM_OPTIONAL(2, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Squeeze(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::SqueezeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::IntOrTupleInt, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Stack(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::StackArgs);
  MNM_REQUIRED(0, value2schema::TupleTensor, x);
  MNM_OPTIONAL(1, value2schema::Int, axis);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Stream(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::StreamArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::Int, stream_tag);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs StreamBarrier(const Array<Value>& values) {
  MNM_PRELUDE(0, 0, schema::StreamBarrierArgs);

  return Attrs(attrs);
}

template <const char* op_name>
Attrs StridedSlice(const Array<Value>& values) {
  MNM_PRELUDE(3, 5, schema::StridedSliceArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, begin);
  MNM_REQUIRED(2, value2schema::IntOrTupleInt, end);
  MNM_OPTIONAL(3, value2schema::IntOrTupleInt, strides);
  MNM_OPTIONAL(4, value2schema::String, slice_mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs StridedSliceDx(const Array<Value>& values) {
  MNM_PRELUDE(4, 6, schema::StridedSliceDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, primal_shape);
  MNM_REQUIRED(2, value2schema::IntOrTupleInt, begin);
  MNM_REQUIRED(3, value2schema::IntOrTupleInt, end);
  MNM_OPTIONAL(4, value2schema::IntOrTupleInt, strides);
  MNM_OPTIONAL(5, value2schema::String, slice_mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Sum(const Array<Value>& values) {
  MNM_PRELUDE(1, 4, schema::SumArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::IntOrTupleInt, axis);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, keepdims);
  MNM_OPTIONAL(3, value2schema::Bool, exclude);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs SumDx(const Array<Value>& values) {
  MNM_PRELUDE(2, 5, schema::SumDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, axis);
  MNM_OPTIONAL(3, value2schema::IntOrTupleInt, keepdims);
  MNM_OPTIONAL(4, value2schema::Bool, exclude);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs SwapAxis(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::SwapAxisArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Int, axis1);
  MNM_REQUIRED(2, value2schema::Int, axis2);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Take(const Array<Value>& values) {
  MNM_PRELUDE(2, 4, schema::TakeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, indices);
  MNM_OPTIONAL(2, value2schema::ArrayLike, axis);
  MNM_OPTIONAL(3, value2schema::String, mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs TakeDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 5, schema::TakeDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_REQUIRED(2, value2schema::Tensor, indices);
  MNM_OPTIONAL(3, value2schema::ArrayLike, axis);
  MNM_OPTIONAL(4, value2schema::String, mode);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Ternary(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::TernaryArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x1);
  MNM_REQUIRED(1, value2schema::ArrayLike, x2);
  MNM_REQUIRED(2, value2schema::ArrayLike, x3);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs TernaryDx(const Array<Value>& values) {
  MNM_PRELUDE(5, 5, schema::TernaryDxArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x1);
  MNM_REQUIRED(1, value2schema::ArrayLike, x2);
  MNM_REQUIRED(2, value2schema::ArrayLike, x3);
  MNM_REQUIRED(3, value2schema::Tensor, y);
  MNM_REQUIRED(4, value2schema::Tensor, dy);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs TernaryUfunc(const Array<Value>& values) {
  MNM_PRELUDE(3, 5, schema::TernaryUfuncArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x1);
  MNM_REQUIRED(1, value2schema::ArrayLike, x2);
  MNM_REQUIRED(2, value2schema::ArrayLike, x3);
  MNM_OPTIONAL(3, value2schema::ArrayLike, out);
  MNM_OPTIONAL(4, value2schema::ArrayLike, where);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ThreefryGenerate(const Array<Value>& values) {
  MNM_PRELUDE(2, 2, schema::ThreefryGenerateArgs);
  MNM_REQUIRED(0, value2schema::Tensor, key);
  MNM_REQUIRED(1, value2schema::IntOrTupleInt, shape);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ThreefrySplit(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::ThreefrySplitArgs);
  MNM_REQUIRED(0, value2schema::Tensor, key);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Threshold(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::ThresholdArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x);
  MNM_OPTIONAL(1, value2schema::Double, threshold);
  MNM_OPTIONAL(2, value2schema::Double, value);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs ThresholdDx(const Array<Value>& values) {
  MNM_PRELUDE(2, 3, schema::ThresholdDxArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x);
  MNM_REQUIRED(1, value2schema::Tensor, dy);
  MNM_OPTIONAL(2, value2schema::Double, threshold);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Topk(const Array<Value>& values) {
  MNM_PRELUDE(1, 6, schema::TopkArgs);
  MNM_REQUIRED(0, value2schema::Tensor, data);
  MNM_OPTIONAL(1, value2schema::Int, k);
  MNM_OPTIONAL(2, value2schema::Int, axis);
  MNM_OPTIONAL(3, value2schema::String, ret_type);
  MNM_OPTIONAL(4, value2schema::Bool, is_ascend);
  MNM_OPTIONAL(5, value2schema::String, dtype);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Transpose(const Array<Value>& values) {
  MNM_PRELUDE(1, 2, schema::TransposeArgs);
  MNM_REQUIRED(0, value2schema::Tensor, x);
  MNM_OPTIONAL(1, value2schema::IntOrTupleInt, axes);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs TransposeDx(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::TransposeDxArgs);
  MNM_REQUIRED(0, value2schema::Tensor, dy);
  MNM_OPTIONAL(1, value2schema::IntOrTupleInt, axes);
  MNM_OPTIONAL(2, value2schema::IntOrTupleInt, primal_shape);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Unary(const Array<Value>& values) {
  MNM_PRELUDE(1, 1, schema::UnaryArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs UnaryDx(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::UnaryDxArgs);
  MNM_REQUIRED(0, value2schema::OptionalArrayLike, x);
  MNM_REQUIRED(1, value2schema::OptionalTensor, y);
  MNM_REQUIRED(2, value2schema::Tensor, dy);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs UnaryUfunc(const Array<Value>& values) {
  MNM_PRELUDE(1, 3, schema::UnaryUfuncArgs);
  MNM_REQUIRED(0, value2schema::ArrayLike, x);
  MNM_OPTIONAL(1, value2schema::ArrayLike, out);
  MNM_OPTIONAL(2, value2schema::ArrayLike, where);
  return Attrs(attrs);
}

template <const char* op_name>
Attrs Where(const Array<Value>& values) {
  MNM_PRELUDE(3, 3, schema::WhereArgs);
  MNM_REQUIRED(0, value2schema::Tensor, condition);
  MNM_REQUIRED(1, value2schema::Tensor, x);
  MNM_REQUIRED(2, value2schema::Tensor, y);
  return Attrs(attrs);
}

#undef MNM_OPTIONAL
#undef MNM_REQUIRED
#undef MNM_PRELUDE

}  // namespace value2schema
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 3.2. Schema field index (for each schema)
namespace mnm {
namespace op {
namespace regs {
namespace schema_field_idx {

template <const char* op_name>
int AdaptivePool(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  if (field == "layout") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int AdaptivePoolDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "shape") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int AdvIndex(const std::string& field) {
  if (field == "inputs") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int AdvIndexDx(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "inputs") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Allgather(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int AllocStorage(const std::string& field) {
  if (field == "size") {
    return 0;
  }
  if (field == "alignment") {
    return 1;
  }
  if (field == "device_type") {
    return 2;
  }
  if (field == "device_id") {
    return 3;
  }
  if (field == "dtype") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int AllocTensor(const std::string& field) {
  if (field == "storage") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  if (field == "dtype") {
    return 2;
  }
  if (field == "assert_shape") {
    return 3;
  }
  if (field == "own") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Allreduce(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "computation") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Arange(const std::string& field) {
  if (field == "start") {
    return 0;
  }
  if (field == "stop") {
    return 1;
  }
  if (field == "step") {
    return 2;
  }
  if (field == "dtype") {
    return 3;
  }
  if (field == "device") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Argsort(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "is_ascend") {
    return 2;
  }
  if (field == "dtype") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Argwhere(const std::string& field) {
  if (field == "condition") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BatchNorm(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "running_mean") {
    return 1;
  }
  if (field == "running_var") {
    return 2;
  }
  if (field == "w") {
    return 3;
  }
  if (field == "b") {
    return 4;
  }
  if (field == "momentum") {
    return 5;
  }
  if (field == "eps") {
    return 6;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BatchNormTrainDxwb(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "x") {
    return 1;
  }
  if (field == "w") {
    return 2;
  }
  if (field == "b") {
    return 3;
  }
  if (field == "eps") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BiasAdd(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "bias") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Binary(const std::string& field) {
  if (field == "x1") {
    return 0;
  }
  if (field == "x2") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BinaryDx(const std::string& field) {
  if (field == "x1") {
    return 0;
  }
  if (field == "x2") {
    return 1;
  }
  if (field == "y") {
    return 2;
  }
  if (field == "dy") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BinaryUfunc(const std::string& field) {
  if (field == "x1") {
    return 0;
  }
  if (field == "x2") {
    return 1;
  }
  if (field == "out") {
    return 2;
  }
  if (field == "where") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Broadcast(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "root") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BroadcastTo(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int BroadcastToLike(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "broadcast_type") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Cast(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "dtype") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int CastLike(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "dtype_like") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Clip(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "a_min") {
    return 1;
  }
  if (field == "a_max") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ClipDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "a_min") {
    return 2;
  }
  if (field == "a_max") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int CollapseLike(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int CommReduce(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "root") {
    return 1;
  }
  if (field == "computation") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Concatenate(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Conv(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "w") {
    return 1;
  }
  if (field == "stride") {
    return 2;
  }
  if (field == "padding") {
    return 3;
  }
  if (field == "dilation") {
    return 4;
  }
  if (field == "groups") {
    return 5;
  }
  if (field == "layout") {
    return 6;
  }
  if (field == "kernel_layout") {
    return 7;
  }
  if (field == "out_layout") {
    return 8;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ConvDxw(const std::string& field) {
  if (field == "x_or_w") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "shape") {
    return 3;
  }
  if (field == "stride") {
    return 4;
  }
  if (field == "padding") {
    return 5;
  }
  if (field == "dilation") {
    return 6;
  }
  if (field == "groups") {
    return 7;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ConvTrans(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "w") {
    return 1;
  }
  if (field == "stride") {
    return 2;
  }
  if (field == "padding") {
    return 3;
  }
  if (field == "output_padding") {
    return 4;
  }
  if (field == "dilation") {
    return 5;
  }
  if (field == "groups") {
    return 6;
  }
  if (field == "layout") {
    return 7;
  }
  if (field == "kernel_layout") {
    return 8;
  }
  if (field == "out_layout") {
    return 9;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ConvTransposeDxw(const std::string& field) {
  if (field == "x_or_w") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "shape") {
    return 3;
  }
  if (field == "stride") {
    return 4;
  }
  if (field == "padding") {
    return 5;
  }
  if (field == "output_padding") {
    return 6;
  }
  if (field == "dilation") {
    return 7;
  }
  if (field == "groups") {
    return 8;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Cumsum(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "dtype") {
    return 2;
  }
  if (field == "exclusive") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int DeviceCopy(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "src_dev_type") {
    return 1;
  }
  if (field == "dst_dev_type") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Dropout(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "p") {
    return 1;
  }
  if (field == "in_states") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int DropoutDx(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "mask") {
    return 1;
  }
  if (field == "reserve_space") {
    return 2;
  }
  if (field == "p") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Embedding(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "indices") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int EmbeddingDx(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "indices") {
    return 1;
  }
  if (field == "num_weight") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Event(const std::string& field) {
  if (field == "event_id") {
    return 0;
  }
  if (field == "stream_id") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ExpandDims(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "num_newaxis") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Free(const std::string& field) {
  if (field == "memory") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Full(const std::string& field) {
  if (field == "fill_value") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  if (field == "dtype") {
    return 2;
  }
  if (field == "device") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int FullLike(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "fill_value") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Gather(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "indices") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int GatherDx(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "indices") {
    return 2;
  }
  if (field == "dy") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int GatherNd(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "indices") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int GatherNdDx(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "indices") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int GetValidCounts(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "score_threshold") {
    return 1;
  }
  if (field == "id_index") {
    return 2;
  }
  if (field == "score_index") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int InferType(const std::string& field) {
  if (field == "func") {
    return 0;
  }
  if (field == "inputs") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int InitOp(const std::string& field) {
  if (field == "shape") {
    return 0;
  }
  if (field == "dtype") {
    return 1;
  }
  if (field == "device") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int InvokeOp(const std::string& field) {
  if (field == "func") {
    return 0;
  }
  if (field == "inputs") {
    return 1;
  }
  if (field == "outputs") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int LayerNorm(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "scale") {
    return 1;
  }
  if (field == "bias") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  if (field == "eps") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int LayerNormDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "scale") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  if (field == "eps") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int LocalResponseNorm(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "size") {
    return 1;
  }
  if (field == "alpha") {
    return 2;
  }
  if (field == "beta") {
    return 3;
  }
  if (field == "k") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Loss(const std::string& field) {
  if (field == "y_true") {
    return 0;
  }
  if (field == "y_pred") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int LossDtp(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "y_true") {
    return 1;
  }
  if (field == "y_pred") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int MeanDx(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "x_shape") {
    return 2;
  }
  if (field == "keepdims") {
    return 3;
  }
  if (field == "exclude") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int MeshGrid(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int NonMaxSuppression(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "valid_count") {
    return 1;
  }
  if (field == "indices") {
    return 2;
  }
  if (field == "max_output_size") {
    return 3;
  }
  if (field == "iou_threshold") {
    return 4;
  }
  if (field == "force_suppress") {
    return 5;
  }
  if (field == "top_k") {
    return 6;
  }
  if (field == "coord_start") {
    return 7;
  }
  if (field == "score_index") {
    return 8;
  }
  if (field == "id_index") {
    return 9;
  }
  if (field == "return_indices") {
    return 10;
  }
  if (field == "invalid_to_bottom") {
    return 11;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int OneHot(const std::string& field) {
  if (field == "indices") {
    return 0;
  }
  if (field == "on_value") {
    return 1;
  }
  if (field == "off_value") {
    return 2;
  }
  if (field == "depth") {
    return 3;
  }
  if (field == "axis") {
    return 4;
  }
  if (field == "dtype") {
    return 5;
  }
  if (field == "device") {
    return 6;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Pad(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "pad_width") {
    return 1;
  }
  if (field == "pad_value") {
    return 2;
  }
  if (field == "pad_mode") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Pool(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "kernel") {
    return 1;
  }
  if (field == "stride") {
    return 2;
  }
  if (field == "padding") {
    return 3;
  }
  if (field == "dilation") {
    return 4;
  }
  if (field == "ceil_mode") {
    return 5;
  }
  if (field == "include_pad") {
    return 6;
  }
  if (field == "layout") {
    return 7;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int PoolDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "kernel") {
    return 3;
  }
  if (field == "stride") {
    return 4;
  }
  if (field == "padding") {
    return 5;
  }
  if (field == "dilation") {
    return 6;
  }
  if (field == "ceil_mode") {
    return 7;
  }
  if (field == "include_pad") {
    return 8;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ProdDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  if (field == "keepdims") {
    return 3;
  }
  if (field == "exclude") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Recv(const std::string& field) {
  if (field == "peer") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  if (field == "dtype") {
    return 2;
  }
  if (field == "token") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Reduce(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "keepdims") {
    return 2;
  }
  if (field == "exclude") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ReduceScatter(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Repeat(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "repeats") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int RepeatDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "repeats") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Reshape(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  if (field == "reverse") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Resize2D(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "size") {
    return 1;
  }
  if (field == "layout") {
    return 2;
  }
  if (field == "method") {
    return 3;
  }
  if (field == "coordinate_transformation_mode") {
    return 4;
  }
  if (field == "rounding_method") {
    return 5;
  }
  if (field == "cubic_alpha") {
    return 6;
  }
  if (field == "cubic_exclude") {
    return 7;
  }
  if (field == "out_dtype") {
    return 8;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Resize2DDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "size") {
    return 2;
  }
  if (field == "layout") {
    return 3;
  }
  if (field == "method") {
    return 4;
  }
  if (field == "coordinate_transformation_mode") {
    return 5;
  }
  if (field == "rounding_method") {
    return 6;
  }
  if (field == "cubic_alpha") {
    return 7;
  }
  if (field == "cubic_exclude") {
    return 8;
  }
  if (field == "out_dtype") {
    return 9;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Reverse(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ReverseSequence(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "sequence_length") {
    return 1;
  }
  if (field == "seq_axis") {
    return 2;
  }
  if (field == "batch_axis") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int RoiAlign(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "rois") {
    return 1;
  }
  if (field == "pooled_size") {
    return 2;
  }
  if (field == "spatial_scale") {
    return 3;
  }
  if (field == "sample_ratio") {
    return 4;
  }
  if (field == "layout") {
    return 5;
  }
  if (field == "mode") {
    return 6;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int RoiAlignDx(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "rois") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "pooled_size") {
    return 3;
  }
  if (field == "spatial_scale") {
    return 4;
  }
  if (field == "sample_ratio") {
    return 5;
  }
  if (field == "layout") {
    return 6;
  }
  if (field == "mode") {
    return 7;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Scatter(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "index") {
    return 1;
  }
  if (field == "src") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ScatterDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "index") {
    return 3;
  }
  if (field == "src") {
    return 4;
  }
  if (field == "axis") {
    return 5;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Send(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "peer") {
    return 1;
  }
  if (field == "token") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int SequenceMask(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "sequence_length") {
    return 1;
  }
  if (field == "mask_value") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int SetShape(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int SetStream(const std::string& field) {
  if (field == "device_id") {
    return 0;
  }
  if (field == "stream_id") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Sgd(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dx") {
    return 1;
  }
  if (field == "v") {
    return 2;
  }
  if (field == "learning_rate") {
    return 3;
  }
  if (field == "mu") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Softmax(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int SoftmaxDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Sort(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "is_ascend") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Split(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "indices_or_sections") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Squeeze(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Stack(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Stream(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "stream_tag") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int StreamBarrier(const std::string& field) {
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int StridedSlice(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "begin") {
    return 1;
  }
  if (field == "end") {
    return 2;
  }
  if (field == "strides") {
    return 3;
  }
  if (field == "slice_mode") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int StridedSliceDx(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "primal_shape") {
    return 1;
  }
  if (field == "begin") {
    return 2;
  }
  if (field == "end") {
    return 3;
  }
  if (field == "strides") {
    return 4;
  }
  if (field == "slice_mode") {
    return 5;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Sum(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis") {
    return 1;
  }
  if (field == "keepdims") {
    return 2;
  }
  if (field == "exclude") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int SumDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  if (field == "keepdims") {
    return 3;
  }
  if (field == "exclude") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int SwapAxis(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axis1") {
    return 1;
  }
  if (field == "axis2") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Take(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "indices") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  if (field == "mode") {
    return 3;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int TakeDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "indices") {
    return 2;
  }
  if (field == "axis") {
    return 3;
  }
  if (field == "mode") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Ternary(const std::string& field) {
  if (field == "x1") {
    return 0;
  }
  if (field == "x2") {
    return 1;
  }
  if (field == "x3") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int TernaryDx(const std::string& field) {
  if (field == "x1") {
    return 0;
  }
  if (field == "x2") {
    return 1;
  }
  if (field == "x3") {
    return 2;
  }
  if (field == "y") {
    return 3;
  }
  if (field == "dy") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int TernaryUfunc(const std::string& field) {
  if (field == "x1") {
    return 0;
  }
  if (field == "x2") {
    return 1;
  }
  if (field == "x3") {
    return 2;
  }
  if (field == "out") {
    return 3;
  }
  if (field == "where") {
    return 4;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ThreefryGenerate(const std::string& field) {
  if (field == "key") {
    return 0;
  }
  if (field == "shape") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ThreefrySplit(const std::string& field) {
  if (field == "key") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Threshold(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "threshold") {
    return 1;
  }
  if (field == "value") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int ThresholdDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "dy") {
    return 1;
  }
  if (field == "threshold") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Topk(const std::string& field) {
  if (field == "data") {
    return 0;
  }
  if (field == "k") {
    return 1;
  }
  if (field == "axis") {
    return 2;
  }
  if (field == "ret_type") {
    return 3;
  }
  if (field == "is_ascend") {
    return 4;
  }
  if (field == "dtype") {
    return 5;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Transpose(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "axes") {
    return 1;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int TransposeDx(const std::string& field) {
  if (field == "dy") {
    return 0;
  }
  if (field == "axes") {
    return 1;
  }
  if (field == "primal_shape") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Unary(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int UnaryDx(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "y") {
    return 1;
  }
  if (field == "dy") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int UnaryUfunc(const std::string& field) {
  if (field == "x") {
    return 0;
  }
  if (field == "out") {
    return 1;
  }
  if (field == "where") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

template <const char* op_name>
int Where(const std::string& field) {
  if (field == "condition") {
    return 0;
  }
  if (field == "x") {
    return 1;
  }
  if (field == "y") {
    return 2;
  }
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}

}  // namespace schema_field_idx
}  // namespace regs
}  // namespace op
}  // namespace mnm

// Part 3.3. FMNMSchema API, uses Part 3.1 and Part 3.2
namespace mnm {
namespace op {
namespace regs {
namespace f_mnm_schema {

#define MNM_BIND_SCHEMA(op_str, op_name, schema) \
  MNM_REGISTER_OP(op_str).set_attr<FMNMSchema>("FMNMSchema", schema<op_name>);

#define MNM_BIND_SCHEMA_FIELD_INDEX(op_str, op_name, schema) \
  MNM_REGISTER_OP(op_str).set_attr<FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex", schema<op_name>);

MNM_BIND_SCHEMA("mnm.op._allgather", names::_allgather,
                value2schema::Allgather);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._allgather", names::_allgather,
                            schema_field_idx::Allgather);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._allreduce", names::_allreduce,
                value2schema::Allreduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._allreduce", names::_allreduce,
                            schema_field_idx::Allreduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._broadcast", names::_broadcast,
                value2schema::Broadcast);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._broadcast", names::_broadcast,
                            schema_field_idx::Broadcast);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._contrib_dropout", names::_contrib_dropout,
                value2schema::Dropout);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._contrib_dropout", names::_contrib_dropout,
                            schema_field_idx::Dropout);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._contrib_dropout_dx", names::_contrib_dropout_dx,
                value2schema::DropoutDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._contrib_dropout_dx", names::_contrib_dropout_dx,
                            schema_field_idx::DropoutDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._recv", names::_recv,
                value2schema::Recv);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._recv", names::_recv,
                            schema_field_idx::Recv);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._reduce", names::_reduce,
                value2schema::CommReduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._reduce", names::_reduce,
                            schema_field_idx::CommReduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._reduce_scatter", names::_reduce_scatter,
                value2schema::ReduceScatter);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._reduce_scatter", names::_reduce_scatter,
                            schema_field_idx::ReduceScatter);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op._send", names::_send,
                value2schema::Send);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op._send", names::_send,
                            schema_field_idx::Send);             // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.abs", names::abs, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.abs", names::abs,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.adaptive_avg_pool2d", names::adaptive_avg_pool2d,
                value2schema::AdaptivePool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.adaptive_avg_pool2d", names::adaptive_avg_pool2d,
                            schema_field_idx::AdaptivePool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.adaptive_avg_pool2d_dx", names::adaptive_avg_pool2d_dx,
                value2schema::AdaptivePoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.adaptive_avg_pool2d_dx", names::adaptive_avg_pool2d_dx,
                            schema_field_idx::AdaptivePoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.adaptive_max_pool2d", names::adaptive_max_pool2d,
                value2schema::AdaptivePool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.adaptive_max_pool2d", names::adaptive_max_pool2d,
                            schema_field_idx::AdaptivePool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.adaptive_max_pool2d_dx", names::adaptive_max_pool2d_dx,
                value2schema::AdaptivePoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.adaptive_max_pool2d_dx", names::adaptive_max_pool2d_dx,
                            schema_field_idx::AdaptivePoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.add", names::add,
                value2schema::BinaryUfunc);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.add", names::add,
                            schema_field_idx::BinaryUfunc);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.add_event", names::add_event,
                value2schema::Event);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.add_event", names::add_event,
                            schema_field_idx::Event);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.adv_index", names::adv_index,
                value2schema::AdvIndex);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.adv_index", names::adv_index,
                            schema_field_idx::AdvIndex);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.adv_index_dx", names::adv_index_dx,
                value2schema::AdvIndexDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.adv_index_dx", names::adv_index_dx,
                            schema_field_idx::AdvIndexDx);        // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.all", names::all, value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.all", names::all,
                            schema_field_idx::Reduce);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.any", names::any, value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.any", names::any,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.arange", names::arange,
                value2schema::Arange);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.arange", names::arange,
                            schema_field_idx::Arange);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.argmax", names::argmax,
                value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.argmax", names::argmax,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.argmin", names::argmin,
                value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.argmin", names::argmin,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.argsort", names::argsort,
                value2schema::Argsort);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.argsort", names::argsort,
                            schema_field_idx::Argsort);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.argwhere", names::argwhere,
                value2schema::Argwhere);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.argwhere", names::argwhere,
                            schema_field_idx::Argwhere);           // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.atan", names::atan, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.atan", names::atan,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.avg_pool2d", names::avg_pool2d,
                value2schema::Pool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.avg_pool2d", names::avg_pool2d,
                            schema_field_idx::Pool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.avg_pool2d_dx", names::avg_pool2d_dx,
                value2schema::PoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.avg_pool2d_dx", names::avg_pool2d_dx,
                            schema_field_idx::PoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_flatten", names::batch_flatten,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_flatten", names::batch_flatten,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_matmul", names::batch_matmul,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_matmul", names::batch_matmul,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_matmul_nt", names::batch_matmul_nt,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_matmul_nt", names::batch_matmul_nt,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_matmul_tn", names::batch_matmul_tn,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_matmul_tn", names::batch_matmul_tn,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_matmul_tt", names::batch_matmul_tt,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_matmul_tt", names::batch_matmul_tt,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_norm_infer", names::batch_norm_infer,
                value2schema::BatchNorm);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_norm_infer", names::batch_norm_infer,
                            schema_field_idx::BatchNorm);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_norm_train", names::batch_norm_train,
                value2schema::BatchNorm);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.batch_norm_train", names::batch_norm_train,
                            schema_field_idx::BatchNorm);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.batch_norm_train_dxwb", names::batch_norm_train_dxwb,
                value2schema::BatchNormTrainDxwb);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX(
    "mnm.op.batch_norm_train_dxwb", names::batch_norm_train_dxwb,
    schema_field_idx::BatchNormTrainDxwb);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.bias_add", names::bias_add,
                value2schema::BiasAdd);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.bias_add", names::bias_add,
                            schema_field_idx::BiasAdd);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.broadcast_to", names::broadcast_to,
                value2schema::BroadcastTo);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.broadcast_to", names::broadcast_to,
                            schema_field_idx::BroadcastTo);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.broadcast_to_like", names::broadcast_to_like,
                value2schema::BroadcastToLike);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.broadcast_to_like", names::broadcast_to_like,
                            schema_field_idx::BroadcastToLike);   // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cast", names::cast, value2schema::Cast);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cast", names::cast,
                            schema_field_idx::Cast);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cast_like", names::cast_like,
                value2schema::CastLike);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cast_like", names::cast_like,
                            schema_field_idx::CastLike);           // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.ceil", names::ceil, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.ceil", names::ceil,
                            schema_field_idx::Unary);             // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.clip", names::clip, value2schema::Clip);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.clip", names::clip,
                            schema_field_idx::Clip);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.clip_dx", names::clip_dx,
                value2schema::ClipDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.clip_dx", names::clip_dx,
                            schema_field_idx::ClipDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.collapse_sum_like", names::collapse_sum_like,
                value2schema::CollapseLike);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.collapse_sum_like", names::collapse_sum_like,
                            schema_field_idx::CollapseLike);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.compiler_begin", names::compiler_begin,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.compiler_begin", names::compiler_begin,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.compiler_end", names::compiler_end,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.compiler_end", names::compiler_end,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.concatenate", names::concatenate,
                value2schema::Concatenate);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.concatenate", names::concatenate,
                            schema_field_idx::Concatenate);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.concatenate_dx", names::concatenate_dx,
                value2schema::Concatenate);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.concatenate_dx", names::concatenate_dx,
                            schema_field_idx::Concatenate);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.conv2d", names::conv2d,
                value2schema::Conv);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.conv2d", names::conv2d,
                            schema_field_idx::Conv);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.conv2d_dw", names::conv2d_dw,
                value2schema::ConvDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.conv2d_dw", names::conv2d_dw,
                            schema_field_idx::ConvDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.conv2d_dx", names::conv2d_dx,
                value2schema::ConvDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.conv2d_dx", names::conv2d_dx,
                            schema_field_idx::ConvDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.conv2d_transpose", names::conv2d_transpose,
                value2schema::ConvTrans);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.conv2d_transpose", names::conv2d_transpose,
                            schema_field_idx::ConvTrans);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.conv2d_transpose_dw", names::conv2d_transpose_dw,
                value2schema::ConvTransposeDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.conv2d_transpose_dw", names::conv2d_transpose_dw,
                            schema_field_idx::ConvTransposeDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.conv2d_transpose_dx", names::conv2d_transpose_dx,
                value2schema::ConvTransposeDxw);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.conv2d_transpose_dx", names::conv2d_transpose_dx,
                            schema_field_idx::ConvTransposeDxw);   // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.copy", names::copy, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.copy", names::copy,
                            schema_field_idx::Unary);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cos", names::cos, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cos", names::cos,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cross_entropy", names::cross_entropy,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cross_entropy", names::cross_entropy,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cross_entropy_dpred", names::cross_entropy_dpred,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cross_entropy_dpred", names::cross_entropy_dpred,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cross_entropy_dtrue", names::cross_entropy_dtrue,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cross_entropy_dtrue", names::cross_entropy_dtrue,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.cumsum", names::cumsum,
                value2schema::Cumsum);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.cumsum", names::cumsum,
                            schema_field_idx::Cumsum);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.dense", names::dense,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.dense", names::dense,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.device_copy", names::device_copy,
                value2schema::DeviceCopy);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.device_copy", names::device_copy,
                            schema_field_idx::DeviceCopy);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.divide", names::divide,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.divide", names::divide,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.embedding", names::embedding,
                value2schema::Embedding);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.embedding", names::embedding,
                            schema_field_idx::Embedding);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.embedding_dx", names::embedding_dx,
                value2schema::EmbeddingDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.embedding_dx", names::embedding_dx,
                            schema_field_idx::EmbeddingDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.equal", names::equal,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.equal", names::equal,
                            schema_field_idx::Binary);           // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.erf", names::erf, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.erf", names::erf,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.erf_dx", names::erf_dx,
                value2schema::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.erf_dx", names::erf_dx,
                            schema_field_idx::UnaryDx);          // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.exp", names::exp, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.exp", names::exp,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.expand_dims", names::expand_dims,
                value2schema::ExpandDims);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.expand_dims", names::expand_dims,
                            schema_field_idx::ExpandDims);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.floor", names::floor,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.floor", names::floor,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.floor_divide", names::floor_divide,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.floor_divide", names::floor_divide,
                            schema_field_idx::Binary);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.full", names::full, value2schema::Full);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.full", names::full,
                            schema_field_idx::Full);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.full_like", names::full_like,
                value2schema::FullLike);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.full_like", names::full_like,
                            schema_field_idx::FullLike);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.gather", names::gather,
                value2schema::Gather);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.gather", names::gather,
                            schema_field_idx::Gather);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.gather_dx", names::gather_dx,
                value2schema::GatherDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.gather_dx", names::gather_dx,
                            schema_field_idx::GatherDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.gather_nd", names::gather_nd,
                value2schema::GatherNd);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.gather_nd", names::gather_nd,
                            schema_field_idx::GatherNd);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.gather_nd_dx", names::gather_nd_dx,
                value2schema::GatherNdDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.gather_nd_dx", names::gather_nd_dx,
                            schema_field_idx::GatherNdDx);         // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.gelu", names::gelu, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.gelu", names::gelu,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.gelu_dx", names::gelu_dx,
                value2schema::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.gelu_dx", names::gelu_dx,
                            schema_field_idx::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.get_kept_dims", names::get_kept_dims,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.get_kept_dims", names::get_kept_dims,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.get_reduce_axis", names::get_reduce_axis,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.get_reduce_axis", names::get_reduce_axis,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.get_valid_counts", names::get_valid_counts,
                value2schema::GetValidCounts);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.get_valid_counts", names::get_valid_counts,
                            schema_field_idx::GetValidCounts);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.greater", names::greater,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.greater", names::greater,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.greater_equal", names::greater_equal,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.greater_equal", names::greater_equal,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.layer_norm", names::layer_norm,
                value2schema::LayerNorm);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.layer_norm", names::layer_norm,
                            schema_field_idx::LayerNorm);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.layer_norm_dx", names::layer_norm_dx,
                value2schema::LayerNormDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.layer_norm_dx", names::layer_norm_dx,
                            schema_field_idx::LayerNormDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.left_shift", names::left_shift,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.left_shift", names::left_shift,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.less", names::less,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.less", names::less,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.less_equal", names::less_equal,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.less_equal", names::less_equal,
                            schema_field_idx::Binary);           // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.log", names::log, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.log", names::log,
                            schema_field_idx::Unary);              // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.log2", names::log2, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.log2", names::log2,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.log_softmax", names::log_softmax,
                value2schema::Softmax);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.log_softmax", names::log_softmax,
                            schema_field_idx::Softmax);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.log_softmax_dx", names::log_softmax_dx,
                value2schema::SoftmaxDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.log_softmax_dx", names::log_softmax_dx,
                            schema_field_idx::SoftmaxDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.logical_and", names::logical_and,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.logical_and", names::logical_and,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.logical_not", names::logical_not,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.logical_not", names::logical_not,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.matmul", names::matmul,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.matmul", names::matmul,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.matmul_nt", names::matmul_nt,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.matmul_nt", names::matmul_nt,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.matmul_tn", names::matmul_tn,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.matmul_tn", names::matmul_tn,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.matmul_tt", names::matmul_tt,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.matmul_tt", names::matmul_tt,
                            schema_field_idx::Binary);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.max", names::max, value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.max", names::max,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.max_pool2d", names::max_pool2d,
                value2schema::Pool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.max_pool2d", names::max_pool2d,
                            schema_field_idx::Pool);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.max_pool2d_dx", names::max_pool2d_dx,
                value2schema::PoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.max_pool2d_dx", names::max_pool2d_dx,
                            schema_field_idx::PoolDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.maximum", names::maximum,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.maximum", names::maximum,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.mean", names::mean,
                value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.mean", names::mean,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.mean_dx", names::mean_dx,
                value2schema::MeanDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.mean_dx", names::mean_dx,
                            schema_field_idx::MeanDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.mesh_grid", names::mesh_grid,
                value2schema::MeshGrid);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.mesh_grid", names::mesh_grid,
                            schema_field_idx::MeshGrid);          // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.min", names::min, value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.min", names::min,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.minimum", names::minimum,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.minimum", names::minimum,
                            schema_field_idx::Binary);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.mod", names::mod, value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.mod", names::mod,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.multiply", names::multiply,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.multiply", names::multiply,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.ndarray_size", names::ndarray_size,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.ndarray_size", names::ndarray_size,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.negative", names::negative,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.negative", names::negative,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.nll_loss", names::nll_loss,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.nll_loss", names::nll_loss,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.nll_loss_dpred", names::nll_loss_dpred,
                value2schema::LossDtp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.nll_loss_dpred", names::nll_loss_dpred,
                            schema_field_idx::LossDtp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.nll_loss_dtrue", names::nll_loss_dtrue,
                value2schema::LossDtp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.nll_loss_dtrue", names::nll_loss_dtrue,
                            schema_field_idx::LossDtp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.non_max_suppression", names::non_max_suppression,
                value2schema::NonMaxSuppression);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.non_max_suppression", names::non_max_suppression,
                            schema_field_idx::NonMaxSuppression);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.not_equal", names::not_equal,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.not_equal", names::not_equal,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.one_hot", names::one_hot,
                value2schema::OneHot);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.one_hot", names::one_hot,
                            schema_field_idx::OneHot);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.ones", names::ones,
                value2schema::InitOp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.ones", names::ones,
                            schema_field_idx::InitOp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.ones_like", names::ones_like,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.ones_like", names::ones_like,
                            schema_field_idx::Unary);          // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.pad", names::pad, value2schema::Pad);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.pad", names::pad,
                            schema_field_idx::Pad);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.power", names::power,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.power", names::power,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.prod", names::prod,
                value2schema::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.prod", names::prod,
                            schema_field_idx::Reduce);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.prod_dx", names::prod_dx,
                value2schema::ProdDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.prod_dx", names::prod_dx,
                            schema_field_idx::ProdDx);             // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.relu", names::relu, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.relu", names::relu,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.relu_dx", names::relu_dx,
                value2schema::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.relu_dx", names::relu_dx,
                            schema_field_idx::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.repeat", names::repeat,
                value2schema::Repeat);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.repeat", names::repeat,
                            schema_field_idx::Repeat);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.repeat_dx", names::repeat_dx,
                value2schema::RepeatDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.repeat_dx", names::repeat_dx,
                            schema_field_idx::RepeatDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.reshape", names::reshape,
                value2schema::Reshape);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.reshape", names::reshape,
                            schema_field_idx::Reshape);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.resize2d", names::resize2d,
                value2schema::Resize2D);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.resize2d", names::resize2d,
                            schema_field_idx::Resize2D);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.resize2d_dx", names::resize2d_dx,
                value2schema::Resize2DDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.resize2d_dx", names::resize2d_dx,
                            schema_field_idx::Resize2DDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.reverse", names::reverse,
                value2schema::Reverse);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.reverse", names::reverse,
                            schema_field_idx::Reverse);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.reverse_sequence", names::reverse_sequence,
                value2schema::ReverseSequence);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.reverse_sequence", names::reverse_sequence,
                            schema_field_idx::ReverseSequence);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.right_shift", names::right_shift,
                value2schema::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.right_shift", names::right_shift,
                            schema_field_idx::Binary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.roi_align", names::roi_align,
                value2schema::RoiAlign);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.roi_align", names::roi_align,
                            schema_field_idx::RoiAlign);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.roi_align_dx", names::roi_align_dx,
                value2schema::RoiAlignDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.roi_align_dx", names::roi_align_dx,
                            schema_field_idx::RoiAlignDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.round", names::round,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.round", names::round,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.rsqrt", names::rsqrt,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.rsqrt", names::rsqrt,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.scatter", names::scatter,
                value2schema::Scatter);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.scatter", names::scatter,
                            schema_field_idx::Scatter);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.scatter_dx", names::scatter_dx,
                value2schema::ScatterDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.scatter_dx", names::scatter_dx,
                            schema_field_idx::ScatterDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sequence_mask", names::sequence_mask,
                value2schema::SequenceMask);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sequence_mask", names::sequence_mask,
                            schema_field_idx::SequenceMask);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.set_stream", names::set_stream,
                value2schema::SetStream);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.set_stream", names::set_stream,
                            schema_field_idx::SetStream);      // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sgd", names::sgd, value2schema::Sgd);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sgd", names::sgd,
                            schema_field_idx::Sgd);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.shape", names::shape,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.shape", names::shape,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sigmoid", names::sigmoid,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sigmoid", names::sigmoid,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sigmoid_dx", names::sigmoid_dx,
                value2schema::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sigmoid_dx", names::sigmoid_dx,
                            schema_field_idx::UnaryDx);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sign", names::sign, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sign", names::sign,
                            schema_field_idx::Unary);            // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sin", names::sin, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sin", names::sin,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.smooth_l1_loss", names::smooth_l1_loss,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.smooth_l1_loss", names::smooth_l1_loss,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.smooth_l1_loss_dpred", names::smooth_l1_loss_dpred,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.smooth_l1_loss_dpred", names::smooth_l1_loss_dpred,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.smooth_l1_loss_dtrue", names::smooth_l1_loss_dtrue,
                value2schema::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.smooth_l1_loss_dtrue", names::smooth_l1_loss_dtrue,
                            schema_field_idx::Loss);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.softmax", names::softmax,
                value2schema::Softmax);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.softmax", names::softmax,
                            schema_field_idx::Softmax);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.softmax_dx", names::softmax_dx,
                value2schema::SoftmaxDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.softmax_dx", names::softmax_dx,
                            schema_field_idx::SoftmaxDx);         // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sort", names::sort, value2schema::Sort);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sort", names::sort,
                            schema_field_idx::Sort);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.split", names::split,
                value2schema::Split);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.split", names::split,
                            schema_field_idx::Split);              // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sqrt", names::sqrt, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sqrt", names::sqrt,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sqrt_dx", names::sqrt_dx,
                value2schema::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sqrt_dx", names::sqrt_dx,
                            schema_field_idx::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.squeeze", names::squeeze,
                value2schema::Squeeze);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.squeeze", names::squeeze,
                            schema_field_idx::Squeeze);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.stack", names::stack,
                value2schema::Stack);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.stack", names::stack,
                            schema_field_idx::Stack);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.stream_barrier", names::stream_barrier,
                value2schema::StreamBarrier);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.stream_barrier", names::stream_barrier,
                            schema_field_idx::StreamBarrier);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.stream_sync", names::stream_sync,
                value2schema::Stream);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.stream_sync", names::stream_sync,
                            schema_field_idx::Stream);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.strided_slice", names::strided_slice,
                value2schema::StridedSlice);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.strided_slice", names::strided_slice,
                            schema_field_idx::StridedSlice);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.strided_slice_dx", names::strided_slice_dx,
                value2schema::StridedSliceDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.strided_slice_dx", names::strided_slice_dx,
                            schema_field_idx::StridedSliceDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.subtract", names::subtract,
                value2schema::BinaryUfunc);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.subtract", names::subtract,
                            schema_field_idx::BinaryUfunc);    // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sum", names::sum, value2schema::Sum);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sum", names::sum,
                            schema_field_idx::Sum);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.sum_dx", names::sum_dx,
                value2schema::SumDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.sum_dx", names::sum_dx,
                            schema_field_idx::SumDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.swap_axis", names::swap_axis,
                value2schema::SwapAxis);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.swap_axis", names::swap_axis,
                            schema_field_idx::SwapAxis);          // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.take", names::take, value2schema::Take);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.take", names::take,
                            schema_field_idx::Take);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.take_dx", names::take_dx,
                value2schema::TakeDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.take_dx", names::take_dx,
                            schema_field_idx::TakeDx);             // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.tanh", names::tanh, value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.tanh", names::tanh,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.tanh_dx", names::tanh_dx,
                value2schema::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.tanh_dx", names::tanh_dx,
                            schema_field_idx::UnaryDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.threefry_generate", names::threefry_generate,
                value2schema::ThreefryGenerate);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.threefry_generate", names::threefry_generate,
                            schema_field_idx::ThreefryGenerate);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.threefry_split", names::threefry_split,
                value2schema::ThreefrySplit);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.threefry_split", names::threefry_split,
                            schema_field_idx::ThreefrySplit);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.threshold", names::threshold,
                value2schema::Threshold);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.threshold", names::threshold,
                            schema_field_idx::Threshold);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.threshold_dx", names::threshold_dx,
                value2schema::ThresholdDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.threshold_dx", names::threshold_dx,
                            schema_field_idx::ThresholdDx);       // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.topk", names::topk, value2schema::Topk);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.topk", names::topk,
                            schema_field_idx::Topk);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.transpose", names::transpose,
                value2schema::Transpose);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.transpose", names::transpose,
                            schema_field_idx::Transpose);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.transpose_dx", names::transpose_dx,
                value2schema::TransposeDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.transpose_dx", names::transpose_dx,
                            schema_field_idx::TransposeDx);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.trunc", names::trunc,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.trunc", names::trunc,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.upper_bound.argwhere", names::upper_bound_argwhere,
                value2schema::Argwhere);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.upper_bound.argwhere", names::upper_bound_argwhere,
                            schema_field_idx::Argwhere);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.vm.alloc_storage", names::vm_alloc_storage,
                value2schema::AllocStorage);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.vm.alloc_storage", names::vm_alloc_storage,
                            schema_field_idx::AllocStorage);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.vm.alloc_tensor", names::vm_alloc_tensor,
                value2schema::AllocTensor);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.vm.alloc_tensor", names::vm_alloc_tensor,
                            schema_field_idx::AllocTensor);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.vm.free", names::vm_free,
                value2schema::Free);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.vm.free", names::vm_free,
                            schema_field_idx::Free);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.vm.infer_type", names::vm_infer_type,
                value2schema::InferType);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.vm.infer_type", names::vm_infer_type,
                            schema_field_idx::InferType);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.vm.invoke_op", names::vm_invoke_op,
                value2schema::InvokeOp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.vm.invoke_op", names::vm_invoke_op,
                            schema_field_idx::InvokeOp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.vm.set_shape", names::vm_set_shape,
                value2schema::SetShape);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.vm.set_shape", names::vm_set_shape,
                            schema_field_idx::SetShape);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.wait_event", names::wait_event,
                value2schema::Event);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.wait_event", names::wait_event,
                            schema_field_idx::Event);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.where", names::where,
                value2schema::Where);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.where", names::where,
                            schema_field_idx::Where);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.zeros", names::zeros,
                value2schema::InitOp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.zeros", names::zeros,
                            schema_field_idx::InitOp);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA("mnm.op.zeros_like", names::zeros_like,
                value2schema::Unary);  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.zeros_like", names::zeros_like,
                            schema_field_idx::Unary);  // NOLINT(whitespace/line_length)

#undef MNM_BIND_SCHEMA
#undef MNM_BIND_SCHEMA_FIELD_INDEX

}  // namespace f_mnm_schema
}  // namespace regs
}  // namespace op
}  // namespace mnm

// The last part: registering schemas
namespace mnm {
namespace op {
namespace schema {
namespace {
MNM_REGISTER_OBJECT_REFLECT(ListArgs);
MNM_REGISTER_OBJECT_REFLECT(AdaptivePoolArgs);
MNM_REGISTER_OBJECT_REFLECT(AdaptivePoolDxArgs);
MNM_REGISTER_OBJECT_REFLECT(AdvIndexArgs);
MNM_REGISTER_OBJECT_REFLECT(AdvIndexDxArgs);
MNM_REGISTER_OBJECT_REFLECT(AllgatherArgs);
MNM_REGISTER_OBJECT_REFLECT(AllocStorageArgs);
MNM_REGISTER_OBJECT_REFLECT(AllocTensorArgs);
MNM_REGISTER_OBJECT_REFLECT(AllreduceArgs);
MNM_REGISTER_OBJECT_REFLECT(ArangeArgs);
MNM_REGISTER_OBJECT_REFLECT(ArgsortArgs);
MNM_REGISTER_OBJECT_REFLECT(ArgwhereArgs);
MNM_REGISTER_OBJECT_REFLECT(BatchNormArgs);
MNM_REGISTER_OBJECT_REFLECT(BatchNormTrainDxwbArgs);
MNM_REGISTER_OBJECT_REFLECT(BiasAddArgs);
MNM_REGISTER_OBJECT_REFLECT(BinaryArgs);
MNM_REGISTER_OBJECT_REFLECT(BinaryDxArgs);
MNM_REGISTER_OBJECT_REFLECT(BinaryUfuncArgs);
MNM_REGISTER_OBJECT_REFLECT(BroadcastArgs);
MNM_REGISTER_OBJECT_REFLECT(BroadcastToArgs);
MNM_REGISTER_OBJECT_REFLECT(BroadcastToLikeArgs);
MNM_REGISTER_OBJECT_REFLECT(CastArgs);
MNM_REGISTER_OBJECT_REFLECT(CastLikeArgs);
MNM_REGISTER_OBJECT_REFLECT(ClipArgs);
MNM_REGISTER_OBJECT_REFLECT(ClipDxArgs);
MNM_REGISTER_OBJECT_REFLECT(CollapseLikeArgs);
MNM_REGISTER_OBJECT_REFLECT(CommReduceArgs);
MNM_REGISTER_OBJECT_REFLECT(ConcatenateArgs);
MNM_REGISTER_OBJECT_REFLECT(ConvArgs);
MNM_REGISTER_OBJECT_REFLECT(ConvDxwArgs);
MNM_REGISTER_OBJECT_REFLECT(ConvTransArgs);
MNM_REGISTER_OBJECT_REFLECT(ConvTransposeDxwArgs);
MNM_REGISTER_OBJECT_REFLECT(CumsumArgs);
MNM_REGISTER_OBJECT_REFLECT(DeviceCopyArgs);
MNM_REGISTER_OBJECT_REFLECT(DropoutArgs);
MNM_REGISTER_OBJECT_REFLECT(DropoutDxArgs);
MNM_REGISTER_OBJECT_REFLECT(EmbeddingArgs);
MNM_REGISTER_OBJECT_REFLECT(EmbeddingDxArgs);
MNM_REGISTER_OBJECT_REFLECT(EventArgs);
MNM_REGISTER_OBJECT_REFLECT(ExpandDimsArgs);
MNM_REGISTER_OBJECT_REFLECT(FreeArgs);
MNM_REGISTER_OBJECT_REFLECT(FullArgs);
MNM_REGISTER_OBJECT_REFLECT(FullLikeArgs);
MNM_REGISTER_OBJECT_REFLECT(GatherArgs);
MNM_REGISTER_OBJECT_REFLECT(GatherDxArgs);
MNM_REGISTER_OBJECT_REFLECT(GatherNdArgs);
MNM_REGISTER_OBJECT_REFLECT(GatherNdDxArgs);
MNM_REGISTER_OBJECT_REFLECT(GetValidCountsArgs);
MNM_REGISTER_OBJECT_REFLECT(InferTypeArgs);
MNM_REGISTER_OBJECT_REFLECT(InitOpArgs);
MNM_REGISTER_OBJECT_REFLECT(InvokeOpArgs);
MNM_REGISTER_OBJECT_REFLECT(LayerNormArgs);
MNM_REGISTER_OBJECT_REFLECT(LayerNormDxArgs);
MNM_REGISTER_OBJECT_REFLECT(LocalResponseNormArgs);
MNM_REGISTER_OBJECT_REFLECT(LossArgs);
MNM_REGISTER_OBJECT_REFLECT(LossDtpArgs);
MNM_REGISTER_OBJECT_REFLECT(MeanDxArgs);
MNM_REGISTER_OBJECT_REFLECT(MeshGridArgs);
MNM_REGISTER_OBJECT_REFLECT(NonMaxSuppressionArgs);
MNM_REGISTER_OBJECT_REFLECT(OneHotArgs);
MNM_REGISTER_OBJECT_REFLECT(PadArgs);
MNM_REGISTER_OBJECT_REFLECT(PoolArgs);
MNM_REGISTER_OBJECT_REFLECT(PoolDxArgs);
MNM_REGISTER_OBJECT_REFLECT(ProdDxArgs);
MNM_REGISTER_OBJECT_REFLECT(RecvArgs);
MNM_REGISTER_OBJECT_REFLECT(ReduceArgs);
MNM_REGISTER_OBJECT_REFLECT(ReduceScatterArgs);
MNM_REGISTER_OBJECT_REFLECT(RepeatArgs);
MNM_REGISTER_OBJECT_REFLECT(RepeatDxArgs);
MNM_REGISTER_OBJECT_REFLECT(ReshapeArgs);
MNM_REGISTER_OBJECT_REFLECT(Resize2DArgs);
MNM_REGISTER_OBJECT_REFLECT(Resize2DDxArgs);
MNM_REGISTER_OBJECT_REFLECT(ReverseArgs);
MNM_REGISTER_OBJECT_REFLECT(ReverseSequenceArgs);
MNM_REGISTER_OBJECT_REFLECT(RoiAlignArgs);
MNM_REGISTER_OBJECT_REFLECT(RoiAlignDxArgs);
MNM_REGISTER_OBJECT_REFLECT(ScatterArgs);
MNM_REGISTER_OBJECT_REFLECT(ScatterDxArgs);
MNM_REGISTER_OBJECT_REFLECT(SendArgs);
MNM_REGISTER_OBJECT_REFLECT(SequenceMaskArgs);
MNM_REGISTER_OBJECT_REFLECT(SetShapeArgs);
MNM_REGISTER_OBJECT_REFLECT(SetStreamArgs);
MNM_REGISTER_OBJECT_REFLECT(SgdArgs);
MNM_REGISTER_OBJECT_REFLECT(SoftmaxArgs);
MNM_REGISTER_OBJECT_REFLECT(SoftmaxDxArgs);
MNM_REGISTER_OBJECT_REFLECT(SortArgs);
MNM_REGISTER_OBJECT_REFLECT(SplitArgs);
MNM_REGISTER_OBJECT_REFLECT(SqueezeArgs);
MNM_REGISTER_OBJECT_REFLECT(StackArgs);
MNM_REGISTER_OBJECT_REFLECT(StreamArgs);
MNM_REGISTER_OBJECT_REFLECT(StreamBarrierArgs);
MNM_REGISTER_OBJECT_REFLECT(StridedSliceArgs);
MNM_REGISTER_OBJECT_REFLECT(StridedSliceDxArgs);
MNM_REGISTER_OBJECT_REFLECT(SumArgs);
MNM_REGISTER_OBJECT_REFLECT(SumDxArgs);
MNM_REGISTER_OBJECT_REFLECT(SwapAxisArgs);
MNM_REGISTER_OBJECT_REFLECT(TakeArgs);
MNM_REGISTER_OBJECT_REFLECT(TakeDxArgs);
MNM_REGISTER_OBJECT_REFLECT(TernaryArgs);
MNM_REGISTER_OBJECT_REFLECT(TernaryDxArgs);
MNM_REGISTER_OBJECT_REFLECT(TernaryUfuncArgs);
MNM_REGISTER_OBJECT_REFLECT(ThreefryGenerateArgs);
MNM_REGISTER_OBJECT_REFLECT(ThreefrySplitArgs);
MNM_REGISTER_OBJECT_REFLECT(ThresholdArgs);
MNM_REGISTER_OBJECT_REFLECT(ThresholdDxArgs);
MNM_REGISTER_OBJECT_REFLECT(TopkArgs);
MNM_REGISTER_OBJECT_REFLECT(TransposeArgs);
MNM_REGISTER_OBJECT_REFLECT(TransposeDxArgs);
MNM_REGISTER_OBJECT_REFLECT(UnaryArgs);
MNM_REGISTER_OBJECT_REFLECT(UnaryDxArgs);
MNM_REGISTER_OBJECT_REFLECT(UnaryUfuncArgs);
MNM_REGISTER_OBJECT_REFLECT(WhereArgs);
}  // namespace
}  // namespace schema
}  // namespace op
}  // namespace mnm
