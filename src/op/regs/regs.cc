/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/regs.cc
 * \brief Auto generated. Do not touch.
 */
#include "./regs_utils.h"
#include "../schema/gemm.h"
#include "../schema/likes.h"
#include "../schema/loss.h"
#include "../schema/nn.h"
#include "../schema/ufunc.h"

namespace mnm {
namespace op {
namespace schema {
namespace {
MNM_REGISTER_OBJECT_REFLECT(BatchNormArgs);
MNM_REGISTER_OBJECT_REFLECT(BatchNormTrainDxwbArgs);
MNM_REGISTER_OBJECT_REFLECT(BiasAddArgs);
MNM_REGISTER_OBJECT_REFLECT(BinaryArgs);
MNM_REGISTER_OBJECT_REFLECT(BinaryDxArgs);
MNM_REGISTER_OBJECT_REFLECT(BinaryUfuncArgs);
MNM_REGISTER_OBJECT_REFLECT(CollapseLikeArgs);
MNM_REGISTER_OBJECT_REFLECT(ConvArgs);
MNM_REGISTER_OBJECT_REFLECT(ConvDxwArgs);
MNM_REGISTER_OBJECT_REFLECT(LocalResponseNormArgs);
MNM_REGISTER_OBJECT_REFLECT(LossArgs);
MNM_REGISTER_OBJECT_REFLECT(LossDxArgs);
MNM_REGISTER_OBJECT_REFLECT(MatmulArgs);
MNM_REGISTER_OBJECT_REFLECT(PoolArgs);
MNM_REGISTER_OBJECT_REFLECT(PoolDxArgs);
MNM_REGISTER_OBJECT_REFLECT(ReshapeLikeArgs);
MNM_REGISTER_OBJECT_REFLECT(SoftmaxArgs);
MNM_REGISTER_OBJECT_REFLECT(SoftmaxDxArgs);
MNM_REGISTER_OBJECT_REFLECT(TernaryArgs);
MNM_REGISTER_OBJECT_REFLECT(TernaryDxArgs);
MNM_REGISTER_OBJECT_REFLECT(TernaryUfuncArgs);
MNM_REGISTER_OBJECT_REFLECT(UnaryArgs);
MNM_REGISTER_OBJECT_REFLECT(UnaryDxArgs);
MNM_REGISTER_OBJECT_REFLECT(UnaryUfuncArgs);
}  // namespace
}  // namespace schema
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace args {
using namespace mnm::ir;
using namespace mnm::value;
#define MNM_REQUIRED(i, norm, name) attrs->name = norm(values[i]);
#define MNM_OPTIONAL(i, norm, name) \
  if (size > i) attrs->name = norm(values[i]);
Attrs BatchNorm(const Array<Value> &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 7);
  auto attrs = make_object<schema::BatchNormArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToTensor, running_mean);
  MNM_REQUIRED(2, args::ToTensor, running_var);
  MNM_OPTIONAL(3, args::ToTensor, w);
  MNM_OPTIONAL(4, args::ToTensor, b);
  MNM_OPTIONAL(5, args::ToDouble, momentum);
  MNM_OPTIONAL(6, args::ToDouble, eps);
  return Attrs(attrs);
}
Attrs BatchNormTrainDxwb(const Array<Value> &values) {
  const int size = values.size();
  CHECK(5 <= size && size <= 5);
  auto attrs = make_object<schema::BatchNormTrainDxwbArgs>();
  MNM_REQUIRED(0, args::ToTensor, dy);
  MNM_REQUIRED(1, args::ToTensor, x);
  MNM_REQUIRED(2, args::ToTensor, w);
  MNM_REQUIRED(3, args::ToTensor, b);
  MNM_REQUIRED(4, args::ToDouble, eps);
  return Attrs(attrs);
}
Attrs BiasAdd(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 3);
  auto attrs = make_object<schema::BiasAddArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToTensor, b);
  MNM_OPTIONAL(2, args::ToInt, axis);
  return Attrs(attrs);
}
Attrs Binary(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  auto attrs = make_object<schema::BinaryArgs>();
  MNM_REQUIRED(0, args::ToAny, x1);
  MNM_REQUIRED(1, args::ToAny, x2);
  return Attrs(attrs);
}
Attrs BinaryDx(const Array<Value> &values) {
  const int size = values.size();
  CHECK(4 <= size && size <= 4);
  auto attrs = make_object<schema::BinaryDxArgs>();
  MNM_REQUIRED(0, args::ToAny, x1);
  MNM_REQUIRED(1, args::ToAny, x2);
  MNM_REQUIRED(2, args::ToTensor, y);
  MNM_REQUIRED(3, args::ToTensor, dy);
  return Attrs(attrs);
}
Attrs BinaryUfunc(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 4);
  auto attrs = make_object<schema::BinaryUfuncArgs>();
  MNM_REQUIRED(0, args::ToAny, x1);
  MNM_REQUIRED(1, args::ToAny, x2);
  MNM_OPTIONAL(2, args::ToAny, out);
  MNM_OPTIONAL(3, args::ToAny, where);
  return Attrs(attrs);
}
Attrs CollapseLike(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  auto attrs = make_object<schema::CollapseLikeArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToIntTuple, shape);
  return Attrs(attrs);
}
Attrs Conv(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 6);
  auto attrs = make_object<schema::ConvArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToTensor, w);
  MNM_OPTIONAL(2, args::ToIntTuple, stride);
  MNM_OPTIONAL(3, args::ToIntTuple, padding);
  MNM_OPTIONAL(4, args::ToIntTuple, dilation);
  MNM_OPTIONAL(5, args::ToInt, groups);
  return Attrs(attrs);
}
Attrs ConvDxw(const Array<Value> &values) {
  const int size = values.size();
  CHECK(8 <= size && size <= 8);
  auto attrs = make_object<schema::ConvDxwArgs>();
  MNM_REQUIRED(0, args::ToTensor, x_or_w);
  MNM_REQUIRED(1, args::ToTensor, y);
  MNM_REQUIRED(2, args::ToTensor, dy);
  MNM_REQUIRED(3, args::ToIntTuple, shape);
  MNM_REQUIRED(4, args::ToIntTuple, stride);
  MNM_REQUIRED(5, args::ToIntTuple, padding);
  MNM_REQUIRED(6, args::ToIntTuple, dilation);
  MNM_REQUIRED(7, args::ToInt, groups);
  return Attrs(attrs);
}
Attrs LocalResponseNorm(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 5);
  auto attrs = make_object<schema::LocalResponseNormArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToInt, size);
  MNM_OPTIONAL(2, args::ToDouble, alpha);
  MNM_OPTIONAL(3, args::ToDouble, beta);
  MNM_OPTIONAL(4, args::ToDouble, k);
  return Attrs(attrs);
}
Attrs Loss(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  auto attrs = make_object<schema::LossArgs>();
  MNM_REQUIRED(0, args::ToTensor, y_true);
  MNM_REQUIRED(1, args::ToTensor, y_pred);
  return Attrs(attrs);
}
Attrs LossDx(const Array<Value> &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 3);
  auto attrs = make_object<schema::LossDxArgs>();
  MNM_REQUIRED(0, args::ToTensor, loss);
  MNM_REQUIRED(1, args::ToTensor, y_true);
  MNM_REQUIRED(2, args::ToTensor, y_pred);
  return Attrs(attrs);
}
Attrs Matmul(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 4);
  auto attrs = make_object<schema::MatmulArgs>();
  MNM_REQUIRED(0, args::ToTensor, a);
  MNM_REQUIRED(1, args::ToTensor, b);
  MNM_OPTIONAL(2, args::ToBool, transpose_a);
  MNM_OPTIONAL(3, args::ToBool, transpose_b);
  return Attrs(attrs);
}
Attrs Pool(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 7);
  auto attrs = make_object<schema::PoolArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToIntTuple, kernel);
  MNM_OPTIONAL(2, args::ToOptionalIntTuple, stride);
  MNM_OPTIONAL(3, args::ToIntTuple, padding);
  MNM_OPTIONAL(4, args::ToIntTuple, dilation);
  MNM_OPTIONAL(5, args::ToBool, ceil_mode);
  MNM_OPTIONAL(6, args::ToBool, include_pad);
  return Attrs(attrs);
}
Attrs PoolDx(const Array<Value> &values) {
  const int size = values.size();
  CHECK(9 <= size && size <= 9);
  auto attrs = make_object<schema::PoolDxArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToTensor, y);
  MNM_REQUIRED(2, args::ToTensor, dy);
  MNM_REQUIRED(3, args::ToIntTuple, kernel);
  MNM_REQUIRED(4, args::ToIntTuple, stride);
  MNM_REQUIRED(5, args::ToIntTuple, padding);
  MNM_REQUIRED(6, args::ToIntTuple, dilation);
  MNM_REQUIRED(7, args::ToBool, ceil_mode);
  MNM_REQUIRED(8, args::ToBool, include_pad);
  return Attrs(attrs);
}
Attrs ReshapeLike(const Array<Value> &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  auto attrs = make_object<schema::ReshapeLikeArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToIntTuple, shape);
  return Attrs(attrs);
}
Attrs Softmax(const Array<Value> &values) {
  const int size = values.size();
  CHECK(1 <= size && size <= 2);
  auto attrs = make_object<schema::SoftmaxArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_OPTIONAL(1, args::ToInt, axis);
  return Attrs(attrs);
}
Attrs SoftmaxDx(const Array<Value> &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 4);
  auto attrs = make_object<schema::SoftmaxDxArgs>();
  MNM_REQUIRED(0, args::ToTensor, x);
  MNM_REQUIRED(1, args::ToTensor, y);
  MNM_REQUIRED(2, args::ToTensor, dy);
  MNM_OPTIONAL(3, args::ToInt, axis);
  return Attrs(attrs);
}
Attrs Ternary(const Array<Value> &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 3);
  auto attrs = make_object<schema::TernaryArgs>();
  MNM_REQUIRED(0, args::ToAny, x1);
  MNM_REQUIRED(1, args::ToAny, x2);
  MNM_REQUIRED(2, args::ToAny, x3);
  return Attrs(attrs);
}
Attrs TernaryDx(const Array<Value> &values) {
  const int size = values.size();
  CHECK(5 <= size && size <= 5);
  auto attrs = make_object<schema::TernaryDxArgs>();
  MNM_REQUIRED(0, args::ToAny, x1);
  MNM_REQUIRED(1, args::ToAny, x2);
  MNM_REQUIRED(2, args::ToAny, x3);
  MNM_REQUIRED(3, args::ToTensor, y);
  MNM_REQUIRED(4, args::ToTensor, dy);
  return Attrs(attrs);
}
Attrs TernaryUfunc(const Array<Value> &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 5);
  auto attrs = make_object<schema::TernaryUfuncArgs>();
  MNM_REQUIRED(0, args::ToAny, x1);
  MNM_REQUIRED(1, args::ToAny, x2);
  MNM_REQUIRED(2, args::ToAny, x3);
  MNM_OPTIONAL(3, args::ToAny, out);
  MNM_OPTIONAL(4, args::ToAny, where);
  return Attrs(attrs);
}
Attrs Unary(const Array<Value> &values) {
  const int size = values.size();
  CHECK(1 <= size && size <= 1);
  auto attrs = make_object<schema::UnaryArgs>();
  MNM_REQUIRED(0, args::ToAny, x);
  return Attrs(attrs);
}
Attrs UnaryDx(const Array<Value> &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 3);
  auto attrs = make_object<schema::UnaryDxArgs>();
  MNM_REQUIRED(0, args::ToAny, x);
  MNM_REQUIRED(1, args::ToTensor, y);
  MNM_REQUIRED(2, args::ToTensor, dy);
  return Attrs(attrs);
}
Attrs UnaryUfunc(const Array<Value> &values) {
  const int size = values.size();
  CHECK(1 <= size && size <= 3);
  auto attrs = make_object<schema::UnaryUfuncArgs>();
  MNM_REQUIRED(0, args::ToAny, x);
  MNM_OPTIONAL(1, args::ToAny, out);
  MNM_OPTIONAL(2, args::ToAny, where);
  return Attrs(attrs);
}
#undef MNM_OPTIONAL
#undef MNM_REQUIRED
}  // namespace args
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace ffi {
using namespace mnm::ir;
using namespace mnm::value;
using registry::TVMArgs;
#define MNM_REQUIRED(i, norm, name) result.push_back(norm(values[i]));
#define MNM_OPTIONAL(i, norm, name) \
  if (size > i) result.push_back(norm(values[i]));
Array<Expr> BatchNorm(const TVMArgs &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 7);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToTensor, running_mean);
  MNM_REQUIRED(2, ffi::ToTensor, running_var);
  MNM_OPTIONAL(3, ffi::ToTensor, w);
  MNM_OPTIONAL(4, ffi::ToTensor, b);
  MNM_OPTIONAL(5, ffi::ToDouble, momentum);
  MNM_OPTIONAL(6, ffi::ToDouble, eps);
  return Array<Expr>(result);
}
Array<Expr> BatchNormTrainDxwb(const TVMArgs &values) {
  const int size = values.size();
  CHECK(5 <= size && size <= 5);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, dy);
  MNM_REQUIRED(1, ffi::ToTensor, x);
  MNM_REQUIRED(2, ffi::ToTensor, w);
  MNM_REQUIRED(3, ffi::ToTensor, b);
  MNM_REQUIRED(4, ffi::ToDouble, eps);
  return Array<Expr>(result);
}
Array<Expr> BiasAdd(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 3);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToTensor, b);
  MNM_OPTIONAL(2, ffi::ToInt, axis);
  return Array<Expr>(result);
}
Array<Expr> Binary(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x1);
  MNM_REQUIRED(1, ffi::ToAny, x2);
  return Array<Expr>(result);
}
Array<Expr> BinaryDx(const TVMArgs &values) {
  const int size = values.size();
  CHECK(4 <= size && size <= 4);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x1);
  MNM_REQUIRED(1, ffi::ToAny, x2);
  MNM_REQUIRED(2, ffi::ToTensor, y);
  MNM_REQUIRED(3, ffi::ToTensor, dy);
  return Array<Expr>(result);
}
Array<Expr> BinaryUfunc(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 4);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x1);
  MNM_REQUIRED(1, ffi::ToAny, x2);
  MNM_OPTIONAL(2, ffi::ToAny, out);
  MNM_OPTIONAL(3, ffi::ToAny, where);
  return Array<Expr>(result);
}
Array<Expr> CollapseLike(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToIntTuple, shape);
  return Array<Expr>(result);
}
Array<Expr> Conv(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 6);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToTensor, w);
  MNM_OPTIONAL(2, ffi::ToIntTuple, stride);
  MNM_OPTIONAL(3, ffi::ToIntTuple, padding);
  MNM_OPTIONAL(4, ffi::ToIntTuple, dilation);
  MNM_OPTIONAL(5, ffi::ToInt, groups);
  return Array<Expr>(result);
}
Array<Expr> ConvDxw(const TVMArgs &values) {
  const int size = values.size();
  CHECK(8 <= size && size <= 8);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x_or_w);
  MNM_REQUIRED(1, ffi::ToTensor, y);
  MNM_REQUIRED(2, ffi::ToTensor, dy);
  MNM_REQUIRED(3, ffi::ToIntTuple, shape);
  MNM_REQUIRED(4, ffi::ToIntTuple, stride);
  MNM_REQUIRED(5, ffi::ToIntTuple, padding);
  MNM_REQUIRED(6, ffi::ToIntTuple, dilation);
  MNM_REQUIRED(7, ffi::ToInt, groups);
  return Array<Expr>(result);
}
Array<Expr> LocalResponseNorm(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 5);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToInt, size);
  MNM_OPTIONAL(2, ffi::ToDouble, alpha);
  MNM_OPTIONAL(3, ffi::ToDouble, beta);
  MNM_OPTIONAL(4, ffi::ToDouble, k);
  return Array<Expr>(result);
}
Array<Expr> Loss(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, y_true);
  MNM_REQUIRED(1, ffi::ToTensor, y_pred);
  return Array<Expr>(result);
}
Array<Expr> LossDx(const TVMArgs &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 3);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, loss);
  MNM_REQUIRED(1, ffi::ToTensor, y_true);
  MNM_REQUIRED(2, ffi::ToTensor, y_pred);
  return Array<Expr>(result);
}
Array<Expr> Matmul(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 4);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, a);
  MNM_REQUIRED(1, ffi::ToTensor, b);
  MNM_OPTIONAL(2, ffi::ToBool, transpose_a);
  MNM_OPTIONAL(3, ffi::ToBool, transpose_b);
  return Array<Expr>(result);
}
Array<Expr> Pool(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 7);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToIntTuple, kernel);
  MNM_OPTIONAL(2, ffi::ToOptionalIntTuple, stride);
  MNM_OPTIONAL(3, ffi::ToIntTuple, padding);
  MNM_OPTIONAL(4, ffi::ToIntTuple, dilation);
  MNM_OPTIONAL(5, ffi::ToBool, ceil_mode);
  MNM_OPTIONAL(6, ffi::ToBool, include_pad);
  return Array<Expr>(result);
}
Array<Expr> PoolDx(const TVMArgs &values) {
  const int size = values.size();
  CHECK(9 <= size && size <= 9);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToTensor, y);
  MNM_REQUIRED(2, ffi::ToTensor, dy);
  MNM_REQUIRED(3, ffi::ToIntTuple, kernel);
  MNM_REQUIRED(4, ffi::ToIntTuple, stride);
  MNM_REQUIRED(5, ffi::ToIntTuple, padding);
  MNM_REQUIRED(6, ffi::ToIntTuple, dilation);
  MNM_REQUIRED(7, ffi::ToBool, ceil_mode);
  MNM_REQUIRED(8, ffi::ToBool, include_pad);
  return Array<Expr>(result);
}
Array<Expr> ReshapeLike(const TVMArgs &values) {
  const int size = values.size();
  CHECK(2 <= size && size <= 2);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToIntTuple, shape);
  return Array<Expr>(result);
}
Array<Expr> Softmax(const TVMArgs &values) {
  const int size = values.size();
  CHECK(1 <= size && size <= 2);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_OPTIONAL(1, ffi::ToInt, axis);
  return Array<Expr>(result);
}
Array<Expr> SoftmaxDx(const TVMArgs &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 4);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToTensor, x);
  MNM_REQUIRED(1, ffi::ToTensor, y);
  MNM_REQUIRED(2, ffi::ToTensor, dy);
  MNM_OPTIONAL(3, ffi::ToInt, axis);
  return Array<Expr>(result);
}
Array<Expr> Ternary(const TVMArgs &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 3);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x1);
  MNM_REQUIRED(1, ffi::ToAny, x2);
  MNM_REQUIRED(2, ffi::ToAny, x3);
  return Array<Expr>(result);
}
Array<Expr> TernaryDx(const TVMArgs &values) {
  const int size = values.size();
  CHECK(5 <= size && size <= 5);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x1);
  MNM_REQUIRED(1, ffi::ToAny, x2);
  MNM_REQUIRED(2, ffi::ToAny, x3);
  MNM_REQUIRED(3, ffi::ToTensor, y);
  MNM_REQUIRED(4, ffi::ToTensor, dy);
  return Array<Expr>(result);
}
Array<Expr> TernaryUfunc(const TVMArgs &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 5);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x1);
  MNM_REQUIRED(1, ffi::ToAny, x2);
  MNM_REQUIRED(2, ffi::ToAny, x3);
  MNM_OPTIONAL(3, ffi::ToAny, out);
  MNM_OPTIONAL(4, ffi::ToAny, where);
  return Array<Expr>(result);
}
Array<Expr> Unary(const TVMArgs &values) {
  const int size = values.size();
  CHECK(1 <= size && size <= 1);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x);
  return Array<Expr>(result);
}
Array<Expr> UnaryDx(const TVMArgs &values) {
  const int size = values.size();
  CHECK(3 <= size && size <= 3);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x);
  MNM_REQUIRED(1, ffi::ToTensor, y);
  MNM_REQUIRED(2, ffi::ToTensor, dy);
  return Array<Expr>(result);
}
Array<Expr> UnaryUfunc(const TVMArgs &values) {
  const int size = values.size();
  CHECK(1 <= size && size <= 3);
  std::vector<Expr> result;
  MNM_REQUIRED(0, ffi::ToAny, x);
  MNM_OPTIONAL(1, ffi::ToAny, out);
  MNM_OPTIONAL(2, ffi::ToAny, where);
  return Array<Expr>(result);
}
#undef MNM_OPTIONAL
#undef MNM_REQUIRED
}  // namespace ffi
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace args {
#define MNM_BIND_SCHEMA(op_name, schema_name) \
  MNM_OP_REGISTER(op_name).set_attr<::mnm::op::FMNMSchema>("FMNMSchema", schema_name);
MNM_BIND_SCHEMA("mnm.op.abs", args::Unary);
MNM_BIND_SCHEMA("mnm.op.add", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.avg_pool2d", args::Pool);
MNM_BIND_SCHEMA("mnm.op.avg_pool2d_dx", args::PoolDx);
MNM_BIND_SCHEMA("mnm.op.batch_flatten", args::Unary);
MNM_BIND_SCHEMA("mnm.op.batch_norm_infer", args::BatchNorm);
MNM_BIND_SCHEMA("mnm.op.batch_norm_train", args::BatchNorm);
MNM_BIND_SCHEMA("mnm.op.batch_norm_train_dxwb", args::BatchNormTrainDxwb);
MNM_BIND_SCHEMA("mnm.op.bias_add", args::BiasAdd);
MNM_BIND_SCHEMA("mnm.op.ceil", args::Unary);
MNM_BIND_SCHEMA("mnm.op.collapse_sum_like", args::CollapseLike);
MNM_BIND_SCHEMA("mnm.op.conv2d", args::Conv);
MNM_BIND_SCHEMA("mnm.op.conv2d_dw", args::ConvDxw);
MNM_BIND_SCHEMA("mnm.op.conv2d_dx", args::ConvDxw);
MNM_BIND_SCHEMA("mnm.op.copy", args::Unary);
MNM_BIND_SCHEMA("mnm.op.cos", args::Unary);
MNM_BIND_SCHEMA("mnm.op.divide", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.equal", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.floor", args::Unary);
MNM_BIND_SCHEMA("mnm.op.greater", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.greater_equal", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.less", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.less_equal", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.log", args::Unary);
MNM_BIND_SCHEMA("mnm.op.log_softmax", args::Softmax);
MNM_BIND_SCHEMA("mnm.op.log_softmax_dx", args::SoftmaxDx);
MNM_BIND_SCHEMA("mnm.op.logical_not", args::UnaryUfunc);
MNM_BIND_SCHEMA("mnm.op.matmul", args::Matmul);
MNM_BIND_SCHEMA("mnm.op.max_pool2d", args::Pool);
MNM_BIND_SCHEMA("mnm.op.max_pool2d_dx", args::PoolDx);
MNM_BIND_SCHEMA("mnm.op.mod", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.multiply", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.negative", args::UnaryUfunc);
MNM_BIND_SCHEMA("mnm.op.nll_loss", args::Loss);
MNM_BIND_SCHEMA("mnm.op.nll_loss_dpred", args::LossDx);
MNM_BIND_SCHEMA("mnm.op.nll_loss_dtrue", args::LossDx);
MNM_BIND_SCHEMA("mnm.op.not_equal", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.relu", args::Unary);
MNM_BIND_SCHEMA("mnm.op.relu_dx", args::UnaryDx);
MNM_BIND_SCHEMA("mnm.op.reshape_like", args::ReshapeLike);
MNM_BIND_SCHEMA("mnm.op.sigmoid", args::Unary);
MNM_BIND_SCHEMA("mnm.op.sigmoid_dx", args::UnaryDx);
MNM_BIND_SCHEMA("mnm.op.softmax", args::Softmax);
MNM_BIND_SCHEMA("mnm.op.softmax_dx", args::SoftmaxDx);
MNM_BIND_SCHEMA("mnm.op.subtract", args::BinaryUfunc);
MNM_BIND_SCHEMA("mnm.op.tanh", args::Unary);
MNM_BIND_SCHEMA("mnm.op.tanh_dx", args::UnaryDx);
#undef MNM_BIND_SCHEMA
}  // namespace args
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace ffi {
using registry::TVMArgs;
using registry::TVMRetValue;
MNM_REGISTER_GLOBAL("mnm.op.sym.abs")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.abs");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.add")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.add");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.avg_pool2d")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.avg_pool2d");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Pool(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.avg_pool2d_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.avg_pool2d_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::PoolDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_flatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_flatten");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_norm_infer")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_norm_infer");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BatchNorm(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_norm_train")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_norm_train");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BatchNorm(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.batch_norm_train_dxwb")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_norm_train_dxwb");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BatchNormTrainDxwb(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.bias_add")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.bias_add");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BiasAdd(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.ceil")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.ceil");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.collapse_sum_like")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.collapse_sum_like");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::CollapseLike(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.conv2d");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Conv(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_dw")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.conv2d_dw");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::ConvDxw(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.conv2d_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.conv2d_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::ConvDxw(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.copy")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.copy");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.cos")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.cos");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.divide")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.divide");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.equal");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.floor")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.floor");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.greater")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.greater");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.greater_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.greater_equal");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.less")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.less");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.less_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.less_equal");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.log")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.log");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.log_softmax")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.log_softmax");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Softmax(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.log_softmax_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.log_softmax_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::SoftmaxDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.logical_not")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.logical_not");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::UnaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.matmul");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Matmul(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.max_pool2d")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.max_pool2d");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Pool(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.max_pool2d_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.max_pool2d_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::PoolDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.mod")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.mod");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.multiply")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.multiply");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.negative")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.negative");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::UnaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.nll_loss")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.nll_loss");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Loss(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.nll_loss_dpred")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.nll_loss_dpred");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::LossDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.nll_loss_dtrue")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.nll_loss_dtrue");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::LossDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.not_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.not_equal");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.relu")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.relu");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.relu_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.relu_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::UnaryDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.reshape_like")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.reshape_like");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::ReshapeLike(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.sigmoid")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.sigmoid");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.sigmoid_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.sigmoid_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::UnaryDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.softmax")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.softmax");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Softmax(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.softmax_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.softmax_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::SoftmaxDx(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.subtract")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.subtract");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::BinaryUfunc(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.tanh")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.tanh");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::Unary(args)),
                                               NullValue<Value>());
});
MNM_REGISTER_GLOBAL("mnm.op.sym.tanh_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.tanh_dx");
  *ret = binding::BindExprValue(CallNode::make(op, ffi::UnaryDx(args)),
                                               NullValue<Value>());
});
}  // namespace ffi
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace ffi {
using registry::TVMArgs;
using registry::TVMRetValue;
using executor::interpreter::Interpret;
MNM_REGISTER_GLOBAL("mnm.op.imp.abs")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.abs");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.add")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.add");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.avg_pool2d")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.avg_pool2d");
  *ret = Interpret(CallNode::make(op, ffi::Pool(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.avg_pool2d_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.avg_pool2d_dx");
  *ret = Interpret(CallNode::make(op, ffi::PoolDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.batch_flatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_flatten");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.batch_norm_infer")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_norm_infer");
  *ret = Interpret(CallNode::make(op, ffi::BatchNorm(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.batch_norm_train")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_norm_train");
  *ret = Interpret(CallNode::make(op, ffi::BatchNorm(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.batch_norm_train_dxwb")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.batch_norm_train_dxwb");
  *ret = Interpret(CallNode::make(op, ffi::BatchNormTrainDxwb(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.bias_add")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.bias_add");
  *ret = Interpret(CallNode::make(op, ffi::BiasAdd(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.ceil")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.ceil");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.collapse_sum_like")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.collapse_sum_like");
  *ret = Interpret(CallNode::make(op, ffi::CollapseLike(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.conv2d");
  *ret = Interpret(CallNode::make(op, ffi::Conv(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_dw")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.conv2d_dw");
  *ret = Interpret(CallNode::make(op, ffi::ConvDxw(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.conv2d_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.conv2d_dx");
  *ret = Interpret(CallNode::make(op, ffi::ConvDxw(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.copy")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.copy");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.cos")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.cos");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.divide")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.divide");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.equal");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.floor")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.floor");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.greater")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.greater");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.greater_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.greater_equal");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.less")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.less");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.less_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.less_equal");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.log")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.log");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.log_softmax")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.log_softmax");
  *ret = Interpret(CallNode::make(op, ffi::Softmax(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.log_softmax_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.log_softmax_dx");
  *ret = Interpret(CallNode::make(op, ffi::SoftmaxDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.logical_not")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.logical_not");
  *ret = Interpret(CallNode::make(op, ffi::UnaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.matmul");
  *ret = Interpret(CallNode::make(op, ffi::Matmul(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.max_pool2d")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.max_pool2d");
  *ret = Interpret(CallNode::make(op, ffi::Pool(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.max_pool2d_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.max_pool2d_dx");
  *ret = Interpret(CallNode::make(op, ffi::PoolDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.mod")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.mod");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.multiply")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.multiply");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.negative")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.negative");
  *ret = Interpret(CallNode::make(op, ffi::UnaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.nll_loss")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.nll_loss");
  *ret = Interpret(CallNode::make(op, ffi::Loss(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.nll_loss_dpred")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.nll_loss_dpred");
  *ret = Interpret(CallNode::make(op, ffi::LossDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.nll_loss_dtrue")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.nll_loss_dtrue");
  *ret = Interpret(CallNode::make(op, ffi::LossDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.not_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.not_equal");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.relu")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.relu");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.relu_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.relu_dx");
  *ret = Interpret(CallNode::make(op, ffi::UnaryDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.reshape_like")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.reshape_like");
  *ret = Interpret(CallNode::make(op, ffi::ReshapeLike(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.sigmoid")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.sigmoid");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.sigmoid_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.sigmoid_dx");
  *ret = Interpret(CallNode::make(op, ffi::UnaryDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.softmax")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.softmax");
  *ret = Interpret(CallNode::make(op, ffi::Softmax(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.softmax_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.softmax_dx");
  *ret = Interpret(CallNode::make(op, ffi::SoftmaxDx(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.subtract")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.subtract");
  *ret = Interpret(CallNode::make(op, ffi::BinaryUfunc(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.tanh")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.tanh");
  *ret = Interpret(CallNode::make(op, ffi::Unary(args)));
});
MNM_REGISTER_GLOBAL("mnm.op.imp.tanh_dx")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  static Op op = Op::Get("mnm.op.tanh_dx");
  *ret = Interpret(CallNode::make(op, ffi::UnaryDx(args)));
});
}  // namespace ffi
}  // namespace op
}  // namespace mnm
