/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/op_utils.h
 * \brief Useful classes of storing op metadata.
 */
#pragma once

#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>
#include "./op.h"
#include "./value.h"

namespace raf {
namespace op {

using namespace raf::value;

template <int n>
static std::vector<int64_t> Pad(const std::vector<int64_t>& a) {
  int size = a.size();
  CHECK(size == 1 || size == n);
  return size == 1 ? std::vector<int64_t>(n, a[0]) : a;
}

static inline void GetPadHW(const std::vector<int64_t>& padding, int64_t* pad_h, int64_t* pad_w) {
  if (padding.size() == 1) {
    *pad_h = padding[0] * 2;
    *pad_w = padding[0] * 2;
  } else if (padding.size() == 2) {
    *pad_h = padding[0] * 2;
    *pad_w = padding[1] * 2;
  } else if (padding.size() == 4) {
    *pad_h = padding[0] + padding[2];
    *pad_w = padding[1] + padding[3];
  } else {
    CHECK_EQ(padding.size(), 4) << " Padding size should be 1, 2 or 4, but got " << padding.size();
  }
}

static inline void GetOutputPadHW(const std::vector<int64_t>& padding, int64_t* pad_h,
                                  int64_t* pad_w) {
  if (padding.size() == 1) {
    *pad_h = padding[0];
    *pad_w = padding[0];
  } else if (padding.size() == 2) {
    *pad_h = padding[0];
    *pad_w = padding[1];
  } else if (padding.size() == 4) {
    *pad_h = (padding[0] + padding[2]) / 2;
    *pad_w = (padding[1] + padding[3]) / 2;
  } else {
    LOG(FATAL) << " Padding size should be 1, 2 or 4, but got " << padding.size();
    throw;
  }
}

inline void GetAdaptivePoolKernel(int64_t ind, int64_t outd, int64_t* kernel_size, int64_t* stride,
                                  int64_t* padding) {
  CHECK_EQ(ind % outd, 0) << "Not supported: input dimension = " << ind
                          << ", output dimension = " << outd;
  *stride = ind / outd;
  *kernel_size = ind - (outd - 1) * *stride;
  *padding = 0;
}

template <class T>
inline std::vector<int64_t> ArrayToInt(const T& arr) {
  std::vector<int64_t> ret;
  for (const ObjectRef i : arr) {
    auto node = i.as<IntImmNode>();
    CHECK(node != nullptr) << "Array elemment " << i << " is not IntImmNode";
    int64_t val = node->value;
    ret.push_back(val);
  }
  return std::move(ret);
}

template <class T>
inline TupleValue ArrayToIntTuple(const T& arr) {
  Array<Value> ret;
  for (int64_t val : ArrayToInt(arr)) {
    ret.push_back(ScalarValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

template <>
inline TupleValue ArrayToIntTuple(const std::vector<int64_t>& arr) {
  Array<Value> ret;
  for (auto val : arr) {
    ret.push_back(ScalarValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

inline bool IsCollectiveOp(const Expr& op) {
  if (auto op_node = op.as<OpNode>()) {
    return op::GetOpAttrOrDefault<TRAFCollective>(GetRef<Op>(op_node), "TRAFCollective", false);
  }
  return false;
}

using OpSet = std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual>;

inline bool IsInOpSet(const Expr& op, const OpSet& op_set) {
  if (auto op_node = op.as<OpNode>()) {
    Op op_ = GetRef<Op>(op_node);
    Op op_n = IsDialectOp(op_) ? GetBaseOp(op_) : op_;
    return op_set.find(op_n) != op_set.end();
  }
  return false;
}

inline bool IsReshapeOp(const Op& op) {
  static std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual> reshape_ops{
      Op::Get("raf.op.reshape"), Op::Get("raf.op.expand_dims"), Op::Get("raf.op.squeeze"),
      Op::Get("raf.op.batch_flatten"), Op::Get("raf.op.reshape_like")};
  return IsInOpSet(op, reshape_ops);
}

inline bool IsNonDeterministicOp(const Op& op) {
  static std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual> non_deterministic_ops{
      Op::Get("raf.op._contrib_dropout"), Op::Get("raf.op._contrib_dropout_dx")};
  return IsInOpSet(op, non_deterministic_ops);
}

inline bool IsMemcpyOp(const Expr& op) {
  static OpSet memcpy_ops = {
      Op::Get("raf.op.fuse_tensor"),
      Op::Get("raf.op.defuse_tensor"),
  };
  return IsInOpSet(op, memcpy_ops);
}

inline bool IsFuseTensorOp(const Expr& op) {
  static OpSet fuse_tensor_ops = {
      Op::Get("raf.op.fuse_tensor"),
  };
  return IsInOpSet(op, fuse_tensor_ops);
}

inline bool IsDefuseTensorOp(const Expr& op) {
  static OpSet defuse_tensor_ops = {
      Op::Get("raf.op.defuse_tensor"),
  };
  return IsInOpSet(op, defuse_tensor_ops);
}

inline size_t GetSizeInBytes(const DLDataType& dtype) {
  return (dtype.bits + 7) / 8;
}

inline std::vector<int64_t> GetShapeVecFromValue(const Value& value) {
  ICHECK(value.defined());
  std::vector<int64_t> shape;
  if (const auto* scalar = value.as<IntValueObj>()) {
    shape.push_back(scalar->value);
  } else if (const auto* tup = value.as<TupleValueObj>()) {
    for (auto field : tup->fields) {
      shape.push_back(GetScalarValueData<int64_t>(field));
    }
  } else if (const auto* tv = value.as<TensorValueObj>()) {
    DLTensor* tensor = GetRef<TensorValue>(tv);
    ICHECK_EQ(tensor->ndim, 1U);
    ICHECK_EQ(tensor->dtype.code, 0U);
    ICHECK_EQ(tensor->dtype.bits, 32U);
    const int32_t* int_ptr = reinterpret_cast<int32_t*>(tensor->data);
    for (size_t i = 0; i < tensor->shape[0]; ++i) {
      shape.push_back(int_ptr[i]);
    }
  } else {
    LOG(FATAL) << "Unsupported value type " << value;
  }
  return shape;
}

inline Array<tvm::PrimExpr> GetShapeExprFromValue(const Value& value) {
  ICHECK(value.defined());
  Array<tvm::PrimExpr> shape;
  if (auto ttv = value.as<TensorTypeValueObj>()) {
    auto ndim = ttv->type->shape[0].as<ir::IntImmNode>();
    ICHECK(ndim) << "Expected IntImm, but got " << ttv->type->shape[0]->GetTypeKey();
    for (size_t i = 0; i < ndim->value; ++i) {
      shape.push_back(Any());
    }
  } else {
    std::vector<int64_t> shape_vec = GetShapeVecFromValue(value);
    for (auto i : shape_vec) {
      shape.push_back(tvm::Integer(i));
    }
  }
  return shape;
}

inline tvm::PrimExpr GetIntExprFromValue(const Value& value) {
  ICHECK(value.defined());
  if (auto tv = value.as<TensorTypeValueObj>()) {
    return Any();
  }
  return tvm::Integer(GetScalarValueData<int64_t>(value));
}

inline std::vector<int64_t> BroadcastShapeVec(const std::vector<int64_t>& x1,
                                              const std::vector<int64_t>& x2) {
  size_t ndim_1 = x1.size();
  size_t ndim_2 = x2.size();
  size_t ndim = std::max(ndim_1, ndim_2);
  std::vector<int64_t> oshape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t lhs = (i < ndim_1) ? x1[ndim_1 - 1 - i] : 1;
    int64_t rhs = (i < ndim_2) ? x2[ndim_2 - 1 - i] : 1;

    if (lhs == 1) {
      oshape[ndim - 1 - i] = rhs;
    } else if (rhs == 1) {
      oshape[ndim - 1 - i] = lhs;
    } else if (lhs == rhs) {
      oshape[ndim - 1 - i] = lhs;
    } else {
      LOG(FATAL) << "Cannot broadcast " << lhs << " and " << rhs;
    }
  }
  return oshape;
}

}  // namespace op
}  // namespace raf
