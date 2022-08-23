/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/common/shape_utils.h
 * \brief Utilities for shape-related manipulation
 */
#pragma once
#include <vector>
#include "raf/ir_ext.h"
#include "raf/op.h"
#include "raf/value.h"

namespace raf {
namespace common {
namespace shape_utils {

template <class T>
inline std::vector<T> GetShape(const DLTensor& dlt) {
  return std::vector<T>(dlt.shape, dlt.shape + dlt.ndim);
}

inline std::vector<int64_t> GetShapeVecFromData(const DLTensor* shape) {
  std::vector<int64_t> raw_shape;
  ICHECK_EQ(shape->device.device_type, kDLCPU);
  ICHECK_EQ(shape->ndim, 1u);
  ICHECK_EQ(shape->dtype.code, 0U) << "The dtype of constant shape must be int32 or int64, but got "
                                   << tvm::runtime::DLDataType2String(shape->dtype);
  ICHECK(shape->dtype.bits == 64 || shape->dtype.bits == 32)
      << "The dtype of constant shape must be int32 or int64, but got"
      << tvm::runtime::DLDataType2String(shape->dtype);

  if (shape->dtype.bits == 32) {
    const int32_t* int_ptr = reinterpret_cast<int32_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  } else if (shape->dtype.bits == 64) {
    const int64_t* int_ptr = reinterpret_cast<int64_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  }
  return raw_shape;
}

template <typename T>
inline std::vector<T> MakeShape(const ir::Array<ir::Integer>& shape) {
  int ndim = shape.size();
  std::vector<T> result(ndim);
  for (int i = 0; i < ndim; ++i) {
    result[i] = shape[i]->value;
  }
  return result;
}

inline ir::Array<ir::Integer> StdVector2Array(const std::vector<int64_t>& shape) {
  ir::Array<ir::Integer> shape_;
  shape_.resize(shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    shape_.Set(i, shape[i]);
  }
  return shape_;
}

template <typename TDest, typename TSrc>
inline std::vector<TDest> Shape2Strides(const std::vector<TSrc>& shape) {
  int ndim = shape.size();
  std::vector<TDest> strides(ndim);
  int64_t carry = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = carry;
    carry *= shape[i];
  }
  return strides;
}

template <typename TDest, typename TSrc>
inline std::vector<TDest> PadDims(const std::vector<TSrc>& shape, int at_least_nd) {
  int n = shape.size();
  if (n >= at_least_nd) {
    std::vector<TDest> res(n);
    for (int i = 0; i < n; ++i) {
      res[i] = shape[i];
    }
    return res;
  }
  std::vector<TDest> res(at_least_nd);
  for (int i = 0; i < n; ++i) {
    res[i] = shape[i];
  }
  int padn = at_least_nd - n;
  for (int i = 0; i < padn; ++i) {
    res[i + n] = 1;
  }
  return res;
}

inline bool IsCompact(const DLTensor& dlt) {
  int ndim = dlt.ndim;
  if (dlt.byte_offset != 0) {
    return false;
  }
  if (ndim == 0) {
    return true;
  }
  if (dlt.strides == nullptr) {
    return true;
  }
  if (dlt.strides[ndim - 1] != 1) {
    return false;
  }
  for (int i = 0; i < ndim - 1; ++i) {
    if (dlt.strides[i] != dlt.strides[i + 1] * dlt.shape[i + 1]) {
      return false;
    }
  }
  return true;
}

inline int64_t BytesCompactTensor(const DLTensor& dlt) {
  CHECK(IsCompact(dlt));
  int64_t nbytes = (dlt.dtype.bits + 7) / 8;
  if (dlt.ndim == 0) {
    return nbytes;
  }
  if (dlt.strides != nullptr) {
    return nbytes * dlt.shape[0] * dlt.strides[0];
  }
  for (int i = 0; i < dlt.ndim; ++i) {
    nbytes *= dlt.shape[i];
  }
  return nbytes;
}

inline int64_t BytesCompactTensor(const ir::TensorTypeNode* type) {
  int64_t size = 1;
  for (auto dim : type->shape) {
    if (auto dim_imm = dim.as<ir::IntImmNode>()) {
      size *= dim_imm->value;
    } else {  // Dynamic shape
      return 0;
    }
  }
  size *= (type->dtype.bits() * type->dtype.lanes() + 7) / 8;
  return size;
}

inline int64_t GetNumel(const DLTensor& dlt) {
  int64_t numel = 1;
  for (int i = 0; i < dlt.ndim; ++i) {
    numel *= dlt.shape[i];
  }
  return numel;
}

inline int64_t GetSizeFromType(ir::Type ty) {
  if (auto tuple_type = ty.as<TupleTypeNode>()) {
    int64_t total_size = 0;
    for (auto field : tuple_type->fields) {
      auto size = GetSizeFromType(field);
      if (size == 0) {
        return 0;
      }
      total_size += size;
    }
    return total_size;
  } else if (auto ttype = ty.as<TensorTypeNode>()) {
    int64_t size = 1;
    for (auto axis : ttype->shape) {
      auto node = axis.as<ir::IntImmNode>();
      CHECK(node != nullptr) << "Axis " << axis << " is not IntImmNode";
      size *= (int64_t)node->value;
    }
    return size;
  }
  LOG(FATAL) << "Unsupported type: " << ty->GetTypeKey();
  throw;
}

inline int64_t GetDimSize(const Expr& expr, const int64_t dim) {
  auto ttype = expr->checked_type().as<TensorTypeNode>();
  ICHECK(ttype != nullptr);
  ICHECK_LT(dim, ttype->shape.size())
      << "Dim to access must be less than the shape size, but got " << dim << " (dim) and "
      << ttype->shape.size() << " (shape size)";
  return ttype->shape[dim].as<IntImmNode>()->value;
}

/*!
 * \brief Calculate the byte compact size of the given type. If the type is a tuple,
 * then the total size of summing up all tensors in the tuple will be returned.
 * Note that size 0 means a tensor with dynamic shape and cannot determine the size.
 * \param type The type to calculate the size of.
 * \return The size of the type in bytes.
 */
inline int64_t BytesCompactType(const Type& type) {
  if (auto tuple_type = type.as<TupleTypeNode>()) {
    int64_t total_size = 0;
    for (auto field : tuple_type->fields) {
      auto size = BytesCompactType(field);
      if (size == 0) {
        return 0;
      }
      total_size += size;
    }
    return total_size;
  } else if (auto ttype = type.as<TensorTypeNode>()) {
    return BytesCompactTensor(ttype);
  }
  LOG(FATAL) << "Unsupported type: " << type->GetTypeKey();
  throw;
}

inline int64_t GetElementNum(const Expr& var) {
  int64_t n;
  CHECK(var->checked_type_.defined());
  if (var->checked_type().as<TupleTypeNode>()) {
    n = 0;
    for (auto field : Downcast<Tuple>(var)->fields) {
      int64_t fn = GetElementNum(field);
      n += fn;
    }
  } else {
    n = 1;
    auto var_type = var->checked_type().as<TensorTypeNode>();
    CHECK(var_type != nullptr);
    for (int i = 0; i < var_type->shape.size(); ++i) {
      PrimExpr k = var_type->shape[i];
      int64_t k_v = k.as<IntImmNode>()->value;
      n *= k_v;
    }
  }
  return n;
}

}  // namespace shape_utils
}  // namespace common
}  // namespace raf
