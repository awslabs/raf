/*!
 * Copyright (c) 2019 by Contributors
 * \file src/common/shape_utils.h
 * \brief Utilities for shape-related manipulation
 */
#pragma once
#include <vector>
#include "mnm/ir.h"

namespace mnm {
namespace common {
namespace shape_utils {

template <class T>
inline std::vector<T> GetShape(const DLTensor& dlt) {
  return std::vector<T>(dlt.shape, dlt.shape + dlt.ndim);
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

inline int64_t GetNumel(const DLTensor& dlt) {
  int64_t numel = 1;
  for (int i = 0; i < dlt.ndim; ++i) {
    numel *= dlt.shape[i];
  }
  return numel;
}

}  // namespace shape_utils
}  // namespace common
}  // namespace mnm
