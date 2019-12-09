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
inline std::vector<T> GetShape(const DLTensor& tensor) {
  return std::vector<T>(tensor.shape, tensor.shape + tensor.ndim);
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

inline bool IsCompact(const DLTensor& dl_tensor) {
  int ndim = dl_tensor.ndim;
  if (ndim == 0) {
    return true;
  }
  if (dl_tensor.byte_offset != 0) {
    return false;
  }
  if (dl_tensor.strides[ndim - 1] != 1) {
    return false;
  }
  for (int i = 0; i < ndim - 1; ++i) {
    if (dl_tensor.strides[i] != dl_tensor.strides[i + 1] * dl_tensor.shape[i + 1]) {
      return false;
    }
  }
  return true;
}

inline int64_t BytesCompactTensor(const DLTensor& dl_tensor) {
  CHECK(IsCompact(dl_tensor));
  if (dl_tensor.ndim) {
    return (dl_tensor.shape[0] * dl_tensor.strides[0] * dl_tensor.dtype.bits - 1) / 8 + 1;
  }
  return (dl_tensor.dtype.bits - 1) / 8 + 1;
}

}  // namespace shape_utils
}  // namespace common
}  // namespace mnm
