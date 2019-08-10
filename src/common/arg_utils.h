#pragma once

#include <vector>

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace common {
namespace arg_utils {

inline std::vector<const DLTensor*> AsVector(ir::Array<value::Value> a) {
  int n = a.size();
  std::vector<const DLTensor*> res(n);
  for (int i = 0; i < n; ++i) {
    res[i] = a[i];
  }
  return res;
}

inline DType DeduceDLType(const std::vector<const DLTensor*>& v) {
  DType res = v[0]->dtype;
  for (int i = 1, e = v.size(); i < e; ++i) {
    if (res != v[i]->dtype) {
      throw;
    }
  }
  return res;
}

inline Context DeduceCtx(const std::vector<const DLTensor*>& v) {
  Context res = v[0]->ctx;
  for (int i = 1, e = v.size(); i < e; ++i) {
    if (res != v[i]->ctx) {
      throw;
    }
  }
  return res;
}

}  // namespace arg_utils
}  // namespace common
}  // namespace mnm
