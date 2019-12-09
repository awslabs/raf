/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/generic/generic_utils.h
 * \brief Utility functions for generics
 */
#pragma once
#include "dmlc/logging.h"

namespace mnm {
namespace op {
namespace generic {
inline int NormalizeAxis(int axis, int ndim) {
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return axis < 0 ? axis + ndim : axis;
}
}  // namespace generic
}  // namespace op
}  // namespace mnm
