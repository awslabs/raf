/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/declare_utils.h
 * \brief Utility functions for declaration
 */
#pragma once
#include "dmlc/logging.h"

namespace mnm {
namespace op {
namespace declare {
inline int NormalizeAxis(int axis, int ndim) {
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return axis < 0 ? axis + ndim : axis;
}

}  // namespace declare
}  // namespace op
}  // namespace mnm
