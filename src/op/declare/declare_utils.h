/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/declare_utils.h
 * \brief Utility functions for declaration
 */
#pragma once
#include "dmlc/logging.h"

namespace raf {
namespace op {
namespace declare {
inline int NormalizeAxis(int axis, int ndim) {
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return axis < 0 ? axis + ndim : axis;
}

template <typename T>
inline void DeclareGeneralDx(const CallValues& call) {
  using namespace raf::value;
  const auto* args = call->args.as<T>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}

}  // namespace declare
}  // namespace op
}  // namespace raf
