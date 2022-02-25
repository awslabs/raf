/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dispatch/cutlass/singleton_ext.cu
 * \brief Extentions for cutlass singleton
 */
#include "cutlass_ext/library/singleton_ext.h"

namespace cutlass {
namespace library {

static std::unique_ptr<SingletonExt> instance;

SingletonExt::SingletonExt() {
  manifest.initialize();
  operation_table.append(manifest);
}

SingletonExt const & SingletonExt::get() {
  if (!instance.get()) {
    instance.reset(new SingletonExt);
  }
  return *instance.get();
}

} // namespace library
} // namespace cutlass
