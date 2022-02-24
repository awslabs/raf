/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/registry.cc
 * \brief RAF registry underlying implementation
 */
#include "raf/registry.h"

namespace raf {
namespace registry {

const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

}  // namespace registry
}  // namespace raf
