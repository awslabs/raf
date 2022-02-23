/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/registry.cc
 * \brief MNM registry underlying implementation
 */
#include "mnm/registry.h"

namespace mnm {
namespace registry {

const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

}  // namespace registry
}  // namespace mnm
