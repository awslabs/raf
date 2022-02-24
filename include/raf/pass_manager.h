/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/pass_manager.h
 * \brief This file implements the sequential pass for raf.
 */

#pragma once

#include <tvm/ir/transform.h>

#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"

namespace raf {
namespace pass {

using namespace raf::ir;
using tvm::transform::Pass;
using tvm::transform::PassInfo;

class RAFSequentialNode;

class RAFSequential : public Pass {
 public:
  /*!
   * \brief The constructor of `RAFSequential`.
   *
   * \param passes The passes to apply.
   * \param pass_info The pass metadata.
   */
  RAFSequential(Array<Pass> passes, PassInfo pass_info);

  /*!
   * \brief The constructor of `RAFSequential`.
   *
   * \param passes The passes to apply.
   * \param name The name of a sequential pass. It's defaulted to "sequential".
   *        This allows users to only provide a list of passes and execute them
   *        under a given context.
   */
  RAFSequential(Array<Pass> passes, String name = "sequential");

  RAFSequential() = default;
  explicit RAFSequential(ObjectPtr<Object> n) : Pass(n) {
  }

  const RAFSequentialNode* operator->() const;
  using ContainerType = RAFSequential;
};

}  // namespace pass
}  // namespace raf
