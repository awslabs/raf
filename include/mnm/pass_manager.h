/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/pass/pass_manager.h
 * \brief This file implements the sequential pass for meta.
 */

#pragma once

#include <tvm/ir/transform.h>

#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"

namespace mnm {
namespace pass {

using namespace mnm::ir;
using tvm::transform::Pass;
using tvm::transform::PassInfo;

class MNMSequentialNode;

class MNMSequential : public Pass {
 public:
  /*!
   * \brief The constructor of `MNMSequential`.
   *
   * \param passes The passes to apply.
   * \param pass_info The pass metadata.
   */
  MNMSequential(Array<Pass> passes, PassInfo pass_info);

  /*!
   * \brief The constructor of `MNMSequential`.
   *
   * \param passes The passes to apply.
   * \param name The name of a sequential pass. It's defaulted to "sequential".
   *        This allows users to only provide a list of passes and execute them
   *        under a given context.
   */
  MNMSequential(Array<Pass> passes, String name = "sequential");

  MNMSequential() = default;
  explicit MNMSequential(ObjectPtr<Object> n) : Pass(n) {
  }

  const MNMSequentialNode* operator->() const;
  using ContainerType = MNMSequential;
};

}  // namespace pass
}  // namespace mnm
