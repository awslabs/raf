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
 * \file print_ir.cc
 * \brief Printing out the IR to LOG(INFO) in a sequential pass.
 */
#include "mnm/ir_ext.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {

Pass PrintIR(const std::string& header, bool show_meta_data) {
  auto pass_func = [header, show_meta_data](ir::IRModule mod, const PassContext& ctx) {
    LOG(INFO) << "PrintIR(" << header << "):\n" << ir::AsText(mod, show_meta_data);
    return mod;
  };
  return CreateModulePass(pass_func, 0, "PrintIR", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.PrintIR").set_body_typed(PrintIR);

}  // namespace pass
}  // namespace mnm
