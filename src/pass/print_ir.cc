/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file print_ir.cc
 * \brief Printing out the IR to LOG(INFO) in a sequential pass.
 */
#include "raf/ir_ext.h"
#include "raf/pass.h"

namespace raf {
namespace pass {

Pass PrintIR(const std::string& header, bool show_meta_data) {
  auto pass_func = [header, show_meta_data](ir::IRModule mod, const PassContext& ctx) {
    LOG(INFO) << "PrintIR(" << header << "):\n" << ir::AsText(mod, show_meta_data);
    return mod;
  };
  return CreateModulePass(pass_func, 0, "PrintIR", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.PrintIR").set_body_typed(PrintIR);

}  // namespace pass
}  // namespace raf
