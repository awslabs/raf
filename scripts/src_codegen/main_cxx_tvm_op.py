# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from .codegen_utils import write_to_file
from ..op_def import topi


def gen_file(filename):
    FILE = """
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
 * Auto generated. Do not touch.
 * \\file {FILENAME}
 * \\brief Register TVM ops.
 */
#include "mnm/ir.h"
#include "mnm/op.h"
namespace mnm {{
namespace op {{
namespace {{
using ir::Array;
using ir::Attrs;
using ir::Op;
using ir::Type;
using tvm::Target;
using tvm::te::Schedule;
using tvm::te::Tensor;
using tvm::relay::FTVMCompute;
using tvm::relay::FTVMSchedule;
#define MNM_TVM_OP(MNM_OP, OP)                                                                  \\
  MNM_REGISTER_OP(MNM_OP)                                                                       \\
      .set_attr<FTVMCompute>("FTVMCompute",                                                     \\
                             [](const Attrs& attrs, const Array<Tensor>& inputs,                \\
                                const Type& out_type) -> Array<Tensor> {{                        \\
                               auto fcompute =                                                  \\
                                   Op::GetAttrMap<FTVMCompute>("FTVMCompute")[Op::Get(OP)];     \\
                               return fcompute(attrs, inputs, out_type);                        \\
                             }})                                                                 \\
      .set_attr<FTVMSchedule>(                                                                  \\
          "FTVMSchedule",                                                                       \\
          [](const Attrs& attrs, const Array<Tensor>& outs, const Target& target) -> Schedule {{ \\
            auto fschedule = Op::GetAttrMap<FTVMSchedule>("FTVMSchedule")[Op::Get(OP)];         \\
            return fschedule(attrs, outs, target);                                              \\
          }})

{REGS}
}}  // namespace
}}  // namespace op
}}  // namespace mnm
""".strip()
    regs = []
    for mnm_op_name in sorted(topi.OP_MAP.keys()):
        relay_op_name, _, _ = topi.OP_MAP[mnm_op_name]
        regs.append(gen_reg(mnm_op_name, relay_op_name))
    regs = "\n".join(regs)
    return FILE.format(REGS=regs, FILENAME=filename)


def gen_reg(mnm_op_name, relay_op_name):
    REG = """
MNM_TVM_OP("{MNM_OP_NAME}", "{RELAY_OP_NAME}");
""".strip()
    return REG.format(MNM_OP_NAME=mnm_op_name, RELAY_OP_NAME=relay_op_name)


def main(path="./src/op/regs/tvm_op_regs.cc"):
    result = gen_file(path)
    write_to_file(path, result)


if __name__ == "__main__":
    main()
