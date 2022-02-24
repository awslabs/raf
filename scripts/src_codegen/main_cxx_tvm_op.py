# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .codegen_utils import write_to_file
from ..op_def import topi


def gen_file(filename):
    FILE = """
/*!
 * Auto generated. Do not touch.
 * \\file {FILENAME}
 * \\brief Register TVM ops.
 */
#include "raf/ir.h"
#include "raf/op.h"
namespace raf {{
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
#define RAF_TVM_OP(RAF_OP, OP)                                                                  \\
  RAF_REGISTER_OP(RAF_OP)                                                                       \\
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
}}  // namespace raf
""".strip()
    regs = []
    for raf_op_name in sorted(topi.OP_MAP.keys()):
        relay_op_name, _, _ = topi.OP_MAP[raf_op_name]
        regs.append(gen_reg(raf_op_name, relay_op_name))
    regs = "\n".join(regs)
    return FILE.format(REGS=regs, FILENAME=filename)


def gen_reg(raf_op_name, relay_op_name):
    REG = """
RAF_TVM_OP("{RAF_OP_NAME}", "{RELAY_OP_NAME}");
""".strip()
    return REG.format(RAF_OP_NAME=raf_op_name, RELAY_OP_NAME=relay_op_name)


def main(path="./src/op/regs/tvm_op_regs.cc"):
    result = gen_file(path)
    write_to_file(path, result)


if __name__ == "__main__":
    main()
