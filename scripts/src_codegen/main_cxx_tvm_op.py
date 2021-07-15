from .codegen_utils import write_to_file
from ..op_def import topi


def gen_file(filename):
    FILE = """
/*!
 * Copyright (c) 2020 by Contributors
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
    return REG.format(MNM_OP_NAME=mnm_op_name,
                      RELAY_OP_NAME=relay_op_name)


def main(path="./src/op/regs/tvm_op_regs.cc"):
    result = gen_file(path)
    write_to_file(path, result)


if __name__ == "__main__":
    main()
