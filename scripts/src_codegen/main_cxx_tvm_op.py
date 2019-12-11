from codegen_utils import write_to_file
from def_tvm_op import OP_MAP


def gen_file(filename):
    FILE = """
/*!
 * Copyright (c) 2019 by Contributors
 * \\file {FILENAME}
 * \\brief Auto generated. Do not touch.
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
using tvm::Schedule;
using tvm::Target;
using tvm::Tensor;
using tvm::relay::FTVMCompute;
using tvm::relay::FTVMSchedule;
using tvm::relay::OpPatternKind;
using tvm::relay::TOpPattern;
#define MNM_TVM_OP(MNM_OP, OP, PATTERN)                                                         \\
  MNM_OP_REGISTER(MNM_OP)                                                                       \\
      .set_attr<FTVMCompute>("FTVMCompute",                                                     \\
                             [](const Attrs& attrs, const Array<Tensor>& inputs,                \\
                                const Type& out_type, const Target& target) -> Array<Tensor> {{  \\
                               auto fcompute =                                                  \\
                                   Op::GetAttr<FTVMCompute>("FTVMCompute")[Op::Get(OP)];        \\
                               return fcompute(attrs, inputs, out_type, target);                \\
                             }})                                                                 \\
      .set_attr<FTVMSchedule>(                                                                  \\
          "FTVMSchedule",                                                                       \\
          [](const Attrs& attrs, const Array<Tensor>& outs, const Target& target) -> Schedule {{ \\
            auto fschedule = Op::GetAttr<FTVMSchedule>("FTVMSchedule")[Op::Get(OP)];            \\
            return fschedule(attrs, outs, target);                                              \\
          }})                                                                                    \\
      .set_attr<TOpPattern>("TOpPattern", tvm::relay::PATTERN);
{REGS}
}}  // namespace
}}  // namespace op
}}  // namespace mnm
""".strip()
    regs = []
    for mnm_op_name in sorted(OP_MAP.keys()):
        relay_op_name, _, pattern = OP_MAP[mnm_op_name]
        regs.append(gen_reg(mnm_op_name, relay_op_name, pattern))
    regs = "\n".join(regs)
    return FILE.format(REGS=regs, FILENAME=filename)


def gen_reg(mnm_op_name, relay_op_name, pattern):
    REG = """
MNM_TVM_OP("{MNM_OP_NAME}", "{RELAY_OP_NAME}", {PATTERN});
""".strip()
    return REG.format(MNM_OP_NAME=mnm_op_name,
                      RELAY_OP_NAME=relay_op_name,
                      PATTERN=pattern)


def main(path="./src/op/regs/tvmjit_regs.cc"):
    result = gen_file(path)
    write_to_file(path, result)


if __name__ == "__main__":
    main()
