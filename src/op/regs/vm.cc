/*!
 * Copyright (c) 2020 by Contributors
 * \file ./src/op/regs/vm.cc
 * \brief VM-related operators.
 */
#include "mnm/ir.h"
#include "mnm/op.h"
namespace mnm {
namespace op {

MNM_OP_REGISTER("mnm.op.vm.alloc_storage");
MNM_OP_REGISTER("mnm.op.vm.alloc_tensor");
MNM_OP_REGISTER("mnm.op.vm.invoke_op");

}  // namespace op
}  // namespace mnm
