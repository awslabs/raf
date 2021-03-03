/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/ty/vm.cc
 * \brief Typing of vm dialect operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "mnm/ir_ext.h"
#include "../schema/vm.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace schema;
using namespace tvm;
using namespace tvm::relay;
using tvm::relay::Type;

Type AllocStorageInfer(const CallValues& value) {
  const auto* args = value->args.as<AllocStorageArgs>();
  CHECK(args != nullptr);
  // Here we assign the type of alloc_storage call to be a scalar type. In
  // fact, it should be a `TypeCall`, but `Module` currently doesn't support
  // ``GlobalTypeVar`. The type is only used for passing type inference.
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType::Scalar(dtype);
}

MNM_OP_TYPE("mnm.op._alloc_storage", "AllocStorage", AllocStorageInfer);

Type AllocTensorInfer(const CallValues& value) {
  const auto* args = value->args.as<AllocTensorArgs>();
  CHECK(args != nullptr);
  // TODO(zhiics) Add the handling of const shape
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  TupleType tt = Downcast<TupleType>(GetType(args->shape));

  int64_t dims = tt->fields.size();
  auto assert_shape = args->assert_shape;
  ICHECK_EQ(assert_shape.size(), dims);
  Array<IndexExpr> out_shape;
  for (int64_t i = 0; i < dims; i++) {
    out_shape.push_back(tvm::Integer(assert_shape[i]));
  }
  return TensorType(out_shape, dtype);
}

MNM_OP_TYPE("mnm.op._alloc_tensor", "AllocTensor", AllocTensorInfer);

Type InvokeOpInfer(const CallValues& value) {
  // The return value of invoke op is implicitly written into the output tensor
  // passed as an arguement. Here we only return an empty type since no real
  // return value is used.
  return TupleType::Empty();
}
MNM_OP_TYPE("mnm.op._invoke_op", "InvokeOp", InvokeOpInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
