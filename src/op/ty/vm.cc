/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/vm.cc
 * \brief Typing of vm dialect operators
 */
#include "raf/type.h"
#include "raf/value.h"
#include "raf/ir_ext.h"
#include "../schema/vm.h"
#include "./utils.h"

namespace raf {
namespace op {
namespace type {

using namespace raf::ir;
using namespace schema;
using namespace raf::value;

Type AllocStorageInfer(const CallValues& value) {
  const auto* args = value->args.as<AllocStorageArgs>();
  CHECK(args != nullptr);
  // Here we assign the type of alloc_storage call to be a scalar type. In
  // fact, it should be a `TypeCall`, but `Module` currently doesn't support
  // ``GlobalTypeVar`. The type is only used for passing type inference.
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType::Scalar(dtype);
}

RAF_OP_TYPE("raf.op.vm.alloc_storage", "AllocStorage", AllocStorageInfer);

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

RAF_OP_TYPE("raf.op.vm.alloc_tensor", "AllocTensor", AllocTensorInfer);

Type EmptyTypeInfer(const CallValues& value) {
  // The return value of certain ops are implicitly written into the output tensor
  // passed as an arguement. Here we only return an empty type since no real
  // return value is used.
  return TupleType::Empty();
}

RAF_OP_TYPE("raf.op.vm.free", "Free", EmptyTypeInfer);
RAF_OP_TYPE("raf.op.vm.invoke_op", "InvokeOp", EmptyTypeInfer);

Type InferTypeInfer(const CallValues& value) {
  static auto fschema = Op::GetAttrMap<op::FRAFSchema>("FRAFSchema");
  const auto* args = value->args.as<InferTypeArgs>();
  CHECK(args != nullptr);
  Type ret_type;
  if (const auto* func = args->func.as<FunctionNode>()) {
    FuncType fty = Downcast<FuncType>(func->checked_type());
    ret_type = fty->ret_type;
  } else {
    OpValue opv = Downcast<OpValue>(args->func);
    auto inputs = Downcast<TupleValue>(args->inputs)->fields;
    auto call_values = CallValues::make(opv, fschema[opv->op](inputs));
    auto fty = Downcast<FuncType>(opv->op->checked_type());
    TypeInference ti = Downcast<TypeInference>(fty->type_constraints[0]);
    ret_type = ti->func(call_values);
  }
  // use fake type for raf values
  Array<Type> ret_tup;
  static auto fake_type = TensorType::Scalar(DataType::Int(64));
  if (ret_type->IsInstance<TensorTypeNode>()) {
    ret_tup.push_back(TupleType({fake_type, fake_type}));
  } else if (const auto* tup = ret_type.as<TupleTypeNode>()) {
    for (size_t i = 0; i < tup->fields.size(); ++i) {
      ret_tup.push_back(TupleType({fake_type, fake_type}));
    }
  }
  return TupleType(ret_tup);
}

RAF_OP_TYPE("raf.op.vm.infer_type", "InferType", InferTypeInfer);

Type SetShapeInfer(const CallValues& value) {
  const auto* args = value->args.as<SetShapeArgs>();
  CHECK(args != nullptr);

  if (const auto tuple = args->shape.as<TupleValueObj>()) {
    // Return the true type if shape is a constant tuple
    Array<PrimExpr> shape;
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      shape.push_back(tvm::Integer(Downcast<IntValue>(tuple->fields[i])->value));
    }
    TensorType data_type = Downcast<TensorType>(GetType(args->data));
    return TensorType(shape, data_type->dtype);
  }

  // Otherwise just return a fake type
  return TensorType::Scalar(DataType::Int(64));
}

RAF_OP_TYPE("raf.op.vm.set_shape", "SetShape", SetShapeInfer);

}  // namespace type
}  // namespace op
}  // namespace raf
