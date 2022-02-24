/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/algorithm.cc
 * \brief Algorithm-related operators bridged from TVM.
 */
#include <raf/op_utils.h>
#include <raf/value.h>
#include <array>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/nn.h"
#include "../../schema/algorithm.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;

std::vector<Value> ArgsortSchema2Args(const ArgsortArgs* args) {
  return {args->data};
}

std::vector<std::string> ArgsortSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs ArgsortSchema2Attrs(const ArgsortArgs* args) {
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = args->axis;
  attrs->is_ascend = args->is_ascend;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey ArgsortHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ArgsortArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->is_ascend;
  key << ir::String2DLDataType(args->dtype);
  return key;
}

RAF_TVM(argsort, Argsort, ArgsortArgs, ArgsortSchema2Args, ArgsortSchemaArgNames,
        ArgsortSchema2Attrs, ArgsortHasher, kOpaque);

std::vector<Value> SortSchema2Args(const SortArgs* args) {
  return {args->data};
}

std::vector<std::string> SortSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs SortSchema2Attrs(const SortArgs* args) {
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = args->axis;
  attrs->is_ascend = args->is_ascend;
  return Attrs(attrs);
}

HashKey SortHasher(const std::vector<Type>& param_types, const Type& y_type, const SortArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->is_ascend;
  return key;
}

RAF_TVM(sort, Sort, SortArgs, SortSchema2Args, SortSchemaArgNames, SortSchema2Attrs, SortHasher,
        kOpaque);

std::vector<Value> TopkSchema2Args(const TopkArgs* args) {
  return {args->data};
}

std::vector<std::string> TopkSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs TopkSchema2Attrs(const TopkArgs* args) {
  auto attrs = make_object<TopKAttrs>();
  int64_t k = args->k.defined() ? GetScalarValueData<int64_t>(args->k) : 1;
  attrs->k = IntImm(tvm::runtime::DataType::Int(64), k);
  attrs->axis = args->axis;
  attrs->ret_type = args->ret_type;
  attrs->is_ascend = args->is_ascend;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey TopkHasher(const std::vector<Type>& param_types, const Type& y_type, const TopkArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (!args->k.defined()) {
    key << int64_t(1);
  } else {
    key << GetScalarValueData<int64_t>(args->k);
  }
  key << args->axis;
  key << args->ret_type;
  key << args->is_ascend;
  key << ir::String2DLDataType(args->dtype);
  return key;
}

RAF_TVM(topk, Topk, TopkArgs, TopkSchema2Args, TopkSchemaArgNames, TopkSchema2Attrs, TopkHasher,
        kOpaque);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
