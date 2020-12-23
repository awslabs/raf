/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/likes.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::op::schema;

MNM_TVMJIT(Add, "mnm.op.add", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Subtract, "mnm.op.subtract", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Divide, "mnm.op.divide", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Multiply, "mnm.op.multiply", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Power, "mnm.op.power", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Greater, "mnm.op.greater", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Maximum, "mnm.op.maximum", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Minimum, "mnm.op.minimum", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);

struct SumAttrs : public tvm::AttrsNode<SumAttrs> {
  Array<Integer> axis;
  Array<Integer> keepdims;
  TVM_DECLARE_ATTRS(SumAttrs, "attrs.SumAttrs") {
    TVM_ATTR_FIELD(axis);
    TVM_ATTR_FIELD(keepdims);
  }
};
TVM_REGISTER_NODE_TYPE(SumAttrs);

std::vector<Value> SumSchema2Args(const SumArgs* args) {
  return {args->x};
}

std::vector<std::string> SumSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs SumSchema2Attrs(const SumArgs* args) {
  auto attrs = make_object<SumAttrs>();
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    attrs->axis.push_back(args->axis[i]);
  }
  for (int i = 0, n = args->keepdims.size(); i < n; ++i) {
    attrs->keepdims.push_back(args->keepdims[i]);
  }
  return Attrs(attrs);
}

HashKey SumHasher(const std::vector<Type>& param_types, const Type& ret_type, const SumArgs* args) {
  HashKey key = GenericHasher<std::nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
    key << args->keepdims[i];
  }
  return key;
}

MNM_TVMJIT(Sum, "mnm.op.sum", SumArgs, SumSchema2Args, SumSchemaArgNames, SumSchema2Attrs,
           SumHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
