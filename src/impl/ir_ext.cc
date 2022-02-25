/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/ir_ext.cc
 * \brief RAF extension to TVM/Relay IR.
 */
#include <printer/text_printer.h>
#include "raf/ir_ext.h"
#include "raf/registry.h"
#include "raf/pass.h"
#include "raf/value.h"

namespace raf {
namespace ir {

using namespace raf::value;

IRModule GlobalModule() {
  static IRModule inst = IRModule();
  return inst;
}

tvm::runtime::NDArray MakeFakeTensor() {
  static int64_t a[1] = {-114514};
  static int64_t b[1] = {1};
  Device dev = raf::Device(raf::DevType::kCPU(), 0);
  DType dtype = raf::DType(raf::DTypeCode::kInt(), 64, 1);
  DLTensor tensor;
  tensor.data = a;
  tensor.device = dev;
  tensor.dtype = dtype;
  tensor.shape = b;
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty({}, dtype, dev);
  array.CopyFrom(&tensor);
  return array;
}

bool ConstantNode::IsTensor() const {
  return value.defined() && value.as<BaseTensorValueObj>();
}

bool ConstantNode::IsScalar() const {
  return value.defined() && value.as<ScalarValueObj>();
}

ObjectPtr<ConstantNode> MakeConstantNode(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  static const auto fake_tensor = MakeFakeTensor();
  n->data = fake_tensor;
  n->value = std::move(node_ref);
  return n;
}

RelayConstant MakeConstant(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = MakeConstantNode(node_ref);
  return RelayConstant(n);
}

RelayConstant MakeNull() {
  return MakeConstant(NullValue<Value>());
}

ObjectRef ConstantExtractValue(RelayConstant _node) {
  const ConstantNode* node = static_cast<const ConstantNode*>(_node.get());
  return node->value;
}

Var MakeVar_(Id vid, Type type_annotation, Var may_share = Var()) {
  ObjectPtr<ExtendedVarNode> n = make_object<ExtendedVarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->may_share = may_share;
  return Var(n);
}

Var MakeVar(const std::string& name_hint, Type type_annotation, Var may_share) {
  return MakeVar_(Id(name_hint), type_annotation, may_share);
}

Var GetMayShare(Expr var) {
  const auto* vn = var.as<ExtendedVarNode>();
  CHECK(vn);
  return vn->may_share;
}

Var TryGetMayShare(Expr var) {
  const auto* vn = var.as<ExtendedVarNode>();
  CHECK(vn);
  while (vn->may_share.defined()) {
    vn = vn->may_share.as<ExtendedVarNode>();
    CHECK(vn);
  }
  return GetRef<Var>(vn);
}

std::string AsText(const ObjectRef& node, bool show_meta_data) {
  auto annotate =
      tvm::runtime::TypedPackedFunc<String(ObjectRef)>([](const ObjectRef& expr) -> String {
        std::ostringstream os;
        const auto* constant = expr.as<ConstantNode>();
        if (constant) {
          if (constant->value.defined()) {
            os << constant->value;
          } else {
            os << "nullptr";
          }
        }
        if (expr.as<CallNode>() && Downcast<Expr>(expr)->checked_type_.defined()) {
          auto meta = tvm::TextMetaDataContext();
          tvm::relay::RelayTextPrinter printer(false, &meta, nullptr);
          os << " /* ty=" << printer.Print(Downcast<Expr>(expr)->checked_type()).str() << " */";
        }
        const auto* extended_var = expr.as<ExtendedVarNode>();
        if (extended_var && extended_var->may_share.defined()) {
          os << "(share: %" << extended_var->may_share->name_hint() << ")";
        }
        return String(os.str());
      });

  std::string ret = tvm::AsText(node, show_meta_data, annotate);
  size_t index = 0;
  while (true) {
    index = ret.find("-114514", index);
    if (index == std::string::npos) {
      break;
    }
    ret.replace(index, 7, "");
  }
  return ret;
}

RAF_REGISTER_GLOBAL("raf.ir.AsText").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  ObjectRef value = args[0];
  bool show_meta_data = args.size() == 2 ? args[1] : false;
  *rv = AsText(value, show_meta_data);
});

RAF_REGISTER_GLOBAL("raf.ir._make.Constant").set_body_typed(MakeConstant);
RAF_REGISTER_GLOBAL("raf.ir._make.Var").set_body_typed(MakeVar);
RAF_REGISTER_GLOBAL("raf.ir.constant.ExtractValue").set_body_typed(ConstantExtractValue);
RAF_REGISTER_GLOBAL("raf.ir.variable.GetMayShare").set_body_typed(GetMayShare);
RAF_REGISTER_GLOBAL("raf.ir.module.Global").set_body_typed(GlobalModule);

}  // namespace ir
}  // namespace raf
