/*!
 * Copyright (c) 2020 by Contributors
 * \file memory_alloc.cc
 * \brief Manifest memory allocation in the IR.
 */
#include <algorithm>
#include <vector>

#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "../op/from_relay/from_relay_utils.h"
#include "tvm/relay/attrs/memory.h"

namespace tvm {
namespace relay {

extern bool IsDynamic(const Type& ty);
extern Expr ToTupleType(const Type& ty, const std::vector<Expr>& exprs);

}  // namespace relay
}  // namespace tvm

namespace mnm {
namespace pass {

namespace manifest_alloc {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::from_relay;

class ManifestAllocMutator : public ExprMutator {
 public:
  ManifestAllocMutator() : scopes_{LetList()} {
  }

  Expr VisitExpr_(const TupleNode* node) {
    auto& scope = scopes_.back();
    Array<Expr> new_fields;
    for (auto field : node->fields) {
      auto new_field = VisitExpr(field);
      if (auto constant_field = field.as<ConstantNode>()) {
        auto const_var = scope.Push(field);
        new_field = const_var;
      }
      new_fields.push_back(new_field);
    }
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const ConstantNode* node) {
    return scopes_.back().Push(GetRef<Expr>(node));
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back();
    auto& scope = scopes_.back();
    Expr body;
    do {
      scope.Push(node->var, VisitExpr(node->value));
      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto new_body = VisitExpr(body);
    auto ret = scopes_.back().Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) {
    const auto* op = node->op.as<OpNode>();
    const auto* func = node->op.as<FunctionNode>();
    if (op || func && func->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
      auto& scope = scopes_.back();
      Array<Expr> new_args;
      for (auto& arg : node->args) {
        new_args.push_back(VisitExpr(arg));
      }
      auto ret_type = node->checked_type();
      auto out_types = tvm::relay::FlattenTupleType(ret_type);
      if (tvm::relay::IsDynamic(ret_type)) {
        LOG(FATAL) << "Dynamic type not supported.";
        return Expr();
      } else {
        std::vector<Expr> outs;
        for (size_t i = 0; i < out_types.size(); i++) {
          outs.push_back(MakeStaticAllocation(&scope, out_types[i].as<TensorTypeNode>()));
        }
        auto invoke =
            Call(Op::Get("mnm.op.vm.invoke_op"),
                 Array<Expr>{scope.Push(node->op), Tuple(new_args), Tuple(Array<Expr>(outs))});
        scope.Push(invoke);
        return tvm::relay::ToTupleType(ret_type, outs);
      }
    } else {
      return ExprMutator::VisitExpr_(node);
    }
  }

  Expr VisitExpr_(const FunctionNode* node) {
    if (node->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
      return GetRef<Expr>(node);
    } else {
      return ExprMutator::VisitExpr_(node);
    }
  }

 private:
  Expr ComputeAlignment(DataType dtype) {
    int64_t align = dtype.bits() / 8 * dtype.lanes();
    if (align < 64) {
      align = 64;
    }
    return MakeConstant(ScalarValue::make(align));
  }

  Expr ComputeStorage(const TensorTypeNode* type) {
    int64_t size = 1;
    for (auto dim : type->shape) {
      auto dim_imm = dim.as<IntImmNode>();
      CHECK(dim_imm);
      size *= dim_imm->value;
    }
    size *= (type->dtype.bits() * type->dtype.lanes() + 7) / 8;
    return MakeConstant(ScalarValue::make(size));
  }

  Expr MakeAllocStorage(const Array<Expr>& args, int device_type, int device_id,
                        const tvm::runtime::DataType& dtype) {
    static const Op& op = Op::Get("mnm.op.vm.alloc_storage");
    Array<Expr> new_args = args;
    new_args.push_back(MakeConstant(ScalarValue::make(device_type)));
    new_args.push_back(MakeConstant(ScalarValue::make(device_id)));
    new_args.push_back(MakeConstant(StringValue::make(DLDataType2String(dtype))));
    return Call(op, new_args);
  }

  Expr MakeAllocTensor(const Array<Expr>& args, const Expr& assert_shape,
                       const tvm::runtime::DataType& dtype) {
    static const Op& op = Op::Get("mnm.op.vm.alloc_tensor");
    Array<Expr> new_args = args;
    new_args.push_back(MakeConstant(StringValue::make(DLDataType2String(dtype))));
    new_args.push_back(assert_shape);
    return Call(op, new_args);
  }

  Expr MakeStaticAllocation(LetList* scope, const TensorTypeNode* type) {
    Expr shape = MakeConstant(type->shape);
    Expr size = ComputeStorage(type);
    Expr alignment = ComputeAlignment(type->dtype);
    auto alloc_storage_attrs = make_object<tvm::relay::AllocStorageAttrs>();
    auto dtype = type->dtype;
    auto target = tvm::Target::Current();
    auto device_type = target.defined() ? target->kind->device_type : kDLCPU;
    int device_id = 0;
    auto storage = scope->Push(MakeAllocStorage(Array<Expr>{size, alignment},
                                                static_cast<int>(device_type), device_id, dtype));
    auto tensor = scope->Push(MakeAllocTensor(Array<Expr>{storage, shape}, shape, dtype));
    return tensor;
  }

  std::vector<LetList> scopes_;
  tvm::runtime::DataType compute_dtype_ = tvm::runtime::DataType::Int(64);
};

}  // namespace manifest_alloc

ir::IRModule ManifestAlloc(ir::IRModule mod) {
  tvm::Map<ir::GlobalVar, ir::BaseFunc> functions;
  for (auto& kv : mod->functions) {
    functions.Set(kv.first, tvm::Downcast<ir::Function>(
                                manifest_alloc::ManifestAllocMutator().Mutate(kv.second)));
  }
  return ir::IRModule(functions);
}

MNM_REGISTER_GLOBAL("mnm.pass_.ManifestAlloc").set_body_typed(ManifestAlloc);

}  // namespace pass
}  // namespace mnm
