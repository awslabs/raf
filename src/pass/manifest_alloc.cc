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
using tvm::relay::LetList;

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
    if (node->op.as<OpNode>()) {
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
          outs.push_back(MakeStaticAllocation(&scope, out_types[i].as<TensorTypeNode>(), i));
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

 private:
  Expr ComputeAlignment(DataType dtype) {
    int64_t align = dtype.bits() / 8 * dtype.lanes();
    if (align < 64) {
      align = 64;
    }
    return MakeConstant(IntValue::make(align));
  }

  Expr ComputeStorage(const TensorTypeNode* type) {
    int64_t size = 1;
    for (auto dim : type->shape) {
      auto dim_imm = dim.as<IntImmNode>();
      CHECK(dim_imm);
      size *= dim_imm->value;
    }
    size *= (type->dtype.bits() * type->dtype.lanes() + 7) / 8;
    return MakeConstant(IntValue::make(size));
  }

  Expr MakeStaticAllocation(LetList* scope, const TensorTypeNode* type, size_t i) {
    Expr shape = MakeConstant(type->shape);
    Expr size = ComputeStorage(type);
    Expr alignment = ComputeAlignment(type->dtype);
    auto alloc_storage_attrs = make_object<tvm::relay::AllocStorageAttrs>();
    alloc_storage_attrs->dtype = type->dtype;
    auto target = tvm::Target::Current();
    alloc_storage_attrs->device_type = target.defined() ? target->id->device_type : kDLCPU;
    alloc_storage_attrs->device_id = 0;
    auto storage = scope->Push(Call(Op::Get("mnm.op.vm.alloc_storage"),
                                    Array<Expr>{size, alignment}, Attrs(alloc_storage_attrs)));
    auto alloc_tensor_attrs = make_object<tvm::relay::AllocTensorAttrs>();
    alloc_tensor_attrs->dtype = type->dtype;
    auto tensor = scope->Push(Call(Op::Get("mnm.op.vm.alloc_tensor"), Array<Expr>{storage, shape},
                                   Attrs(alloc_tensor_attrs)));
    return tensor;
  }

  std::vector<LetList> scopes_;
  tvm::runtime::DataType compute_dtype_ = tvm::runtime::DataType::Int(64);
};

}  // namespace manifest_alloc

ir::Module ManifestAlloc(ir::Module mod) {
  tvm::Map<ir::GlobalVar, ir::Function> functions;
  for (auto& kv : mod->functions) {
    functions.Set(kv.first, tvm::Downcast<ir::Function>(
                                manifest_alloc::ManifestAllocMutator().Mutate(kv.second)));
  }
  return ir::Module::make(functions);
}

MNM_REGISTER_GLOBAL("mnm.pass_.ManifestAlloc").set_body_typed(ManifestAlloc);

}  // namespace pass
}  // namespace mnm
