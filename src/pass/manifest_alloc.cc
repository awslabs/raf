/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_alloc.cc
 * \brief Manifest memory allocation in the IR.
 */
#include <algorithm>
#include <vector>

#include "raf/device.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "raf/pass.h"
#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"
#include "tvm/relay/attrs/memory.h"

namespace raf {
namespace pass {
namespace manifest_alloc {

using namespace raf::ir;
using namespace raf::value;
using common::shape_utils::BytesCompactTensor;

class InplaceVisitor : public MixedModeVisitor {
 public:
  void VisitExpr_(const LetNode* node) override {
    auto pre_visit = [this](const LetNode* node) {
      auto var = node->var.as<ExtendedVarNode>();
      if (var->may_share.defined()) {
        auto call = node->value.as<CallNode>();
        auto tup_get = node->value.as<TupleGetItemNode>();
        if (call) {
          // If the value of the var is a call node and this var has may_share defined,
          // it can only be a TensorType. We add the mapping from this var to its may_share var.
          var_share_map.emplace(node->var, std::vector<Var>{var->may_share});
        } else if (tup_get) {
          // If variables in a tuple share with others, we record the mapping from the tuple var to
          // a list of variables that the tuple items share with. The list could contain empty vars,
          // indicating that the corresponding item doesn't share memory with others.
          auto tup = Downcast<Var>(tup_get->tuple);
          if (var_share_map.count(tup) == 0) {
            size_t num_fields = Downcast<TupleType>(tup->checked_type())->fields.size();
            std::vector<Var> shares(num_fields);
            var_share_map.emplace(tup, shares);
          }
          var_share_map[tup][tup_get->index] = var->may_share;
        }
      }
    };
    auto post_visit = [this](const LetNode* node) {
      VisitExpr(node->value);
      VisitExpr(node->body);
    };
    ExpandANormalForm(node, pre_visit, post_visit);
  }

  /*! \brief Mapping from a var to a list of vars that it shares memory with.
   *  When the var is a tuple, #vars in the list must be the same as #items in the tuple.
   */
  std::unordered_map<Var, std::vector<Var>, ObjectPtrHash, ObjectPtrEqual> var_share_map;
};

class ManifestAllocMutator : public ExprMutator {
 public:
  ManifestAllocMutator() {
    scopes_.emplace_back(new LetList);
  }

  Expr VisitExpr_(const TupleNode* node) {
    // Previously `scopes_` is defined as `std::vector<LetList>` and
    // `auto& scope = scopes_.back();` is heavily used to access the inner most scope.
    // However, this pattern is erron prone and thus not recommended:
    // scopes_.back() returns a reference to the last element, which is invalidated when
    // reallocation happens in scopes_. See https://stackoverflow.com/questions/20098454/
    // weird-behavior-of-reference-to-vector-back-after-vector-is-modified for reference.
    auto scope = scopes_.back().get();
    Array<Expr> new_fields;
    for (auto field : node->fields) {
      auto new_field = VisitExpr(field);
      if (auto constant_field = field.as<ConstantNode>()) {
        auto const_var = scope->Push(field);
        new_field = const_var;
      }
      new_fields.push_back(new_field);
    }
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const RelayConstantNode* node) {
    auto var = scopes_.back()->Push(GetRef<Expr>(node));
    /*
     * After constant folding, sometimes the IR return a single ConstantNode.
     * In this case, get the body from LetList and return, otherwise there will be free vars.
     */
    if (scopes_.size() == 1) {
      auto ret = scopes_.back()->Get(var);
      scopes_.pop_back();
      return ret;
    }
    return var;
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      let_binding_.emplace(node->value, node->var);
      scope->Push(node->var, VisitExpr(node->value));
      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto new_body = VisitExpr(body);
    auto ret = scopes_.back()->Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) {
    static std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual> exclude_ops{
        Op::Get("raf.op.set_stream"), Op::Get("raf.op.wait_event"), Op::Get("raf.op.add_event"),
        Op::Get("raf.op.stream_barrier")};
    static auto vm_set_shape_op = Op::Get("raf.op.vm.set_shape");

    const auto* op = node->op.as<OpNode>();
    const auto* func = node->op.as<FunctionNode>();
    if ((op && !exclude_ops.count(GetRef<Op>(op))) ||
        (func && func->HasNonzeroAttr(attr::kPrimitive))) {
      Call call = GetRef<Call>(node);
      // change the op which uses upper-bound memory to its upper-bound dialect op
      bool use_upper_bound = false;
      static auto upper_bound_map = Op::GetAttrMap<Op>("TRAFUpperBoundOp");
      if (op && upper_bound_map.count(GetRef<Op>(op))) {
        call = Call(upper_bound_map[GetRef<Op>(op)], node->args);
        call = Downcast<Call>(pass::InferType(call));
        op = call->op.as<OpNode>();
        use_upper_bound = true;
      }
      auto scope = scopes_.back().get();
      Var bind_var = let_binding_[GetRef<Call>(node)];

      auto ret_type = call->checked_type();
      auto out_types = tvm::relay::FlattenTupleType(ret_type);
      Array<Expr> new_args;
      if (op::IsReshapeOp(GetRef<Op>(op))) {
        // generate vm.set_shape for reshape ops to avoid unnecessary kernels and allocations
        CHECK_EQ(out_types.size(), 1U);
        // first push the input tensor
        new_args.push_back(VisitExpr(call->args[0]));

        // the new shape is determined in run time if it is dynamic;
        // otherwise the new shape is just the output shape
        if (tvm::relay::IsDynamic(ret_type)) {
          for (size_t i = 1; i < call->args.size(); ++i) {
            new_args.push_back(VisitExpr(call->args[i]));
          }
          auto ins = scope->Push(Tuple(new_args));
          auto op_var = scope->Push(call->op);
          auto infer_type = Call(Op::Get("raf.op.vm.infer_type"), Array<Expr>{op_var, ins});
          auto out_type_exprs = scope->Push(infer_type);
          auto out_type_expr = scope->Push(TupleGetItem(out_type_exprs, 1));
          auto shape = scope->Push(TupleGetItem(out_type_expr, 0));
          return Call(vm_set_shape_op, {new_args[0], shape});
        }
        auto tensor_ty_node = out_types[0].as<TensorTypeNode>();
        new_args.push_back(MakeConstant(op::ArrayToIntTuple(tensor_ty_node->shape)));
        return Call(vm_set_shape_op, new_args);
      } else {
        // allocate necessary memory buffers and invoke ops
        for (auto& arg : call->args) {
          new_args.push_back(VisitExpr(arg));
        }

        // Determine the device for output tensor allocation.
        auto device = GetOutputDevice(call);

        std::vector<Expr> outs;
        if (tvm::relay::IsDynamic(ret_type)) {
          outs = DynamicInvoke(scope, bind_var, call->op, new_args, out_types, device);
        } else {
          outs = StaticInvoke(scope, bind_var, call->op, new_args, out_types, device);
        }

        // if op uses upper bound shape, reshapes its results
        if (op && use_upper_bound) {
          return Call(vm_set_shape_op, outs);
        }
        return tvm::relay::ToTupleType(ret_type, outs);
      }
    } else {
      return ExprMutator::VisitExpr_(node);
    }
  }

  Expr VisitExpr_(const FunctionNode* node) {
    if (node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Expr>(node);
    } else {
      return ExprMutator::VisitExpr_(node);
    }
  }

  Expr operator()(const Expr& expr) {
    inplace_.VisitExpr(expr);
    return Mutate(expr);
  }

 private:
  Expr ComputeAlignment(DataType dtype) {
    int64_t align = dtype.bits() / 8 * dtype.lanes();
    if (align < 64) {
      align = 64;
    }
    return MakeConstant(ScalarValue::make(align));
  }

  Expr MakeAllocStorage(const Array<Expr>& args, int device_type, int device_id,
                        const DataType& dtype) {
    static const Op& op = Op::Get("raf.op.vm.alloc_storage");
    Array<Expr> new_args = args;
    new_args.push_back(MakeConstant(ScalarValue::make(device_type)));
    new_args.push_back(MakeConstant(ScalarValue::make(device_id)));
    new_args.push_back(MakeConstant(StringValue::make(DLDataType2String(dtype))));
    return Call(op, new_args);
  }

  Expr MakeAllocTensor(const Array<Expr>& args, const Expr& assert_shape, const DataType& dtype) {
    static const Op& op = Op::Get("raf.op.vm.alloc_tensor");
    Array<Expr> new_args = args;
    new_args.push_back(MakeConstant(StringValue::make(DLDataType2String(dtype))));
    new_args.push_back(assert_shape);
    return Call(op, new_args);
  }

  Expr MakeAllocationCommon(LetList* scope, const TensorTypeNode* type, const Expr& shape,
                            const Expr& size, const Device& device) {
    Expr alignment = ComputeAlignment(type->dtype);
    auto dtype = type->dtype;

    Device target_device(DevType::kCPU(), 0);
    if (device.device_type() != DevType::kUnknown()) {
      target_device = device;
    }
    auto storage = scope->Push(MakeAllocStorage(Array<Expr>{size, alignment},
                                                static_cast<int>(target_device.device_type()),
                                                target_device.device_id(), dtype));
    auto tensor = scope->Push(MakeAllocTensor(Array<Expr>{storage, shape}, shape, dtype));
    return tensor;
  }

  Expr MakeStaticAllocation(LetList* scope, const TensorTypeNode* type, const Device& device) {
    Expr shape = MakeConstant(type->shape);
    Expr size = MakeConstant(ScalarValue::make(BytesCompactTensor(type)));
    return MakeAllocationCommon(scope, type, shape, size, device);
  }

  Expr MakeDynamicAllocation(LetList* scope, const TensorTypeNode* type, const Expr& out_type_expr,
                             const Device& device) {
    Expr shape = scope->Push(TupleGetItem(out_type_expr, 0));
    Expr size = scope->Push(TupleGetItem(out_type_expr, 1));
    return MakeAllocationCommon(scope, type, shape, size, device);
  }

  std::vector<Expr> DynamicInvoke(LetList* scope, const Var& bind_var, const Expr& op,
                                  const Array<Expr>& new_args,
                                  const std::vector<TensorType>& out_types, const Device& device) {
    auto ins = scope->Push(Tuple(new_args));
    auto op_var = scope->Push(op);
    auto infer_type = Call(Op::Get("raf.op.vm.infer_type"), Array<Expr>{op_var, ins});
    auto out_type_exprs = scope->Push(infer_type);
    std::vector<Expr> outs;
    auto it = inplace_.var_share_map.find(bind_var);
    if (it != inplace_.var_share_map.end()) {
      // some outputs have inplace update
      auto share = it->second;
      CHECK_EQ(share.size(), out_types.size());
      for (size_t i = 0; i < out_types.size(); ++i) {
        // check if the output shares the memory with input
        if (share[i].defined()) {
          outs.push_back(share[i]);
        } else {
          Expr out_type_expr = scope->Push(TupleGetItem(out_type_exprs, i + 1));
          outs.push_back(MakeDynamicAllocation(scope, out_types[i].as<TensorTypeNode>(),
                                               out_type_expr, device));
        }
      }
    } else {
      for (size_t i = 0; i < out_types.size(); i++) {
        Expr out_type_expr = scope->Push(TupleGetItem(out_type_exprs, i + 1));
        outs.push_back(
            MakeDynamicAllocation(scope, out_types[i].as<TensorTypeNode>(), out_type_expr, device));
      }
    }
    Call invoke_op;
    if (op->IsInstance<OpNode>()) {
      invoke_op = Call(Op::Get("raf.op.vm.invoke_op"),
                       Array<Expr>{op_var, ins, scope->Push(Tuple(Array<Expr>(outs)))});
    } else {
      ICHECK(op->IsInstance<FunctionNode>());
      invoke_op = Call(Op::Get("raf.op.vm.invoke_op"),
                       Array<Expr>{scope->Push(TupleGetItem(out_type_exprs, 0)), ins,
                                   scope->Push(Tuple(Array<Expr>(outs)))});
    }
    scope->Push(invoke_op);
    return outs;
  }

  std::vector<Expr> StaticInvoke(LetList* scope, const Var& bind_var, const Expr& op,
                                 const Array<Expr>& new_args,
                                 const std::vector<TensorType>& out_types, const Device& device) {
    std::vector<Expr> outs;
    auto it = inplace_.var_share_map.find(bind_var);
    if (it != inplace_.var_share_map.end()) {
      // some outputs have inplace update
      auto share = it->second;
      CHECK_EQ(share.size(), out_types.size());
      for (size_t i = 0; i < out_types.size(); ++i) {
        // check if the output shares the memory with input
        if (share[i].defined()) {
          outs.push_back(share[i]);
        } else {
          outs.push_back(MakeStaticAllocation(scope, out_types[i].as<TensorTypeNode>(), device));
        }
      }
    } else {
      for (size_t i = 0; i < out_types.size(); i++) {
        outs.push_back(MakeStaticAllocation(scope, out_types[i].as<TensorTypeNode>(), device));
      }
    }
    auto invoke = Call(Op::Get("raf.op.vm.invoke_op"),
                       Array<Expr>{scope->Push(op), scope->Push(Tuple(new_args)),
                                   scope->Push(Tuple(Array<Expr>(outs)))});
    scope->Push(invoke);
    return outs;
  }

  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The mapping from expr to let bound var. */
  std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  /*! \breif Inplace visitor to check the may_share information. */
  InplaceVisitor inplace_;
};

}  // namespace manifest_alloc

Pass ManifestAlloc() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<ir::Function>(manifest_alloc::ManifestAllocMutator()(f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "ManifestAlloc", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ManifestAlloc").set_body_typed(ManifestAlloc);

}  // namespace pass
}  // namespace raf
