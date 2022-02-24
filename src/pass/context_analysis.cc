/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/context_analysis.cc
 * \brief A pass for analyzing device attribute of each IR node.
 *
 * We use union-find data structures to analyze the context information of each
 * sub-expression in a Relay program in this pass. Only the device copy node in
 * Relay directly contains bidiretional device information. We use it to
 * bidirectionally propagate the device info of its inputs and outputs.
 *
 * However, to support dynamism (e.g dynamic inputs), Relay introduces several
 * concepts to compute the shape of tensors and operators at runtime, i.e.
 * shape_of, shape_func, and reshape_tensor. These nodes are also referred to as
 * VM dialects as we have native VM instructions for them. These dialects are
 * intrinsically CPU friendly, therefore, they are only designed to be
 * executed on CPU. We, hence, unify their inputs and outputs to CPU as well.
 * Note the input of shape_of is a tensor and we only need the tensor shape.
 * Therefore, the input could be sitting on GPU as well since no real data is
 * needed. The context of the input would be propagated from its other
 * consumers or fallback to the default device.
 *
 * Another type of dialect is used fo memory allocation, namely, alloc_storage
 * and alloc_tensor. alloc_storage contains a context field to indicate where
 * the chunk of memory is allocated. Therefore, we unify the context of
 * alloc_storage with the context field. Other inputs, such as size and
 * alignment, are left on CPU.
 *
 * Based on the above rules, we keep unifying the connected expressions and
 * propagating their device information. An error will be raised whenever there
 * is a unification conflict. All IR nodes that are not propagated with device
 * context will fallback to the specified device.
 */

#include <raf/binding.h>
#include <raf/ir.h>
#include <raf/op.h>
#include <raf/pass.h>
#include <raf/value.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/op_attr_types.h>

#include "../op/schema/memory.h"

namespace raf {
namespace pass {

using namespace raf::ir;
using namespace raf::op::schema;
using namespace raf::value;

using AnalysisResultMap = Map<Expr, Device>;

namespace context_analysis {

// Cache ops
static const Op& device_copy_op = Op::Get("raf.op.device_copy");
static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
static const Op& alloc_tensor_op = Op::Get("raf.op.vm.alloc_tensor");
static const Op& invoke_op = Op::Get("raf.op.vm.invoke_op");

class DeviceDomain;
using DeviceDomainPtr = std::shared_ptr<DeviceDomain>;

/*
 * \brief A class to represent the device of a domain, i.e. a segment of relay program.
 */
class DeviceDomain {
 public:
  // Construct an empty domain.
  DeviceDomain() {
  }

  // Construct a domain based on a given context.
  explicit DeviceDomain(const Device& dev) : device_(dev) {
  }

  // Check if the current domain is empty.
  bool IsEmptyDomain() const {
    return device_.device_type() == DevType::kUnknown() && device_.device_id() == -1;
  }

  // Check if the current domain equals the other one.
  bool operator==(const DeviceDomain& other) const {
    return device_.device_type() == other.device_.device_type() &&
           device_.device_id() == other.device_.device_id();
  }

  bool operator!=(const DeviceDomain& other) const {
    return !(*this == other);
  }

 private:
  // Create a hash for a domain.
  struct Hash {
    size_t operator()(const DeviceDomainPtr& domain) const {
      if (domain->IsEmptyDomain()) {
        return (size_t)(domain.get());
      } else {
        size_t const h1(std::hash<int>()(static_cast<int>(domain->device_.device_type())));
        size_t const h2(std::hash<int>()(domain->device_.device_id()));
        return h1 ^ (h2 << 1);
      }
    }
  };

  // Create an equality for domains.
  struct Equal {
   public:
    bool operator()(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) const {
      // We compare the pointer for empty domains.
      if (lhs->IsEmptyDomain() && rhs->IsEmptyDomain()) return lhs.get() == rhs.get();

      // Otherwise device type and id are used to check equality.
      return (*lhs.get() == *rhs.get());
    }
  };

  /* \brief The device to be assigned to the current domain. */
  Device device_;

  friend DeviceDomainPtr Join(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs);
  friend class ContextAnalyzer;
};

// Join two domains.
DeviceDomainPtr Join(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) {
  if (lhs->IsEmptyDomain() && rhs->IsEmptyDomain()) {
    return lhs;
  } else if (lhs->IsEmptyDomain()) {
    return rhs;
  } else if (rhs->IsEmptyDomain()) {
    return lhs;
  } else {
    CHECK(*lhs.get() == *rhs.get()) << "All expressions must have a singular device to unify";
    return lhs;
  }
}

/*
 * \brief Compute on which device each sub-expression will execute. A union find
 * algorithm is used to assign and merge the context domains.
 */
class ContextAnalyzer : public MixedModeVisitor {
 public:
  ContextAnalyzer(const IRModule& mod, const GlobalVar& current_func,
                  const Device& default_device)
      : MixedModeVisitor(9),  // the number of repeated visits a node can perform
        mod_(mod),
        current_func_(current_func),
        default_device_(default_device) {
  }

  // Create an empty domain.
  // This usually happens when we enter a new scope, i.e. Function.
  DeviceDomainPtr Bottom() {
    return std::make_shared<DeviceDomain>(DeviceDomain());
  }

  // Create a domain with the given device context.
  DeviceDomainPtr DeviceType(const Device& dev) {
    return std::make_shared<DeviceDomain>(DeviceDomain(dev));
  }

  // Find the root of a device.
  DeviceDomainPtr Lookup(DeviceDomainPtr device) {
    while (device_uf_.count(device) && device != device_uf_[device]) {
      // Path compression
      if (device_uf_.count(device_uf_[device])) {
        device_uf_[device] = device_uf_[device_uf_[device]];
      }
      device = device_uf_[device];
    }
    return device;
  }

  // Unify two domains.
  DeviceDomainPtr Unify(DeviceDomainPtr lhs, DeviceDomainPtr rhs) {
    lhs = Lookup(lhs);
    rhs = Lookup(rhs);
    auto unified_device = Join(lhs, rhs);
    if (lhs != unified_device) {
      device_uf_[lhs] = unified_device;
    }

    if (rhs != unified_device) {
      device_uf_[rhs] = unified_device;
    }

    return unified_device;
  }

  // Unify the domain for two IR nodes.
  DeviceDomainPtr UnifyExpr(const Expr& lhs, const Expr& rhs) {
    auto lhs_dom = DeviceFor(lhs);
    auto rhs_dom = DeviceFor(rhs);
    return Unify(lhs_dom, rhs_dom);
  }

  // Lookup or insert an IR node to device domain map.
  DeviceDomainPtr DeviceFor(const Expr& expr) {
    auto it = expr_to_device_.find(expr);
    if (it == expr_to_device_.end()) {
      auto bottom = Bottom();
      expr_to_device_[expr] = bottom;
      return bottom;
    } else {
      return it->second;
    }
  }

  // Unify the device context for a device copy node. Device copy node is
  // the only node that carries bidirectional devices in the input program. The device
  // attribute of other nodes can be propagated from it.
  void UnifyDeviceCopy(const std::vector<Expr>& inps, const std::vector<Expr>& outputs,
                       DevType src_dev_type, DevType dst_dev_type) {
    Device src_ctx = Device(src_dev_type, 0);
    auto src_domain = DeviceType(src_ctx);
    for (const auto& it : inps) {
      auto lhs = DeviceFor(it);
      Unify(lhs, src_domain);
    }

    Device dst_ctx = Device(dst_dev_type, 0);
    auto dst_domain = DeviceType(dst_ctx);
    for (const auto& it : outputs) {
      auto lhs = DeviceFor(it);
      Unify(lhs, dst_domain);
    }
  }

  // Unify the domain of inputs and outputs of a relay call.
  //
  // For most call nodes, the op, inputs, and outputs should all be in the
  // same domain, i.e. having the same context. However, device_copy call node
  // needs to be handled differently as it copies data from one device to
  // another.
  DeviceDomainPtr UnifyCall(const Expr& call_op, const Array<Expr>& inps,
                            const Array<Expr>& outputs, DeviceDomainPtr device) {
    device = Unify(device, DeviceFor(call_op));

    for (const auto& it : inps) {
      device = Unify(device, DeviceFor(it));
    }

    for (const auto& it : outputs) {
      device = Unify(device, DeviceFor(it));
    }

    return device;
  }

  void VisitExpr_(const CallNode* cn) final {
    Call call = GetRef<Call>(cn);

    if (IsDeviceCopy(call)) {
      UnifyDeviceCopyCall(cn);
    } else if (call->op == alloc_storage_op) {
      UnifyAllocStorageCall(cn);
    } else if (call->op == alloc_tensor_op) {
      UnifyAllocTensorCall(cn);
    } else if (call->op == invoke_op) {
      UnifyInvokeOpCall(cn);
    } else if (call->op.as<FunctionNode>()) {
      UnifyFunctionCall(cn);
    } else if (call->op.as<GlobalVarNode>()) {
      UnifyGlobalVarCall(cn);
    } else if (call->op.as<VarNode>()) {
      UnifyVarCall(cn);
    } else {
      UnifyCall(call, cn->args, {call}, Bottom());
      MixedModeVisitor::VisitExpr_(cn);
    }
  }

  void VisitExpr_(const LetNode* ln) final {
    Expr expr = GetRef<Let>(ln);
    // Iteratively visit let nodes to avoid stack overflow.
    while (expr->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(expr);
      // Save currying/closures since they will be invoked later
      auto ty = let->value->checked_type();
      if (ty->IsInstance<FuncTypeNode>()) {
        auto gv = ExtractClosure(let);
        CHECK(gv.defined() && gv->IsInstance<GlobalVarNode>());
        closures_[let->var] = Downcast<GlobalVar>(gv);
      }

      // Unify let var, value, and body
      Unify(DeviceFor(let->var), DeviceFor(let->value));
      UnifyExpr(let, let->body);
      MixedModeVisitor::VisitExpr(let->value);
      expr = let->body;
    }
    // Visit the last body
    MixedModeVisitor::VisitExpr(expr);
  }

  void VisitExpr_(const FunctionNode* fn) final {
    auto func = GetRef<Function>(fn);
    // No need to step into fused primitive functions as they are handled as
    // a whole.
    if (fn->HasNonzeroAttr(attr::kPrimitive)) {
      return;
    }

    auto device = Unify(DeviceFor(func), DeviceFor(fn->body));
    for (const auto& it : fn->params) {
      DeviceFor(it);
    }
    MixedModeVisitor::VisitExpr(fn->body);
  }

  void VisitExpr_(const TupleNode* tn) final {
    // We only support tuple with the same of device.
    Tuple tup = GetRef<Tuple>(tn);
    if (tn->fields.size() > 0) {
      auto device = DeviceFor(tup->fields[0]);
      for (size_t i = 1; i < tup->fields.size(); i++) {
        device = Unify(device, DeviceFor(tup->fields[i]));
      }
      Unify(device, DeviceFor(tup));
    }
    MixedModeVisitor::VisitExpr_(tn);
  }

  void VisitExpr_(const TupleGetItemNode* tn) final {
    TupleGetItem item = GetRef<TupleGetItem>(tn);

    Unify(DeviceFor(item), DeviceFor(item->tuple));

    MixedModeVisitor::VisitExpr_(tn);
  }

  void VisitExpr_(const GlobalVarNode* gvn) final {
    DeviceFor(GetRef<GlobalVar>(gvn));
  }

  void VisitExpr_(const VarNode* vn) {
    DeviceFor(GetRef<Var>(vn));
  }

  void VisitExpr_(const RelayConstantNode* cn) final {
    DeviceFor(GetRef<Constant>(cn));
  }

  // Return the analysis results.
  AnalysisResultMap Results() {
    AnalysisResultMap ret;
    for (const auto& it : expr_to_device_) {
      auto device = Lookup(it.second);
      if (device->IsEmptyDomain()) {
        ret.Set(it.first, default_device_);
      } else {
        ret.Set(it.first, device->device_);
      }
    }

    return ret;
  }

 private:
  Expr ExtractClosure(Expr expr) const {
    while (expr->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(expr);
      expr = let->value;
      if (expr->IsInstance<GlobalVarNode>()) {
        return expr;
      } else {
        const auto* cn = expr.as<CallNode>();
        if (cn && cn->op->IsInstance<GlobalVarNode>()) {
          return cn->op;
        }
      }
    }
    return Expr(nullptr);
  }

  // Check if an expression is a device copy call.
  bool IsDeviceCopy(const Expr& expr) const {
    if (!expr->IsInstance<CallNode>()) return false;

    Call call = Downcast<Call>(expr);
    if (call->op == device_copy_op) return true;

    // Fused function with device copy op as the body
    // device copy op is opaque therefore the fused function only has one node.
    if (const FunctionNode* fn = call->op.as<FunctionNode>()) {
      if (const CallNode* cn = fn->body.as<CallNode>()) {
        return cn->op == device_copy_op;
      }
    }

    return false;
  }

  // Check if a function is a closure.
  bool IsClosure(const Function& func) {
    return func->GetAttr<Integer>(attr::kClosure, 0) != 0;
  }

  // Check if a function is a currying function.
  bool IsCurrying(const Function& func) {
    if (const auto* let = func->body.as<LetNode>()) {
      return closures_.find(let->var) != closures_.end();
    }
    return false;
  }

  inline Device ToDevice(const ObjectRef& val) {
    const auto* str_val = val.as<StringValueObj>();
    CHECK(str_val != nullptr);
    const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
    return Device((tvm::Device)(*str2dev)(str_val->value));
  }

  // Process device copy call node
  void UnifyDeviceCopyCall(const CallNode* call) {
    CHECK_EQ(call->args.size(), 3U);

    std::vector<Expr> inps{call->args[0]};
    // The call and its src and dst are allocated with the same device type as
    // the destination.
    std::vector<Expr> outs{GetRef<Call>(call), call->args[1], call->args[2]};
    DevType src_dev_type, dst_dev_type;
    const DeviceCopyArgs* attrs = nullptr;
    if (const auto* fn = call->op.as<FunctionNode>()) {
      // TODO(zhiics) Check how to handle this
      inps.push_back(fn->params[0]);
      outs.push_back(call->op);
      Expr body = fn->body;
      CHECK(body->IsInstance<CallNode>() && IsDeviceCopy(body));
      Call call_body = Downcast<Call>(body);
      attrs = call_body->args.as<DeviceCopyArgs>();
    } else {
      const ConstantNode* src_const = call->args[1].as<ConstantNode>();
      const ConstantNode* dst_const = call->args[2].as<ConstantNode>();
      CHECK(src_const != nullptr && dst_const != nullptr);
      auto src_device = ToDevice(src_const->value);
      auto dst_device = ToDevice(dst_const->value);
      src_dev_type = src_device->device_type;
      dst_dev_type = dst_device->device_type;
    }

    //  Device copy op only has one input which is now annotated with the
    //  same device to the source device type of the device copy op.
    //  The call itself has the same device type to the destination.
    UnifyDeviceCopy(inps, outs, src_dev_type, dst_dev_type);
    MixedModeVisitor::VisitExpr_(call);
  }

  void UnifyAllocStorageCall(const CallNode* call) {
    // [size, alignment]
    CHECK_EQ(call->args.size(), 2U);

    // The arguments of alloc storage should be on CPU.
    for (int i = 0; i < 2; i++) {
      Unify(DeviceFor(call->args[i]), DeviceType(cpu_ctx_));
      MixedModeVisitor::VisitExpr(call->args[i]);
    }
    const auto* attrs = call->attrs.as<tvm::relay::AllocStorageAttrs>();
    Device dev(attrs->virtual_device->ToDevice());
    Unify(DeviceFor(GetRef<Call>(call)), DeviceType(dev));
  }

  void UnifyAllocTensorCall(const CallNode* call) {
    // [storage, offset, shape]
    CHECK_EQ(call->args.size(), 3U);

    Expr storage = call->args[0];
    Expr shape = call->args[1];
    Unify(DeviceFor(storage), DeviceFor(GetRef<Call>(call)));

    // The shape for alloc_tensor should be on CPU.
    Unify(DeviceFor(shape), DeviceType(cpu_ctx_));
    MixedModeVisitor::VisitExpr(shape);
  }

  void UnifyInvokeOpCall(const CallNode* call) {
    // [op, inputs, outputs]
    CHECK_EQ(call->args.size(), 3U);
    Tuple inps = Downcast<Tuple>(call->args[1]);
    Tuple outputs = Downcast<Tuple>(call->args[2]);
    UnifyCall(call->args[0], inps->fields, outputs->fields, Bottom());
    MixedModeVisitor::VisitExpr_(call);
  }

  void UnifyFunctionCall(const CallNode* call) {
    auto device = DeviceFor(GetRef<Call>(call));
    // Unify the arguments of the caller.
    for (const auto& arg : call->args) {
      device = Unify(device, DeviceFor(arg));
      MixedModeVisitor::VisitExpr(arg);
    }

    // Unify the parameters of the callee.
    if (!call->op->IsInstance<FunctionNode>()) return;
    Function func = Downcast<Function>(call->op);
    for (const auto& param : func->params) {
      device = Unify(device, DeviceFor(param));
      MixedModeVisitor::VisitExpr(param);
    }

    // Unify the function expression and its body
    Unify(device, DeviceFor(call->op));
    Unify(device, DeviceFor(func->body));

    // Step into the callee. It will be skipped if the callee if a primitive
    // function
    MixedModeVisitor::VisitExpr(call->op);
  }

  // Invoke a global function.
  void UnifyGlobalVarCall(const CallNode* call) {
    auto device = DeviceFor(GetRef<Call>(call));
    CHECK(mod_.defined()) << "Cannot analyze context on a globalvar without module";
    GlobalVar gv = Downcast<GlobalVar>(call->op);
    auto func = Downcast<Function>(mod_->Lookup(gv));
    CHECK_EQ(call->args.size(), func->params.size())
        << "The number of arguments doesn't match the number of parameters of the function.";

    for (size_t i = 0; i < call->args.size(); i++) {
      Expr arg = call->args[i];
      Expr param = func->params[i];
      MixedModeVisitor::VisitExpr(arg);

      // Save the the arg to function mapping for closures as it will
      // be invoked/unified later.
      CHECK(arg->checked_type().defined())
          << "Type inference is required to run the context analysis passes.";
      if (arg->checked_type()->IsInstance<FuncTypeNode>()) {
        auto it = closures_.find(arg);
        if (it != closures_.end()) {
          closures_[param] = it->second;
        } else {
          CHECK(arg->IsInstance<GlobalVarNode>());
          closures_[param] = Downcast<GlobalVar>(arg);
        }
      }
      Unify(DeviceFor(arg), DeviceFor(param));
    }
    device = Unify(device, DeviceFor(call->op));
    device = Unify(device, DeviceFor(func));
    device = Unify(device, DeviceFor(func->body));

    // Step into the callee. We need to skip recursive calls, otherwise, it
    // would be a infinite loop.
    //
    // TODO(@zhiics) This may cause problem for mutual recursive calls as well.
    auto cur_func = current_func_;
    current_func_ = gv;
    if (cur_func->name_hint != gv->name_hint) {
      MixedModeVisitor::VisitExpr(func);
    }
    // Exit the frame.
    current_func_ = cur_func;
  }

  void UnifyVarCall(const CallNode* call) {
    // It is a closure when we call a var.
    // Unify the corresponding arguement and parameter.
    auto device = DeviceFor(GetRef<Call>(call));
    auto it = closures_.find(call->op);
    CHECK(it != closures_.end()) << "Cannot find var: " << call->op;
    auto glb_var = it->second;
    CHECK(mod_.defined()) << "Cannot analyze context on a globalvar without module";
    Function func = Downcast<Function>(mod_->Lookup(glb_var));
    // Unify the underlying function for clousre or currying functions.
    while (IsClosure(func) || IsCurrying(func)) {
      device = Unify(device, DeviceFor(func));
      if (IsClosure(func)) {
        func = Downcast<Function>(func->body);
      } else if (IsCurrying(func)) {
        Let let = Downcast<Let>(func->body);
        func = Downcast<Function>(mod_->Lookup(closures_[let->var]));
      } else {
        LOG(FATAL) << "func is expected to be a closure or a currying function";
      }
    }

    CHECK_EQ(call->args.size(), func->params.size());
    for (size_t i = 0; i < call->args.size(); i++) {
      Unify(DeviceFor(call->args[i]), DeviceFor(func->params[i]));
      MixedModeVisitor::VisitExpr(call->args[i]);
    }
    device = Unify(device, DeviceFor(call->op));
    device = Unify(device, DeviceFor(glb_var));
    device = Unify(device, DeviceFor(func));

    // Step into the global function.
    auto cur_func = current_func_;
    current_func_ = glb_var;
    if (cur_func->name_hint != glb_var->name_hint) {
      MixedModeVisitor::VisitExpr(func);
    }
    current_func_ = cur_func;
  }

 private:
  /* \brief The cpu context. */
  Device cpu_ctx_ = Device(DevType::kCPU(), 0);
  /* \brief The module that helps context analysis. */
  const IRModule& mod_;
  /* \brief The current function that is being analyzed. */
  GlobalVar current_func_;
  /* \brief The default device that could be attached to an expression. */
  Device default_device_;
  /* \brief The IR node to device domain mapping. */
  std::unordered_map<Expr, DeviceDomainPtr, tvm::ObjectHash, tvm::ObjectEqual> expr_to_device_;
  /* \brief The domain map for union-find. */
  std::unordered_map<DeviceDomainPtr, DeviceDomainPtr, DeviceDomain::Hash, DeviceDomain::Equal>
      device_uf_;
  /*
   * \brief The expr to global var map. It saves the closures/currying that
   * will be invoked lazily.
   */
  std::unordered_map<Expr, GlobalVar, tvm::ObjectHash, tvm::ObjectEqual> closures_;
};

}  // namespace context_analysis

AnalysisResultMap ContextAnalysis(const IRModule& mod, const Device& default_device) {
  auto entry = mod->GetGlobalVar("main");
  auto ca = context_analysis::ContextAnalyzer(mod, entry, default_device);
  auto expr = mod->Lookup(entry);
  ca.VisitExpr(expr);
  return ca.Results();
}

RAF_REGISTER_GLOBAL("raf.pass_.ContextAnalysis").set_body_typed(ContextAnalysis);

}  // namespace pass
}  // namespace raf
