/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/interpreter.cc
 * \brief RAF interpreter, a naive implementation of executor
 */
#include "raf/executor.h"
#include "raf/ir.h"
#include "raf/memory_pool.h"
#include "raf/op.h"
#include "raf/pass.h"
#include "raf/registry.h"
#include "raf/tensor.h"
#include "raf/value.h"
#include "raf/binding.h"
#include "raf/profiler.h"
#include "raf/communicator.h"
#include "dmlc/thread_local.h"
#include "../common/shape_utils.h"
#include "../requests.h"
#include "../op/schema/reduce.h"

#include <list>

namespace raf {
namespace executor {
namespace interpreter {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op;
using binding::BindingEntry;
using binding::BindNDArray;
using binding::DeTuple;
using binding::LookupBinding;
using binding::NDArrayBinding;
using binding::SymbolBindingObj;
using common::shape_utils::BytesCompactTensor;
using memory_pool::Memory;
using requests::Requests;
using stream_pool::Stream;
using tensor::Tensor;

class SymbolTable {
 public:
  std::unordered_map<const VarNode*, std::vector<Value>> tab;

  Value Lookup(const Var& var) {
    auto iter = tab.find(var.operator->());
    if (iter != tab.end() && !iter->second.empty()) {
      return iter->second.back();
    }
    BindingEntry entry = LookupBinding(var.operator->());
    if (!entry.defined()) {
      LOG(FATAL) << "could not find variable binding for " << var->name_hint();
      throw;
    }
    if (const auto* sym = entry.as<SymbolBindingObj>()) {
      CHECK(sym->expr.defined());
      return Interpret(sym->expr);
    }
    return Downcast<NDArrayBinding>(entry)->value;
  }

  class AddVar {
   public:
    SymbolTable& st;
    Var var;
    explicit AddVar(SymbolTable& st, const Var& var, const Value& value) : st(st), var(var) {
      st.Extend_(var, value);
    }
    ~AddVar() {
      st.Pop_(var);
    }
  };

  class LocalFrame {
   public:
    SymbolTable& st;
    Map<Var, Value> frame;
    explicit LocalFrame(SymbolTable& st, Map<Var, Value>&& frame) : st(st), frame(frame) {
      for (auto iter : frame) {
        st.Extend_(iter.first, iter.second);
      }
    }
    ~LocalFrame() {
      for (auto iter : frame) {
        st.Pop_(iter.first);
      }
    }
  };

 private:
  void Extend_(const Var& var, const Value& value) {
    tab[var.operator->()].push_back(value);
  }

  void Pop_(const Var& var) {
    std::vector<Value>& values = tab.at(var.operator->());
    CHECK(!values.empty());
    values.pop_back();
  }
};

class Interpreter final : public ExprFunctor<Value(const Expr& n)>, public Executor {
 public:
  SymbolTable st;
  IRModule mod{nullptr};

 public:
  Interpreter() = default;
  ~Interpreter() = default;

  Value Eval(const Expr& expr) {
    return ExprFunctor<Value(const Expr& n)>::VisitExpr(expr);
  }

  Value VisitExpr(const Expr& expr) override {
    return Eval(expr);
  }

  Value VisitExpr_(const VarNode* node) override {
    return st.Lookup(GetRef<Var>(node));
  }

  Value VisitExpr_(const GlobalVarNode* node) override {
    return Eval(mod->Lookup(GetRef<GlobalVar>(node)));
  }

  Value VisitExpr_(const OpNode* node) override {
    // Q: Why not do eta-expansion?
    // A: Sometimes the frontend may be interested in knowning the op.
    return OpValue::make(GetRef<Op>(node));
  }

  Value MakeClosure(const Function& func, Var letrec_name = Var()) {
    Map<Var, Value> captured_mod;
    Array<Var> free_vars = pass::FreeVars(func);
    for (const auto& var : free_vars) {
      // Evaluate the free var (which could be a function call) if it hasn't
      // shown up in a let-binding that has invoked the function. This is usually happens
      // for local recursive function that the free var points to the function itself.
      if (letrec_name.defined() && letrec_name == var) {
        continue;
      }
      captured_mod.Set(var, Eval(var));
    }
    if (letrec_name.defined()) {
      return ClosureValue::make(captured_mod, func, letrec_name);
    }
    return ClosureValue::make(captured_mod, func);
  }

  Value VisitExpr_(const FunctionNode* node) override {
    const Function& func = GetRef<Function>(node);
    return MakeClosure(func);
  }

  Value VisitExpr_(const CallNode* node) override {
    static auto fschema = Op::GetAttrMap<op::FRAFSchema>("FRAFSchema");
    const Call& call = GetRef<Call>(node);
    Array<Value> args;
    for (auto arg : call->args) {
      args.push_back(Eval(arg));
    }
    CallValues call_values = CallValues::make();
    call_values->callee = Eval(call->op);
    if (call_values->callee->IsInstance<ClosureValueObj>()) {
      call_values->args = MakeListArgs(args);
      return InvokeClosure(call_values);
    } else if (const auto* opv = call_values->callee.as<OpValueObj>()) {
      call_values->args = fschema[opv->op](args);
      Value output_value;
      WITH_BASE_PROFILER(call_values->device, opv->op->name, "SchedulingCommunication", {},
                         { output_value = InvokePrimitive(call_values); });
      return output_value;
    }
    LOG(FATAL) << "ValueError: type " << call_values->callee->GetTypeKey() << " is not callable";
    throw;
  }

  Value VisitExpr_(const RelayConstantNode* _node) override {
    const ConstantNode* node = static_cast<const ConstantNode*>(_node);
    return node->value.defined() ? Downcast<Value>(node->value) : NullValue<Value>();
  }

  Value VisitExpr_(const LetNode* node) override {
    Expr body = GetRef<Let>(node);
    // Iteratively visit let nodes to avoid stack overflow.
    std::list<SymbolTable::AddVar> add_vars;
    while (body->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(body);
      if (auto func = let->value.as<FunctionNode>()) {
        auto clo = MakeClosure(GetRef<Function>(func), let->var);
        add_vars.emplace_back(st, let->var, clo);
      } else {
        add_vars.emplace_back(st, let->var, Eval(let->value));
      }
      body = let->body;
    }
    return Eval(body);
  }

  Value VisitExpr_(const IfNode* node) override {
    bool result = GetScalarValueData<bool>(Eval(node->cond));
    return result ? Eval(node->true_branch) : Eval(node->false_branch);
  }

  Value VisitExpr_(const TupleNode* node) override {
    std::vector<Value> values;
    for (const Expr& field : node->fields) {
      values.push_back(Eval(field));
    }
    return TupleValue::make(values);
  }

  Value VisitExpr_(const TupleGetItemNode* node) override {
    TupleValue tuple = Downcast<TupleValue>(Eval(node->tuple));
    int index = node->index;
    int size = static_cast<int>(tuple->fields.size());
    CHECK(0 <= index && index < size) << "IndexError: tuple index out of range";
    Value sub_value = tuple->fields[index];
    if (sub_value->op_env == nullptr) {
      sub_value->op_env = tuple->op_env;
    }
    return sub_value;
  }

  Value VisitExpr_(const RefCreateNode* node) override {
    return RefValue::make(Eval(node->value));
  }

  Value VisitExpr_(const RefReadNode* node) override {
    return Downcast<RefValue>(Eval(node->ref))->value;
  }

  Value VisitExpr_(const RefWriteNode* node) override {
    Downcast<RefValue>(Eval(node->ref))->value = Eval(node->value);
    return TupleValue::make(tvm::Array<Value>({}));
  }

 public:
  Value InvokePrimitive(const CallValues& call) {
    const Op& op = Downcast<OpValue>(call->callee)->op;
    bool use_upper_bound = false;
    static auto upper_bound_map = Op::GetAttrMap<Op>("TRAFUpperBoundOp");
    if (upper_bound_map.count(op)) {
      call->callee = OpValue::make(upper_bound_map[op]);
      use_upper_bound = true;
    }
    RunDeclare(call);
    if (!call->callee.defined()) {
      return call->out;
    }
    ICHECK(call->out.defined()) << "ValueError: Tensor compute of " << op->name
                                << " is not implemented.";
    AllocOutputBuffer(call->out);
    std::shared_ptr<OpEnv> op_env = Dispatch(call);
    if (op_env != nullptr) {
      InvokePrimitiveOpEnv(std::move(op_env), call, use_upper_bound);
    } else {
      LOG(FATAL) << "ValueError: Cannot dispatch " << op->name << "@" << call->device.c_str();
      throw;
    }
    return call->out;
  }

  void RunDeclare(const CallValues& call) {
    static const auto f_op_make_output = Op::GetAttrMap<FRAFDeclare>("FRAFDeclare");
    const Op& op = Downcast<OpValue>(call->callee)->op;
    const auto& f = f_op_make_output[op];
    f(call);
  }

  void InvokePrimitiveOpEnv(std::shared_ptr<OpEnv> op_env, const CallValues& call,
                            bool use_upper_bound) {
    const Op& op = Downcast<OpValue>(call->callee)->op;
    std::shared_ptr<Requests> req = op_env->GetRequests();
    {
      // note: Request workspace, workspace is kind of special memory which will be freed once
      // this op is done.
      WITH_BASE_PROFILER(call->device, op->name, "WorkspaceRequest",
                         {"Count: " + std::to_string(req->workspace.size())}, {
                           for (int i = 0, n = req->workspace.size(); i < n; ++i) {
                             RequestWorkspace(req.get(), i);
                           }
                         });

      // note: Request stream, every op will run on a given stream. For op that executed on
      // cuda, the default one is cuda DefautlStream. Currently, all ops are running on default
      // stream.
      WITH_BASE_PROFILER(call->device, op->name, "StreamRequest",
                         {"Count: " + std::to_string(req->stream.size())}, {
                           for (int i = 0, n = req->stream.size(); i < n; ++i) {
                             RequestStream(req.get(), i);
                           }
                         });

      // note: Request distributed resources, operators like allreduce needs such resources.
      // Currently, the distributed resources only contain a communicator.
      WITH_BASE_PROFILER(call->device, op->name, "DistributedRequest",
                         {"Count: " + std::to_string(req->distributed.size())}, {
                           for (int i = 0, n = req->distributed.size(); i < n; ++i) {
                             RequestDistributed(req.get(), i);
                           }
                         });
    }

    // note: Execute the Operator.
    WITH_BASE_PROFILER(call->device, op->name, "CUDA_CALL", {}, { op_env->Execute(call); });

    {
      // note: Force op to run synchronously.
      for (int i = 0, n = req->stream.size(); i < n; ++i) {
        req->stream[i].stream->Wait();
      }
      // note: Free the workspace of this op.
      WITH_BASE_PROFILER(call->device, op->name, "WorkspaceClear", {}, {
        req->workspace.clear();
        req->workspace.shrink_to_fit();
      });

      req->stream.clear();
      req->stream.shrink_to_fit();
    }

    // note: The next op holds a reference to this op. It will make sure that the memories requested
    // by this op will not be freed after the return of this op.
    call->out->op_env = std::move(op_env);

    if (use_upper_bound) {
      auto tup = Downcast<TupleValue>(call->out);
      auto data = Downcast<TensorValue>(tup->fields[0]);
      auto shape_data = Downcast<TensorValue>(tup->fields[1]);
      shape_data = Downcast<TensorValue>(CopyTo(shape_data, Device(DevType::kCPU(), 0)));
      auto shape = common::shape_utils::GetShapeVecFromData(shape_data);
      auto new_out = data.CreateView(shape);
      call->out = new_out;
    }
  }

 public:
  Value InvokeClosure(const CallValues& call) {
    const auto* node = call->callee.as<ClosureValueObj>();
    const Function& func = node->func;
    const Array<Value>& call_args = GetListArgs(call->args);
    Map<Var, Value> locals;
    CHECK_EQ(func->params.size(), call_args.size());
    int n_args = call_args.size();
    for (int i = 0; i < n_args; ++i) {
      locals.Set(func->params[i], call_args[i]);
    }
    for (auto it = node->env.begin(); it != node->env.end(); ++it) {
      locals.Set((*it).first, (*it).second);
    }
    if (node->bind.defined()) {
      locals.Set(node->bind.value(), call->callee);
    }
    {
      SymbolTable::LocalFrame lf(st, std::move(locals));
      return Eval(func->body);
    }
  }

 public:
  void OnBind(const op::OpEnv* op_env) override {
  }

  void OnDestruct(const op::OpEnv* op_env) override {
  }

  void RequestWorkspace(Requests* req, int index) override {
    Requests::WorkspaceRequest& entry = req->workspace[index];
    CHECK(entry.memory == nullptr);
    std::shared_ptr<Memory> memory = Memory::Alloc(entry.device, entry.nbytes);
    *entry.dest = memory->data;
    entry.memory = memory;
  }

  void RequestStream(Requests* req, int index) override {
    Requests::StreamRequest& entry = req->stream[index];
    std::shared_ptr<Stream> stream = Stream::Get(entry.device, entry.tag_idx, entry.stream_idx);
    *entry.dest = stream->data();
    entry.stream = stream;
  }

  void RequestDistributed(Requests* req, int index) override {
    Requests::DistributedRequest& entry = req->distributed[index];
    *entry.dest = distributed::communicator::CommunicatorManager::Get()->GetCommunicator();
  }

 private:
  void AllocOutputBuffer(Value& out) {
    std::vector<DLTensor*> out_tensors;
    std::vector<TensorValue> out_tvs;
    if (out->IsInstance<TensorValueObj>()) {
      DLTensor* t = out;
      out_tensors.emplace_back(t);
      out_tvs.push_back(Downcast<TensorValue>(out));
    } else if (const auto* tv = out.as<TupleValueObj>()) {
      for (const auto& v : tv->fields) {
        DLTensor* t = v;
        out_tensors.emplace_back(t);
        out_tvs.push_back(Downcast<TensorValue>(v));
      }
    } else if (out->IsInstance<VoidValueObj>()) {
      // do nothing.
    } else {
      LOG(FATAL) << "InternalError: Interpreter does not deal with " << out->GetTypeKey();
      throw;
    }
    CHECK_EQ(out_tensors.size(), out_tvs.size());
    for (size_t i = 0; i < out_tensors.size(); ++i) {
      DLTensor* dlt = out_tensors[i];
      TensorValue tv = out_tvs[i];
      if (dlt->data == nullptr) {
        std::shared_ptr<Memory> memory = Memory::Alloc(dlt->device, BytesCompactTensor(*dlt));
        dlt->data = memory->data;
        tv->mem = std::move(memory);
      }
    }
  }
};

class IntrpThreadEntry {
 public:
  IntrpThreadEntry() = default;

  static Interpreter* ThreadLocal() {
    using TLS = dmlc::ThreadLocalStore<IntrpThreadEntry>;
    return &TLS::Get()->exec;
  }
  Interpreter exec;
};

Value Interpret(Expr expr, Optional<IRModule> mod) {
  Interpreter* intrp = IntrpThreadEntry::ThreadLocal();
  intrp->mod = mod.defined() ? mod.value() : GlobalModule();
  auto ret = intrp->Eval(expr);
  intrp->mod = {};
  intrp->st.tab = {};
  return ret;
}

Value InvokePrimitive(const CallValues& call) {
  Interpreter* intrp = IntrpThreadEntry::ThreadLocal();
  auto ret = intrp->InvokePrimitive(call);
  intrp->mod = {};
  intrp->st.tab = {};
  return ret;
}

Value InvokeClosure(const CallValues& call) {
  Interpreter* intrp = IntrpThreadEntry::ThreadLocal();
  auto ret = intrp->InvokeClosure(call);
  intrp->mod = {};
  intrp->st.tab = {};
  return ret;
}

ObjectRef _Interpret(Expr expr, Optional<IRModule> mod) {
  return DeTuple(Interpret(expr, mod));
}

RAF_REGISTER_GLOBAL("raf.executor.Interpret").set_body_typed(_Interpret);
}  // namespace interpreter
}  // namespace executor
}  // namespace raf
