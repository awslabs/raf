#include <mnm/executor.h>
#include <mnm/ir.h>
#include <mnm/memory_pool.h>
#include <mnm/op.h>
#include <mnm/pass.h>
#include <mnm/registry.h>
#include <mnm/tensor.h>
#include <mnm/value.h>
#include "../requests.h"

namespace mnm {
namespace interpreter {

using namespace mnm::ir;
using namespace mnm::value;
using executor::Executor;
using memory_pool::Memory;
using op::MakeOutput;
using op::OpDispatch;
using op::OpEnv;
using op::OpInfo;
using registry::TypedPackedFunc;
using requests::Requests;
using stream_pool::Stream;
using tensor::Tensor;

class Stack {
  using Frame = Map<Var, Value>;

 public:
  std::vector<Frame> frames;

  Stack() : frames() {
    frames.push_back(Frame({}));
  }

  void Extend(const Var& var, const Value& value) {
    frames.back().Set(var, value);
  }

  Value Lookup(const Var& local) {
    for (auto frame = frames.rbegin(); frame != frames.rend(); frame++) {
      auto elem = frame->find(local);
      if (elem != frame->end()) {
        return (*elem).second;
      }
    }
    LOG(FATAL) << "could not find variable binding for " << local->name_hint();
    return Value();
  }

  class LocalFrame {
   public:
    Stack& st;
    explicit LocalFrame(Stack& st, const Frame& fr) : st(st) {
      st.frames.push_back(fr);
    }
    ~LocalFrame() {
      st.frames.pop_back();
    }
  };
};

class Interpreter final : public ExprFunctor<Value(const Expr& n)>, public Executor {
 public:
  Module mod;
  Stack stack;
  std::unordered_map<const ExprNode*, const BoundExprNode*> bindings;

 public:
  Interpreter(Module mod) : mod(mod) {
  }

  ~Interpreter() = default;

  Value Eval(const Expr& expr) {
    const ExprNode* node = expr.as_derived<ExprNode>();
    CHECK(node != nullptr);
    if (bindings.count(node)) {
      return bindings[node]->value;
    }
    return ExprFunctor<Value(const Expr& n)>::VisitExpr(expr);
  }

  Value VisitExpr(const Expr& expr) override {
    return Eval(expr);
  }

  Value VisitExpr_(const VarNode* node) override {
    return stack.Lookup(GetRef<Var>(node));
  }

  Value VisitExpr_(const GlobalVarNode* node) override {
    return Eval(mod->Lookup(GetRef<GlobalVar>(node)));
  }

  Value VisitExpr_(const OpNode* node) override {
    // Q: Why not do eta-expansion?
    // A: Sometimes the frontend may be interested in knowning the op.
    return OpValue::make(GetRef<Op>(node));
  }

  Value VisitExpr_(const FunctionNode* node) override {
    const Function& func = GetRef<Function>(node);
    Map<Var, Value> captured_mod;
    Array<Var> free_vars = pass::FreeVars(func);
    for (const auto& var : free_vars) {
      captured_mod.Set(var, Eval(var));
    }
    return ClosureValue::make(captured_mod, func);
  }

  Value VisitExpr_(const CallNode* node) override {
    const Call& call = GetRef<Call>(node);
    Array<Value> args;
    for (auto arg : call->args) {
      args.push_back(Eval(arg));
    }
    Value fn = Eval(call->op);
    if (const auto* closure = fn.as<ClosureValueNode>()) {
      return InvokeClosure(closure, args, call->attrs);
    } else if (const auto* op = fn.as<OpValueNode>()) {
      return InvokePrimitive(op, args, call->attrs);
    }
    LOG(FATAL) << "InternalError: " << fn->type_key();
    throw;
  }

  Value VisitExpr_(const RelayConstantNode* _node) override {
    const ConstantNode* node = static_cast<const ConstantNode*>(_node);
    return Downcast<Value>(node->value);
  }

  Value VisitExpr_(const LetNode* node) override {
    stack.Extend(node->var, Eval(node->value));
    return Eval(node->body);
  }

  Value VisitExpr_(const IfNode* node) override {
    bool result = Downcast<BoolValue>(Eval(node->cond))->data;
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
    return TupleValue::make({});
  }

 public:
  Value InvokePrimitive(const OpValueNode* node, Array<Value> args, const Attrs& attrs) {
    const Op& op = node->op;
    OpInfo info = MakeOutput(op, args, attrs);
    if (!info->computational) {
      return info->output;
    }
    // TODO(@junrushao1994): resources with non-computational ops
    std::shared_ptr<OpEnv> op_env = OpDispatch::Dispatch(op, info, args, attrs);
    std::shared_ptr<Requests> req = op_env->GetRequests();
    {
      for (int i = 0, n = req->memory.size(); i < n; ++i) {
        this->RequestMemory(req.get(), i);
      }
      for (int i = 0, n = req->workspace.size(); i < n; ++i) {
        this->RequestWorkspace(req.get(), i);
      }
      for (int i = 0, n = req->stream.size(); i < n; ++i) {
        this->RequestStream(req.get(), i);
      }
    }
    op_env->Execute(args, info, attrs);
    {
      req->workspace.clear();
      req->workspace.shrink_to_fit();
      req->stream.clear();
      req->stream.shrink_to_fit();
    }
    info->output->op_env = std::move(op_env);
    return info->output;
  }

  Value InvokeClosure(const ClosureValueNode* node, Array<Value> args, const Attrs& attrs) {
    const Function& func = node->func;
    Map<Var, Value> locals;
    CHECK_EQ(func->params.size(), args.size());
    int n_args = args.size();
    for (int i = 0; i < n_args; ++i) {
      locals.Set(func->params[i], args[i]);
    }
    for (auto it = node->env.begin(); it != node->env.end(); ++it) {
      locals.Set((*it).first, (*it).second);
    }
    {
      Stack::LocalFrame lf(stack, std::move(locals));
      return Eval(func->body);
    }
  }

 public:
  void OnBind(const op::OpEnv* op_env) override {
  }
  void OnDestruct(const op::OpEnv* op_env) override {
  }

  void OnBind(const BoundExprNode* bound_expr) override {
    const ExprNode* expr = bound_expr->expr.as_derived<ExprNode>();
    CHECK(expr != nullptr);
    CHECK_EQ(bindings.count(expr), 0);
    bindings[expr] = bound_expr;
  }

  void OnDestruct(const BoundExprNode* bound_expr) override {
    const ExprNode* expr = bound_expr->expr.as_derived<ExprNode>();
    CHECK(expr != nullptr);
    CHECK_NE(bindings.count(expr), 0);
    bindings.erase(expr);
  }

  void RequestMemory(Requests* req, int index) override {
    Requests::MemoryRequest& entry = req->memory[index];
    CHECK(entry.memory == nullptr);
    std::shared_ptr<Memory> memory = Memory::Alloc(entry.ctx, entry.nbytes);
    *entry.dest = memory->data;
    entry.memory = memory;
  }

  void RequestWorkspace(Requests* req, int index) override {
    Requests::WorkspaceRequest& entry = req->workspace[index];
    CHECK(entry.memory == nullptr);
    std::shared_ptr<Memory> memory = Memory::Alloc(entry.ctx, entry.nbytes);
    *entry.dest = memory->data;
    entry.memory = memory;
  }

  void RequestStream(Requests* req, int index) override {
    Requests::StreamRequest& entry = req->stream[index];
    std::shared_ptr<Stream> stream = Stream::Get(entry.ctx, entry.tag_idx, entry.stream_idx);
    *entry.dest = stream->data();
    entry.stream = stream;
  }
};

static NodeRef DeTuple(const Expr& expr, const Value& value, Executor* executor) {
  // TODO(@junrushao1994): dispatch by type key?
  // make nested lists of BoundExpr
  if (value->derived_from<ScalarValueNode>()) {
    return value;
  }
  if (value->is_type<TensorValueNode>()) {
    BoundExpr ret = BoundExpr::make(expr, value);
    ret->BindExecutor(executor);
    return std::move(ret);
  }
  if (const auto* tuple = value.as<TupleValueNode>()) {
    Array<NodeRef> result;
    int n = static_cast<int>(tuple->fields.size());
    for (int i = 0; i < n; ++i) {
      Expr sub_expr = ir::TupleGetItemNode::make(expr, i);
      Value sub_value = tuple->fields[i];
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      result.push_back(DeTuple(expr, value, executor));
    }
    return std::move(result);
  }
  LOG(FATAL) << "ValueError: cannot de-tuple " << value->type_key();
  throw;
}

TypedPackedFunc<NodeRef(Expr)> CreateInterpreter(Module module) {
  auto intrp = std::make_shared<Interpreter>(module);
  auto packed = [intrp](Expr expr) {
    Value value = intrp->Eval(expr);
    return DeTuple(expr, value, intrp.get());
  };
  return TypedPackedFunc<NodeRef(Expr)>(packed);
}

TVM_REGISTER_API("mnm.executor.CreateInterpreter").set_body_typed(CreateInterpreter);

}  // namespace interpreter
}  // namespace mnm
