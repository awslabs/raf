/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/vm/compiler.cc
 * \brief The RAF virtual machine compiler.
 */
#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/target/target.h>
#include <tvm/tir/op.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/binding.h"
#include "raf/type.h"
#include "raf/pass.h"
#include "raf/dist_config.h"
#include "./compiler.h"

namespace tvm {
namespace relay {
namespace vm {
bool IsClosure(const Function& func);
}  // namespace vm
}  // namespace relay
}  // namespace tvm

namespace raf {
namespace executor {
namespace vm {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using binding::LookupBinding;
using binding::NDArrayBinding;
using raf::distributed::DistConfig;
using tvm::relay::Shape;

/*!
 * \brief Bind params to function by using name
 * \param func Relay function
 * \param params params dict
 * \return Function
 */
inline Function BindParamsByName(Function func,
                                 const std::unordered_map<std::string, Value>& params) {
  std::unordered_map<std::string, Var> name_dict;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> repeat_var;
  for (auto arg : func->params) {
    const auto& name = arg->name_hint();
    if (name_dict.count(name)) {
      repeat_var.insert(arg);
    } else {
      name_dict[name] = arg;
    }
  }

  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> bind_dict;
  for (auto& kv : params) {
    if (name_dict.count(kv.first) == 0) {
      continue;
    }
    auto arg = name_dict.at(kv.first);
    if (repeat_var.count(arg)) {
      LOG(FATAL) << "Multiple args in the function have name " << kv.first;
    }
    bind_dict[arg] = MakeConstant(kv.second);
  }
  Expr bound_expr = tvm::relay::Bind(func, bind_dict);
  Function ret = Downcast<Function>(bound_expr);
  CHECK(ret.defined()) << "The returning type is expected to be a Relay Function."
                       << "\n";
  return ret;
}

/*! \brief A helper class for matching and rewriting operators. */
template <typename R>
class OpMatch {
 public:
  using MatchFunc =
      std::function<R(const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_args)>;

  /*! \brief Match an operator with the given name.
   *  \param op_name The name of the operator to match.
   *  \param func The function to execute when it matches.
   *  \return A self-reference for builder style API.
   */
  inline OpMatch& Match(const std::string& op_name, MatchFunc func) {
    auto op = Op::Get(op_name);
    match_map_.insert({op, func});
    return *this;
  }

  /*! \brief Rewrite a call operation based on the operator and the registered
   *  match functions.
   * \param call The call to rewrite.
   * \return The result of rewriting.
   */
  inline R operator()(const Call& call) {
    auto it = match_map_.find(Downcast<Op>(call->op));
    if (it != match_map_.end()) {
      return it->second(call->args, call->attrs, call->type_args);
    } else {
      if (default_ != nullptr) {
        return default_(call->args, call->attrs, call->type_args);
      } else {
        LOG(FATAL) << "unexpected operation " << call->op;
      }
    }
  }

 private:
  /*! \brief The match function map. */
  std::unordered_map<Op, MatchFunc, ObjectPtrHash, ObjectPtrEqual> match_map_;
  /*! \brief An optional default case. */
  MatchFunc default_;
};

// Represent a runtime object that's going to be matched by pattern match expressions
struct MatchValue {
  virtual ~MatchValue() {
  }
};
using MatchValuePtr = std::shared_ptr<MatchValue>;

// A runtime object that resides in a register
struct RegisterValue : MatchValue {
  // The register num
  RegName rergister_num;

  explicit RegisterValue(RegName reg) : rergister_num(reg) {
  }

  ~RegisterValue() {
  }
};

// The value is a field of another runtime object
struct AccessField : MatchValue {
  MatchValuePtr parent;
  // Field index
  size_t index;
  // Runtime register num after compiling the access field path
  RegName reg{-1};

  AccessField(MatchValuePtr parent, size_t index) : parent(parent), index(index) {
  }

  ~AccessField() {
  }
};

/*!
 * \brief Condition in a decision tree
 */
struct ConditionNode {
  virtual ~ConditionNode() {
  }
};

using ConditionObjectPtr = std::shared_ptr<ConditionNode>;

/*!
 * \brief A var binding condition
 */
struct VarBinding : ConditionNode {
  Var var;
  MatchValuePtr val;

  VarBinding(Var var, MatchValuePtr val) : var(var), val(val) {
  }

  ~VarBinding() {
  }
};

/*!
 * \brief Compare the tag of the object
 */
struct TagCompare : ConditionNode {
  /*! \brief The object to be examined */
  MatchValuePtr obj;

  /*! \brief The expected tag */
  int target_tag;

  TagCompare(MatchValuePtr obj, size_t target) : obj(obj), target_tag(target) {
  }

  ~TagCompare() {
  }
};

class VMFunctionCompiler : ExprFunctor<void(const Expr& expr)> {
 public:
  VMFunctionCompiler(VMCompilerContext* context, DeviceMap device_map)
      : last_register_(0), registers_num_(0), context_(context), device_map_(device_map) {
  }

  VMFunction Compile(const GlobalVar& var, const Function& func) {
    size_t i = 0;
    // We then assign register num to the free variables
    for (auto param : func->params) {
      auto arg_register = NewRegister();
      CHECK_EQ(i, arg_register);
      var_register_map_.insert({param, arg_register});
      params_.push_back(param->name_hint());
      ++i;
    }

    if (tvm::relay::vm::IsClosure(func)) {
      Function inner_func;
      if (const auto* fn = func->body.as<FunctionNode>()) {
        inner_func = GetRef<Function>(fn);
      } else if (const auto* ln = func->body.as<LetNode>()) {
        CHECK_EQ(ln->var, ln->body);
        inner_func = Downcast<Function>(ln->value);
      }
      for (auto param : inner_func->params) {
        auto arg_register = NewRegister();
        CHECK_EQ(i, arg_register);
        var_register_map_.insert({param, arg_register});
        params_.push_back(param->name_hint());
        ++i;
      }
      this->VisitExpr(inner_func->body);
    } else {
      this->VisitExpr(func->body);
    }
    instructions_.push_back(Instruction::Ret(last_register_));
    return VMFunction(var->name_hint, params_, instructions_, registers_num_);
  }

 protected:
  size_t NewRegister() {
    return registers_num_++;
  }

  inline void Emit(const Instruction& instr) {
    DLOG(INFO) << "VMCompiler::Emit: instr=" << instr;
    CHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
    switch (instr.op) {
      case Opcode::AllocTuple:
      case Opcode::AllocTensor:
      case Opcode::AllocTensorReg:
      case Opcode::GetField:
      case Opcode::LoadConst:
      case Opcode::LoadConsti:
      case Opcode::InvokeFunc:
      case Opcode::AllocClosure:
      case Opcode::AllocStorage:
      case Opcode::Move:
      case Opcode::InvokeClosure:
      case Opcode::InferType:
      case Opcode::SetShape:
        last_register_ = instr.dst;
        break;
      case Opcode::InvokePacked:
      case Opcode::InvokeJit:
      case Opcode::Free:
      case Opcode::If:
      case Opcode::Ret:
      case Opcode::Goto:
      case Opcode::Fatal:
      case Opcode::CudaSetStream:
      case Opcode::CudaAddEvent:
      case Opcode::CudaWaitEvent:
      case Opcode::CudaStreamBarrier:
        last_register_ = -1;
        break;
    }
    instructions_.push_back(instr);
  }

  void VisitExpr_(const ConstantNode* const_node) {
    size_t konst_idx = context_->constants.size();
    context_->constants.push_back(Downcast<Value>(const_node->value));
    Emit(Instruction::LoadConst(konst_idx, NewRegister()));
  }

  void VisitExpr_(const RelayConstantNode* const_node) {
    VisitExpr_(static_cast<const ConstantNode*>(const_node));
  }

  void VisitExpr_(const OpNode* op_node) {
    VisitExpr(MakeConstant(OpValue::make(GetRef<Op>(op_node))));
  }

  void VisitExpr_(const VarNode* var_node) {
    auto var = GetRef<Var>(var_node);
    auto reg_it = this->var_register_map_.find(var);
    CHECK(reg_it != this->var_register_map_.end());
    last_register_ = reg_it->second;
  }

  void VisitExpr_(const TupleNode* tuple_node) {
    auto tuple = GetRef<Tuple>(tuple_node);
    std::vector<Index> fields_registers;

    for (auto& field : tuple->fields) {
      this->VisitExpr(field);
      fields_registers.push_back(last_register_);
    }

    Emit(Instruction::AllocTuple(fields_registers, NewRegister()));
  }

  void VisitExpr_(const LetNode* let_node) {
    Expr body = GetRef<Let>(let_node);
    // Iteratively visit let nodes to avoid stack overflow.
    while (body->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(body);
      DLOG(INFO) << PrettyPrint(let->value);
      expr_map_[let->var] = let->value;
      this->VisitExpr(let->value);
      var_register_map_.insert({let->var, this->last_register_});
      body = let->body;
    }
    this->VisitExpr(body);
  }

  void VisitExpr_(const TupleGetItemNode* get_node) {
    auto get = GetRef<TupleGetItem>(get_node);
    this->VisitExpr(get->tuple);
    auto tuple_register = last_register_;
    Emit(Instruction::GetField(tuple_register, get->index, NewRegister()));
  }

  void VisitExpr_(const GlobalVarNode* gvar) {
    auto var = GetRef<GlobalVar>(gvar);
    auto func = context_->module->Lookup(var);
    auto it = context_->global_map.find(var);
    CHECK(it != context_->global_map.end());
    // Allocate closure with zero free vars
    Emit(Instruction::AllocClosure(it->second, {}, NewRegister()));
  }

  void VisitExpr_(const IfNode* if_node) {
    this->VisitExpr(if_node->cond);

    size_t test_register = last_register_;

    this->Emit(Instruction::LoadConsti(1, NewRegister()));
    auto after_cond = instructions_.size();
    auto target_register = last_register_;
    this->Emit(Instruction::If(test_register, target_register, 0, 0));
    this->VisitExpr(if_node->true_branch);

    // It saves the result of If-Else expression.
    auto merge_register = NewRegister();
    Emit(Instruction::Move(last_register_, merge_register));
    Emit(Instruction::Goto(0));

    // Finally store how many instructions there are in the
    // true branch.
    auto after_true = this->instructions_.size();

    this->VisitExpr(if_node->false_branch);

    size_t false_register = last_register_;

    // In else-branch, override the then-branch register
    Emit(Instruction::Move(false_register, merge_register));
    // Compute the total number of instructions
    // after generating false.
    auto after_false = this->instructions_.size();

    // Now we will compute the jump targets in order
    // to properly patch the instruction with the
    // the requiste targets.

    // After we emit the true body, and false body,
    // we patch up the if instruction, and goto.
    auto true_offset = 1;
    auto false_offset = after_true - after_cond;
    instructions_[after_cond].if_op.true_offset = true_offset;
    instructions_[after_cond].if_op.false_offset = false_offset;

    // Patch the Goto.
    this->instructions_[after_true - 1].pc_offset = (after_false - after_true) + 1;

    this->last_register_ = merge_register;
  }

  void VisitExpr_(const CallNode* call_node) {
    Expr op = call_node->op;

    // First we handle the case in which we are using an opaque
    // operator used to define a sub-dialect, such as memory
    // allocation operations.
    if (op.as<OpNode>()) {
      OpMatch<void> matcher;
      matcher
          .Match("raf.op.vm.invoke_op",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 3);
                   EmitInvokeOp(args[0], args[1], args[2]);
                 })
          .Match("raf.op.vm.infer_type",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 2);
                   EmitInferType(args[0], args[1], NewRegister());
                 })
          .Match("raf.op.vm.alloc_tensor",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   bool own = true;
                   if (args.size() == 5) {
                     // The last "own" argument is usually specified by the MemoryPlan pass
                     // to indicate that this tensor is not the final output so it should not
                     // own the memory pointer.
                     CHECK(args[4].as<ConstantNode>());
                     auto own_val = args[4].as<ConstantNode>()->value;
                     CHECK(own_val->IsInstance<BoolValueObj>());
                     own = own_val.as<BoolValueObj>()->value;
                   } else {
                     CHECK_EQ(args.size(), 4);
                   }

                   // The storage will be passed dynamically.
                   this->VisitExpr(args[0]);
                   auto storage_register = last_register_;

                   // If the shape is constant then we will emit a static tensor allocation
                   // instruction.
                   auto const_shape = args[1].as<ConstantNode>();

                   // dtype
                   CHECK(args[2]->IsInstance<ConstantNode>());
                   auto dtype_val = args[2].as<ConstantNode>()->value;
                   CHECK(dtype_val->IsInstance<StringValueObj>());
                   std::string dtype_s = dtype_val.as<StringValueObj>()->value;
                   DataType dtype(String2DLDataType(dtype_s));

                   if (const_shape) {
                     Shape shape = Downcast<Shape>(const_shape->value);
                     std::vector<int64_t> raw_shape;
                     for (auto dim : shape) {
                       auto imm = dim.as<tvm::IntImmNode>();
                       CHECK(imm);
                       raw_shape.push_back(imm->value);
                     }
                     // Add context field.
                     Emit(Instruction::AllocTensor(storage_register, 0, raw_shape, dtype,
                                                   NewRegister(), own));
                   } else {
                     this->VisitExpr(args[1]);
                     Emit(Instruction::AllocTensorReg(storage_register, 0, last_register_, dtype,
                                                      NewRegister(), own));
                   }
                 })
          .Match("raf.op.vm.alloc_storage",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK(args.size() == 5 || args.size() == 6);
                   // Compute the size of the allocation.
                   this->VisitExpr(args[0]);
                   auto size_register = last_register_;

                   // Alignment
                   CHECK(args[1]->IsInstance<ConstantNode>());
                   auto align_val = args[1].as<ConstantNode>()->value;
                   CHECK(align_val->IsInstance<IntValueObj>());
                   Index alignment = align_val.as<IntValueObj>()->value;

                   // device type
                   CHECK(args[2]->IsInstance<ConstantNode>());
                   auto device_type_val = args[2].as<ConstantNode>()->value;
                   CHECK(device_type_val->IsInstance<IntValueObj>());
                   Index device_type = device_type_val.as<IntValueObj>()->value;

                   // device id
                   CHECK(args[3]->IsInstance<ConstantNode>());
                   auto device_id_val = args[3].as<ConstantNode>()->value;
                   CHECK(device_id_val->IsInstance<IntValueObj>());
                   Index device_id = device_id_val.as<IntValueObj>()->value;

                   // dtype
                   CHECK(args[4]->IsInstance<ConstantNode>());
                   auto dtype_val = args[4].as<ConstantNode>()->value;
                   CHECK(dtype_val->IsInstance<StringValueObj>());
                   std::string dtype_s = dtype_val.as<StringValueObj>()->value;
                   DataType dtype(String2DLDataType(dtype_s));

                   // alloc_async
                   bool alloc_async = true;
                   if (args.size() == 6) {
                     CHECK(args[5]->IsInstance<ConstantNode>());
                     auto async_val = args[5].as<ConstantNode>()->value;
                     CHECK(async_val->IsInstance<BoolValueObj>());
                     alloc_async = async_val.as<BoolValueObj>()->value;
                   }

                   Emit(Instruction::AllocStorage(size_register, alignment, dtype, device_type,
                                                  device_id, NewRegister(), alloc_async));
                 })
          .Match("raf.op.vm.free",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 1);

                   this->VisitExpr(args[0]);
                   auto memory_register = last_register_;
                   Emit(Instruction::Free(memory_register));
                 })
          .Match("raf.op.vm.set_shape",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 2);
                   this->VisitExpr(args[0]);
                   auto data_reg = last_register_;
                   // The shape argument may be a constant or a tensor
                   this->VisitExpr(args[1]);
                   auto shape_reg = last_register_;
                   Emit(Instruction::SetShape(data_reg, shape_reg, NewRegister()));
                 })
          .Match(
              "raf.op.set_stream",
              [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                CHECK_EQ(args.size(), 2);
                this->VisitExpr(args[0]);
                Expr device_id_expr;
                if (args[0].as<VarNode>()) {
                  device_id_expr = expr_map_[GetRef<Var>(args[0].as<VarNode>())];
                } else {
                  device_id_expr = args[0];
                }
                Index device_id = device_id_expr.as<ConstantNode>()->value.as<IntValueObj>()->value;
                this->VisitExpr(args[1]);
                Expr stream_id_expr;
                if (args[1].as<VarNode>()) {
                  stream_id_expr = expr_map_[GetRef<Var>(args[1].as<VarNode>())];
                } else {
                  stream_id_expr = args[1];
                }
                Index stream_id = stream_id_expr.as<ConstantNode>()->value.as<IntValueObj>()->value;
                Emit(Instruction::CudaSetStream(device_id, stream_id));
              })
          .Match("raf.op.add_event",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK(args.size() == 1 || args.size() == 2);
                   Expr event_id_expr;
                   if (args[0].as<VarNode>()) {
                     event_id_expr = expr_map_[GetRef<Var>(args[0].as<VarNode>())];
                   } else {
                     event_id_expr = args[0];
                   }
                   Index event_id =
                       event_id_expr.as<ConstantNode>()->value.as<IntValueObj>()->value;
                   Index stream_id;
                   if (args.size() == 2) {
                     Expr stream_id_expr;
                     if (args[1].as<VarNode>()) {
                       stream_id_expr = expr_map_[GetRef<Var>(args[1].as<VarNode>())];
                     } else {
                       stream_id_expr = args[1];
                     }
                     stream_id = stream_id_expr.as<ConstantNode>()->value.as<IntValueObj>()->value;
                   } else {
                     // default stream_id to -1
                     stream_id = -1;
                   }
                   Emit(Instruction::CudaAddEvent(event_id, stream_id));
                 })
          .Match("raf.op.wait_event",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK(args.size() == 1 || args.size() == 2);
                   Expr event_id_expr;
                   if (args[0].as<VarNode>()) {
                     event_id_expr = expr_map_[GetRef<Var>(args[0].as<VarNode>())];
                   } else {
                     event_id_expr = args[0];
                   }
                   Index event_id =
                       event_id_expr.as<ConstantNode>()->value.as<IntValueObj>()->value;
                   Index stream_id;
                   if (args.size() == 2) {
                     Expr stream_id_expr;
                     if (args[1].as<VarNode>()) {
                       stream_id_expr = expr_map_[GetRef<Var>(args[1].as<VarNode>())];
                     } else {
                       stream_id_expr = args[1];
                     }
                     stream_id = stream_id_expr.as<ConstantNode>()->value.as<IntValueObj>()->value;
                   } else {
                     // default stream_id to -1
                     stream_id = -1;
                   }
                   Emit(Instruction::CudaWaitEvent(event_id, stream_id));
                 })
          .Match("raf.op.stream_barrier",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 0);
                   Emit(Instruction::CudaStreamBarrier());
                 });
      matcher(GetRef<Call>(call_node));
      return;
    }

    // In the case its not one of these specialized operators we will generate code
    // for one of the "standard" cases.
    std::vector<Index> args_registers;

    for (auto arg : call_node->args) {
      this->VisitExpr(arg);
      args_registers.push_back(last_register_);
    }

    if (auto global_node = op.as<GlobalVarNode>()) {
      // In the case we are invoking a global we need to find its
      // global ID, and then check whether it is closure invocation
      // or whether it is a standard global, and emit the correct
      // calling convention.
      auto global = GetRef<GlobalVar>(global_node);
      auto it = context_->global_map.find(global);
      CHECK(it != context_->global_map.end());
      DLOG(INFO) << "VisitExpr_: generating invoke for " << global->name_hint
                 << " with func_index=" << it->second;

      // TODO(tvm-team):
      // Think about mixed call into global that is not a relay::Function
      // perhaps establish as an invariance(all functions in mod must be relay::Function)
      auto func = Downcast<Function>(context_->module->Lookup(global));

      if (tvm::relay::vm::IsClosure(func)) {
        auto arity = func->params.size();
        Emit(Instruction::AllocClosure(it->second, args_registers, NewRegister()));
      } else {
        Emit(Instruction::InvokeFunc(it->second, args_registers, NewRegister()));
      }
    } else if (auto var_node = op.as<VarNode>()) {
      // If we are calling a variable, it must be the case that it is a closure so we
      // emit invoke closure here.
      VisitExpr(GetRef<Var>(var_node));
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else if (auto inner_call_node = op.as<CallNode>()) {
      VisitExpr(GetRef<Call>(inner_call_node));
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else {
      // Finally if there are any other cases this is a bug.
      LOG(FATAL) << "internal error: unreachable code,"
                 << "should be transformed away by previous passes"
                 << PrettyPrint(GetRef<Expr>(call_node));
    }
  }

  void VisitExpr_(const FunctionNode* func_node) {
    if (func_node->HasNonzeroAttr(attr::kPrimitive)) {
      VisitExpr(MakeConstant(ClosureValue::make({}, GetRef<Function>(func_node))));
    } else {
      LOG(FATAL) << "local functions should have been removed by lambda lifting:" << std::endl
                 << "Program: " << ir::AsText(GetRef<Function>(func_node), false) << std::endl
                 << "AST: " << GetRef<Function>(func_node);
    }
  }

  /*!
   * \brief Compile a match value
   * Generate byte code that compute the value specificed in val
   *
   * \return The register number assigned for the final value
   */
  RegName CompileMatchValue(MatchValuePtr val) {
    if (std::dynamic_pointer_cast<RegisterValue>(val)) {
      auto r = std::dynamic_pointer_cast<RegisterValue>(val);
      return r->rergister_num;
    } else {
      auto path = std::dynamic_pointer_cast<AccessField>(val);
      auto p = CompileMatchValue(path->parent);
      Emit(Instruction::GetField(p, path->index, NewRegister()));
      path->reg = last_register_;
      return path->reg;
    }
  }

  void EmitInvokeOp(const Expr& op, const Expr& inputs, const Expr& outputs) {
    VisitExpr(op);
    std::vector<Index> argument_registers;
    auto op_reg = last_register_;

    CHECK(inputs.as<VarNode>() && outputs.as<VarNode>())
        << "internal error: invoke_op inputs/outputs must be binded";

    auto input_var = Downcast<Var>(inputs);
    CHECK_GT(expr_map_.count(input_var), 0)
        << "internal error: cannot find the input value in the expression map";
    auto input_tuple = expr_map_[input_var].as<TupleNode>();
    CHECK(input_tuple) << "internal error: invoke_op inputs must be a tuple,"
                       << "please file a bug in the memory manifestation pass";

    auto output_var = Downcast<Var>(outputs);
    CHECK_GT(expr_map_.count(output_var), 0)
        << "internal error: cannot find the output value in the expression map";
    auto output_tuple = expr_map_[output_var].as<TupleNode>();
    CHECK(output_tuple) << "internal error: invoke_op outputs must be a tuple,"
                        << "please file a bug in the memory manifestation pass";
    for (auto input : input_tuple->fields) {
      if (input.as<VarNode>()) {
        auto reg = var_register_map_.find(Downcast<Var>(input));
        CHECK(reg != var_register_map_.end())
            << "internal error: all variables should be in the register mapping";
        argument_registers.push_back(reg->second);
      } else {
        // We have to cover this special case because we may run InferType, which does
        // constant folding, after ManifestAlloc, so the IR may not be a strict ANF.
        CHECK(input.as<ConstantNode>());
        VisitExpr(input);
        argument_registers.push_back(last_register_);
      }
    }

    for (auto output : output_tuple->fields) {
      auto reg = var_register_map_.find(Downcast<Var>(output));
      CHECK(reg != var_register_map_.end())
          << "internal error: all variables should be in the register mapping";
      argument_registers.push_back(reg->second);
    }

    CHECK_EQ(device_map_.size(), 1U)
        << "Currently VM compiler doesn't support heterogeneous compilation";
    Emit(Instruction::InvokeJit(op_reg, argument_registers.size(), output_tuple->fields.size(),
                                argument_registers));
  }

  void EmitInferType(const Expr& op, const Expr& inputs, RegName dst) {
    VisitExpr(op);
    std::vector<Index> argument_registers;
    auto op_reg = last_register_;

    auto input_var = Downcast<Var>(inputs);
    CHECK_GT(expr_map_.count(input_var), 0)
        << "internal error: cannot find the input value in the expression map";
    auto input_tuple = expr_map_[input_var].as<TupleNode>();
    CHECK(input_tuple) << "internal error: infer_type inputs must be a tuple,"
                       << "please file a bug in the memory manifestation pass";

    for (auto input : input_tuple->fields) {
      if (input.as<VarNode>()) {
        auto reg = var_register_map_.find(Downcast<Var>(input));
        CHECK(reg != var_register_map_.end())
            << "internal error: all variables should be in the register mapping";
        argument_registers.push_back(reg->second);
      } else {
        // We have to cover this special case because we may run InferType, which does
        // constant folding, after ManifestAlloc, so the IR may not be a strict ANF.
        CHECK(input.as<ConstantNode>());
        VisitExpr(input);
        argument_registers.push_back(last_register_);
      }
    }

    Emit(Instruction::InferType(op_reg, argument_registers, dst));
  }

 protected:
  /*! \brief Store the expression a variable points to. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief Instructions in the VMFunction. */
  std::vector<Instruction> instructions_;
  /*! \brief Parameter names of the function. */
  std::vector<std::string> params_;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectPtrHash, ObjectPtrEqual> var_register_map_;
  /*! \brief Last used register number. */
  Index last_register_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_;
  /*! \brief Global shared meta data */
  VMCompilerContext* context_;
  /*! \brief Device map. */
  DeviceMap device_map_;
};

void VMCompiler::SetParam(const std::string& name, Value data_in) {
  params_[name] = data_in;
}

void VMCompiler::Lower(IRModule mod, const DeviceMap& device_map) {
  CHECK_EQ(device_map.size(), 1U)
      << "Currently VM compiler doesn't support heterogeneous compilation";
  if (params_.size()) {
    BaseFunc base_func = mod->Lookup("main");
    CHECK(base_func->IsInstance<FunctionNode>())
        << "VM compiler expects to compile relay::Function";
    auto f = BindParamsByName(Downcast<Function>(base_func), params_);
    auto gvar = mod->GetGlobalVar("main");
    mod->Add(gvar, f, true);
  }

  exec_ = make_object<Executable>();
  device_map_ = device_map;

  // Run the optimizations necessary to target the VM.
  context_.module = OptimizeModule(mod, device_map_);

  // Populate the global map.
  //
  // This maps global variables to a global index
  // in the VMFunction table.
  PopulateGlobalMap();

  // Next we get ready by allocating space for
  // the global state.
  exec_->functions.resize(context_.module->functions.size());

  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    if (auto* n = named_func.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(n);
      VMFunctionCompiler func_compiler(&context_, device_map_);
      auto vm_func = func_compiler.Compile(gvar, func);

      size_t func_index = context_.global_map.at(gvar);
      CHECK(func_index < exec_->functions.size());
      exec_->functions[func_index] = vm_func;
    }
  }

#if USE_RELAY_DEBUG
  for (auto vm_func : exec_->functions) {
    DLOG(INFO) << vm_func << "-------------";
  }
#endif  // USE_RELAY_DEBUG

  // populate constants
  for (const Value& data : context_.constants) {
    exec_->constants.push_back(data);
  }

  // update global function map
  for (auto gv : context_.global_map) {
    exec_->global_map.insert({gv.first->name_hint, gv.second});
  }
}

IRModule VMCompiler::OptimizeModule(const IRModule& mod, const DeviceMap& device_map) {
  CHECK_EQ(device_map.size(), 1U)
      << "Currently VM compiler doesn't support heterogeneous compilation";
  const auto& it = device_map.begin();
  tvm::With<Device> dctx((*it).second);
  pass::PassContext pass_ctx = pass::PassContext::Current();
  tvm::With<pass::PassContext> ctx(pass_ctx);
  auto dcfg = DistConfig::Global();
  auto device_t = (*it).second.device_type();
  Array<pass::Pass> pass_seqs;

  // optimization passes that work on ANF
  pass_seqs.push_back(pass::GradInputSelect());
  pass_seqs.push_back(pass::InlineLet());
  pass_seqs.push_back(pass::DeadCodeElimination());
  // enable group all gather for ZeRO.
  if (dcfg->zero_opt_level > 1 && dcfg->group_bucket_size > 1 && device_t == DevType::kCUDA()) {
    pass_seqs.push_back(pass::GroupAllgather());
  }

  bool enable_stream_schedule = true;
  if (!pass_ctx->GetConfig("raf.vm.optimize.anf_only", Bool(false)).value()) {
    // optimization passes that work on BBNF
    pass_seqs.push_back(pass::ToGraphNormalForm());
    pass_seqs.push_back(pass::ToBasicBlockNormalForm());
    pass_seqs.push_back(pass::SimplifyExpr());
    pass_seqs.push_back(pass::InferType());
    pass_seqs.push_back(pass::FuseDialect());
    pass_seqs.push_back(pass::FuseTVM());
    pass_seqs.push_back(pass::DispatchDialect());
    // We need to erase the type after dialect dispatching because dialect ops may have different
    // output type than the base ops.
    pass_seqs.push_back(pass::EraseType());

    // optimization passes that transform BBNF into ANF
    if (device_t == DevType::kCUDA()) {
      if (DistConfig::Global()->enable_data_parallel) {
        // The current design of EnforceSync assumes ops are executed on multiple CUDA streams:
        // all computation ops are executed on a computation stream, and all communication
        // collectives are executed on another communication stream. Memory copy ops added in
        // AnnotateCollectiveOps are executed on fuse and defuse stream. This ensures that no two
        // collectives and memory copies can execute concurrently, and we can enforce strictly
        // identical order of the operators across different devices. (There is a potential problem
        // if NCCL collectives are executed in parallel, see e.g.
        // https://github.com/NVIDIA/nccl/issues/522, https://github.com/NVIDIA/nccl/issues/195).
        // Thus currently distributed learning and the multi-stream passes are mutually exclusive.
        pass_seqs.push_back(pass::DataParallelSchedule());
        pass_seqs.push_back(pass::AnnotateCollectiveOps());
        pass_seqs.push_back(pass::EnforceSync());
      } else {
        auto policy_name =
            pass_ctx->GetConfig<tvm::String>("raf.stream_schedule.policy", "sequential");
        if (policy_name == "sequential") {
          enable_stream_schedule = false;
          pass_seqs.push_back(pass::ToANormalForm());
        } else if (policy_name == "wavefront") {
          pass_seqs.push_back(pass::WavefrontStreamSchedule());
        } else if (policy_name == "asap") {
          pass_seqs.push_back(pass::ASAPStreamSchedule());
        } else if (policy_name == "ios") {
          pass_seqs.push_back(pass::InferType());
          pass_seqs.push_back(pass::IOSStreamSchedule());
        } else {
          LOG(FATAL) << "Cannot recognize schedule policy: " << policy_name << ", candidates are \n"
                     << "  sequential, wavefront, asap, and ios" << std::endl;
        }
      }
    } else {
      enable_stream_schedule = false;
      pass_seqs.push_back(pass::ToANormalForm());
    }
  } else {
    enable_stream_schedule = false;
  }

  // optimization passes that work on ANF
  pass_seqs.push_back(pass::InlinePrimitives());
  pass_seqs.push_back(pass::InferType());
  pass_seqs.push_back(pass::InplaceUpdate());

  if (pass_ctx->GetConfig("raf.use_multi_func", Bool(false)).value()) {
    // The memory-related passes below do not support multi-function, so we need to inline
    // all functions here. This one pass actually runs LambdaLift, inline, and DCE.
    pass_seqs.push_back(pass::FullInline());
  }

  if (!enable_stream_schedule) {
    // TODO(@comaniac): Support rematerialization with multi-streaming.
    pass_seqs.push_back(pass::InferType());
    pass_seqs.push_back(pass::MemorySchedule());
    pass_seqs.push_back(pass::InferType());
    pass_seqs.push_back(pass::Rematerialization());
  }
  // TODO(@hzfan): Currently disable the ValidateInplaceUpdate pass because it removes the may_share
  // attr in some cases without any error messages.
  // pass_seqs.push_back(pass::ValidateInplaceUpdate(true));
  // pass_seqs.push_back(pass::InferType());
  pass_seqs.push_back(pass::LambdaLift());
  pass_seqs.push_back(pass::InferType());
  pass_seqs.push_back(pass::ManifestAlloc());
  pass_seqs.push_back(pass::MemoryPlan());

  pass::RAFSequential seq(pass_seqs, "vm_compiler_optimize");
  return seq(mod);
}

void VMCompiler::PopulateGlobalMap() {
  // First we populate global map.
  size_t global_index = 0;
  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    context_.global_map.insert({gvar, global_index++});
  }
}

tvm::runtime::Module CreateVMCompiler() {
  auto exec = make_object<VMCompiler>();
  return tvm::runtime::Module(exec);
}

PackedFunc VMCompiler::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "lower") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 2);
      IRModule mod = args[0];
      this->Lower(mod, args[1]);
    });
  } else if (name == "get_executable") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = tvm::runtime::Module(exec_); });
  } else if (name == "set_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, Var> params = args[0];
      for (const auto& kv : params) {
        auto entry = LookupBinding(kv.second.operator->());
        if (!entry.defined()) {
          LOG(FATAL) << "could not find variable binding for " << kv.second->name_hint();
          throw;
        }
        auto value = Downcast<NDArrayBinding>(entry)->value;
        this->SetParam(kv.first, value);
      }
    });
  } else if (name == "get_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> ret;
      for (const auto& kv : params_) {
        ret.Set(kv.first, Constant(kv.second));
      }
      *rv = ret;
    });
  } else if (name == "optimize") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 2);
      *rv = this->OptimizeModule(args[0], args[1]);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

TVM_REGISTER_PASS_CONFIG_OPTION("raf.vm.optimize.anf_only", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.use_multi_func", Bool);

RAF_REGISTER_GLOBAL("raf.vm.VMCompiler").set_body_typed(CreateVMCompiler);

}  // namespace vm
}  // namespace executor
}  // namespace raf
