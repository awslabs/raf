/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/vm/compiler.cc
 * \brief The Meta virtual machine compiler.
 */
#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/target/target.h>
#include <tvm/tir/op.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "mnm/type.h"
#include "mnm/pass.h"
#include "./compiler.h"

namespace tvm {
namespace relay {
namespace vm {
bool IsClosure(const Function& func);
}  // namespace vm
}  // namespace relay
}  // namespace tvm

namespace mnm {
namespace executor {
namespace vm {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;
using namespace mnm::type;
using namespace tvm;
using binding::LookupBinding;
using binding::NDArrayBinding;
using relay::Shape;

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
  std::unordered_map<Op, MatchFunc, ObjectHash, ObjectEqual> match_map_;
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
  VMFunctionCompiler(VMCompilerContext* context, TargetsMap targets, Target target_host)
      : last_register_(0),
        registers_num_(0),
        context_(context),
        targets_(targets),
        target_host_(target_host) {
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
      Function inner_func = Downcast<Function>(func->body);
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
        last_register_ = instr.dst;
        break;
      case Opcode::InvokePacked:
      case Opcode::InvokeJit:
      case Opcode::If:
      case Opcode::Ret:
      case Opcode::Goto:
      case Opcode::Fatal:
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
    DLOG(INFO) << PrettyPrint(let_node->value);
    this->VisitExpr(let_node->value);
    var_register_map_.insert({let_node->var, this->last_register_});
    this->VisitExpr(let_node->body);
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
          .Match("mnm.op.vm.invoke_op",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 3);
                   EmitInvokeOp(args[0], args[1], args[2]);
                 })
          .Match("mnm.op.vm.alloc_tensor",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 2);

                   // Get the attributes.
                   auto alloc_attrs = attrs.as<tvm::relay::AllocTensorAttrs>();
                   CHECK(alloc_attrs != nullptr) << "must be the alloc tensor attrs";
                   auto dtype = alloc_attrs->dtype;

                   // The storage will be passed dynamically.
                   this->VisitExpr(args[0]);
                   auto storage_register = last_register_;

                   // If the shape is constant then we will emit a static tensor allocation
                   // instruction.
                   auto const_shape = args[1].as<ConstantNode>();

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
                                                   NewRegister()));
                   } else {
                     LOG(FATAL) << "Not suported";
                   }
                 })
          .Match("mnm.op.vm.alloc_storage",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   CHECK_EQ(args.size(), 2);
                   // Compute the size of the allocation.
                   this->VisitExpr(args[0]);
                   auto size_register = last_register_;

                   CHECK(args[1]->IsInstance<ConstantNode>());
                   auto align_val = args[1].as<ConstantNode>()->value;
                   CHECK(align_val->IsInstance<IntValueObj>());
                   Index alignment = align_val.as<IntValueObj>()->data;

                   // Get the dtype hint from the attributes.
                   auto alloc_attrs = attrs.as<tvm::relay::AllocStorageAttrs>();
                   CHECK(alloc_attrs != nullptr) << "must be the alloc tensor attrs";
                   auto dtype = alloc_attrs->dtype;

                   Emit(Instruction::AllocStorage(size_register, alignment, dtype,
                                                  alloc_attrs->device_type, alloc_attrs->device_id,
                                                  NewRegister()));
                 })
          .Match("memory.kill",
                 [](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   LOG(FATAL) << "memory.kill is not yet supported";
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
    if (func_node->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
      VisitExpr(MakeConstant(ClosureValue::make({}, GetRef<Function>(func_node))));
    } else {
      LOG(FATAL) << "local functions should have been removed by lambda lifting:" << std::endl
                 << "Program: " << AsText(GetRef<Function>(func_node), false) << std::endl
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
    auto input_tuple = inputs.as<TupleNode>();
    CHECK(input_tuple) << "internal error: invoke_op inputs must be a tuple,"
                       << "please file a bug in the memory manifestation pass";

    auto output_tuple = outputs.as<TupleNode>();
    CHECK(output_tuple) << "internal error: invoke_op outputs must be a tuple,"
                        << "please file a bug in the memory manifestation pass";
    for (auto input : input_tuple->fields) {
      if (input.as<VarNode>()) {
        auto reg = var_register_map_.find(Downcast<Var>(input));
        CHECK(reg != var_register_map_.end())
            << "internal error: all variables should be in the register mapping";
        argument_registers.push_back(reg->second);
      } else {
        // FIXME: use ANF so that we don't need such special cases
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

    Target target;
    if (targets_.size() == 1) {
      // homogeneous execution.
      const auto& it = targets_.begin();
      target = (*it).second;
    } else {
      // heterogeneous execution.
      LOG(FATAL) << "Currently VM compiler doesn't support heterogeneous compilation";
    }
    Emit(Instruction::InvokeJit(op_reg, argument_registers.size(), output_tuple->fields.size(),
                                argument_registers));
  }

 protected:
  /*! \brief Store the expression a variable points to. */
  std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> expr_map_;
  /*! \brief Instructions in the VMFunction. */
  std::vector<Instruction> instructions_;
  /*! \brief Parameter names of the function. */
  std::vector<std::string> params_;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectHash, ObjectEqual> var_register_map_;
  /*! \brief Last used register number. */
  Index last_register_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_;
  /*! \brief Global shared meta data */
  VMCompilerContext* context_;
  /*! \brief Target devices. */
  TargetsMap targets_;
  /*! \brief Host target. */
  Target target_host_;
};

void VMCompiler::SetParam(const std::string& name, Value data_in) {
  params_[name] = data_in;
}

void VMCompiler::Lower(Module mod, const TargetsMap& targets, const tvm::Target& target_host) {
  CHECK_EQ(targets.size(), 1) << "Currently VM compiler doesn't support heterogeneous compilation";
  if (params_.size()) {
    BaseFunc base_func = mod->Lookup("main");
    CHECK(base_func->IsInstance<FunctionNode>())
        << "VM compiler expects to compile relay::Function";
    auto f = BindParamsByName(Downcast<Function>(base_func), params_);
    auto gvar = mod->GetGlobalVar("main");
    mod->Add(gvar, f, true);
  }

  exec_ = make_object<Executable>();
  targets_ = targets;
  target_host_ = target_host;

  // Run the optimizations necessary to target the VM.
  context_.module = OptimizeModule(mod, targets_);

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
      VMFunctionCompiler func_compiler(&context_, targets_, target_host_);
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

Module VMCompiler::OptimizeModule(const Module& mod, const TargetsMap& targets) {
  auto m = pass::InferType(mod);
  CHECK_EQ(targets.size(), 1) << "Currently VM compiler doesn't support heterogeneous compilation";
  const auto& it = targets.begin();
  With<tvm::Target> tctx((*it).second);
  m = pass::ManifestAlloc(m);
  return pass::InplaceUpdate(m);
}

void VMCompiler::PopulateGlobalMap() {
  // First we populate global map.
  size_t global_index = 0;
  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    context_.global_map.insert({gvar, global_index++});
  }
}

runtime::Module CreateVMCompiler() {
  auto exec = make_object<VMCompiler>();
  return runtime::Module(exec);
}

PackedFunc VMCompiler::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "lower") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 3);
      Module mod = args[0];
      this->Lower(mod, args[1], args[2]);
    });
  } else if (name == "get_executable") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = runtime::Module(exec_); });
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

MNM_REGISTER_GLOBAL("mnm.vm.VMCompiler").set_body_typed(CreateVMCompiler);

}  // namespace vm
}  // namespace executor
}  // namespace mnm
