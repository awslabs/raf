#include "raf/pass.h"
#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"
#include "raf/op_utils.h"


namespace raf {
namespace pass {
namespace group_comm {

using namespace raf::op;

class CommGrouper : public ExprMutator {
 public:
  CommGrouper(const Function& func) : func_(func) {
//    std::cout << "old body is \n" << PrettyPrint(func->body) << "\n" << std::flush;
    auto ell = ExplicitLetList::make(func->body);
    //for (size_t i = 0; i < ell->vars.size(); ++i) {
    //  var_to_expr_.Set(ell->vars[i], ell->exprs[i]);
    //}
    // Assume output is a tuple of (forward out,step, update_param0, update_param1....)
    auto ret = ell->exprs.back().as<TupleNode>();
    ret_var_ = ell->vars.back();
    // std::cout << "----- ret is " << PrettyPrint(ell->exprs.back()) << " \n" << std::flush;
    for (int i = 2; i < ret->fields.size(); ++i) {
      params_.Set(Downcast<Var>(ret->fields[i]), Expr());
    }
	scopes_.emplace_back(new LetList);
  }

  Function Group() {
    //std::cout << " in Group\n " << std::flush;
    //std::cout << " param size is " <<params_.size() <<"\n " << std::flush;
    if (params_.empty()) {
      return func_;
    }
    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    //std::cout << "-----------------------------------new_body is\n " << PrettyPrint(new_body)
    //          << "\n"
    //          << std::flush;
    auto f = Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
    std::cout << "before return in Group\n " << std::flush;
    return f; }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    //std::cout << " in LetNdoe ------------\n" << std::flush;
    static auto add_op = Op::Get("raf.op.add");
    static auto allgather_op = Op::Get("raf.op._allgather");
    static auto slice_op = Op::Get("raf.op.strided_slice");
    static auto cast_op = Op::Get("raf.op.cast");
    static auto zeros_op = Op::Get("raf.op.zeros");
    static auto group_cast = Op::Get("raf.op.group_cast");
    static auto group_allgather = Op::Get("raf.op._group_allgather");
    Expr body;
    do {
      auto curr_var = node->var;
      auto value = VisitExpr(node->value);

      bool comm_node = false;
      if (IsCast(node)) {
        auto gather_node = node->body.as<LetNode>();
        if (IsAllgather(gather_node)) {
          auto slice_node = gather_node->body.as<LetNode>();
          if (IsSlice(slice_node)) {
            auto add_node = slice_node->body.as<LetNode>();
            if (IsAdd_update(add_node)) {
              auto update_var = add_node->var;
              if (params_.count(update_var)) {
                comm_node = true;
                cast_allgather_ = true;
                auto size = NElement(update_var);
                auto cast_call = Downcast<Call>(node->value);
                auto slice_call = Downcast<Call>(slice_node->value);
                auto gather_var = gather_node->var;
                auto var_type = gather_var->checked_type_.as<TensorTypeNode>();
                //auto input = scope->Push(Call(zeros_op, {gather_var}));

                auto input = scope->Push(Call(
                    zeros_op, {MakeConstant(ArrayToIntTuple(var_type->shape)),
                               MakeConstant(StringValue::make(DLDataType2String(var_type->dtype))),
                               MakeConstant(StringValue::make("gpu"))}));

                if (curr_size_ + size < allgather_bucket_size_) {
                  curr_size_ += size;
                  allgather_input_.push_back(cast_call->args[0]);
                  allgather_output_.push_back(input);
                  slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                  update_params_.push_back(update_var);
                } else {
                  curr_size_ = size;
                  auto gather_call = Downcast<Call>(gather_node->value);
                  auto cast_input = scope->Push(Tuple(allgather_input_));
                  auto gather_output = scope->Push(Tuple(allgather_output_));
                  auto cast_output =
                      scope->Push(Call(group_cast, {cast_input, cast_call->args[1]}));
                  auto output = scope->Push(
                      Call(group_allgather, {cast_output, gather_call->args[1], gather_output}));
                  for (int i = 0; i < allgather_input_.size(); ++i) {
                    auto out_tensor = scope->Push(TupleGetItem(output, i));
                    if (slice_dic_.count(i)) {
                      out_tensor = scope->Push(Call(
                          slice_op,
                          {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                           slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(0)}))}));
                    }
                    params_.Set(update_params_[i], out_tensor);
                  }
                  allgather_input_ = {cast_call->args[0]};
                  allgather_output_ = {input};
                  slice_dic_.clear();
                  slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                  update_params_ = {update_var};
                }
                node = add_node;
              }
            }
          } else {
            // no slice
            auto add_node = slice_node;
            if (IsAdd_update(add_node)) {
              auto update_var = add_node->var;
              if (params_.count(update_var)) {
                comm_node = true;
                cast_allgather_ = true;
                auto size = NElement(update_var);
                auto cast_call = Downcast<Call>(node->value);
                auto add_call = Downcast<Call>(add_node->value); // auto slice_call = Downcast<Call>(slice_node->value);
                auto gather_var = gather_node->var;
                auto var_type = gather_var->checked_type_.as<TensorTypeNode>();

                if (curr_size_ + size < allgather_bucket_size_) {
                  curr_size_ += size;
                  allgather_input_.push_back(cast_call->args[0]);
                  allgather_output_.push_back(add_call->args[2]);
                  // slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                  update_params_.push_back(update_var);
                } else {
                  curr_size_ = size;
                  auto gather_call = Downcast<Call>(gather_node->value);
                  auto cast_input = scope->Push(Tuple(allgather_input_));
                  auto gather_output = scope->Push(Tuple(allgather_output_));
                  auto cast_output =
                      scope->Push(Call(group_cast, {cast_input, cast_call->args[1]}));

                  auto output = scope->Push(
                      Call(group_allgather, {cast_output, gather_call->args[1], gather_output}));
                  for (int i = 0; i < allgather_input_.size(); ++i) {
                    auto out_tensor = scope->Push(TupleGetItem(output, i));
                    if (slice_dic_.count(i)) {
                      out_tensor = scope->Push(Call(
                          slice_op,
                          {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                           slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(0)}))}));
                    }
                    params_.Set(update_params_[i], out_tensor);
                  }
                  allgather_input_ = {cast_call->args[0]};
                  allgather_output_ = {add_call->args[2]};
                  slice_dic_.clear();
                  // slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                  update_params_ = {update_var};
                }
                node = add_node;
              }
            }
          }
        }
      } else if (IsAllgather(node)) {
        std::cout << "------->>>>> in allgather node \n" << std::flush;
        auto slice_node = node->body.as<LetNode>();
        if (IsSlice(slice_node)) {
        std::cout << "------->>>>> in is slice node \n" << std::flush;
          auto add_node = slice_node->body.as<LetNode>();
          if (IsAdd_update(add_node)) {
            std::cout << "------->>>>> in add update node \n" << std::flush;
            auto update_var = add_node->var;
            if (params_.count(update_var)) {
              comm_node = true;
              auto size = NElement(update_var);
              auto gather_call = Downcast<Call>(node->value);
              auto slice_call = Downcast<Call>(slice_node->value);
              auto gather_var = node->var;
              auto var_type = gather_var->checked_type_.as<TensorTypeNode>();

              //auto input = scope->Push(Call(zeros_op, {gather_var}));
              auto input = scope->Push(Call(
                  zeros_op, {MakeConstant(ArrayToIntTuple(var_type->shape)),
                             MakeConstant(StringValue::make(DLDataType2String(var_type->dtype))),
                             MakeConstant(StringValue::make("gpu"))}));

              if (curr_size_ + size < allgather_bucket_size_) {
                curr_size_ += size;
                allgather_input_.push_back(gather_call->args[0]);
                allgather_output_.push_back(input);
                slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                update_params_.push_back(update_var);
              } else {
                curr_size_ = size;
                auto gather_input = scope->Push(Tuple(allgather_input_));
                auto gather_output = scope->Push(Tuple(allgather_output_));
                auto output = scope->Push(
                    Call(group_allgather, {gather_input, gather_call->args[1], gather_output}));
                for (int i = 0; i < allgather_input_.size(); ++i) {
                  auto out_tensor = scope->Push(TupleGetItem(output, i));
                  if (slice_dic_.count(i)) {
                    out_tensor = scope->Push(Call(
                        slice_op,
                        {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                         slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(0)}))}));
                  }
                  params_.Set(update_params_[i], out_tensor);
                }
                allgather_input_ = {gather_call->args[0]};
                allgather_output_ = {input};
                slice_dic_.clear();
                slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                update_params_ = {update_var};
              }
              node = add_node;
            }
          }
        } else {
          // no slice
          auto add_node = slice_node;
          if (IsAdd_update(add_node)) {
            auto update_var = add_node->var;
            if (params_.count(update_var)) {
              comm_node = true;
              auto size = NElement(update_var);
              auto gather_call = Downcast<Call>(node->value);
              // auto slice_call = Downcast<Call>(slice_node->value);
              auto add_call = Downcast<Call>(add_node->value);
              auto gather_var = node->var;
              auto var_type = gather_var->checked_type_.as<TensorTypeNode>();

              if (curr_size_ + size < allgather_bucket_size_) {
                curr_size_ += size;
                allgather_input_.push_back(gather_call->args[0]);
                allgather_output_.push_back(add_call->args[2]);
                // slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                update_params_.push_back(update_var);
              } else {
                curr_size_ = size;
                auto gather_call = Downcast<Call>(node->value);
                auto gather_input = scope->Push(Tuple(allgather_input_));
                auto gather_output = scope->Push(Tuple(allgather_output_));
                // auto cast_output = scope->Push(Call(group_cast, {cast_input,
                // cast_call->args[1]}));

                auto output = scope->Push(
                    Call(group_allgather, {gather_input, gather_call->args[1], gather_output}));
                for (int i = 0; i < allgather_input_.size(); ++i) {
                  auto out_tensor = scope->Push(TupleGetItem(output, i));
                  if (slice_dic_.count(i)) {
                    out_tensor = scope->Push(Call(
                        slice_op,
                        {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                         slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(0)}))}));
                  }
                  params_.Set(update_params_[i], out_tensor);
                }
                allgather_input_ = {gather_call->args[0]};
                allgather_output_ = {add_call->args[2]};
                slice_dic_.clear();
                // slice_dic_[allgather_input_.size() - 1] = slice_call->args[2];
                update_params_ = {update_var};
              }

              node = add_node;
            }
          }
        }
      }

      else if (curr_var == ret_var_) {
        comm_node = true;
			  //std::cout << "--------------->>>>>in return branch  \n " << std::flush;
        if (allgather_input_.size() > 1) {
          // auto gather_call = Downcast<Call>(gather_node->value);
          auto gather_input = scope->Push(Tuple(allgather_input_));
          auto gather_output = scope->Push(Tuple(allgather_output_));
          if (cast_allgather_) {
            gather_input = scope->Push(
                Call(group_cast, {gather_input, MakeConstant(StringValue::make("float16"))}));
          }

          auto output = scope->Push(Call(
              group_allgather, {gather_input, MakeConstant(ScalarValue::make(0)), gather_output}));
          for (int i = 0; i < allgather_input_.size(); ++i) {
            auto out_tensor = scope->Push(TupleGetItem(output, i));
            if (slice_dic_.count(i)) {
              out_tensor = scope->Push(
                  Call(slice_op,
                       {out_tensor, MakeConstant(TupleValue::make({ScalarValue::make(0)})),
                        slice_dic_[i], MakeConstant(TupleValue::make({ScalarValue::make(0)}))}));
            }
            params_.Set(update_params_[i], out_tensor);
          }
        }
        Array<Expr> tuple;
        //std::cout << "-------->>>>> value is " << PrettyPrint(value) << "\n" << std::flush;
        auto ret_value = value.as<TupleNode>();
        //std::cout << "-------->>>>>after tule node while \n" << std::flush;
        tuple.push_back(ret_value->fields[0]);
        tuple.push_back(ret_value->fields[1]);
        for (int j = 2; j < ret_value->fields.size(); ++j) {
          auto key = Downcast<Var>(ret_value->fields[j]);
          if (params_[key].defined()) {
            tuple.push_back(params_[key]);
          } else {
            tuple.push_back(key);
          }
          // std::cout << " key is " << PrettyPrint(key) << " value is " <<
          // PrettyPrint(params_[key])
          //          << "\n"
          //          << std::flush;
        }
        //std::cout << "-------->>>>>tmp  is " << PrettyPrint(tmp) << "\n" << std::flush;
        //auto ret_tuple = scope->Push(Tuple(tuple));
        scope->Push(curr_var, Tuple(tuple));

        //std::cout << "-------->>>>>ret_tuple  is " << PrettyPrint(ret_tuple) << "\n" << std::flush;
      }
      // else {
      //  scope->Push(curr_var, value);
      //}
      if (comm_node == false) {
        scope->Push(curr_var, value);
      }
      body = node->body;
      node = body.as<LetNode>();

  } while (node);
  std::cout << "-------->>>>>out do while \n" << std::flush;
  auto ret = scopes_.back()->Get(this->Mutate(body));
 // std::cout << "-------->>>>>ret  is " << PrettyPrint(ret) << "\n" << std::flush;
  scopes_.pop_back();
  return ret;
  }

 private:
  bool IsCast(const LetNode* node){
  static auto cast_op = Op::Get("raf.op.cast");
    if (node->value.as<CallNode>()) {
      auto call = Downcast<Call>(node->value);
      auto opn = Downcast<Op>(call->op);
      if (opn == cast_op) {
        return true;
      }
    }
    return false; }
 
  bool IsAllgather(const LetNode* node){
  static auto allgather_op = Op::Get("raf.op._allgather");
    if (node->value.as<CallNode>()) {
      auto call = Downcast<Call>(node->value);
      auto opn = Downcast<Op>(call->op);
      if (opn == allgather_op) {
        return true;
      }
    }
    return false;
  }

  bool IsSlice(const LetNode* node){
  static auto slice_op = Op::Get("raf.op.strided_slice");
    if (node->value.as<CallNode>()) {
      auto call = Downcast<Call>(node->value);
      auto opn = Downcast<Op>(call->op);
      if (opn == slice_op) {
        return true;
      }
    }
    return false;
  }

  bool IsAdd_update(const LetNode* node) {
    static auto add_op = Op::Get("raf.op.add");
    if (node->value.as<CallNode>()) {
      auto call = Downcast<Call>(node->value);
      auto opn = Downcast<Op>(call->op);
      if (opn == add_op && call->args.size() > 2) {
       //auto const_node = call->args[1].as<ConstantNode>();
       //auto value = const_node->value.as<FloatValueObj>()->value;
       //std::cout << "in Add_update 4\n " << std::flush;
       //if (value == 0) {
       return true;
       //}
      }
    }
    //std::cout << "in Add_update return false\n " << std::flush;
    return false;
  }

  inline int64_t NElement(const Var& var) {
    int64_t n = 1;
    TensorType var_type = Downcast<TensorType>(var->checked_type());
    for (int i = 0; i < var_type->shape.size(); ++i) {
     PrimExpr k = var_type->shape[i];
     int64_t k_v = k.as<IntImmNode>()->value;
     n *= k_v;
    }
    return n;
  }
  Function func_;
  std::vector<std::unique_ptr<LetList>> scopes_;
  Map<Var, Expr> var_to_expr_;
  Map<Var, Expr> params_;
  std::vector<Expr> allgather_input_;
  std::vector<Expr> allgather_output_;
  std::vector<Var> update_params_;
  std::unordered_map<size_t, Expr> slice_dic_;
  size_t allgather_bucket_size_ = 5000000000;
  size_t curr_size_ = 0;
  bool cast_allgather_ = false;
  Var ret_var_;
};
}  // namespace group_comm

Pass GroupComm(){
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return group_comm::CommGrouper(f).Group(); };
  auto group_comm_pass = CreateRAFFunctionPass(pass_func, 0, "GroupComm", {});

  return RAFSequential({InferType(), group_comm_pass, InferType()}, "GroupComm");
  //return RAFSequential({InferType(), group_comm_pass}, "GroupComm");
}  // namespace pass

RAF_REGISTER_GLOBAL("raf.pass_.GroupComm").set_body_typed(GroupComm);


}  // namespace pass
}  // namespace raf
