/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/serialization.cc
 * \brief MNM serialization underlying implementation
 */
#include <tvm/node/serialization.h>
#include "mnm/pass.h"
#include "mnm/registry.h"
#include "mnm/serialization.h"

namespace mnm {
namespace ir {
namespace serialization {

using namespace value;

class IRRewrite4Loader : public ir::ExprMutator {
 public:
  ir::Expr VisitExpr(const ir::Expr& expr) override {
    auto ret = ir::ExprMutator::VisitExpr(expr);
    ret->checked_type_ = expr->checked_type_;
    return ret;
  }

  ir::Expr VisitExpr_(const tvm::relay::ConstantNode* _node) override {
    const ir::ConstantNode* node = static_cast<const ir::ConstantNode*>(_node);
    ir::ObjectPtr<serialization::ConstantNode> n = ir::make_object<serialization::ConstantNode>();
    n->data = node->data;
    n->value = node->value;
    return ir::Expr(n);
  }
};

std::string SaveJSON(const ir::IRModule& mod) {
  ir::IRModule inst = ir::IRModule();
  for (auto kv : mod->functions) {
    ir::Expr func = IRRewrite4Loader()(kv.second);
    inst->Add(kv.first, Downcast<ir::Function>(func));
  }
  return tvm::SaveJSON(inst);
}

std::string SaveJSON(const ir::Expr& expr) {
  auto e = IRRewrite4Loader().VisitExpr(expr);
  return tvm::SaveJSON(e);
}

std::string SaveJSON(const ir::ObjectRef& n) {
  if (const ir::IRModuleNode* m = n.as<ir::IRModuleNode>()) {
    return SaveJSON(ir::GetRef<ir::IRModule>(m));
  } else if (const ir::FunctionNode* f = n.as<ir::FunctionNode>()) {
    return SaveJSON(ir::GetRef<ir::Function>(f));
  } else if (const ir::ExprNode* e = n.as<ir::ExprNode>()) {
    return SaveJSON(ir::GetRef<ir::Expr>(e));
  }
  return tvm::SaveJSON(n);
}

ir::ObjectPtr<ir::Object> CreateConstantNode(const std::string& s) {
  return ir::MakeConstantNode(tvm::LoadJSON(s));
}

void SerializeValue(dmlc::Stream* strm, const Value& value) {
  if (!value.defined()) {
    strm->Write(static_cast<uint8_t>(kNullptr));
  } else if (auto ival = value.as<IntValueObj>()) {
    strm->Write(static_cast<uint8_t>(kIntValue));
    strm->Write(ival->dtype.operator DLDataType());
    strm->Write(ival->value);
  } else if (auto fval = value.as<FloatValueObj>()) {
    strm->Write(static_cast<uint8_t>(kFloatValue));
    strm->Write(fval->dtype.operator DLDataType());
    strm->Write(fval->value);
  } else if (auto bval = value.as<BoolValueObj>()) {
    strm->Write(static_cast<uint8_t>(kBoolValue));
    strm->Write(bval->value);
  } else if (auto sval = value.as<StringValueObj>()) {
    strm->Write(static_cast<uint8_t>(kStringValue));
    strm->Write(sval->value);
  } else if (auto tval = value.as<TensorValueObj>()) {
    strm->Write(static_cast<uint8_t>(kTensorValue));
    DLTensor* dlt = Downcast<TensorValue>(value);
    SaveDLTensor(strm, dlt);
  } else if (auto tup = value.as<TupleValueObj>()) {
    strm->Write(static_cast<uint8_t>(kTupleValue));
    strm->Write(static_cast<uint64_t>(tup->fields.size()));
    for (auto v : tup->fields) {
      SerializeValue(strm, v);
    }
  } else if (auto op = value.as<OpValueObj>()) {
    strm->Write(static_cast<uint8_t>(kOpValue));
    strm->Write(SaveJSON(op->op));
  } else if (auto clo = value.as<ClosureValueObj>()) {
    strm->Write(static_cast<uint8_t>(kClosureValue));
    strm->Write(SaveJSON((clo->func)));
    strm->Write(static_cast<uint64_t>(clo->env.size()));
    for (auto it : clo->env) {
      strm->Write(SaveJSON(it.first));
      SerializeValue(strm, it.second);
    }
  } else if (value.as<NoGradValueObj>()) {
    strm->Write(static_cast<uint8_t>(kNoGradValue));
  } else if (value.as<VoidValueObj>()) {
    strm->Write(static_cast<uint8_t>(kVoidValue));
  } else {
    LOG(ERROR) << "Currently don't support the serialization of " << value->GetTypeKey();
  }
}

Value DeserializeValue(dmlc::Stream* strm) {
  uint8_t v_type;
  strm->Read(&v_type, sizeof(v_type));
  ValueType value_type = static_cast<ValueType>(v_type);
  DLDataType dtype;
  std::string str;
  switch (value_type) {
    case value::kNullptr:
      return Value{};
    case kIntValue: {
      int64_t ival;
      strm->Read(&dtype);
      strm->Read(&ival);
      return IntValue::make(DataType(dtype), ival);
    }
    case kFloatValue: {
      double fval;
      strm->Read(&dtype);
      strm->Read(&fval);
      return FloatValue::make(DataType(dtype), fval);
    }
    case kBoolValue: {
      bool bval;
      strm->Read(&bval);
      return BoolValue::make(bval);
    }
    case kStringValue: {
      strm->Read(&str);
      return StringValue::make(str);
    }
    case kTensorValue: {
      tensor::Tensor tensor;
      tensor.Load(strm);
      return TensorValue::make(tensor);
    }
    case value::kTupleValue: {
      uint64_t size;
      strm->Read(&size);
      Array<Value> fields;
      for (uint64_t i = 0; i < size; ++i) {
        fields.push_back(DeserializeValue(strm));
      }
      return TupleValue::make(fields);
    }
    case kOpValue: {
      strm->Read(&str);
      Op op = Downcast<Op>(tvm::LoadJSON(str));
      return OpValue::make(op);
    }
    case kClosureValue: {
      strm->Read(&str);
      auto func = Downcast<Function>(tvm::LoadJSON(str));
      uint64_t cnt;
      std::unordered_map<Var, Value, ObjectPtrHash, ObjectPtrEqual> env;
      strm->Read(&cnt);
      for (uint64_t i = 0; i < cnt; ++i) {
        strm->Read(&str);
        Var var = Downcast<Var>(tvm::LoadJSON(str));
        Value val = DeserializeValue(strm);
        env.emplace(var, val);
      }
      return ClosureValue::make(env, func);
    }
    case kNoGradValue:
      return NoGradValue::make();
    case kVoidValue:
      return VoidValue::make();
    default:
      LOG(FATAL) << "Currently don't support deserialization for value type "
                 << ValueType2String(value_type);
      return Value{};
  }
}

MNM_REGISTER_OBJECT_REFLECT(ConstantNode)
    .set_creator(CreateConstantNode)
    .set_repr_bytes([](const ir::Object* n) -> std::string {
      return tvm::SaveJSON(static_cast<const ConstantNode*>(n)->value);
    });

MNM_REGISTER_GLOBAL("mnm.ir.serialization.SaveJSON")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
      CHECK(args.size() == 1);
      ir::ObjectRef obj = args[0].operator ir::ObjectRef();
      *ret = SaveJSON(obj);
    });

}  // namespace serialization
}  // namespace ir
}  // namespace mnm
