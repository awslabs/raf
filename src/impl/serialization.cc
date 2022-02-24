/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/serialization.cc
 * \brief RAF serialization underlying implementation
 */
#include <tvm/node/serialization.h>
#include "raf/pass.h"
#include "raf/registry.h"
#include "raf/serialization.h"

namespace raf {
namespace ir {
namespace serialization {

using namespace value;

class IRRewrite4Saver : public ir::ExprMutator {
 public:
  ir::Expr VisitExpr(const ir::Expr& expr) override {
    auto ret = ir::ExprMutator::VisitExpr(expr);
    ret->checked_type_ = expr->checked_type_;
    return ret;
  }

  Expr VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      this->memo_[GetRef<Expr>(op)] = Let(var, value, body);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  ir::Expr VisitExpr_(const tvm::relay::ConstantNode* _node) override {
    const ir::ConstantNode* node = static_cast<const ir::ConstantNode*>(_node);
    ir::ObjectPtr<serialization::ConstantNode> n = ir::make_object<serialization::ConstantNode>();
    n->data = node->data;
    n->value = node->value;
    return ir::Expr(n);
  }
};

class IRRewrite4Loader : public ir::ExprMutator {
 public:
  ir::Expr VisitExpr(const ir::Expr& expr) override {
    auto ret = ir::ExprMutator::VisitExpr(expr);
    ret->checked_type_ = expr->checked_type_;
    return ret;
  }

  Expr VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      this->memo_[GetRef<Expr>(op)] = Let(var, value, body);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  ir::Expr VisitExpr_(const VarNode* _node) override {
    return MakeVar(_node->vid->name_hint, _node->type_annotation);
  }
};

template <typename T>
ir::ObjectRef Normalize(const ir::ObjectRef& n) {
  if (const ir::IRModuleNode* mod = n.as<ir::IRModuleNode>()) {
    ir::IRModule updated_mod = ir::IRModule();
    for (auto kv : mod->functions) {
      ir::Expr func = T()(kv.second);
      // updated_mod->Add cannot be used, which runs InferType automatically
      // However, InferType cannot visit serialization::ConstantNode
      updated_mod->AddUnchecked(kv.first, Downcast<ir::Function>(func));
    }
    return updated_mod;
  } else if (const ExprNode* e = n.as<ExprNode>()) {
    return T().VisitExpr(GetRef<ir::Expr>(e));
  } else if (const ir::ArrayNode* a = n.as<ir::ArrayNode>()) {
    Array<ir::ObjectRef, ir::ObjectRef> updated_array;
    for (const auto& x : *a) {
      updated_array.push_back(Normalize<T>(x));
    }
    return updated_array;
  } else if (const ir::MapNode* m = n.as<ir::MapNode>()) {
    Map<ir::ObjectRef, ir::ObjectRef> updated_map;
    for (const auto& kv : *m) {
      updated_map.Set(kv.first, Normalize<T>(kv.second));
    }
    return updated_map;
  }
  return n;
}

std::string SaveJSON(const ir::ObjectRef& n) {
  return tvm::SaveJSON(Normalize<IRRewrite4Saver>(n));
}

ir::ObjectRef LoadJSON(const std::string& n) {
  return Normalize<IRRewrite4Loader>(tvm::LoadJSON(n));
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
    strm->Write(serialization::SaveJSON(op->op));
  } else if (auto clo = value.as<ClosureValueObj>()) {
    strm->Write(static_cast<uint8_t>(kClosureValue));
    strm->Write(serialization::SaveJSON((clo->func)));
    strm->Write(static_cast<uint64_t>(clo->env.size()));
    for (auto it : clo->env) {
      strm->Write(serialization::SaveJSON(it.first));
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

RAF_REGISTER_OBJECT_REFLECT(ConstantNode)
    .set_creator(CreateConstantNode)
    .set_repr_bytes([](const ir::Object* n) -> std::string {
      return tvm::SaveJSON(static_cast<const ConstantNode*>(n)->value);
    });

RAF_REGISTER_GLOBAL("raf.ir.serialization.SaveJSON")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
      CHECK(args.size() == 1);
      ir::ObjectRef obj = args[0].operator ir::ObjectRef();
      *ret = serialization::SaveJSON(obj);
    });

RAF_REGISTER_GLOBAL("raf.ir.serialization.LoadJSON")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* ret) {
      CHECK(args.size() == 1);
      auto obj = args[0].operator ir::String();
      *ret = serialization::LoadJSON(obj);
    });

}  // namespace serialization
}  // namespace ir
}  // namespace raf
