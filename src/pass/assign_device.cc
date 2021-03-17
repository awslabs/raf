/*!
 * Copyright (c) 2021 by Contributors
 * \file assign_device.cc
 * \brief Assign the target device to init and constant ops.
 */
#include "mnm/op.h"
#include "mnm/ir.h"

#include "mnm/pass.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/op_attr_types.h"
#include "../op/schema/init.h"
#include "../op/schema/transform.h"
#include "../op/schema/nn.h"

namespace mnm {
namespace pass {
namespace assign_device {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::op::schema;
using namespace mnm::value;

Expr AssignDeviceFullOp(const CallNode* node, const Array<Expr> args,
                        std::string target_device_str) {
  Array<Expr> new_args;

  // Get the device of the current node.
  const auto* str2ctx = tvm::runtime::Registry::Get("mnm._core.core_utils.str2ctx");
  Device call_device;
  if (node->args.size() < 4) {
    call_device = Device(static_cast<TVMContext>((*str2ctx)("cpu")));
  } else {
    auto device_name_node = (node->args[3]).as<ir::ConstantNode>();
    CHECK(device_name_node);
    auto device_name_string_obj = device_name_node->value.as<StringValueObj>();
    CHECK(device_name_string_obj);
    std::string device_name_str = device_name_string_obj->data;
    call_device = Device(static_cast<TVMContext>((*str2ctx)(device_name_str)));
  }

  // Get the target device.
  Device target_device = Device(static_cast<TVMContext>((*str2ctx)(target_device_str)));

  // Current node is not on the desired device, adjust the device argument.
  if (target_device.device_type != call_device.device_type) {
    size_t arg_idx = 0;
    // Do nothing with required arguments.
    for (; arg_idx < 2; ++arg_idx) {
      new_args.push_back(args[arg_idx]);
    }

    // Process optional arguments.
    for (; arg_idx < 4; ++arg_idx) {
      if (arg_idx == 3) {
        // Set/Override the target device.
        new_args.push_back(MakeConstant(StringValue::make(target_device_str)));
      } else if (arg_idx < node->args.size()) {
        // Optional argument is specified.
        new_args.push_back(args[arg_idx]);
      } else if (arg_idx == 2) {
        // Make up the default argument value for dtype.
        new_args.push_back(MakeConstant(StringValue::make("int")));
      }
    }
  } else {
    new_args = args;
  }
  return Call(node->op, new_args);
}

Expr AssignDeviceOneHotOp(const CallNode* node, const Array<Expr> args,
                          std::string target_device_str) {
  Array<Expr> new_args;

  // Get the device of the current node.
  const auto* str2ctx = tvm::runtime::Registry::Get("mnm._core.core_utils.str2ctx");
  Device call_device;
  if (node->args.size() < 7) {
    call_device = Device(static_cast<TVMContext>((*str2ctx)("cpu")));
  } else {
    call_device = Device(static_cast<TVMContext>((*str2ctx)(node->args[6])));
  }

  // Get the target device.
  Device target_device = Device(static_cast<TVMContext>((*str2ctx)(target_device_str)));

  // Current node is not on the desired device, adjust the device argument.
  if (target_device.device_type != call_device.device_type) {
    size_t arg_idx = 0;
    // Do nothing with required arguments.
    for (; arg_idx < 4; ++arg_idx) {
      new_args.push_back(args[arg_idx]);
    }

    // Process optional arguments.
    for (; arg_idx < 7; ++arg_idx) {
      if (arg_idx == 6) {
        // Set/Override the target device.
        new_args.push_back(MakeConstant(StringValue::make(target_device_str)));
      } else if (arg_idx < node->args.size()) {
        // Optional argument is specified.
        new_args.push_back(args[arg_idx]);
      } else if (arg_idx == 4) {
        // Make up the default argument value for axis.
        new_args.push_back(MakeConstant(IntValue::make(-1)));
      } else if (arg_idx == 5) {
        // Make up the default argument value for dtype.
        new_args.push_back(MakeConstant(StringValue::make("int")));
      }
    }
  } else {
    new_args = args;
  }
  return Call(node->op, new_args);
}

Expr AssignDeviceInitOp(const CallNode* node, const Array<Expr> args,
                        std::string target_device_str) {
  Array<Expr> new_args;

  // Get the device of the current node.
  const auto* str2ctx = tvm::runtime::Registry::Get("mnm._core.core_utils.str2ctx");
  Device call_device;
  if (node->args.size() < 3) {
    call_device = Device(static_cast<TVMContext>((*str2ctx)("cpu")));
  } else {
    call_device = Device(static_cast<TVMContext>((*str2ctx)(node->args[2])));
  }

  // Get the target device.
  Device target_device = Device(static_cast<TVMContext>((*str2ctx)(target_device_str)));

  // Current node is not on the desired device, adjust the device argument.
  if (target_device.device_type != call_device.device_type) {
    size_t arg_idx = 0;
    // Do nothing with required arguments.
    for (; arg_idx < 1; ++arg_idx) {
      new_args.push_back(args[arg_idx]);
    }

    // Process optional arguments.
    for (; arg_idx < 3; ++arg_idx) {
      if (arg_idx == 2) {
        // Set/Override the target device.
        new_args.push_back(MakeConstant(StringValue::make(target_device_str)));
      } else if (arg_idx < node->args.size()) {
        // Optional argument is specified.
        new_args.push_back(args[arg_idx]);
      } else if (arg_idx == 1) {
        // Make up the default argument value for dtype.
        new_args.push_back(MakeConstant(StringValue::make("int")));
      }
    }
  } else {
    new_args = args;
  }
  return Call(node->op, new_args);
}

typedef Expr (*AssignDeviceOpFuncType)(const CallNode* node, const Array<Expr> args,
                                       std::string target_device);
std::unordered_map<String, AssignDeviceOpFuncType> fmap = {
    {"mnm.op.full", &AssignDeviceFullOp},      {"mnm.op.one_hot", &AssignDeviceOneHotOp},
    {"mnm.op.zeros", &AssignDeviceInitOp},     {"mnm.op.ones", &AssignDeviceInitOp},
    {"mnm.op.ones_like", &AssignDeviceInitOp}, {"mnm.op.zeros_like", &AssignDeviceInitOp}};

class DeviceAssigner : public ExprMutator {
 public:
  DeviceAssigner(std::string device) : device_str_(device){};

  Expr VisitExpr_(const RelayConstantNode* node) final {
    auto value = Downcast<Value>(ConstantExtractValue(GetRef<Constant>(node)));

    // Only focus on constant tensor.
    if (value.as<TensorValueObj>()) {
      DLTensor* dlt = value;

      const auto* str2ctx = tvm::runtime::Registry::Get("mnm._core.core_utils.str2ctx");
      TVMContext target_tvm_ctx = (*str2ctx)(device_str_);
      Device target_device = Device(target_tvm_ctx);

      // Do nothing if the constant is already on the target device.
      if (target_tvm_ctx.device_type == dlt->ctx.device_type) {
        return GetRef<Expr>(node);
      }

      std::vector<int64_t> shape;
      DType dtype = DType(DLDataType(dlt->dtype));
      for (auto i = 0; i < dlt->ndim; ++i) {
        shape.push_back(dlt->shape[i]);
      }

      auto array = tvm::runtime::NDArray::Empty(shape, dtype, target_device);

      // Move tensor to the target device.
      array.CopyFrom(dlt);
      auto tv = TensorValue::Assemble(target_device, dtype, shape);
      tv->tensor = std::move(array);
      return MakeConstant(tv);
    }
    return GetRef<Expr>(node);
  }

  Expr VisitExpr_(const CallNode* node) final {
    if (node->op.as<OpNode>() == nullptr) {
      return ExprMutator::VisitExpr_(node);
    }

    const Op& node_op = Downcast<Op>(node->op);
    CHECK(node_op.defined());

    if (fmap.count(node_op->name) != 0) {
      Array<Expr> visited_args;
      for (auto arg : node->args) {
        visited_args.push_back(this->Mutate(arg));
      }
      return (*fmap[node_op->name])(node, visited_args, device_str_);
    }
    return ExprMutator::VisitExpr_(node);
  }

 private:
  /*! \brief The target device string. */
  std::string device_str_;
};
}  // namespace assign_device

ir::Expr AssignDevice(ir::Expr expr, std::string device) {
  return assign_device::DeviceAssigner(device).Mutate(expr);
}

// TODO - Cleanup when pass manager is introduced.
ir::IRModule AssignDevice(ir::IRModule mod, std::string device) {
  ir::IRModule updated_mod = ir::IRModule(mod->functions);
  std::vector<std::pair<ir::GlobalVar, ir::Function>> updated_funcs;
  auto assigner = assign_device::DeviceAssigner(device);

  for (auto kv : updated_mod->functions) {
    if (kv.second.as<ir::FunctionNode>()) {
      auto func = tvm::runtime::Downcast<ir::Function>(assigner.Mutate(kv.second));
      updated_funcs.emplace_back(kv.first, func);
    }
  }

  for (const auto& it : updated_funcs) {
    updated_mod->Add(it.first, it.second, true);
  }
  return updated_mod;
}

MNM_REGISTER_GLOBAL("mnm.pass_.AssignDevice")
    .set_body_typed([](ir::IRModule mod, std::string device) { return AssignDevice(mod, device); });

}  // namespace pass
}  // namespace mnm
