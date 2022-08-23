/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file assign_device.cc
 * \brief Assign the target device to init and constant ops.
 */
#include "raf/op.h"
#include "raf/ir_ext.h"

#include "raf/pass.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/op_attr_types.h"
#include "../op/schema/init.h"
#include "../op/schema/transform.h"
#include "../op/schema/nn.h"
#ifdef RAF_CXX_USE_CUDNN
#include "../op/dialect/cudnn/cudnn_utils.h"
#endif

namespace raf {
namespace pass {
namespace assign_device {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::op::schema;
using namespace raf::value;

raf::Device GetDeviceFromConstExpr(const Expr& expr) {
  static auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  auto device_name_node = expr.as<ir::ConstantNode>();
  CHECK(device_name_node);
  auto device_name_string_obj = device_name_node->value.as<StringValueObj>();
  CHECK(device_name_string_obj);
  std::string device_name_str = device_name_string_obj->value;
  return Device(static_cast<tvm::Device>((*str2dev)(device_name_str)));
}

/*!
 * \brief The helper function to mutate a call node to be in the target device.
 * \param node The input call node.
 * \param args The mutated args.
 * \param target_device_str A string of the target device.
 * \param device_arg_idx The argument index of the target device of the op.
 * \param default_vals The default values of the op. This array must have the same length of the
 * arguments. If an argument is required and does not have a default value, an undefined Expr has
 * to be provided as a placeholder.
 *
 * \return A mutated call node with the target device.
 */
Expr AssignDeviceHelper(const CallNode* node, const Array<Expr> args, std::string target_device_str,
                        size_t device_arg_idx, Array<Expr> default_vals) {
  Array<Expr> new_args;

  // Get the device of the current node. If not specified, the default is always CPU.
  const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  Device call_device;
  if (device_arg_idx >= args.size()) {
    call_device = Device(static_cast<tvm::Device>((*str2dev)("cpu")));
  } else {
    call_device = GetDeviceFromConstExpr(args[device_arg_idx]);
  }

  // Get the target device.
  Device target_device = Device(static_cast<tvm::Device>((*str2dev)(target_device_str)));

  // Current node is not on the desired device, adjust the device argument.
  if (target_device.device_type() != call_device.device_type()) {
    for (auto arg_idx = 0; arg_idx < default_vals.size(); ++arg_idx) {
      if (!default_vals[arg_idx].defined()) {
        // Do nothing with required arguments.
        new_args.push_back(args[arg_idx]);
      } else if (arg_idx >= args.size() || arg_idx == device_arg_idx) {
        // Make up the default argument value.
        new_args.push_back(default_vals[arg_idx]);
      } else {
        // Optional argument is specified.
        new_args.push_back(args[arg_idx]);
      }
    }
    CHECK_EQ(new_args.size(), default_vals.size());
  } else {
    new_args = args;
  }
  return Call(node->op, new_args);
}

Expr AssignDeviceFullOp(const CallNode* node, const Array<Expr> args,
                        std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* fill value */
      Expr(),                                            /* shape */
      MakeConstant(StringValue::make("int")),            /* target dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 3, default_vals);
}

Expr AssignDeviceOneHotOp(const CallNode* node, const Array<Expr> args,
                          std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* indices */
      Expr(),                                            /* on_value */
      Expr(),                                            /* off_value */
      Expr(),                                            /* depth */
      MakeConstant(ScalarValue::make(-1)),               /* axis */
      MakeConstant(StringValue::make("int")),            /* dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 6, default_vals);
}

Expr AssignDeviceInitOp(const CallNode* node, const Array<Expr> args,
                        std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* shape */
      MakeConstant(StringValue::make("int")),            /* dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 2, default_vals);
}

Expr AssignDeviceArangeOp(const CallNode* node, const Array<Expr> args,
                          std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* start */
      Expr(),                                            /* stop */
      Expr(),                                            /* step */
      MakeConstant(StringValue::make("float")),          /* dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 4, default_vals);
}

typedef Expr (*AssignDeviceOpFuncType)(const CallNode* node, const Array<Expr> args,
                                       std::string target_device);
std::unordered_map<String, AssignDeviceOpFuncType> fmap = {
    {"raf.op.full", &AssignDeviceFullOp},
    {"raf.op.one_hot", &AssignDeviceOneHotOp},
    {"raf.op.zeros", &AssignDeviceInitOp},
    {"raf.op.ones", &AssignDeviceInitOp},
    {"raf.op.arange", &AssignDeviceArangeOp}};

class DeviceAssigner : public ExprMutator {
 public:
  DeviceAssigner(std::string device) : device_str_(device){};

  Expr VisitExpr_(const RelayConstantNode* node) final {
    auto value = Downcast<Value>(ConstantExtractValue(GetRef<Constant>(node)));

    // Only focus on constant tensor.
    if (value.as<TensorValueObj>()) {
      DLTensor* dlt = value;

      const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
      tvm::Device target_tvm_ctx = (*str2dev)(device_str_);
      Device target_device = Device(target_tvm_ctx);

      // Do nothing if the constant is already on the target device.
      if (target_tvm_ctx.device_type == dlt->device.device_type) {
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

Pass AssignDevice(std::string device) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto assigner = assign_device::DeviceAssigner(device);
    return Downcast<Function>(assigner.Mutate(f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "AssignDevice", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.AssignDevice").set_body_typed(AssignDevice);

}  // namespace pass
}  // namespace raf
