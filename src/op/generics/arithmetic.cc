#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace arith {

using ir::Array;
using ir::Attrs;
using value::BoolValue;
using value::BoolValueNode;
using value::FloatValue;
using value::FloatValueNode;
using value::IntValue;
using value::IntValueNode;
using value::ScalarValue;
using value::Value;

#define SWITCH_SCALAR(var, value, body)                        \
  do                                                           \
    if (const auto* var = value.as<IntValueNode>()) {          \
      body;                                                    \
    } else if (const auto* var = value.as<FloatValueNode>()) { \
      body;                                                    \
    } else if (const auto* var = value.as<BoolValueNode>()) {  \
      body;                                                    \
    }                                                          \
  while (0);

#define SCALAR(data) OpInfo::make(ScalarValue::make(data), Context(DevType::kCPU(), 0), false)

OpInfo Add(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  SWITCH_SCALAR(x1, values[0], SWITCH_SCALAR(x2, values[1], return SCALAR(x1->data + x2->data);));
  LOG(FATAL) << "NotImplementedError";
  throw;
}

OpInfo Subtract(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  SWITCH_SCALAR(x1, values[0], SWITCH_SCALAR(x2, values[1], return SCALAR(x1->data - x2->data);));
  LOG(FATAL) << "NotImplementedError";
  throw;
}

OpInfo Multiply(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  SWITCH_SCALAR(x1, values[0], SWITCH_SCALAR(x2, values[1], return SCALAR(x1->data * x2->data);));
  LOG(FATAL) << "NotImplementedError";
  throw;
}

OpInfo Divide(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  SWITCH_SCALAR(x1, values[0], SWITCH_SCALAR(x2, values[1], return SCALAR(x1->data / x2->data);));
  LOG(FATAL) << "NotImplementedError";
  throw;
}

OpInfo Mod(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  if (const auto* x1 = values[0].as<IntValueNode>()) {
    if (const auto* x2 = values[1].as<IntValueNode>()) {
      return SCALAR(x1->data % x2->data);
    }
  }
  LOG(FATAL) << "NotImplementedError";
  throw;
}

OpInfo Negative(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  SWITCH_SCALAR(x1, values[0], return SCALAR(-x1->data););
  LOG(FATAL) << "NotImplementedError";
  throw;
}

#define MAKE_CMP(op_name, func_name, op)                                               \
  OpInfo func_name(const Array<Value>& values, const Attrs& attrs) {                   \
    CHECK_EQ(values.size(), 2);                                                        \
    SWITCH_SCALAR(x1, values[0],                                                       \
                  SWITCH_SCALAR(x2, values[1], return SCALAR(x1->data op x2->data);)); \
    LOG(FATAL) << "NotImplementedError";                                               \
    throw;                                                                             \
  }                                                                                    \
  MNM_REGISTER_OP(op_name).set_num_inputs(2).set_attr<FOpMakeOutput>("FOpMakeOutput", func_name);

MAKE_CMP("mnm.op.less", Less, <);
MAKE_CMP("mnm.op.greater", Greater, >);
MAKE_CMP("mnm.op.less_equal", LessEqual, <=);
MAKE_CMP("mnm.op.greater_equal", GreaterEqual, >=);
MAKE_CMP("mnm.op.equal", Equal, ==);
MAKE_CMP("mnm.op.not_equal", NotEqual, !=);

MNM_REGISTER_OP("mnm.op.add")
    .describe(R"code(This is Add.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Add);

MNM_REGISTER_OP("mnm.op.subtract")
    .describe(R"code(This is Subtract.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Subtract);

MNM_REGISTER_OP("mnm.op.multiply")
    .describe(R"code(This is Multiply.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Multiply);

MNM_REGISTER_OP("mnm.op.divide")
    .describe(R"code(This is Divide.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Divide);

MNM_REGISTER_OP("mnm.op.mod")
    .describe(R"code(This is Mod.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Mod);

MNM_REGISTER_OP("mnm.op.negative")
    .describe(R"code(This is Negative.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Negative);

}  // namespace arith
}  // namespace op
}  // namespace mnm
