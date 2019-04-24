#pragma once

#include <tvm/relay/interpreter.h>

namespace mnm {
namespace value {

using Value = tvm::relay::Value;
using ValueNode = tvm::relay::ValueNode;

class TensorValue;
class TensorValueNode;

class TupleValue;
class TupleValueNode;

class ClosureValue;
class ClosureValueNode;

class RefValue;
class RefValueNode;

class ConstructorValue;
class ConstructorValueNode;

}  // namespace value
}  // namespace mnm
