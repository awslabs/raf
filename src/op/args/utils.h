#pragma once
#include <mnm/op.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace args {

std::vector<int64_t> NormalizeTupleOrInt(const value::Value& value);
bool ToBool(const value::Value &self);
int ToInt(const value::Value &self);
int64_t ToInt64(const value::Value &self);
float ToFloat(const value::Value &self);
double ToDouble(const value::Value &self);
std::string ToString(const value::Value &self);

}  // namespace args
}  // namespace op
}  // namespace mnm
