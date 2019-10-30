#include "./utils.h"

namespace mnm {
namespace op {
namespace args {

std::vector<int64_t> NormalizeTupleOrInt(const value::Value& value) {
  using value::IntValueNode;
  using value::TupleValueNode;
  if (const auto* v = value.as<IntValueNode>()) {
    return {v->data};
  }
  if (const auto* v = value.as<TupleValueNode>()) {
    int n = v->fields.size();
    std::vector<int64_t> result;
    for (int i = 0; i < n; ++i) {
      if (const auto* vv = value.as<IntValueNode>()) {
        result.push_back(vv->data);
      }
      LOG(FATAL) << "Requires tuple of integers, but element " << i << " is not an integer";
      throw;
    }
    return result;
  }
  LOG(FATAL) << "Requires tuple of integers, but get " << value->type_key();
  throw;
}

using namespace mnm::value;

#define MNM_SWITCH_SCALAR(var, value, body)                      \
  do                                                             \
    if (const auto* var = (value).as<IntValueNode>()) {          \
      body;                                                      \
    } else if (const auto* var = (value).as<FloatValueNode>()) { \
      body;                                                      \
    } else if (const auto* var = (value).as<BoolValueNode>()) {  \
      body;                                                      \
    }                                                            \
  while (0);

bool ToBool(const Value &self) {
  MNM_SWITCH_SCALAR(value, self, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int";
  throw;
}

int ToInt(const Value &self) {
  MNM_SWITCH_SCALAR(value, self, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int";
  throw;
}

int64_t ToInt64(const Value &self) {
  MNM_SWITCH_SCALAR(value, self, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int64_t";
  throw;
}

float ToFloat(const Value &self) {
  MNM_SWITCH_SCALAR(value, self, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to float";
  throw;
}

double ToDouble(const Value &self) {
  MNM_SWITCH_SCALAR(value, self, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to double";
  throw;
}

std::string ToString(const Value &self) {
  if (const auto* value = self.as<StringValueNode>()) {
    return value->data;
  }
  LOG(FATAL) << "InternalError: cannot be converted to std::string";
  throw;
}

}  // namespace args
}  // namespace op
}  // namespace mnm
