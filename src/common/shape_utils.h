#include <vector>

#include <mnm/rly.h>

namespace mnm {
namespace common {
namespace shape_utils {

inline std::vector<int64_t> MakeShape(const rly::Array<rly::Integer>& shape) {
  int ndim = shape.size();
  std::vector<int64_t> result(ndim);
  for (int i = 0; i < ndim; ++i) {
    result[i] = shape[i]->value;
  }
  return result;
}

inline std::vector<int64_t> Shape2Strides(const std::vector<int64_t>& shape) {
  int ndim = shape.size();
  std::vector<int64_t> strides(ndim);
  int64_t carry = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = carry;
    carry *= shape[i];
  }
  return strides;
}

}  // namespace shape_utils
}  // namespace common
}  // namespace mnm
