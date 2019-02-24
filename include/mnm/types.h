#pragma once

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dlpack/dlpack.h>
#include <limits>
#include <ostream>
#include <sstream>
#include <vector>
// TODO(@junrushao1994): replace CHECK with detailed errors
// TODO(@junrushao1994): should we enable overflow checks only in DEBUG mode?

namespace mnm {
namespace types {
namespace details {

#define MNM_DIM_T_CHECK_OOB(U, v)                          \
  {                                                        \
    constexpr int64_t min = std::numeric_limits<U>::min(); \
    constexpr int64_t max = std::numeric_limits<U>::max(); \
    CHECK(v >= min) << "Underflow";                        \
    CHECK(v <= max) << "Overflow";                         \
  }
#define MNM_DIM_T_DEFINE_UNARY(op)   \
  TSelf operator op() const {        \
    MNM_DIM_T_CHECK_OOB(T, (op v_)); \
    return TSelf(op v_);             \
  }

#define MNM_DIM_T_DEFINE_BINARY(op)           \
  TSelf operator op(const TSelf &rhs) const { \
    MNM_DIM_T_CHECK_OOB(T, (v_ op rhs.v_));   \
    return TSelf(v_ op rhs.v_);               \
  }

#define MNM_DIM_T_DEFINE_ASSIGN(op, base_op)     \
  TSelf &operator op(const TSelf &rhs) {         \
    MNM_DIM_T_CHECK_OOB(T, (v_ base_op rhs.v_)); \
    v_ op rhs.v_;                                \
    return *this;                                \
  }

#define MNM_DIM_T_DEFINE_BOOL_UNARY(op) \
  bool operator op() const { return op v_; }

#define MNM_DIM_T_DEFINE_BOOL_BINARY(op) \
  bool operator op(const TSelf &rhs) const { return v_ op rhs.v_; }

template <typename T>
class dim_t_base final {
  using TSelf = dim_t_base<T>;

 public:
  template <typename U>
  explicit dim_t_base(const U &v) = delete;
  explicit dim_t_base(const T &v) : v_(v) {}
  explicit dim_t_base(const int &v) : v_(v) {}
  dim_t_base(const TSelf &) = default;
  dim_t_base(TSelf &&) = default;
  TSelf &operator=(const TSelf &) = default;
  TSelf &operator=(TSelf &&) = default;

 public:
  template <typename U>
  U As() const {
    MNM_DIM_T_CHECK_OOB(U, v_);
    return static_cast<U>(v_);
  }
  MNM_DIM_T_DEFINE_UNARY(~);
  MNM_DIM_T_DEFINE_BINARY(+);
  MNM_DIM_T_DEFINE_BINARY(-);
  MNM_DIM_T_DEFINE_BINARY(*);
  MNM_DIM_T_DEFINE_BINARY(/);
  MNM_DIM_T_DEFINE_BINARY(%);
  MNM_DIM_T_DEFINE_BINARY(&);
  MNM_DIM_T_DEFINE_BINARY(|);
  MNM_DIM_T_DEFINE_BINARY(^);
  MNM_DIM_T_DEFINE_BOOL_UNARY(!);
  MNM_DIM_T_DEFINE_BOOL_BINARY(<);
  MNM_DIM_T_DEFINE_BOOL_BINARY(>);
  MNM_DIM_T_DEFINE_BOOL_BINARY(<=);
  MNM_DIM_T_DEFINE_BOOL_BINARY(>=);
  MNM_DIM_T_DEFINE_BOOL_BINARY(==);
  MNM_DIM_T_DEFINE_BOOL_BINARY(!=);
  MNM_DIM_T_DEFINE_BOOL_BINARY(&&);
  MNM_DIM_T_DEFINE_BOOL_BINARY(||);
  MNM_DIM_T_DEFINE_ASSIGN(+=, +);
  MNM_DIM_T_DEFINE_ASSIGN(-=, -);
  MNM_DIM_T_DEFINE_ASSIGN(*=, *);
  MNM_DIM_T_DEFINE_ASSIGN(/=, /);
  MNM_DIM_T_DEFINE_ASSIGN(%=, %);
  MNM_DIM_T_DEFINE_ASSIGN(&=, &);
  MNM_DIM_T_DEFINE_ASSIGN(|=, |);
  MNM_DIM_T_DEFINE_ASSIGN(^=, ^);
  operator std::string() const { return std::to_string(v_); }
  friend std::ostream &operator<<(std::ostream &os, const TSelf &self) {
    return os << std::to_string(self.v_);
  }

 private:
  T v_;
};  // dim_t_base
#undef MNM_CHECK_OOB
#undef MNM_DEFINE_UNARY
#undef MNM_DEFINE_BINARY
#undef MNM_DEFINE_BOOL_UNARY
#undef MNM_DEFINE_BOOL_BINARY
#undef MNM_DEFINE_ASSIGN

using dim_t = dim_t_base<int64_t>;

}  // namespace details
}  // namespace types
}  // namespace mnm

namespace mnm {
namespace types {
namespace details {

using dim_t = mnm::types::details::dim_t;

class shape_t final {
 public:
  shape_t() : data_() {}
  shape_t(shape_t &&other) : data_(std::move(other.data_)) {}
  shape_t(const shape_t &other) : data_(other.data_) {}
  shape_t &operator=(shape_t &&other) {
    data_ = std::move(other.data_);
    return *this;
  }
  shape_t &operator=(const shape_t &other) {
    data_ = other.data_;
    return *this;
  }
  template <typename T>
  shape_t(const std::vector<T> &init) {
    Assign(init.begin(), init.end());
  }
  template <typename T>
  shape_t(std::initializer_list<T> init) {
    Assign(init.begin(), init.end());
  }
  friend std::ostream &operator<<(std::ostream &os, const shape_t &self) {
    os << '(';
    bool is_first = true;
    for (const dim_t v : self.data_) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      os << v;
    }
    os << ')';
    return os;
  }
  operator std::string() const {
    std::ostringstream os;
    os << *this;
    return os.str();
  }
  int Ndim() const { return static_cast<int>(data_.size()); }
  dim_t Numel() const {
    dim_t result(1);
    for (const dim_t x : data_) {
      result *= x;
    }
    return result;
  }
  dim_t operator[](const int axis) const {
    CHECK(0 <= axis && axis < Ndim())
        << "InternalError: please call shape_t::NormalizeAxis before indexing";
    return data_[axis];
  }
  std::vector<dim_t> get() const { return data_; }

 public:
  int NormalizeAxis(const int axis) const {
    const int ndim = Ndim();
    if (axis < -ndim || axis >= ndim) {
      LOG(FATAL) << "IndexError: indexing shape " << std::string(*this) << " with axis " << axis;
    }
    return axis < 0 ? axis + ndim : axis;
  }
  static shape_t Broadcast(const shape_t &a, const shape_t &b) {
    const int a_ndim = a.Ndim();
    const int b_ndim = b.Ndim();
    const int ndim = std::max(a_ndim, b_ndim);
    std::vector<dim_t> result;
    result.reserve(ndim);
    int a_axis = a_ndim - ndim;
    int b_axis = b_ndim - ndim;
    for (int i = 0; i < ndim; ++i, ++a_axis, ++b_axis) {
      const dim_t a_val = (a_axis >= 0) ? a.data_[a_axis] : dim_t(1);
      const dim_t b_val = (b_axis >= 0) ? b.data_[b_axis] : dim_t(1);
      if (a_val == b_val) {
        result.emplace_back(a_val);
      } else if (a_val == dim_t(1)) {
        result.emplace_back(b_val);
      } else if (b_val == dim_t(1)) {
        result.emplace_back(a_val);
      } else {
        LOG(FATAL) << "ValueError: could not be broadcast together with shapes: " << std::string(a)
                   << ", " << std::string(b) << "]";
      }
    }
    return shape_t(result);
  }

 private:
  template <typename IterType>
  void Assign(IterType begin, IterType end) {
    std::vector<dim_t> result;
    for (IterType it = begin; it != end; ++it) {
      result.push_back(dim_t(*it));
    }
    data_ = std::move(result);
  }

 private:
  std::vector<dim_t> data_;
};  // shape_t
}  // namespace details
}  // namespace types
}  // namespace mnm

namespace mnm {
namespace types {

using dim_t = mnm::types::details::dim_t;
using shape_t = mnm::types::details::shape_t;
// TODO(@junrushao1994): make these enum strong-typed with std::string overloaded
using DataType = TVMType;
using Context = TVMContext;
using DeviceType = DLDeviceType;
using Args = tvm::runtime::TVMArgs;
using RetValue = tvm::runtime::TVMRetValue;

auto DeviceName =  tvm::runtime::DeviceName;
}  // namespace types
}  // namespace mnm
