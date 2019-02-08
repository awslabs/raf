#pragma once

#include <vector>
#include <limits>
#include <cassert>
#include <ostream>
#include <sstream>

// TODO(@junrushao1994): replace assert with dmlc-core's CHECK
// TODO(@junrushao1994): should we enable overflow checks only in DEBUG mode?

namespace mnm {
namespace types {
namespace details {

#define MNM_DIM_T_CHECK_OOB(U, v)                             \
  {                                                           \
    constexpr int64_t min = std::numeric_limits<U>::min();    \
    constexpr int64_t max = std::numeric_limits<U>::max();    \
    assert(min <= v && v <= max);                             \
  }
#define MNM_DIM_T_DEFINE_UNARY(op)                            \
  TSelf operator op () const {                                \
    MNM_DIM_T_CHECK_OOB(T, (op v_));                          \
    return TSelf(op v_);                                      \
  }

#define MNM_DIM_T_DEFINE_BINARY(op)                           \
  TSelf operator op (const TSelf &rhs) const {                \
    MNM_DIM_T_CHECK_OOB(T, (v_ op rhs.v_));                   \
    return TSelf(v_ op rhs.v_);                               \
  }

#define MNM_DIM_T_DEFINE_ASSIGN(op, base_op)                  \
  TSelf& operator op (const TSelf &rhs) {                     \
    MNM_DIM_T_CHECK_OOB(T, (v_ base_op rhs.v_));              \
    v_ op rhs.v_;                                             \
    return *this;                                             \
  }

#define MNM_DIM_T_DEFINE_BOOL_UNARY(op)                       \
  bool operator op () const {                                 \
    return op v_;                                             \
  }

#define MNM_DIM_T_DEFINE_BOOL_BINARY(op)                      \
  bool operator op (const TSelf &rhs) const {                 \
    return v_ op rhs.v_;                                      \
  }

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
  TSelf& operator = (const TSelf &) = default;
  TSelf& operator = (TSelf &&) = default;
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
  operator std::string() const {
    return std::to_string(v_);
  }
  friend std::ostream& operator << (std::ostream &os, const TSelf &self) {
    return os << std::to_string(self.v_);
  }
private:
  T v_;
}; // dim_t_base
#undef MNM_CHECK_OOB
#undef MNM_DEFINE_UNARY
#undef MNM_DEFINE_BINARY
#undef MNM_DEFINE_BOOL_UNARY
#undef MNM_DEFINE_BOOL_BINARY
#undef MNM_DEFINE_ASSIGN

using dim_t = dim_t_base<int64_t>;

} // details
} // types
} // mnm

namespace mnm {
namespace types {
namespace details {

using dim_t = mnm::types::details::dim_t;

class shape_t final {
public:
  shape_t(): data_() {}
  shape_t(shape_t &&other): data_(std::move(other.data_)) {}
  shape_t(const shape_t &other) : data_(other.data_) {}
  shape_t& operator = (shape_t &&other) {
    data_ = std::move(other.data_);
    return *this;
  }
  shape_t& operator = (const shape_t &other) {
    data_ = other.data_;
    return *this;
  }
  template<typename IterType>
  shape_t(IterType begin, IterType end) {
    Assign(begin, end);
  }
  template <typename T>
  shape_t(const std::vector<T> &init) {
    Assign(init.begin(), init.end());
  }
  template<typename T>
  shape_t(std::initializer_list<T> init) {
    Assign(init.begin(), init.end());
  }
  friend std::ostream& operator << (std::ostream &os, const shape_t &self) {
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
  int Ndim() const {
    return static_cast<int>(data_.size());
  }
  dim_t Numel() const {
    dim_t result(1);
    for (const dim_t x : data_) {
      result *= x;
    }
    return result;
  }
  dim_t operator [] (const int axis) const {
    assert(-Ndim() <= axis && axis < Ndim());
    return data_[axis < 0 ? axis + Ndim() : axis];
  }
  std::vector<dim_t> get() const {
    return data_;
  }
private:
  template<typename IterType>
  void Assign(IterType begin, IterType end) {
    std::vector<dim_t> result;
    for (IterType it = begin; it != end; ++it) {
      result.push_back(dim_t(*it));
    }
    data_ = std::move(result);
  }
private:
  std::vector<dim_t> data_;
}; // shape_t
} // details
} // types
} // mnm

namespace mnm {
namespace types {

using dim_t = mnm::types::details::dim_t;
using shape_t = mnm::types::details::shape_t;

} // types
} // mnm
