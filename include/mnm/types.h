#pragma once

#include <limits>
#include <cassert>
#include <sstream>


namespace mnm {
namespace types {
namespace details {
namespace dim_t {

#define MNM_DIM_T_CHECK_OOB(U, v)                         \
  {                                                       \
    int64_t min = std::numeric_limits<U>::min();          \
    int64_t max = std::numeric_limits<U>::max();          \
    assert(min <= v && v <= max);                         \
  }
#define MNM_DIM_T_CHECK_OOB_1(op, a)
#define MNM_DIM_T_CHECK_OOB_2(a, op, b)
#define MNM_DIM_T_DEFINE_UNARY(op)                        \
  dim_t_base operator op () const {                       \
    MNM_DIM_T_CHECK_OOB_1(op, v_);                        \
    return dim_t_base(op v_);                             \
  }

#define MNM_DIM_T_DEFINE_BINARY(op)                       \
  dim_t_base operator op (const dim_t_base &rhs) const {  \
    MNM_DIM_T_CHECK_OOB_2(v_, op, rhs.v_);                \
    return dim_t_base(v_ op rhs.v_);                      \
  }

#define MNM_DIM_T_DEFINE_ASSIGN(op, base_op)              \
  dim_t_base& operator op (const dim_t_base &rhs) {       \
    MNM_DIM_T_CHECK_OOB_2(v_, base_op, rhs.v_);           \
    v_ op rhs.v_;                                         \
    return *this;                                         \
  }

#define MNM_DIM_T_DEFINE_BOOL_UNARY(op)                   \
  bool operator op () const {                             \
    return op v_;                                         \
  }

#define MNM_DIM_T_DEFINE_BOOL_BINARY(op)                  \
  bool operator op (const dim_t_base &rhs) const {        \
    return v_ op rhs.v_;                                  \
  }

template <typename T>
class dim_t_base final {
public:
  template <typename U>
  explicit dim_t_base(const U &v) = delete;
  explicit dim_t_base(const T &v) : v_(v) {}
  explicit dim_t_base(const int &v) : v_(v) {}
  dim_t_base(const dim_t_base &) = default;
  dim_t_base(dim_t_base &&) = default;
  dim_t_base& operator = (const dim_t_base &) = default;
  dim_t_base& operator = (dim_t_base &&) = default;
public:
  template <typename U>
  U get() const {
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
  friend std::ostream& operator << (std::ostream &os, const dim_t_base &self) {
    return os << std::to_string(self.v_);
  }
private:
  T v_;
};

using dim_t = dim_t_base<int64_t>;

#undef MNM_CHECK_OOB
#undef MNM_CHECK_OOB_1
#undef MNM_CHECK_OOB_2
#undef MNM_DEFINE_UNARY
#undef MNM_DEFINE_BINARY
#undef MNM_DEFINE_BOOL_UNARY
#undef MNM_DEFINE_BOOL_BINARY
#undef MNM_DEFINE_ASSIGN

} // dim_t
} // details
} // types
} // mnm


namespace mnm {
namespace types {

using details::dim_t::dim_t;

} // types
} // mnm
