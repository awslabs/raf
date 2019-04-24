#pragma once

#include <limits>
#include <ostream>
#include <sstream>
#include <vector>

#include <dlpack/dlpack.h>
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
// TODO(@junrushao1994): replace CHECK with detailed errors
// TODO(@junrushao1994): should we enable overflow checks only in DEBUG mode?

namespace mnm {
namespace types {

// TODO(@junrushao1994): make these enum strong-typed with std::string overloaded
using DataType = TVMType;
using Context = TVMContext;
using DeviceType = DLDeviceType;
using Args = tvm::runtime::TVMArgs;
using RetValue = tvm::runtime::TVMRetValue;

inline const char* DeviceName(int type) {
  switch (type) {
    case kDLCPU:
      return "cpu";
    case kDLGPU:
      return "gpu";
    case kDLOpenCL:
      return "opencl";
    case kDLSDAccel:
      return "sdaccel";
    case kDLAOCL:
      return "aocl";
    case kDLVulkan:
      return "vulkan";
    case kDLMetal:
      return "metal";
    case kDLVPI:
      return "vpi";
    case kDLROCM:
      return "rocm";
    case kOpenGL:
      return "opengl";
    case kDLExtDev:
      return "ext_dev";
    default:
      LOG(FATAL) << "unknown type =" << type;
      return "Unknown";
  }
}

}  // namespace types
}  // namespace mnm

namespace mnm {
namespace types {
namespace details {

#define MNM_IDX_T_CHECK_OOB(U, v)                          \
  {                                                        \
    constexpr int64_t min = std::numeric_limits<U>::min(); \
    constexpr int64_t max = std::numeric_limits<U>::max(); \
    CHECK(v >= min) << "Underflow";                        \
    CHECK(v <= max) << "Overflow";                         \
  }
#define MNM_IDX_T_DEFINE_UNARY(op)   \
  TSelf operator op() const {        \
    MNM_IDX_T_CHECK_OOB(T, (op v_)); \
    return TSelf(op v_);             \
  }

#define MNM_IDX_T_DEFINE_BINARY(op)           \
  TSelf operator op(const TSelf& rhs) const { \
    MNM_IDX_T_CHECK_OOB(T, (v_ op rhs.v_));   \
    return TSelf(v_ op rhs.v_);               \
  }

#define MNM_IDX_T_DEFINE_ASSIGN(op, base_op)     \
  TSelf& operator op(const TSelf& rhs) {         \
    MNM_IDX_T_CHECK_OOB(T, (v_ base_op rhs.v_)); \
    v_ op rhs.v_;                                \
    return *this;                                \
  }

#define MNM_IDX_T_DEFINE_BOOL_UNARY(op) \
  bool operator op() const {            \
    return op v_;                       \
  }

#define MNM_IDX_T_DEFINE_BOOL_BINARY(op)     \
  bool operator op(const TSelf& rhs) const { \
    return v_ op rhs.v_;                     \
  }

template <typename T>
class index_t_base final {
  using TSelf = index_t_base<T>;

 public:
  template <typename U>
  explicit index_t_base(const U& v) = delete;
  explicit index_t_base(const T& v) : v_(v) {
  }
  explicit index_t_base(const int& v) : v_(v) {
  }
  index_t_base(const TSelf&) = default;
  index_t_base(TSelf&&) = default;
  TSelf& operator=(const TSelf&) = default;
  TSelf& operator=(TSelf&&) = default;

 public:
  template <typename U>
  U As() const {
    MNM_IDX_T_CHECK_OOB(U, v_);
    return static_cast<U>(v_);
  }
  MNM_IDX_T_DEFINE_UNARY(~);
  MNM_IDX_T_DEFINE_BINARY(+);
  MNM_IDX_T_DEFINE_BINARY(-);
  MNM_IDX_T_DEFINE_BINARY(*);
  MNM_IDX_T_DEFINE_BINARY(/);
  MNM_IDX_T_DEFINE_BINARY(%);
  MNM_IDX_T_DEFINE_BINARY(&);
  MNM_IDX_T_DEFINE_BINARY(|);
  MNM_IDX_T_DEFINE_BINARY(^);
  MNM_IDX_T_DEFINE_BOOL_UNARY(!);
  MNM_IDX_T_DEFINE_BOOL_BINARY(<);
  MNM_IDX_T_DEFINE_BOOL_BINARY(>);
  MNM_IDX_T_DEFINE_BOOL_BINARY(<=);
  MNM_IDX_T_DEFINE_BOOL_BINARY(>=);
  MNM_IDX_T_DEFINE_BOOL_BINARY(==);
  MNM_IDX_T_DEFINE_BOOL_BINARY(!=);
  MNM_IDX_T_DEFINE_BOOL_BINARY(&&);
  MNM_IDX_T_DEFINE_BOOL_BINARY(||);
  MNM_IDX_T_DEFINE_ASSIGN(+=, +);
  MNM_IDX_T_DEFINE_ASSIGN(-=, -);
  MNM_IDX_T_DEFINE_ASSIGN(*=, *);
  MNM_IDX_T_DEFINE_ASSIGN(/=, /);
  MNM_IDX_T_DEFINE_ASSIGN(%=, %);
  MNM_IDX_T_DEFINE_ASSIGN(&=, &);
  MNM_IDX_T_DEFINE_ASSIGN(|=, |);
  MNM_IDX_T_DEFINE_ASSIGN(^=, ^);
  operator std::string() const {
    return std::to_string(v_);
  }
  friend std::ostream& operator<<(std::ostream& os, const TSelf& self) {
    return os << std::to_string(self.v_);
  }

 private:
  T v_;
};  // index_t_base
#undef MNM_CHECK_OOB
#undef MNM_DEFINE_UNARY
#undef MNM_DEFINE_BINARY
#undef MNM_DEFINE_BOOL_UNARY
#undef MNM_DEFINE_BOOL_BINARY
#undef MNM_DEFINE_ASSIGN

}  // namespace details

using index_t = details::index_t_base<int64_t>;

}  // namespace types
}  // namespace mnm
