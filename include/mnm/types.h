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

#include <mnm/enum_base.h>
// TODO(@junrushao1994): replace CHECK with detailed errors
// TODO(@junrushao1994): should we enable overflow checks only in DEBUG mode?

namespace mnm {
namespace types {

class DTypeCode final : public EnumBase<DTypeCode, 4, int32_t, DLDataTypeCode> {
 public:
  ENUM_DEF_HEADER(DTypeCode, 0, plain + 1);
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, kUnknown, 0, kDLInt, "???");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, kInt, 1, kDLInt, "i");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, kUInt, 2, kDLUInt, "u");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, kFloat, 3, kDLFloat, "f");
};

class DeviceType final : public EnumBase<DeviceType, 13, int32_t, int> {
 public:
  ENUM_DEF_HEADER(DeviceType, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kUnknown, 0, 0, "???");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kCPU, 1, (int)kDLCPU, "cpu");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kGPU, 2, (int)kDLGPU, "gpu");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kCPUPinned, 3, (int)kDLCPUPinned, "cpupinned");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kOpenCL, 4, (int)kDLOpenCL, "opencl");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kAOCL, 5, (int)kDLAOCL, "aocl");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kSDAccel, 6, (int)kDLSDAccel, "sdaccel");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kVulkan, 7, (int)kDLVulkan, "vulkan");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kMetal, 8, (int)kDLMetal, "metal");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kVPI, 9, (int)kDLVPI, "vpi");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kROCM, 10, (int)kDLROCM, "rocm");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kOpenGL, 11, (int)::kOpenGL, "opengl");
  ENUM_DEF_ENTRY_WITH_NAME(DeviceType, kExtDev, 12, (int)kDLExtDev, "extdev");
  DeviceType(DLDeviceType code) : EnumBase(code) {
  }
  DeviceType(TVMDeviceExtType code) : EnumBase(code) {
  }
  explicit operator DLDeviceType() const {
    return static_cast<DLDeviceType>(v);
  }
  explicit operator TVMDeviceExtType() const {
    return static_cast<TVMDeviceExtType>(v);
  }
};

class Context {
 public:
  Context() = default;
  Context(DeviceType device_type, int device_id) : device_type(device_type), device_id(device_id) {
  }
  Context(TVMContext context) : device_type(context.device_type), device_id(context.device_id) {
  }
  operator TVMContext() const {
    return TVMContext{DLDeviceType(device_type), device_id};
  }
  const char* c_str() const {
    thread_local char buffer[128];
    sprintf(buffer, "%s(%d)", device_type.c_str(), device_id);
    return buffer;
  }

 public:
  DeviceType device_type{DeviceType::kUnknown()};
  int device_id{-1};
};

class DType {
 public:
  DType() = default;
  DType(DTypeCode code, int bits, int lanes) : code(code), bits(bits), lanes(lanes) {
  }
  DType(DLDataType dtype)
      : code(static_cast<DLDataTypeCode>(dtype.code)), bits(dtype.bits), lanes(dtype.lanes) {
  }
  operator DLDataType() const {
    return DLDataType{
        /*code=*/static_cast<uint8_t>(DLDataTypeCode(code)),
        /*bits=*/static_cast<uint8_t>(bits),
        /*lanes=*/static_cast<uint16_t>(lanes),
    };
  }
  const char* c_str() const {
    thread_local char buffer[128];
    sprintf(buffer, "%s(bits = %d, lanes = %d)", code.c_str(), bits, lanes);
    return buffer;
  }

 public:
  DTypeCode code{DTypeCode::kUnknown()};
  int bits{-1};
  int lanes{-1};
};

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
