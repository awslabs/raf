#pragma once

#include <limits>
#include <ostream>
#include <sstream>
#include <vector>

#include <dlpack/dlpack.h>
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>

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
  const char* c_str(bool allow_abbr_cpu = false) const {
    thread_local char buffer[128];
    if (allow_abbr_cpu && device_type == DeviceType::kCPU() && device_id == 0) {
      sprintf(buffer, "%s", device_type.c_str());
    } else {
      sprintf(buffer, "%s(%d)", device_type.c_str(), device_id);
    }
    return buffer;
  }

 public:
  DeviceType device_type{DeviceType::kUnknown()};
  int device_id{-1};
};

class DType {
 public:
  DType() = default;
  DType(DTypeCode code, int bits, int lanes = 1) : code(code), bits(bits), lanes(lanes) {
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
    if (lanes == 1) {
      sprintf(buffer, "%s%d", code.c_str(), bits);
    } else {
      sprintf(buffer, "%s%dx%d", code.c_str(), bits, lanes);
    }
    return buffer;
  }

 public:
  DTypeCode code{DTypeCode::kUnknown()};
  int bits{-1};
  int lanes{-1};
};

}  // namespace types
}  // namespace mnm
