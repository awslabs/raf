#pragma once

#include <cstdio>

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#include <mnm/enum_base.h>

namespace mnm {

class DTypeCode final : public EnumBase<DTypeCode, 4, int32_t, DLDataTypeCode> {
 public:
  ENUM_DEF_HEADER(DTypeCode, 0, plain + 1);
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 0, kUnknown, kDLInt, "???");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 1, kInt, kDLInt, "i");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 2, kUInt, kDLUInt, "u");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 3, kFloat, kDLFloat, "f");
};

class DevType final : public EnumBase<DevType, 13, int32_t, int> {
 public:
  ENUM_DEF_HEADER(DevType, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 0, kUnknown, 0, "???");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 1, kCPU, (int)kDLCPU, "cpu");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 2, kGPU, (int)kDLGPU, "gpu");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 3, kCPUPinned, (int)kDLCPUPinned, "cpupinned");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 4, kOpenCL, (int)kDLOpenCL, "opencl");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 5, kAOCL, (int)kDLAOCL, "aocl");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 6, kSDAccel, (int)kDLSDAccel, "sdaccel");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 7, kVulkan, (int)kDLVulkan, "vulkan");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 8, kMetal, (int)kDLMetal, "metal");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 9, kVPI, (int)kDLVPI, "vpi");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 10, kROCM, (int)kDLROCM, "rocm");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 11, kOpenGL, (int)::kOpenGL, "opengl");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 12, kExtDev, (int)kDLExtDev, "extdev");
  DevType(DLDeviceType code) : EnumBase(code) {
  }
  DevType(TVMDeviceExtType code) : EnumBase(code) {
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
  Context(DevType device_type, int device_id) : device_type(device_type), device_id(device_id) {
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
  DevType device_type{DevType::kUnknown()};
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

}  // namespace mnm
