/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file device.h
 * \brief Definition of device related data structure.
 */
#pragma once
#include <string>
#include "dlpack/dlpack.h"
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/support/with.h"
#include "./enum_base.h"
#include "./ir.h"

namespace raf {

using namespace raf::ir;

constexpr int64_t kDefaultMemoryAlignment = 64;

class DTypeCode final : public EnumBase<DTypeCode, 5, int32_t, DLDataTypeCode> {
 public:
  ENUM_DEF_HEADER(DTypeCode, 0, plain < 3 ? plain + 1 : plain);
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 0, kUnknown, kDLInt, "???");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 1, kInt, kDLInt, "i");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 2, kUInt, kDLUInt, "u");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 3, kFloat, kDLFloat, "f");
  ENUM_DEF_ENTRY_WITH_NAME(DTypeCode, 4, kBFloat, kDLBfloat, "bf");
};

class DevType final : public EnumBase<DevType, 13, int32_t, int> {
 public:
  ENUM_DEF_HEADER(DevType, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 0, kUnknown, 0, "???");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 1, kCPU, (int)kDLCPU, "cpu");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 2, kCUDA, (int)kDLCUDA, "cuda");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 3, kCUDAHost, (int)kDLCUDAHost, "cuda_host");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 4, kOpenCL, (int)kDLOpenCL, "opencl");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 5, kAOCL, (int)kDLAOCL, "aocl");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 6, kSDAccel, (int)kDLSDAccel, "sdaccel");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 7, kVulkan, (int)kDLVulkan, "vulkan");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 8, kMetal, (int)kDLMetal, "metal");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 9, kVPI, (int)kDLVPI, "vpi");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 10, kROCM, (int)kDLROCM, "rocm");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 11, kOpenGL, (int)::kOpenGL, "opengl");
  ENUM_DEF_ENTRY_WITH_NAME(DevType, 12, kExtDev, (int)kDLExtDev, "extdev");
  DevType(DLDeviceType code) : EnumBase(code) {  // NOLINT(runtime/explicit)
  }
  DevType(TVMDeviceExtType code) : EnumBase(code) {  // NOLINT(runtime/explicit)
  }
  explicit operator DLDeviceType() const {
    return static_cast<DLDeviceType>(v);
  }
  explicit operator TVMDeviceExtType() const {
    return static_cast<TVMDeviceExtType>(v);
  }
};

class DeviceObj : public Object {
 public:
  /*! \brief The device type. */
  DevType device_type;
  /*! \brief The device ID. */
  int device_id;

  void VisitAttrs(tvm::AttrVisitor* v) {
    int device_type_value = device_type;
    v->Visit("device_type", &device_type_value);
    v->Visit("device_id", &device_id);
  }

 public:
  static constexpr const char* _type_key = "raf.device.Device";
  RAF_FINAL_OBJECT(DeviceObj, Object);

  friend class Device;
};

class Device : public ObjectRef {
 public:
  static Device make(Integer device_type_value, Integer device_id);
  /*! \brief Default constructor, which behaves as Device(kUnknown, -1). Note that
   * this object is non-nullable and defined() is always true. */
  Device();
  Device(DevType device_type, int device_id);
  Device(tvm::Device dev);

  operator tvm::Device() const {
    return tvm::Device{DLDeviceType(self()->device_type), self()->device_id};
  }

  tvm::Target tvm_target() const {
    auto dl_device_type = tvm::runtime::String(self()->device_type.c_str());
    if (dl_device_type == "cpu") {
      // Device type in DLDevice does not recognize "cpu" but only "llvm".
      dl_device_type = tvm::runtime::String("llvm");
    }
    return tvm::Target(dl_device_type);
  }

  const char* c_str() const {
    thread_local char buffer[128];
    snprintf(buffer, sizeof(buffer), "%s(%d)", self()->device_type.c_str(), self()->device_id);
    return buffer;
  }

  bool operator==(const Device& other_dev) {
    auto other = other_dev.operator->();
    return self()->device_type == other->device_type && self()->device_id == other->device_id;
  }

  bool operator!=(const Device& other) {
    return !(*this == other);
  }

  DevType device_type() const {
    return self()->device_type;
  }

  void set_device_type(const DevType dev_type) const {
    self()->device_type = dev_type;
  }

  int device_id() const {
    return self()->device_id;
  }

  void set_device_id(int dev_id) const {
    self()->device_id = dev_id;
  }

  /*!
   * \brief Push a new device context onto the thread local stack.
   *  The Device on top of the stack is used to determine which
   *  specialization to use when invoking a GenericFunc.
   */
  void EnterWithScope();
  /*!
   * \brief Pop a device off the thread local context stack,
   *  restoring the previous device as the current context.
   */
  void ExitWithScope();

  /*!
   * \brief Get the current device context from thread local storage.
   * \param allow_default If the context stack is empty and this is set to true, a
   *   default Device will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The device that is the current context. The default device is returned if
   * allow_default is true.
   */
  static Device Current(bool allow_default = true);

 public:
  RAF_NOTNULLABLE_OBJECT_REF(Device, ir::ObjectRef, DeviceObj);

 private:
  inline DeviceObj* self() const {
    return this->operator->();
  }

  friend class tvm::With<Device>;
};

class DType {
 public:
  DType() = default;
  DType(DTypeCode code, int bits, int lanes = 1) : code(code), bits(bits), lanes(lanes) {
  }
  DType(DLDataType dtype)  // NOLINT(runtime/explicit)
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
      snprintf(buffer, sizeof(buffer), "%s%d", code.c_str(), bits);
    } else {
      snprintf(buffer, sizeof(buffer), "%s%dx%d", code.c_str(), bits, lanes);
    }
    return buffer;
  }

  bool operator==(const DType& other) {
    return code == other.code && bits == other.bits && lanes == other.lanes;
  }

  bool operator!=(const DType& other) {
    return !(*this == (other));
  }

 public:
  DTypeCode code{DTypeCode::kUnknown()};
  int bits{-1};
  int lanes{-1};

  template <typename T>
  inline operator T() const;
};

}  // namespace raf
