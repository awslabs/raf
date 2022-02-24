/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/device.cc
 * \brief Device implementation.
 */
#include <stack>
#include <dmlc/thread_local.h>
#include <tvm/runtime/object.h>
#include <tvm/node/repr_printer.h>

#include "raf/device.h"
#include "raf/registry.h"

namespace raf {

using namespace raf::ir;
using tvm::ReprPrinter;

struct DeviceThreadLocalEntry {
  /*! \brief The current device context */
  std::stack<Device> context_stack;
};

/*! \brief Thread local store to hold the Device context stack. */
using DeviceThreadLocalStore = dmlc::ThreadLocalStore<DeviceThreadLocalEntry>;

Device Device::make(Integer device_type_value, Integer device_id) {
  ObjectPtr<DeviceObj> n = make_object<DeviceObj>();
  n->device_type = DevType(device_type_value);
  n->device_id = device_id;
  return Device(n);
}

Device::Device() {
  ObjectPtr<DeviceObj> n = make_object<DeviceObj>();
  n->device_type = DevType::kUnknown();
  n->device_id = -1;
  data_ = std::move(n);
}

Device::Device(DevType device_type, int device_id) {
  ObjectPtr<DeviceObj> n = make_object<DeviceObj>();
  n->device_type = device_type;
  n->device_id = device_id;
  data_ = std::move(n);
}

Device::Device(tvm::Device tvm_dev) {
  ObjectPtr<DeviceObj> n = make_object<DeviceObj>();
  n->device_type = tvm_dev.device_type;
  n->device_id = tvm_dev.device_id;
  data_ = std::move(n);
}

void Device::EnterWithScope() {
  DeviceThreadLocalEntry* entry = DeviceThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void Device::ExitWithScope() {
  DeviceThreadLocalEntry* entry = DeviceThreadLocalStore::Get();
  ICHECK(!entry->context_stack.empty());
  entry->context_stack.pop();
}

Device Device::Current(bool allow_default) {
  DeviceThreadLocalEntry* entry = DeviceThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  ICHECK(allow_default) << "Current device is empty and default is not allowed";

  return Device();
}

static Target GetTVMTarget(Device device) {
  return device.tvm_target();
}

static void EnterScope(Device device) {
  device.EnterWithScope();
};
static void ExitScope(Device device) {
  device.ExitWithScope();
};

RAF_REGISTER_GLOBAL("raf.device.Device").set_body_typed(Device::make);
RAF_REGISTER_GLOBAL("raf.device.GetTVMTarget").set_body_typed(GetTVMTarget);
RAF_REGISTER_GLOBAL("raf.device.DeviceEnterScope").set_body_typed(EnterScope);
RAF_REGISTER_GLOBAL("raf.device.DeviceExitScope").set_body_typed(ExitScope);
RAF_REGISTER_GLOBAL("raf.device.DeviceCurrent").set_body_typed(Device::Current);
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DeviceObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DeviceObj*>(ref.get());
      p->stream << "Device(" << GetRef<Device>(node).c_str() << ")";
    });

RAF_REGISTER_OBJECT_REFLECT(DeviceObj);

}  // namespace raf
