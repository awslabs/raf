/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/device_api.cc
 * \brief Device api manager
 */
#include "raf/device_api.h"
#include "raf/registry.h"

namespace raf {
namespace device_api {

using registry::GetPackedFunc;
using registry::PerDevTypeStore;

class DeviceAPIManager {
 public:
  static DeviceAPIManager* Get() {
    static DeviceAPIManager* instance = new DeviceAPIManager();
    return instance;
  }

  static DeviceAPI* CreateDeviceAPI(DevType device_type) {
    thread_local char creator_name[128];
    snprintf(creator_name, sizeof(creator_name), "raf.device_api._make.%s", device_type.c_str());
    void* ret = GetPackedFunc(creator_name)();
    return static_cast<DeviceAPI*>(ret);
  }

 public:
  PerDevTypeStore<DeviceAPI, false> reg;
};

std::shared_ptr<DeviceAPI> DeviceAPI::Get(DevType device_type) {
  DeviceAPIManager* mgr = DeviceAPIManager::Get();
  std::shared_ptr<DeviceAPI>& result = mgr->reg.Get(device_type);
  if (result == nullptr) {
    std::lock_guard<std::mutex> lock(mgr->reg.mutex_);
    if (result == nullptr) {
      result.reset(DeviceAPIManager::CreateDeviceAPI(device_type));
    }
  }
  return result;
}

}  // namespace device_api
}  // namespace raf
