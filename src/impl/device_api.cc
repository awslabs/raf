/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/impl/device_api.cc
 * \brief Device api manager
 */
#include "mnm/device_api.h"
#include "mnm/registry.h"

namespace mnm {
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
    snprintf(creator_name, sizeof(creator_name), "mnm.device_api._make.%s", device_type.c_str());
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
}  // namespace mnm
