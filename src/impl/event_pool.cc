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
 * \file src/impl/event_pool.cc
 * \brief MNM event pool underlying implementation
 */
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "mnm/device_api.h"
#include "mnm/registry.h"
#include "mnm/event_pool.h"

namespace mnm {
namespace event_pool {

using device_api::DeviceAPI;
using registry::PerDeviceStore;

class Event::Impl {
 public:
  explicit Impl(Device dev, uint32_t flags, void* event)
      : device_(std::move(dev)), flags_(flags), event_(event) {
  }

  ~Impl() {
    if (event_ != nullptr) {
      EventPool::Get(device_)->RecycleEvent(flags_, event_);
    }
  }

 public:
  Device device_;
  uint32_t flags_;
  void* event_;
};

Event::Event(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

void* Event::data() const {
  return impl_ ? impl_->event_ : nullptr;
}

EventPool::~EventPool() {
  for (auto& pr : freed_events_) {
    for (void* event : pr.second) {
      api_->FreeEvent(device_, event);
    }
  }
}

std::shared_ptr<Event> EventPool::GetEvent(uint32_t flags) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& pool = freed_events_[flags];
  void* event;
  if (!pool.empty()) {
    event = pool.back();
    pool.pop_back();
  } else {
    event = api_->CreateEvent(device_, flags);
  }
  return std::shared_ptr<Event>(new Event(std::make_unique<Event::Impl>(device_, flags, event)));
}

EventPool::EventPool(const Device& dev) : device_(dev), api_(DeviceAPI::Get(dev.device_type())) {
}

void EventPool::RecycleEvent(uint32_t flags, void* event) {
  std::lock_guard<std::mutex> lock(mutex_);
  freed_events_[flags].push_back(event);
}

std::shared_ptr<EventPool> EventPool::Get(const Device& dev) {
  static auto* per_device = new PerDeviceStore<EventPool, false>();
  std::shared_ptr<EventPool>& ret = per_device->Get(dev);
  if (ret == nullptr) {
    std::lock_guard<std::mutex> lock(per_device->mutex_);
    if (ret == nullptr) {
      ret = std::shared_ptr<EventPool>(new EventPool(dev));
    }
  }
  return ret;
}

Event::~Event() = default;

}  // namespace event_pool
}  // namespace mnm
