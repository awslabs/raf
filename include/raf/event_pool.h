/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file event_pool.h
 * \brief Event pool API
 */
#pragma once
#include <memory>
#include <string>
#include "./device.h"
#include "device_api.h"

namespace raf {
namespace event_pool {

using namespace device_api;

class EventPool;

/*!
 * \brief A representation of event on a device. The events can be used to describe the dependency
 * among kernels on different streams. They can also be used to measure the latency of kernels on
 * device without blocking the entire device.
 *
 * This class implements the event managed by the event pool, which can reduce the real event
 * allocation on device through recycling.
 */
class Event final {
 public:
  ~Event();

  /*!
   * \brief Get the pointer to the event, which can be used by DeviceAPI member functions.
   * \return The pointer to the event.
   */
  void* data() const;

 private:
  class Impl;

  explicit Event(std::unique_ptr<Impl> impl);

  /*! \brief The internal implementation of Event. */
  std::unique_ptr<Impl> impl_;

  friend Impl;
  friend EventPool;
};

class EventPool {
 public:
  ~EventPool();

  /*!
   * \brief Get the event pool for given device.
   * \param dev The device of the memory pool.
   * \return The memory pool for given device.
   */
  static std::shared_ptr<EventPool> Get(const Device& dev);

  /*!
   * \brief Get a new event with given flags. The new event can be a new event created by device api
   * or an event that has been recycled previously.
   * \param flags The flags of the event. The flags depends on the underlying device. For cuda
   * device, refers to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html for
   * available flags.
   * \return The new event with given flags.
   */
  std::shared_ptr<Event> GetEvent(uint32_t flags = 0);

 private:
  /*!
   * \brief Create an event pool for given device.
   * \param dev The device.
   */
  explicit EventPool(const Device& dev);

  /*!
   * \brief Recycle an event. The event will be reused later when user tries to get a new event with
   * the same flags.
   *
   * \param flags The flags of the recycled event.
   * \param event The event to be recycled.
   */
  void RecycleEvent(uint32_t flags, void* event);

  /*! \brief The device of the memory pool. */
  Device device_;
  /*! \brief The device api of memory pool's device. */
  std::shared_ptr<DeviceAPI> api_;
  /*! \brief The freed events. These events would be used for new GetEvent request. */
  std::unordered_map<uint32_t, std::vector<void*>> freed_events_;
  /*! \brief Mutex for exclusive access to the memory pool for multi threads. */
  std::mutex mutex_;

  friend Event;
};

}  // namespace event_pool
}  // namespace raf