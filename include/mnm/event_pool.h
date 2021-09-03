/*!
 * Copyright (c) 2021 by Contributors
 * \file event_pool.h
 * \brief Event pool API
 */
#pragma once
#include <memory>
#include <string>
#include "./device.h"

namespace mnm {
namespace event_pool {

class Event final {
 public:
  class Impl;
  friend Impl;

 public:
  Event() = default;

  explicit Event(Impl* impl);

  explicit Event(std::unique_ptr<Impl> impl);

  ~Event();

  void* data() const;

  /*!
   * Create an event on given device with flags.
   * \param dev The device to create the event.
   * \param flags The flags of the event. The flags depends on the underlying device. For cuda
   * device, refers to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html for
   * available flags.
   * \return
   */
  static std::shared_ptr<Event> Create(const Device& dev, uint32_t flags = 0);

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace event_pool
}  // namespace mnm