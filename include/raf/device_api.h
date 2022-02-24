/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file device_api.h
 * \brief Unified low-level API for heterogeneous devices
 */
#pragma once
#include <memory>
#include "./device.h"

namespace raf {
namespace device_api {

// TODO(@junrushao1994): To pass flags to stream/event/..., do we add thread_local flags?
class DeviceAPI {
 public:
  virtual ~DeviceAPI() = default;

  /*!
   * \brief Get the number of devices.
   * \return The number of devices.
   */
  virtual int GetDeviceCount() = 0;

  /*!
   * \brief Allocate a chuck of memory.
   * \param nbytes The size of memory in bytes to allocate.
   * \param alignment The alignment size.
   * \return The allocated memory.
   */
  virtual void* AllocMemory(int64_t nbytes, int64_t alignment) = 0;

  /*!
   * \brief Allocate a chuck of memory asynchronously.
   * \param nbytes The size of memory in bytes to allocate.
   * \param stream The stream to place the allocation on.
   * \param alignment The alignment size.
   * \return The allocated memory.
   */
  virtual void* AllocMemoryAsync(int64_t nbytes, void* stream, int64_t alignment) = 0;

  /*!
   * \brief Free the allocated memory.
   * \param ptr The allocated memory to be freed.
   */
  virtual void FreeMemory(void* ptr) = 0;

  /*!
   * \brief Free the memory asynchronously.
   * \param ptr The allocated memory to be freed.
   * \param stream The stream to place the free operation on.
   */
  virtual void FreeMemoryAsync(void* ptr, void* stream) = 0;

  /*!
   * \brief Copy data from one place to another
   * \param from The source array.
   * \param to The target array.
   * \param stream Optional stream.
   */
  virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream = nullptr) = 0;

  /*!
   * \brief Query the memory pool size of the underlying memory pool of this device, if applicable.
   * \return <used, allocated> 'used' is the number of bytes that has been allocated to the user,
   * and the 'allocated' is the number of bytes that has been allocated from the system.
   */
  virtual std::pair<int64_t, int64_t> GetPoolSize() {
    return std::make_pair(0, 0);
  };

  /*!
   * \brief Set the device ID for memory allocation. This API is for GPU only.
   * \param dev_id The device id.
   */
  virtual void SetDevice(const int device_id) = 0;

  /*!
   * \brief Create a stream on given device.
   * \param dev The device to create the stream.
   * \return The created stream.
   */
  virtual void* CreateStream(const Device& dev) = 0;

  /*!
   * \brief Free a stream.
   * \param dev The device to free the stream.
   * \param stream The stream.
   */
  virtual void FreeStream(const Device& dev, void* stream) = 0;

  /*!
   * \brief Set the stream for executing computation ops. This API is for GPU only.
   * \param dev The device to set the stream.
   * \param stream The stream to set.
   */
  virtual void SetStream(const Device& dev, void* stream) = 0;

  /*!
   * \brief Get the current stream for executing computation ops. This API is for GPU only.
   * \param dev The device to set the stream.
   * \param stream The stream to set.
   */
  virtual void* GetStream() = 0;

  /*!
   * \brief Create an event on given device.
   * \param dev The device to create the event.
   * \param flags The flags of the event. The value depends on the underlying device. For CUDA
   * device, see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html for
   * the available flags.
   * \return The created event.
   */
  virtual void* CreateEvent(const Device& dev, uint32_t flags = 0) = 0;

  /*!
   * \brief Free an event.
   * \param dev The device of the event.
   * \param event The event.
   */
  virtual void FreeEvent(const Device& dev, void* event) = 0;

  /*!
   * \brief Get the elapsed time between two events in milliseconds.
   * \param start_event The start event.
   * \param end_event The end event.
   * \return The elapsed time in milliseconds represented by a float number.
   */
  virtual float EventElapsedTimeInMilliSeconds(void* start_event, void* end_event) = 0;

  /*!
   * \brief Place an event on given stream. It would record the pending workloads on that stream.
   * \param event The event to record workloads.
   * \param stream The stream to be recorded.
   */
  virtual void EventRecordOnStream(void* event, void* stream) = 0;

  /*!
   * \brief Let a stream wait for an event. This call is asynchronous. All workloads issued to given
   * stream would be executed after the workloads recorded by the event.
   * \param stream The stream to wait for the event.
   * \param event The event to be waited for.
   */
  virtual void StreamWaitEvent(void* stream, void* event) = 0;

  // Granularity of synchronization
  /*!
   * \brief Synchronize the device. It would block the host thread until all pending workloads on
   * the given device finished.
   * \param dev The device to wait.
   */
  virtual void WaitDevice(const Device& dev) = 0;

  /*!
   * \brief Synchronize the stream. It would block the host thread until all pending workloads on
   * the given stream finished.
   * \param stream The stream to wait.
   */
  virtual void WaitStream(void* stream) = 0;

  /*!
   * \brief  Synchronize the event. It would block the host thread until the the workloads captured
   * by the given event finished.
   * \param event The event to wait.
   */
  virtual void WaitEvent(void* event) = 0;

  /*!
   * \brief The the device api of given device type
   * \param device_type The device type.
   * \return The device api.
   */
  static std::shared_ptr<DeviceAPI> Get(DevType device_type);
};

}  // namespace device_api
}  // namespace raf
