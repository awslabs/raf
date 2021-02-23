/*!
 * Copyright (c) 2019 by Contributors
 * \file profiler.h
 * \brief profiler
 */
#pragma once
#include <dmlc/concurrentqueue.h>
#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include "base.h"
#include "device_api.h"

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <unistd.h>
#endif

#define WITH_BASE_PROFILER(DEVICE, NAME, CAT, ARGS, CODE_SNIPPET)                                 \
  {                                                                                               \
    bool profiling = profiler::Profiler::Get()->IsProfiling();                                    \
    if (profiling) {                                                                              \
      profiler::ProfilerBaseHelper phelper(profiling, DEVICE.device_id, DEVICE.device_type, NAME, \
                                           CAT, ARGS);                                            \
      phelper.start();                                                                            \
      CODE_SNIPPET                                                                                \
      phelper.stop();                                                                             \
    } else {                                                                                      \
      CODE_SNIPPET                                                                                \
    }                                                                                             \
  }

namespace mnm {
namespace profiler {

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
inline size_t current_process_id() {
  return ::GetCurrentProcessId();
}
#else
inline size_t current_process_id() {
  return getpid();
}
#endif

/*!
 * \brief Event type as used for chrome://tracing support
 * \note Tracing formats:
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview NOLINT(*)
 */
enum EventType {
  kDurationBegin = 'B',
  kDurationEnd = 'E',
  kComplete = 'X',
  kInstant = 'i',
  kCounter = 'C',
  kAsyncNestableStart = 'b',
  kAsyncNestableInstant = 'n',
  kAsyncNestableEnd = 'e',
  kFlowStart = 's',
  kFlowStep = 't',
  kFlowEnd = 'f',
  kSample = 'P',
  kObjectCreated = 'N',
  kObjectSnapshot = 'O',
  kObjectDestroyed = 'D',
  kMetadata = 'M',
  kMemoryDumpGlobal = 'V',
  kMemoryDumpProcess = 'v',
  kMark = 'R',
  kClockSync = 'c',
  kContextEnter = '(',
  kContextLeave = ')'
};

struct ProfileSubEvent {
  /*! \brief whether this sub-event object is enabled */
  bool enabled_ = false;
  /*! \brief Type of the sub-event */
  EventType event_type_;
  /*! \brief Timestamp of sub-event */
  uint64_t timestamp_;
};

class ProfileStat {
 public:
  enum DurationStatIndex { kStart, kStop };
  std::string name_;
  std::string categories_;
  std::string args_string = "=";

  size_t process_id_ = current_process_id();
  std::thread::id thread_id_ = std::this_thread::get_id();

  /*! \brief Sub-events (ie begin, end, etc.) */
  ProfileSubEvent items_[3];  // Don't use vector in order to avoid memory allocation

 public:
  ProfileStat(std::string categories, std::string name, uint64_t start_time, uint64_t end_time,
              const std::vector<std::string>& args);

  /*!
   * \brief Get current tick count in microseconds
   * \return Current arbitrary tick count in microseconds
   */
  static inline uint64_t NowInMicrosec() {
#if defined(_MSC_VER) && _MSC_VER <= 1800
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return counter.QuadPart * 1000000 / frequency.QuadPart;
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
#endif
  }

  void EmitEvents(std::ostream* os);
};

class DeviceStats {
  using TQueue = dmlc::moodycamel::ConcurrentQueue<ProfileStat*>;

 public:
  /* data */
  /*! \brief device name */
  std::string dev_name_;
  /*! \brief operation execution statistics on this device */
  std::shared_ptr<TQueue> opr_exec_stats_ = std::make_shared<TQueue>();

 public:
  ~DeviceStats();
};

class Profiler {
 public:
  Profiler();
  ~Profiler();
  static Profiler* Get(std::shared_ptr<Profiler>* sp = nullptr);
  void SetConfig(bool profiling);
  void AddNewProfileStat(std::string categories, std::string name, uint64_t start_time,
                         uint64_t end_time, const std::vector<std::string>& args);
  std::string GetProfile();
  std::vector<ProfileStat> GetProfileStats();
  inline bool IsProfiling() {
    return profiling_;
  }

 private:
  DeviceStats profile_stats_;
  bool profiling_{false};
  std::recursive_mutex m_;
  uint64_t init_time_;
};

class ProfilerBaseHelper {
 public:
  ProfilerBaseHelper(bool profiling, uint32_t dev_id, mnm::DevType dev_type, std::string name,
                     std::string categories, std::vector<std::string> args = {})
      : profiling_(profiling),
        device_(Device(dev_type, dev_id)),
        name_(name),
        categories_(categories),
        args(args) {
    if (dev_type != mnm::DevType::kUnknown()) {
      dev_api_ = device_api::DeviceAPI::Get(dev_type);
    }
  }
  void start();
  void stop();

 protected:
  /*! \brief whether the profiler is recording */
  const bool profiling_;
  /*! \brief the device on which profiled code runs */
  Device device_;
  /*! \brief the name annotation of profiling results */
  std::string name_;
  /*! \brief the category annotation of profiling results */
  std::string categories_;
  /*! \brief profiler start time */
  uint64_t start_time_;
  /*! \brief profiler end time */
  uint64_t end_time_;
  /*! \brief the api of the device on which profiled code runs */
  std::shared_ptr<device_api::DeviceAPI> dev_api_;
  /*! \brief the arguments annotation */
  std::vector<std::string> args;
};

inline void ProfilerBaseHelper::start() {
  if (profiling_) {
    if (dev_api_) {
      dev_api_->WaitDevice(device_);
    }
    start_time_ = ProfileStat::NowInMicrosec();
  }
}

inline void ProfilerBaseHelper::stop() {
  if (profiling_) {
    if (dev_api_) {
      dev_api_->WaitDevice(device_);
    }
    end_time_ = ProfileStat::NowInMicrosec();
    Profiler::Get()->AddNewProfileStat(categories_, name_, start_time_, end_time_, args);
  }
}

}  // namespace profiler
}  // namespace mnm
