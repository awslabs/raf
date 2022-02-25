/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file profiler.h
 * \brief profiler
 */
#pragma once
#include <dmlc/concurrentqueue.h>
#include <cstdint>
#include <array>
#include <utility>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include "device.h"
#include "device_api.h"

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <unistd.h>
#endif

#define WITH_BASE_PROFILER_LEVEL(LEVEL, DEVICE, NAME, CAT, ARGS, CODE_SNIPPET)                \
  {                                                                                           \
    bool _profiling = raf::profiler::Profiler::Get()->IsProfiling(LEVEL);                     \
    if (_profiling) {                                                                         \
      auto& _pool = raf::profiler::Profiler::Get()->HelperPool();                             \
      auto _phelper_index = _pool.size();                                                     \
      _pool.push_back(raf::profiler::ProfilerHelper(DEVICE.device_id(), DEVICE.device_type(), \
                                                    NAME, CAT, ARGS));                        \
      _pool[_phelper_index].start();                                                          \
      CODE_SNIPPET                                                                            \
      _pool[_phelper_index].stop();                                                           \
    } else {                                                                                  \
      CODE_SNIPPET                                                                            \
    }                                                                                         \
  }

#define WITH_BASE_PROFILER(DEVICE, NAME, CAT, ARGS, CODE_SNIPPET) \
  WITH_BASE_PROFILER_LEVEL(1, DEVICE, NAME, CAT, ARGS, CODE_SNIPPET)

namespace raf {
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
  std::string args_string = "";

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

class ProfilerHelper {
 public:
  ProfilerHelper(int dev_id, raf::DevType dev_type, std::string name, std::string categories,
                 std::vector<std::string> args = {})
      : device_(Device(dev_type, dev_id)),
        name_(std::move(name)),
        categories_(std::move(categories)),
        args(std::move(args)) {
    if (dev_type != raf::DevType::kUnknown()) {
      dev_api_ = device_api::DeviceAPI::Get(dev_type);
    }
  }

  virtual ~ProfilerHelper() {
  }

  inline void start();
  inline void stop();
  inline void collect();

 protected:
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

class Profiler {
 public:
  ~Profiler();
  static Profiler* Get();  // std::shared_ptr<Profiler>* sp = nullptr);
  void AddNewProfileStat(std::string categories, std::string name, uint64_t start_time,
                         uint64_t end_time, const std::vector<std::string>& args);
  std::string GetProfile();
  std::vector<ProfileStat> GetProfileStats();

  inline bool IsProfiling(int level) {
    return profile_level_ >= level;
  }

  void CollectStat() {
    for (int i = 0; i < helpers_.size(); i++) {
      helpers_[i].collect();
    }
    helpers_.clear();
  }

  inline int profile_level() const {
    return profile_level_;
  }

  inline void set_profile_level(int profile_level = 0) {
    profile_level_ = profile_level;
  }

  std::vector<ProfilerHelper>& HelperPool() {
    return helpers_;
  }

 private:
  Profiler();

  /*! \brief Profile statistics. */
  DeviceStats profile_stats_;
  /*! \brief Profiling level. */
  int profile_level_{0};
  /*! \brief Mutex for multi-threading. */
  std::recursive_mutex m_;
  /*! \brief The helper pool. */
  std::vector<ProfilerHelper> helpers_;
};

inline void ProfilerHelper::start() {
  if (dev_api_) {
    dev_api_->WaitDevice(device_);
  }
  start_time_ = ProfileStat::NowInMicrosec();
}

inline void ProfilerHelper::stop() {
  if (dev_api_) {
    dev_api_->WaitDevice(device_);
  }
  end_time_ = ProfileStat::NowInMicrosec();
}

inline void ProfilerHelper::collect() {
  Profiler::Get()->AddNewProfileStat(categories_, name_, start_time_, end_time_, args);
}

/*!
 * \brief A helper class and macro to profile the execution time of a scope (e.g., function).
 * This is used for debugging purpose. For example:
 * void some_func() {
 *   RAF_TIMED("some_func")
 *   // do something;
 * }
 * The profiled time is then the life time of the created TimeSection object, and will be
 * logged to stderr.
 */
class TimedSection {
 public:
  explicit TimedSection(std::string name) : name_(name), start_(ProfileStat::NowInMicrosec()) {
  }

  ~TimedSection() {
    uint64_t now = ProfileStat::NowInMicrosec();
    LOG(WARNING) << "Timed " << name_ << ": " << (now - start_) << " ms";
  }

 private:
  std::string name_;
  uint64_t start_;
};
#define RAF_TIMED(name) raf::profiler::TimedSection timed_section(name);

}  // namespace profiler
}  // namespace raf
