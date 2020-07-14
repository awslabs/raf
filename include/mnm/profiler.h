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

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <unistd.h>
#endif

#define WITH_BASE_PROFILER(CTX, NAME, CAT, ARGS, CODE_SNIPPET)                                \
  {                                                                                              \
    bool profiling = profiler::Profiler::Get()->IsProfiling();                                   \
    if (profiling) {                                                                             \
      profiler::ProfilerBaseHelper phelper(profiling, CTX.device_id, CTX.device_type, NAME, CAT, \
                                           ARGS);                                                \
      phelper.start();                                                                           \
      CODE_SNIPPET                                                                               \
      phelper.stop();                                                                            \
    } else {                                                                                     \
      CODE_SNIPPET                                                                               \
    }                                                                                            \
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
  inline bool IsProfiling() {
    return profiling_;
  }

 private:
  DeviceStats profile_stats_;
  bool profiling_;
  std::recursive_mutex m_;
  uint64_t init_time_;
};

class ProfilerBaseHelper {
 public:
  ProfilerBaseHelper(bool profiling, uint32_t dev_id, mnm::DevType dev_type, std::string name,
                     std::string categories, std::vector<std::string> args = {})
      : profiling_(profiling),
        dev_id_(dev_id),
        dev_type_(dev_type),
        name_(name),
        categories_(categories),
        args(args) {
  }
  void start();
  void stop();

 protected:
  const bool profiling_;
  uint32_t dev_id_;
  mnm::DevType dev_type_;
  std::string name_;
  std::string categories_;
  uint64_t start_time_;
  uint64_t end_time_;
  std::vector<std::string> args;
};

inline void ProfilerBaseHelper::start() {
  if (profiling_) {
    start_time_ = ProfileStat::NowInMicrosec();
  }
}

inline void ProfilerBaseHelper::stop() {
  if (profiling_) {
    end_time_ = ProfileStat::NowInMicrosec();
    Profiler::Get()->AddNewProfileStat(categories_, name_, start_time_, end_time_, args);
  }
}

}  // namespace profiler
}  // namespace mnm
