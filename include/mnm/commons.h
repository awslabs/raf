#pragma once

#include <mutex>

#define LOCKED_IF(cond, mutex_, stmt)         \
  if (cond) {                                 \
    std::lock_guard<std::mutex> lock(mutex_); \
    if (cond) {                               \
      stmt;                                   \
    }                                         \
  }
