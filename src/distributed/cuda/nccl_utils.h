/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 * See: https://github.com/pytorch/pytorch/blob/master/torch/csrc/distributed/c10d/FileStore.cpp
 */

/*!
 * \file src/distributed/cuda/nccl_utils.h
 * \brief Simple inter-process file store used to sync NCCL unique id.
 */

#include <sys/types.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <system_error>
#include <thread>

#include "dmlc/logging.h"

namespace raf {
namespace distributed {

static constexpr std::chrono::milliseconds kDefaultTimeout = std::chrono::seconds(3);
static constexpr std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds::zero();

#define SYSASSERT(rv, ...)                                                                \
  if ((rv) < 0) {                                                                         \
    LOG(FATAL) << std::system_error(errno, std::system_category(), ##__VA_ARGS__).what(); \
    throw;                                                                                \
  }

template <typename F>
typename std::result_of<F()>::type syscall(F fn) {
  while (true) {
    auto rv = fn();
    if (rv == -1) {
      if (errno == EINTR) {
        continue;
      }
    }
    return rv;
  }
}

// For a comprehensive overview of file locking methods,
// see: https://gavv.github.io/blog/file-locks/.
// We stick to flock(2) here because we don't care about
// locking byte ranges and don't want locks to be process-wide.

// RAII wrapper around flock(2)
class Lock {
 public:
  explicit Lock(int fd, int operation) : fd_(fd) {
    flock(operation);
  }

  ~Lock() {
    unlock();
  }

  Lock(const Lock& that) = delete;

  Lock& operator=(Lock&& other) noexcept {
    if (this != &other) {
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  Lock(Lock&& other) noexcept {
    *this = std::move(other);
  }

  void unlock() {
    if (fd_ >= 0) {
      flock(LOCK_UN);
      fd_ = -1;
    }
  }

 protected:
  int fd_{-1};

  void flock(int operation) {
#ifdef _WIN32
    auto rv = syscall(std::bind(::flock_, fd_, operation));
#else
    auto rv = syscall(std::bind(::flock, fd_, operation));
#endif
    SYSASSERT(rv, "flock");
  }
};

class File {
 public:
  explicit File(const std::string& path, int flags, std::chrono::milliseconds timeout) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      fd_ = syscall(std::bind(::open, path.c_str(), flags, 0644));
      // Only retry when the file doesn't exist, since we are waiting for the
      // file to be created in this case to address the following issue:
      // https://github.com/pytorch/pytorch/issues/13750
      if (fd_ >= 0 || errno != ENOENT) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (timeout != kNoTimeout && elapsed > timeout) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    SYSASSERT(fd_, "open(" + path + ")");
  }

  ~File() {
    ::close(fd_);
  }

  Lock lockShared() {
    return Lock(fd_, LOCK_SH);
  }

  Lock lockExclusive() {
    return Lock(fd_, LOCK_EX);
  }

  off_t seek(off_t offset, int whence) {
    auto rv = syscall(std::bind(lseek, fd_, offset, whence));
    SYSASSERT(rv, "lseek");
    return rv;
  }

  off_t tell() {
    auto rv = syscall(std::bind(lseek, fd_, 0, SEEK_CUR));
    SYSASSERT(rv, "lseek");
    return rv;
  }

  off_t size() {
    auto pos = tell();
    auto size = seek(0, SEEK_END);
    seek(pos, SEEK_SET);
    return size;
  }

  void write(const void* buf, size_t count) {
    while (count > 0) {
      auto rv = syscall(std::bind(::write, fd_, buf, count));
      SYSASSERT(rv, "write");
      buf = (uint8_t*)buf + rv;
      count -= rv;
    }
  }

  void read(void* buf, size_t count) {
    while (count > 0) {
      auto rv = syscall(std::bind(::read, fd_, buf, count));
      SYSASSERT(rv, "read");
      buf = (uint8_t*)buf + rv;
      count -= rv;
    }
  }

  void write(const std::string& str) {
    uint32_t len = str.size();
    CHECK_LE(len, std::numeric_limits<decltype(len)>::max());
    write(&len, sizeof(len));
    write(str.c_str(), len);
  }

  void write(const std::vector<uint8_t>& data) {
    uint32_t len = data.size();
    CHECK_LE(len, std::numeric_limits<decltype(len)>::max());
    write(&len, sizeof(len));
    write(data.data(), len);
  }

  void read(std::string& str) {
    uint32_t len;
    read(&len, sizeof(len));
    std::vector<uint8_t> buf(len);
    read(buf.data(), len);
    str.assign(buf.begin(), buf.end());
  }

  void read(std::vector<uint8_t>& data) {
    uint32_t len;
    read(&len, sizeof(len));
    data.resize(len);
    read(data.data(), len);
  }

 protected:
  int fd_;
};

class SimpleFileStore {
 public:
  explicit SimpleFileStore(const std::string& path) : path_(path) {
  }

  ~SimpleFileStore() {
    // If the file does not exist - exit.
    int res = syscall([filepath = path_.c_str()] { return ::access(filepath, F_OK); });
    if (res == -1) {
      return;
    }
    // Best effort removal without checking the return
    std::remove(path_.c_str());
  }

  void Set(const std::vector<uint8_t>& value) {
    std::unique_lock<std::mutex> l(mutex_);
    File file(path_, O_RDWR | O_CREAT | O_TRUNC, timeout_);
    auto lock = file.lockExclusive();
    auto size = file.size();
    CHECK_EQ(size, 0LL);
    file.write(value);
  }

  std::vector<uint8_t> Get() {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      std::unique_lock<std::mutex> l(mutex_);
      File file(path_, O_RDONLY, timeout_);
      auto lock = file.lockShared();
      auto size = file.size();
      if (size == 0) {
        // No data; release the shared lock and sleep for a bit
        lock.unlock();
        l.unlock();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start);
        if (timeout_ != kNoTimeout && elapsed > timeout_) {
          LOG(FATAL) << "Timeout after " << timeout_.count() << " ms";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      std::vector<uint8_t> value;
      file.read(value);
      return value;
    }
  }

 private:
  std::string path_;
  std::mutex mutex_;
  std::chrono::milliseconds timeout_ = kDefaultTimeout;
};

}  // namespace distributed
}  // namespace raf
