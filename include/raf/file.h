/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file file.h
 * \brief File/Directory manipulation functions.
 */
#pragma once

#include <sys/stat.h>
#include <fstream>
#include <cerrno>
#include <cstring>
#include "dmlc/logging.h"

namespace raf {
inline void CreateDir(const std::string& path) {
  if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno != EEXIST) {
      LOG(FATAL) << "Failed to create directory " << path << ": " << strerror(errno);
      throw;
    }
  }
}

inline bool DirExists(const std::string& path) {
  std::ifstream ifs(path);
  auto ret = ifs.good();
  ifs.close();
  return ret;
}
}  // namespace raf
