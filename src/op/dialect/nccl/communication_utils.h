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
 * \file src/op/dialect/cuda/communication_utils.h
 * \brief Helper functions for communicaton operators
 */
#pragma once
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <nccl.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include "mnm/device.h"
#include "mnm/op.h"
#include "mnm/enum_base.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/tensor.h"
#include "mnm/communicator.h"
#include "mnm/stream_pool.h"
#include "../../../common/shape_utils.h"

#define NCCL_CALL(cmd)                                                                             \
  do {                                                                                             \
    ncclResult_t e = cmd;                                                                          \
    if (e != ncclSuccess) {                                                                        \
      LOG(FATAL) << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << ncclGetErrorString(e); \
    }                                                                                              \
  } while (0)

namespace mnm {

template <>
inline DType::operator ncclDataType_t() const {
  switch (code) {
    case kDLInt:
      if (bits == 8) return ncclInt8;
      break;
    case kDLUInt:
      if (bits == 8) return ncclUint8;
      break;
    case kDLFloat:
      if (bits == 16) return ncclFloat16;
      if (bits == 32) return ncclFloat32;
      if (bits == 64) return ncclFloat64;
  }
  LOG(FATAL) << "NotImplementedError: " << c_str();
  throw;
}

}  // namespace mnm
