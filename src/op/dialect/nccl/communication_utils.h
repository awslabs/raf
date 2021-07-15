/*!
 * Copyright (c) 2021 by Contributors
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
#include "mnm/base.h"
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
