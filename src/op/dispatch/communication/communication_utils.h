/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/communication/communication_utils.h
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
