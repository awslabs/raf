/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file nccl_communicator.h
 * \brief NCCL Communicator.
 */
#pragma once
#include <algorithm>
#include <nccl.h>
#include "raf/communicator.h"
#include "raf/op_utils.h"
#include "raf/value.h"

namespace raf {
namespace distributed {
namespace communicator {

class NCCLCommunicatorObj final : public CommunicatorObj {
 public:
  ncclComm_t nccl_comm;
  Communicator parent_comm;  // Prevent MPI Communicator from releasing in advanced
  static constexpr const char* _type_key = "raf.distributed.NCCLCommunicator";
  ~NCCLCommunicatorObj();
  RAF_FINAL_OBJECT(NCCLCommunicatorObj, CommunicatorObj);
};

class NCCLCommunicator final : public Communicator {
 public:
  static NCCLCommunicator make(Value rank_list);
  RAF_OBJECT_REF(NCCLCommunicator, Communicator, NCCLCommunicatorObj);
};

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
