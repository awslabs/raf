/*!
 * Copyright (c) 2022 by Contributors
 * \file src/distributed/nccl_communicator.cc
 * \brief NCCL Communicator
 */
#pragma once
#include <algorithm>
#include <nccl.h>
#include "mnm/communicator.h"
#include "mnm/op_utils.h"
#include "mnm/value.h"

namespace mnm {
namespace distributed {
namespace communicator {

class NCCLCommunicatorObj final : public CommunicatorObj {
 public:
  ncclComm_t nccl_comm;
  Communicator parent_comm; // Prevent MPI Communicator from releasing
  static constexpr const char* _type_key = "mnm.distributed.NCCLCommunicator";
  MNM_FINAL_OBJECT(NCCLCommunicatorObj, CommunicatorObj);
};

class NCCLCommunicator : public Communicator {
 public:
  static NCCLCommunicator make(value::TupleValue rank_list);
  virtual ~NCCLCommunicator();
  MNM_OBJECT_REF(NCCLCommunicator, Communicator, NCCLCommunicatorObj);
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
