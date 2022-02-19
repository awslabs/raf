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
  Communicator parent_comm;  // Prevent MPI Communicator from releasing in advanced
  static constexpr const char* _type_key = "mnm.distributed.NCCLCommunicator";
  virtual ~NCCLCommunicatorObj();
  MNM_FINAL_OBJECT(NCCLCommunicatorObj, CommunicatorObj);
};

class NCCLCommunicator final : public Communicator {
 public:
  static NCCLCommunicator make(value::TupleValue rank_list);
  MNM_OBJECT_REF(NCCLCommunicator, Communicator, NCCLCommunicatorObj);
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
