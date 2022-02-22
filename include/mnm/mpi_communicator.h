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
 * \file mpi_communicator.h
 * \brief MPI Communicator.
 */
#pragma once
#include <mpi.h>
#include "mnm/communicator.h"
#include <string>
#include <functional>

namespace mnm {
namespace distributed {
namespace communicator {

class MPICommunicatorObj final : public CommunicatorObj {
 public:
  const MPI_Comm mpi_comm = MPI_COMM_WORLD;
  static constexpr const char* _type_key = "mnm.distributed.MPICommunicator";
  virtual ~MPICommunicatorObj();
  MNM_FINAL_OBJECT(MPICommunicatorObj, CommunicatorObj);
};

class MPICommunicator final : public Communicator {
 public:
  static MPICommunicator make(TupleValue rank_list);
  MNM_OBJECT_REF(MPICommunicator, Communicator, MPICommunicatorObj);
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
