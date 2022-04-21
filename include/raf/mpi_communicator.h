/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file mpi_communicator.h
 * \brief MPI Communicator.
 */
#pragma once
#include <mpi.h>
#include "raf/communicator.h"
#include <string>
#include <functional>

namespace raf {
namespace distributed {
namespace communicator {

class MPICommunicatorObj final : public CommunicatorObj {
 public:
  static constexpr const char* _type_key = "raf.distributed.MPICommunicator";
  ~MPICommunicatorObj();
  RAF_FINAL_OBJECT(MPICommunicatorObj, CommunicatorObj);
};

class MPICommunicator final : public Communicator {
 public:
  static MPICommunicator make(Value rank_list);
  RAF_OBJECT_REF(MPICommunicator, Communicator, MPICommunicatorObj);
};

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
