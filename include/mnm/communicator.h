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
 * \file communicator.h
 * \brief Communication resources.
 */
#pragma once
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <memory>
#include "dmlc/logging.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "mnm/op_utils.h"

typedef std::pair<std::string, std::vector<int64_t>> CommunicatorID;

namespace mnm {
namespace distributed {
namespace communicator {

using registry::GetPackedFunc;
using namespace mnm::value;

// #ifdef MNM_USE_MPI
#include <mpi.h>
#define MPI_CALL(cmd)                                                         \
  do {                                                                        \
    int e = cmd;                                                              \
    if (e != MPI_SUCCESS) {                                                   \
      LOG(FATAL) << "Failed: MPI error " << __FILE__ << ":" << __LINE__ << e; \
    }                                                                         \
  } while (0)
// #endif

#ifdef MNM_USE_NCCL
#define NCCL_CALL(cmd)                                                                            \
  do {                                                                                            \
    ncclResult_t e = cmd;                                                                         \
    if (e != ncclSuccess) {                                                                       \
      LOG(INFO) << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << ncclGetErrorString(e); \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)
#endif

class CommunicatorObj : public Object {
 public:
  int local_size;
  int local_rank;
  int size;
  int rank;
  int world_size;
  int world_rank;
  int root_rank;
  std::vector<uint64_t> host_ids;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("local_size", &local_size);
    v->Visit("local_rank", &local_rank);
    v->Visit("size", &size);
    v->Visit("rank", &rank);
    v->Visit("world_size", &world_size);
    v->Visit("world_rank", &world_rank);
    v->Visit("root_rank", &root_rank);
  }

  virtual ~CommunicatorObj() = default;

  static constexpr const char* _type_key = "mnm.distributed.Communicator";
  MNM_BASE_OBJECT(CommunicatorObj, Object);
};

class Communicator : public ObjectRef {
 public:
  static Communicator Get(const std::string& name = "", const std::vector<int64_t>& rank_list = {});
  static void InitSubCommunicator(Communicator sub_comm, const TupleValue rank_list,
                                  const Communicator global_comm);
  static uint64_t GetHostID();

  MNM_OBJECT_REF(Communicator, ObjectRef, CommunicatorObj);
};

class VoidCommunicatorObj final : public CommunicatorObj {
 public:
  static constexpr const char* _type_key = "mnm.distributed.VoidCommunicator";
  virtual ~VoidCommunicatorObj() = default;
  MNM_FINAL_OBJECT(VoidCommunicatorObj, CommunicatorObj);
};

class VoidCommunicator final : public Communicator {
 public:
  static VoidCommunicator make(TupleValue rank_list);
  MNM_OBJECT_REF(VoidCommunicator, Communicator, VoidCommunicatorObj);
};

class CommunicatorPool {
 public:
  CommunicatorPool() {
  }

  static CommunicatorPool* Get() {
    static CommunicatorPool* instance = new CommunicatorPool();
    return instance;
  }

  Communicator GetCommunicator(const std::string& name = "",
                               const std::vector<int64_t>& rank_list = {}) {
#ifdef MNM_USE_NCCL
    auto default_name = "nccl";
#else
    auto default_name = "void";
#endif
    auto comm_name = name.empty() ? default_name : name;
    auto id = CommunicatorID(comm_name, rank_list);

    if (comm_.count(id) == 0) {
        const std::string prefix = "mnm.distributed.communicator._make.";
        auto func_name = prefix + comm_name;
        Communicator comm = GetPackedFunc(func_name)(op::ArrayToIntTuple(rank_list));
        comm_[id] = std::move(comm);
    }
    return comm_[id];
  }

  void Remove() {
    comm_.clear();
  }

 public:
  std::map<CommunicatorID, Communicator> comm_;
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
