/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file communicator.h
 * \brief Communication resources.
 */
#pragma once
#include <unistd.h>
#include <stdint.h>
#include <string>
#include <set>
#include <memory>
#include "dmlc/logging.h"
#include "raf/registry.h"
#include "raf/value.h"
#include "raf/op_utils.h"

typedef std::pair<std::string, std::vector<std::vector<int64_t>>> CommunicatorID;

namespace raf {
namespace distributed {
namespace communicator {

using registry::GetPackedFunc;
using namespace raf::value;

#ifdef RAF_USE_MPI
#include <mpi.h>
#define MPI_CALL(cmd)                                                         \
  do {                                                                        \
    int e = cmd;                                                              \
    if (e != MPI_SUCCESS) {                                                   \
      LOG(FATAL) << "Failed: MPI error " << __FILE__ << ":" << __LINE__ << e; \
    }                                                                         \
  } while (0)
#endif

#ifdef RAF_USE_NCCL
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
  int group_id;
  int group_size;
  std::vector<uint64_t> host_ids;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("local_size", &local_size);
    v->Visit("local_rank", &local_rank);
    v->Visit("size", &size);
    v->Visit("rank", &rank);
    v->Visit("world_size", &world_size);
    v->Visit("world_rank", &world_rank);
    v->Visit("root_rank", &root_rank);
    v->Visit("group_id", &group_id);
    v->Visit("group_size", &group_size);
  }

  virtual ~CommunicatorObj() = default;

  static constexpr const char* _type_key = "raf.distributed.Communicator";
  RAF_BASE_OBJECT(CommunicatorObj, Object);
};

class Communicator : public ObjectRef {
 public:
  static Communicator Get(const std::string& name, const Value rank_list = NullValue<Value>());
  static void InitSubCommunicator(CommunicatorObj* sub_comm, const Value rank_list,
                                  const Communicator global_comm);
  static uint64_t GetHostID();

  RAF_MUTABLE_OBJECT_REF(Communicator, ObjectRef, CommunicatorObj);
};

class VoidCommunicatorObj final : public CommunicatorObj {
 public:
  static constexpr const char* _type_key = "raf.distributed.VoidCommunicator";
  RAF_FINAL_OBJECT(VoidCommunicatorObj, CommunicatorObj);
};

class VoidCommunicator final : public Communicator {
 public:
  static VoidCommunicator make(Value rank_list);
  RAF_MUTABLE_OBJECT_REF(VoidCommunicator, Communicator, VoidCommunicatorObj);
};

class CommunicatorPool {
 public:
  CommunicatorPool() {
  }

  static CommunicatorPool* Get() {
    static CommunicatorPool instance;
    return &instance;
  }

  Communicator GetCommunicator(const std::string& name, const Value rank_list) {
    std::vector<std::vector<int64_t>> rank_list_;
    std::set<int64_t> rank_set_;
    if (rank_list.defined()) {
      for (auto group : Downcast<TupleValue>(rank_list)->fields) {
        std::vector<int64_t> group_;
        for (auto rank : Downcast<TupleValue>(group)->fields) {
          auto rank_val = Downcast<IntValue>(rank)->value;
          CHECK(rank_set_.count(rank_val) == 0) << "Each rank can only appear on rank_list once";
          group_.push_back(rank_val);
          rank_set_.insert(rank_val);
        }
        rank_list_.push_back(group_);
      }
    }

    CommunicatorID id(name, rank_list_);

    if (comm_.count(id) == 0) {
      const std::string prefix = "raf.distributed.communicator._make.";
      auto func_name = prefix + name;
      Communicator comm = GetPackedFunc(func_name)(rank_list);
      comm_[id] = std::move(comm);
    }
    return comm_[id];
  }

  void Remove() {
    comm_.clear();
  }

 private:
  std::map<CommunicatorID, Communicator> comm_;
};

Communicator GetGlobalCommunicator();

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
