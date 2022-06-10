/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/nccl_communicator.cc
 * \brief NCCL Communicator.
 */

#include <numeric>
#include "raf/mpi_communicator.h"
#include "raf/nccl_communicator.h"
#include "./nccl_utils.h"

namespace raf {
namespace distributed {
namespace communicator {

#define NCCL_UNIQUE_ID_BYTES 128

class NCCLIdSyncHelper {
 public:
  NCCLIdSyncHelper() {
    const char* temp = getenv("RAF_FILE_STORE_PATH");
    if (temp == nullptr) {
      LOG(WARNING) << "RAF_FILE_STORE_PATH is no set.";
      base_path_ = "/tmp/.raf_file_store";
      int res = syscall(std::bind(::access, base_path_.c_str(), F_OK));
      // create the directory
      if (res < 0) {
        int rv = syscall(std::bind(mkdir, base_path_.c_str(), S_IRWXU | S_IRWXG | S_IRWXO));
        SYSASSERT(rv, "mkdir");
      }
    } else {
      base_path_ = std::string(temp);
    }
    this->Append();
  }

  void Sync(ncclUniqueId* nccl_id, int rank, std::vector<int>& rank_vec) {
    // rank not in this group, do nothing
    if (std::find(rank_vec.begin(), rank_vec.end(), rank) == rank_vec.end()) {
      return;
    }

    if (rank == rank_vec[0]) {
      // root rank, store NCCL id in the file
      auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(nccl_id),
                                      reinterpret_cast<uint8_t*>(nccl_id) + NCCL_UNIQUE_ID_BYTES);
      fs_vec_.back()->Set(vec);
    } else {
      // other ranks, read NCCL id from the file
      auto vec = fs_vec_.back()->Get();
      CHECK(vec.size() == NCCL_UNIQUE_ID_BYTES);
      std::memcpy(nccl_id, vec.data(), vec.size());
    }
  }

  void Append() {
    count_ += 1;
    fs_vec_.push_back(std::make_unique<SimpleFileStore>(base_path_ + "/" + std::to_string(count_)));
  }

 private:
  std::list<std::unique_ptr<SimpleFileStore>> fs_vec_;
  std::string base_path_;
  int count_{0};
};

NCCLCommunicatorObj::~NCCLCommunicatorObj() {
  NCCL_CALL(ncclCommDestroy(nccl_comm));
}

NCCLCommunicator NCCLCommunicator::make(Value rank_list) {
  auto global_comm = GetGlobalCommunicator();
  auto obj = make_object<NCCLCommunicatorObj>();

  std::unique_ptr<NCCLIdSyncHelper> helper{nullptr};
  if (global_comm->IsInstance<VoidCommunicatorObj>()) {
    helper = std::make_unique<NCCLIdSyncHelper>();
  }

  ncclUniqueId nccl_id;
  NCCL_CALL(ncclGetUniqueId(&nccl_id));

  if (!rank_list.defined()) {
    // Create Global Communicator
    obj->local_size = global_comm->local_size;
    obj->local_rank = global_comm->local_rank;
    obj->size = global_comm->size;
    obj->rank = global_comm->rank;
    obj->world_size = global_comm->world_size;
    obj->world_rank = global_comm->world_rank;
    obj->root_rank = global_comm->root_rank;
    obj->group_id = -1;
    obj->group_size = 0;
    obj->host_ids = global_comm->host_ids;
    obj->parent_comm = global_comm;
    cudaSetDevice(obj->local_rank);
    // sync NCCL id between ranks
    if (global_comm->IsInstance<MPICommunicatorObj>()) {
      MPI_CALL(MPI_Bcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), MPI_BYTE,
                         obj->root_rank, MPI_COMM_WORLD));
    } else {
      CHECK(global_comm->IsInstance<VoidCommunicatorObj>());
      std::vector<int> vec(obj->size);
      std::iota(vec.begin(), vec.end(), 0);
      helper->Sync(&nccl_id, obj->rank, vec);
    }
    NCCL_CALL(ncclCommInitRank(&obj->nccl_comm, obj->size, nccl_id, obj->rank));
  } else {
    // Create Sub-communicator
    InitSubCommunicator(obj.get(), rank_list, global_comm);
    cudaSetDevice(global_comm->local_rank);

    obj->parent_comm = global_comm;

    // sync NCCL id between ranks
    ncclUniqueId& root_nccl_id = nccl_id;
    if (global_comm->IsInstance<MPICommunicatorObj>()) {
      std::vector<ncclUniqueId> nccl_ids(obj->group_size);
      std::vector<int> counts(obj->world_size, 0);
      std::vector<int> displacements(obj->world_size);

      int offset = 0;

      for (auto group : Downcast<TupleValue>(rank_list)->fields) {
        auto root_rank = Downcast<TupleValue>(group)->fields[0];
        auto root_rank_ = Downcast<IntValue>(root_rank)->value;
        counts[root_rank_] = sizeof(nccl_id);
      }

      for (int i = 0; i < obj->world_size; ++i) {
        displacements[i] = offset;
        if (counts[i] > 0) offset += sizeof(nccl_id);
      }

      MPI_CALL(MPI_Allgatherv(reinterpret_cast<void*>(&nccl_id), counts[obj->world_rank], MPI_BYTE,
                              reinterpret_cast<void*>(&nccl_ids[0]),
                              reinterpret_cast<int*>(&counts[0]),
                              reinterpret_cast<int*>(&displacements[0]), MPI_BYTE, MPI_COMM_WORLD));
      if (obj->group_id != -1) {
        root_nccl_id = nccl_ids[obj->group_id];
      }
    } else {
      CHECK(global_comm->IsInstance<VoidCommunicatorObj>());
      for (auto group : Downcast<TupleValue>(rank_list)->fields) {
        std::vector<int> vec;
        for (auto rank : Downcast<TupleValue>(group)->fields) {
          auto rank_val = Downcast<IntValue>(rank)->value;
          vec.push_back(rank_val);
        }
        helper->Append();
        helper->Sync(&nccl_id, global_comm->rank, vec);
      }
    }
    NCCL_CALL(ncclCommInitRank(&obj->nccl_comm, obj->size, root_nccl_id, obj->rank));
  }

  return NCCLCommunicator(obj);
}

RAF_REGISTER_GLOBAL("raf.distributed.communicator._make.nccl")
    .set_body_typed(NCCLCommunicator::make);

RAF_REGISTER_OBJECT_REFLECT(NCCLCommunicatorObj);

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
