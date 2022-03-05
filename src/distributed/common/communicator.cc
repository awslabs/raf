/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/communicator.cc
 * \brief Implementation of Communicator.
 */

#include "raf/communicator.h"

namespace raf {
namespace distributed {
namespace communicator {

Communicator Communicator::Get(const std::string& name, const std::vector<int64_t>& rank_list) {
  return CommunicatorPool::Get()->GetCommunicator(name, rank_list);
}

void Communicator::InitSubCommunicator(CommunicatorObj* sub_comm, const TupleValue rank_list,
                                       const Communicator global_comm) {
  std::vector<int64_t> rank_list_;
  for (auto i : rank_list->fields) {
    auto val = Downcast<value::IntValue>(i);
    rank_list_.push_back(val->value);
  }

  int size = rank_list_.size();
  int rank;
  CHECK_LE(size, global_comm->size);
  for (rank = 0; rank < size; ++rank) {
    if (rank_list_[rank] == global_comm->rank) break;
  }
  if (rank == size) {
    // This rank is not in rank_list
    sub_comm->local_size = 1;
    sub_comm->local_rank = 0;
    sub_comm->size = 1;
    sub_comm->rank = 0;
    sub_comm->world_size = global_comm->size;
    sub_comm->world_rank = global_comm->rank;
    sub_comm->root_rank = global_comm->rank;
    sub_comm->host_ids.push_back(global_comm->host_ids[global_comm->rank]);
  } else {
    // This rank is in rank_list
    int local_size = 0;
    int local_rank = 0;
    for (auto i : rank_list_) {
      sub_comm->host_ids.push_back(global_comm->host_ids[i]);
    }
    for (int p = 0; p < size; ++p) {
      if (p == rank) break;
      if (sub_comm->host_ids[p] == sub_comm->host_ids[rank]) local_rank++;
    }
    for (int p = 0; p < size; ++p) {
      if (sub_comm->host_ids[p] == sub_comm->host_ids[rank]) local_size++;
    }
    sub_comm->local_size = local_size;
    sub_comm->local_rank = local_rank;
    sub_comm->size = size;
    sub_comm->rank = rank;
    sub_comm->world_size = global_comm->size;
    sub_comm->world_rank = global_comm->rank;
    sub_comm->root_rank = rank_list_[0];
  }
}

uint64_t Communicator::GetHostID() {
  // Prevent confusion if all the nodes share the same hostname
  auto hostid = std::to_string(gethostid());

  char hostname[1024];
  gethostname(hostname, 1024);
  size_t hash = std::hash<std::string>{}(std::string(hostname) + hostid);
  return hash;
}

VoidCommunicator VoidCommunicator::make(TupleValue rank_list) {
  auto obj = make_object<VoidCommunicatorObj>();

  if (rank_list->fields.empty()) {
    obj->local_size = 1;
    obj->local_rank = 0;
    obj->size = 1;
    obj->rank = 0;
    obj->world_size = 1;
    obj->world_rank = 0;
    obj->root_rank = 0;
    obj->host_ids.push_back(GetHostID());
  } else {
    InitSubCommunicator(obj.get(), rank_list, Communicator::Get("void"));
  }

  return VoidCommunicator(obj);
}

RAF_REGISTER_GLOBAL("raf.distributed.communicator._make.void")
    .set_body_typed(VoidCommunicator::make);

RAF_REGISTER_GLOBAL("raf.distributed.RemoveCommunicator").set_body_typed([]() {
  CommunicatorPool::Get()->Remove();
});

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
