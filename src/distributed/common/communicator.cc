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

Communicator Communicator::Get(const std::string& name, const Value rank_list) {
  return CommunicatorPool::Get()->GetCommunicator(name, rank_list);
}

void Communicator::InitSubCommunicator(CommunicatorObj* sub_comm, const Value rank_list,
                                       const Communicator global_comm) {
  std::vector<std::vector<int64_t>> rank_list_;
  if (rank_list.defined()) {
    for (auto group : Downcast<TupleValue>(rank_list)->fields) {
      std::vector<int64_t> group_;
      for (auto rank : Downcast<TupleValue>(group)->fields) {
        group_.push_back(Downcast<IntValue>(rank)->value);
      }
      rank_list_.push_back(group_);
    }
  }

  int group_size = rank_list_.size();
  int group_id;
  int rank, size;

  for (group_id = 0; group_id < group_size; ++group_id) {
    auto& group = rank_list_[group_id];
    size = group.size();
    for (rank = 0; rank < size; ++rank) {
      if (group[rank] == global_comm->rank) break;
    }
    if (rank != size) break;
  }

  if (rank == size && group_id == group_size) {
    // This rank is not in rank_list
    sub_comm->local_size = 1;
    sub_comm->local_rank = 0;
    sub_comm->size = 1;
    sub_comm->rank = 0;
    sub_comm->world_size = global_comm->size;
    sub_comm->world_rank = global_comm->rank;
    sub_comm->root_rank = global_comm->rank;
    sub_comm->group_id = -1;
    sub_comm->group_size = group_size;
    sub_comm->host_ids.push_back(global_comm->host_ids.at(global_comm->rank));
  } else {
    // This rank is in rank_list
    auto& group = rank_list_[group_id];
    int local_size = 0;
    int local_rank = 0;
    for (auto i : group) {
      sub_comm->host_ids.push_back(global_comm->host_ids.at(i));
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
    sub_comm->root_rank = group[0];
    sub_comm->group_id = group_id;
    sub_comm->group_size = group_size;
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

VoidCommunicator VoidCommunicator::make(Value rank_list) {
  auto obj = make_object<VoidCommunicatorObj>();

  if (!rank_list.defined()) {
    obj->local_size = 1;
    obj->local_rank = 0;
    obj->size = 1;
    obj->rank = 0;
    obj->world_size = 1;
    obj->world_rank = 0;
    obj->root_rank = 0;
    obj->group_id = -1;
    obj->group_size = 0;
    obj->host_ids.push_back(GetHostID());
  } else {
    InitSubCommunicator(obj.get(), rank_list, GetGlobalCommunicator());
  }

  return VoidCommunicator(obj);
}

class GlobalCommunicatorEntry {
 public:
  GlobalCommunicatorEntry() = default;

  static GlobalCommunicatorEntry* Get() {
    static GlobalCommunicatorEntry entry;
    return &entry;
  }
  Communicator comm;
};

Communicator GetGlobalCommunicator() {
  auto entry = GlobalCommunicatorEntry::Get();
  if (!entry->comm.defined()) {
#ifdef RAF_USE_MPI
    Communicator comm = Communicator::Get("mpi");
#else
    Communicator comm = Communicator::Get("void");
#endif
    entry->comm = comm;
  }
  return entry->comm;
}

void SetDefaultCommunicator(std::string name) {
  auto entry = GlobalCommunicatorEntry::Get();
  entry->comm = Communicator::Get(name);
}

void SetGlobalRank(int rank) {
  CHECK(GetGlobalCommunicator()->IsInstance<communicator::VoidCommunicatorObj>())
      << "Only VoidCommunicator is mutable";
  auto comm = GetGlobalCommunicator();
  comm->rank = rank;
  comm->world_rank = rank;
}

void SetGlobalSize(int size) {
  CHECK(GetGlobalCommunicator()->IsInstance<communicator::VoidCommunicatorObj>())
      << "Only VoidCommunicator is mutable";
  auto comm = GetGlobalCommunicator();
  comm->size = size;
  comm->world_size = size;
  // TODO: make `host_ids` configurable.
  comm->host_ids = std::vector<uint64_t>(size, comm->host_ids[0]);
}

void SetGlobalLocalRank(int local_rank) {
  CHECK(GetGlobalCommunicator()->IsInstance<communicator::VoidCommunicatorObj>())
      << "Only VoidCommunicator is mutable";
  GetGlobalCommunicator()->local_rank = local_rank;
}

void SetGlobalLocalSize(int local_size) {
  CHECK(GetGlobalCommunicator()->IsInstance<communicator::VoidCommunicatorObj>())
      << "Only VoidCommunicator is mutable";
  GetGlobalCommunicator()->local_size = local_size;
}

RAF_REGISTER_GLOBAL("raf.distributed.communicator._make.void")
    .set_body_typed(VoidCommunicator::make);

RAF_REGISTER_GLOBAL("raf.distributed.RemoveCommunicator").set_body_typed([]() {
  CommunicatorPool::Get()->Remove();
});

RAF_REGISTER_GLOBAL("raf.distributed.GetGlobalCommunicator").set_body_typed(GetGlobalCommunicator);
RAF_REGISTER_GLOBAL("raf.distributed.SetDefaultCommunicator")
    .set_body_typed(SetDefaultCommunicator);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalRank").set_body_typed(SetGlobalRank);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalSize").set_body_typed(SetGlobalSize);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalLocalRank").set_body_typed(SetGlobalLocalRank);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalLocalSize").set_body_typed(SetGlobalLocalSize);

RAF_REGISTER_OBJECT_REFLECT(VoidCommunicatorObj);

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
