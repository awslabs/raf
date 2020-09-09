/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/vm/storage.h
 * \brief The storage object of VM.
 */
#pragma once
#include "mnm/ir.h"
#include "mnm/memory_pool.h"

namespace mnm {
namespace executor {
namespace vm {
using namespace mnm::ir;

/*! \brief An object representing a storage allocation. */
class StorageObj : public Object {
 public:
  /*! \brief The index into the VM function table. */
  std::shared_ptr<memory_pool::Memory> buffer;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "vm.Storage";
  TVM_DECLARE_FINAL_OBJECT_INFO(StorageObj, Object);
};

/*! \brief reference to storage. */
class Storage : public ObjectRef {
 public:
  explicit Storage(std::shared_ptr<memory_pool::Memory> buffer);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Storage, ObjectRef, StorageObj);
};

}  // namespace vm
}  // namespace executor
}  // namespace  mnm
