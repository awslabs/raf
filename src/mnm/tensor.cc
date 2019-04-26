#include <mnm/memory_pool.h>
#include <mnm/tensor.h>
#include <tvm/runtime/ndarray.h>

namespace mnm {
namespace tensor {

using TSelf = mnm::tensor::Tensor;
using TSuper = tvm::runtime::NDArray;
using mnm::memory_pool::MemoryPoolManager;
using mnm::types::Context;
using mnm::types::DType;
using mnm::types::DTypeCode;
using mnm::types::index_t;

static std::shared_ptr<MemoryPoolManager> mem_mgr = MemoryPoolManager::Global();

// TODO(@junrushao1994): put this in better place
constexpr int kAllocAlignment = 64;

inline bool IsBoolean(DType dtype) {
  return dtype.code == DTypeCode::kUInt() && dtype.bits == 1;
}

inline index_t Numel(const std::vector<int64_t>& shape) {
  index_t result(1);
  for (int64_t x : shape) {
    result *= index_t(x);
  }
  return result;
}

inline void VerifyDataType(DType dtype) {
  // dtype.lanes >= 1
  CHECK_GE(dtype.lanes, 1);
  // special case: bool
  if (IsBoolean(dtype)) {
    return;
  }
  // dtype.bits % 8 == 0
  CHECK_EQ(dtype.bits % 8, 0);
  // dtype.bits == 2 ^ k
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline index_t GetDataAlignment(const DLTensor& arr) {
  if (IsBoolean(arr.dtype)) {
    return index_t(kAllocAlignment);
  }
  return index_t(std::max(                     //
      (arr.dtype.bits / 8) * arr.dtype.lanes,  //  dtype.bits must be divisible by 8
      kAllocAlignment));
}

class TSelf::Container::Impl {
 public:
  static void DefaultDeleter(TSuper::Container* super_ptr) {
    // This is used for NDArrays managed by TVM or MNM
    TSelf::Container* ptr = static_cast<TSelf::Container*>(super_ptr);
    if (ptr->manager_ctx != nullptr) {
      static_cast<TSuper::Container*>(ptr->manager_ctx)->DecRef();
    } else if (ptr->dl_tensor.data != nullptr) {
      mem_mgr->Dealloc(ptr->dl_tensor.ctx, ptr->memory_chunk_);
    }
    delete ptr;
  }

  static TSelf CreateMeta(std::vector<int64_t> shape, DType dtype, Context ctx) {
    VerifyDataType(dtype);
    // critical zone begins, couldn't fail
    TSelf::Container* data = new TSelf::Container();
    data->deleter = TSelf::Container::Impl::DefaultDeleter;
    TSelf ret(data);
    // critical zone ends, RAII now in effect
    data->shape_ = std::vector<int64_t>(shape);
    data->dl_tensor.data = nullptr;
    data->dl_tensor.ctx = ctx;
    data->dl_tensor.ndim = shape.size();
    data->dl_tensor.dtype = dtype;
    data->dl_tensor.shape = dmlc::BeginPtr(data->shape_);
    data->dl_tensor.strides = nullptr;
    data->dl_tensor.byte_offset = 0;
    return ret;
  }

  static void DLPackDeleter(TSuper::Container* ptr) {
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }

  static void NDArrayDecRefDeleter(DLManagedTensor* tensor) {
    static_cast<TSelf::Container*>(tensor->manager_ctx)->DecRef();
    delete tensor;
  }

  static TSelf Empty(std::vector<int64_t> shape, DType dtype, Context ctx) {
    TSelf ret(CreateMeta(shape, dtype, ctx));
    TSelf::Container* data = static_cast<TSelf::Container*>(ret.data_);
    // TODO(@junrushao1994)
    index_t align = GetDataAlignment(data->dl_tensor);
    index_t nbytes = Numel(shape) * align;
    mem_mgr->Alloc(ctx, nbytes.As<size_t>(), align.As<size_t>(), dtype);
    return ret;
  }
  static TSelf FromFromDLManagedTensorPtr(DLManagedTensor* tensor) {
    // critical zone begins, couldn't fail
    TSelf::Container* data = new TSelf::Container();
    data->deleter = TSelf::Container::Impl::DLPackDeleter;
    TSelf ret(data);
    // critical zone ends, RAII now in effect
    data->dl_tensor = tensor->dl_tensor;
    data->manager_ctx = tensor;
    if (data->dl_tensor.shape != nullptr) {
      data->shape_ = std::vector<int64_t>(data->dl_tensor.shape,  //
                                          data->dl_tensor.shape + data->dl_tensor.ndim);
    } else {
      CHECK_EQ(data->dl_tensor.ndim, 0);
    }
    if (data->dl_tensor.strides != nullptr) {
      data->strides_ = std::vector<int64_t>(data->dl_tensor.strides,
                                            data->dl_tensor.strides + data->dl_tensor.ndim);
    }
    return ret;
  }
};

TSelf TSelf::Empty(std::vector<int64_t> shape, DType dtype, Context ctx) {
  return TSelf::Container::Impl::Empty(shape, dtype, ctx);
}

TSelf TSelf::MoveFromDLTensor(DLTensor&& other) {
  DLManagedTensor tensor;
  tensor.dl_tensor = other;
  tensor.manager_ctx = nullptr;
  tensor.deleter = nullptr;
  return TSelf::MoveFromDLManagedTensor(std::move(tensor));
}

TSelf TSelf::CreateFromDLTensor(const DLTensor& other) {
  DLManagedTensor tensor;
  tensor.dl_tensor = other;
  tensor.manager_ctx = nullptr;
  tensor.deleter = nullptr;
  return TSelf::CreateFromDLManagedTensor(std::move(tensor));
}

TSelf TSelf::CopyFromDLTensor(const DLTensor& other) {
  DLManagedTensor tensor;
  tensor.dl_tensor = other;
  tensor.manager_ctx = nullptr;
  tensor.deleter = nullptr;
  // TODO(@were)
  // return TSelf::CopyFromDLManagedTensor(std::move(tensor));
  return TSelf();
}

DLTensor TSelf::CreateToDLTensor() const {
  TSelf::Container* self = static_cast<TSelf::Container*>(this->data_);
  return self->dl_tensor;
}

TSelf TSelf::MoveFromDLManagedTensor(DLManagedTensor&& other) {
  return TSelf::Container::Impl::FromFromDLManagedTensorPtr(new DLManagedTensor(std::move(other)));
}

TSelf TSelf::CreateFromDLManagedTensor(const DLManagedTensor& other) {
  return TSelf::Container::Impl::FromFromDLManagedTensorPtr(new DLManagedTensor(other));
}

DLManagedTensor TSelf::MoveToDLManagedTensor() {
  DLManagedTensor ret;
  ret.dl_tensor = data_->dl_tensor;
  ret.manager_ctx = data_;
  ret.deleter = TSelf::Container::Impl::NDArrayDecRefDeleter;
  this->data_ = nullptr;
  return ret;
}

DLManagedTensor TSelf::CreateToDLManagedTensor() const {
  DLManagedTensor ret;
  ret.dl_tensor = data_->dl_tensor;
  ret.manager_ctx = data_;
  ret.deleter = TSelf::Container::Impl::NDArrayDecRefDeleter;
  this->data_->IncRef();
  return ret;
}

}  // namespace tensor
}  // namespace mnm
