/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/sharding.cc
 * \brief Description of RAF sharding specifications
 */
#include <tvm/runtime/data_type.h>
#include "raf/ir_ext.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/type.h"
#include "raf/registry.h"
#include "raf/sharding.h"
#include "raf/communicator.h"
#include "../op/ty/utils.h"
#include "../op/schema/ufunc.h"
#include "../op/dialect/tvm/tvm_utils.h"
#include "../op/dialect/tvm/tvm_attrs.h"
#include <string>

namespace raf {
namespace sharding {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::op::schema;
using namespace raf::value;
using namespace raf::distributed;
using namespace raf::distributed::communicator;

static inline int64_t GetRankIdx(Array<Integer> ranks) {
  // returns: the index of the current rank in the given rank array
  for (int64_t i = 0; i < ranks.size(); ++i) {
    if (GetGlobalCommunicator()->rank == ranks[i]->value) {
      return i;
    }
  }
  return -1;
}

ShardSpec ShardSpec::make(Array<Integer> ranks, Array<Integer> phy_shape,
                          Array<Integer> subgroup_shape, bool mutable_) {
  CHECK_EQ(phy_shape.size(), subgroup_shape.size());
  auto ndim = phy_shape.size();
  auto subgroup_index = std::vector<Integer>(ndim);
  auto phy_index = std::vector<Integer>(ndim);
  auto logic_index = std::vector<Integer>(ndim);
  auto logic_shape = std::vector<Integer>(ndim);
  auto rank_idx = GetRankIdx(ranks);
  int64_t nshard = 1, ngroup = 1;

  auto t1 = rank_idx;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    phy_index[i] = t1 % phy_shape[i]->value;
    t1 /= phy_shape[i]->value;

    logic_shape[i] = phy_shape[i]->value / subgroup_shape[i]->value;
    logic_index[i] = phy_index[i]->value / subgroup_shape[i]->value;
    nshard *= logic_shape[i]->value;

    subgroup_index[i] = phy_index[i]->value % subgroup_shape[i]->value;
    ngroup *= subgroup_shape[i]->value;
  }

  auto spec = make_object<ShardSpecObj>();
  spec->mutable_ = mutable_;
  spec->ndim_ = ndim;
  spec->nshard_ = nshard;
  spec->ngroup_ = ngroup;
  spec->ranks = std::move(ranks);
  spec->subgroup_shape = std::move(subgroup_shape);
  spec->phy_shape = std::move(phy_shape);
  spec->logic_shape = Array<Integer>(logic_shape);
  if (rank_idx == -1) {
    spec->subgroup_index_ = NullValue<Array<Integer>>();
    spec->phy_index_ = NullValue<Array<Integer>>();
    spec->logic_index_ = NullValue<Array<Integer>>();
  } else {
    spec->subgroup_index_ = Array<Integer>(subgroup_index);
    spec->phy_index_ = Array<Integer>(phy_index);
    spec->logic_index_ = Array<Integer>(logic_index);
  }

  return ShardSpec(spec);
}

Attrs ShardOpCallAttrs::make(Array<BaseShardSpec> sin, Array<BaseShardSpec> sout) {
  auto attrs = make_object<ShardOpCallAttrs>();
  attrs->sin = std::move(sin);
  attrs->sout = std::move(sout);
  return Attrs(attrs);
}

void Reshard(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op._reshard", Reshard);

Type Reshard_Infer(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  auto data = Downcast<TensorType>(GetType(args->x));
  return TensorType(data->shape, data->dtype);
}

RAF_OP_TYPE("raf.op._reshard", "Reshard", Reshard_Infer);

RAF_REGISTER_GLOBAL("raf.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.UnsetShardSpec").set_body_typed(UnsetShardSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.ShardOpCallAttrs").set_body_typed(ShardOpCallAttrs::make);

RAF_REGISTER_OBJECT_NO_REFLECT(BaseShardSpecObj);
RAF_REGISTER_OBJECT_REFLECT(ShardSpecObj);
RAF_REGISTER_OBJECT_REFLECT(UnsetShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;

std::string PrintAllocTable(const ObjectRef& ref) {
  size_t dev_idx = 0;
  const auto spec = Downcast<ShardSpec>(ref);
  const auto ndim = spec->ndim_;

  std::stringstream ss;

  auto subgroup_index = std::vector<Integer>(ndim);
  auto phy_index = std::vector<Integer>(ndim);
  auto logic_index = std::vector<Integer>(ndim);

  ss << "| Rank | Physical Index | Logic Index | Subgroup Index |" << std::endl;

  for (int64_t rank_idx = 0; rank_idx < spec->ranks.size(); ++rank_idx) {
    auto t1 = rank_idx;
    for (int64_t i = ndim - 1; i >= 0; --i) {
      phy_index[i] = t1 % spec->phy_shape[i]->value;
      t1 /= spec->phy_shape[i]->value;
      logic_index[i] = phy_index[i]->value / spec->subgroup_shape[i]->value;
      subgroup_index[i] = phy_index[i]->value % spec->subgroup_shape[i]->value;
    }
    ss << "| " << spec->ranks[rank_idx]->value << " | ";
    for (auto arr : {phy_index, logic_index, subgroup_index}) {
      ss << "(";
      for (auto e : arr) {
        ss << e << ", ";
      }
      ss.seekp(-2, std::ios_base::end);
      ss << ") | ";
    }
    ss << std::endl;
  }

  return ss.str();
}

RAF_REGISTER_GLOBAL("raf.sharding.PrintAllocTable").set_body_typed(PrintAllocTable);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ShardSpec>(ref);
      auto ndim = r->ndim_;
      if (r->nshard_ == 1) {
        p->stream << "ShardSpec(Replicated)";
      } else {
        p->stream << "ShardSpec("
                  << "[";
        for (size_t i = 0; i < ndim; ++i) {
          auto nshard_on_dim = r->logic_shape[i]->value;
          auto ngroup_on_dim = r->subgroup_shape[i]->value;
          p->stream << (nshard_on_dim == 1 ? ":" : std::to_string(nshard_on_dim))
                    << (ngroup_on_dim == 1 ? "" : "(x" + std::to_string(ngroup_on_dim) + ")")
                    << (i != ndim - 1 ? ", " : "");
        }
        p->stream << "])";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UnsetShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<UnsetShardSpec>(ref);
      p->stream << "UnsetShardSpec()";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardOpCallAttrs>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* n = static_cast<const ShardOpCallAttrs*>(ref.get());
      p->stream << "ShardOpCallAttrs("
                << "in=" << n->sin << ", out=" << n->sout << ")";
    });

TVM_REGISTER_NODE_TYPE(ShardOpCallAttrs);

}  // namespace sharding
}  // namespace raf
