/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file sharding.h
 * \brief Definition of sharding specifications
 */
#pragma once
#include "./value.h"
#include <sstream>

namespace raf {
namespace sharding {

using namespace raf::ir;
using namespace raf::value;

class BaseShardSpecObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
  }
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.sharding.BaseShardSpec";
  RAF_BASE_OBJECT(BaseShardSpecObj, ValueObj);
};

class BaseShardSpec : public Value {
 public:
  RAF_OBJECT_REF(BaseShardSpec, Value, BaseShardSpecObj);
};

class ShardSpecObj final : public BaseShardSpecObj {
 public:
  /*!
   * \brief When this flag is False, it disallows Sharding Propagation Pass to reshard this tensor.
   * During the propagation, it is likely to reshard an intermediate variable to get more
   * opportunities of finding a new sharding solution. It is recommended to make the specs
   * of inputs and outputs immutable to get an sharding solution with expected input and
   * output shape.
   */
  bool mutable_;

  /*! \brief Number of dimensions. */
  int64_t ndim_;

  /*! \brief Number of shards. */
  int64_t nshard_;

  /*! \brief Number of subgroups. */
  int64_t ngroup_;

  /*!
   * \brief The list of ranks that participate in the computation. When ranks is set to an integer
   * N, it is equivalent to [0...N-1]. By default, it will utilize all available devices specified
   * by the launcher (e.g. mpirun).
   */
  Array<Integer> ranks;

  /*!
   * \brief The shape of the logical subgroup grid. For example, for a 2D tensor, if there are 4
   * devices in total, phy_shape is set to [2, 2], subgroup_shape is set to [1, 2], the subgroup g0,
   * g1, the subgroup grid will be [[d0, d1]], [[d2, d3]], [[g0], [g1]] respectively, and the tensor
   * will be partitioned into [[x0], [x1]], where g0 will hold x0, and g1 will hold x1
   * correspondingly.
   */
  Array<Integer> logic_shape;

  /*! \brief The index of current rank in the logical subgroup grid. */
  Array<Integer> logic_index_;

  /*!
   * \brief The shape of the physical device mesh. For example, for a 2D tensor, if there are 4
   * devices in total and phy_shape is set to [2, 2] (subgrouping is not enabled), the tensor will
   * be partitioned into [[x0, x1], [x2, x3]], where the device 0-4 will hold x0-4 respectively.
   */
  Array<Integer> phy_shape;

  /*! \brief The index of current rank in the physical device mesh. */
  Array<Integer> phy_index_;

  /*!
   * \brief The shape of the subgroup. For example, for a 2D tensor, if there are 4 devices
   * in total, phy_shape is set to [2, 2], subgroup_shape is set to [1, 2], the subgroup g0, g1,
   * the subgroup grid will be [[d0, d1]], [[d2, d3]], [[g0], [g1]] respectively,
   * and the tensor will be partitioned into [[x0], [x1]]. Since the data shard will be
   * replicated within the subgroup, d0, d1 will hold x0 and d2, d3 will hold x1 correspondingly.
   */
  Array<Integer> subgroup_shape;

  /*! \brief The index of current rank in the subgroup. */
  Array<Integer> subgroup_index_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mutable", &mutable_);
    v->Visit("ndim", &ndim_);
    v->Visit("nshard", &nshard_);
    v->Visit("ngroup", &ngroup_);
    v->Visit("ranks", &ranks);
    v->Visit("logic_shape", &logic_shape);
    v->Visit("logic_index", &logic_index_);
    v->Visit("phy_shape", &logic_shape);
    v->Visit("phy_index", &logic_index_);
    v->Visit("subgroup_shape", &subgroup_shape);
    v->Visit("subgroup_index", &subgroup_index_);
  }

  bool SEqualReduce(const ShardSpecObj* other, tvm::SEqualReducer equal) const {
    return equal(ranks, other->ranks) && equal(phy_shape, other->phy_shape) &&
           equal(subgroup_shape, other->subgroup_shape) && equal(mutable_, other->mutable_);
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(ranks);
    hash_reduce(phy_shape);
    hash_reduce(subgroup_shape);
    hash_reduce(mutable_);
  }

  static constexpr const char* _type_key = "raf.sharding.ShardSpec";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(ShardSpecObj, BaseShardSpecObj);
};

class ShardSpec final : public BaseShardSpec {
 public:
  static ShardSpec make(Array<Integer> ranks, Array<Integer> phy_shape,
                        Array<Integer> subgroup_shape, bool mutable_);
  RAF_OBJECT_REF(ShardSpec, BaseShardSpec, ShardSpecObj);
};

class UnsetShardSpecObj final : public BaseShardSpecObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
  }

  bool SEqualReduce(const UnsetShardSpecObj* other, tvm::SEqualReducer equal) const {
    return true;
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "raf.sharding.UnsetShardSpec";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(UnsetShardSpecObj, BaseShardSpecObj);
};

class UnsetShardSpec final : public BaseShardSpec {
 public:
  static UnsetShardSpec make() {
    auto n = make_object<UnsetShardSpecObj>();
    return UnsetShardSpec(n);
  };
  RAF_OBJECT_REF(UnsetShardSpec, BaseShardSpec, BaseShardSpecObj);
};

struct ShardOpCallAttrs : public tvm::AttrsNode<ShardOpCallAttrs> {
  static Attrs make(Array<BaseShardSpec> sin, Array<BaseShardSpec> sout);
  Array<BaseShardSpec> sin, sout;
  TVM_DECLARE_ATTRS(ShardOpCallAttrs, "raf.attrs.ShardOpCallAttrs") {
    TVM_ATTR_FIELD(sin)
        .set_default(NullValue<Array<BaseShardSpec>>())
        .describe("Sharding Specifications of inputs");
    TVM_ATTR_FIELD(sout)
        .set_default(NullValue<Array<BaseShardSpec>>())
        .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace raf
