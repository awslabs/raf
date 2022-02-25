/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/random.cc
 * \brief Declaration of random operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/random.h"
namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

RAF_OP_DECLARE("raf.op.threefry_generate", [](const CallValues& call) {
  const auto* args = call->args.as<ThreefryGenerateArgs>();
  CHECK(args != nullptr);
  DLTensor* key = args->key;
  CHECK_EQ(tvm::runtime::DLDataType2String(key->dtype), "uint64")
      << "The type of key must be uint64";
  std::vector<int64_t> kshape(key->shape, key->shape + key->ndim);
  std::vector<int64_t> oshape(args->shape.begin(), args->shape.end());

  TensorValue new_key = TensorValue::Assemble(/*dev=*/key->device,
                                              /*dtype=*/DType(DTypeCode::kUInt(), 64),
                                              /*shape=*/kshape);
  TensorValue random_array = TensorValue::Assemble(/*dev=*/key->device,
                                                   /*dtype=*/DType(DTypeCode::kUInt(), 64),
                                                   /*shape=*/oshape);
  call->out = TupleValue::make({new_key, random_array});
  call->device = key->device;
});

RAF_OP_DECLARE("raf.op.threefry_split", [](const CallValues& call) {
  const auto* args = call->args.as<ThreefrySplitArgs>();
  CHECK(args != nullptr);
  DLTensor* key = args->key;
  CHECK_EQ(tvm::runtime::DLDataType2String(key->dtype), "uint64")
      << "The type of key must be uint64";
  std::vector<int64_t> kshape(key->shape, key->shape + key->ndim);

  TensorValue new_key = TensorValue::Assemble(/*dev=*/key->device,
                                              /*dtype=*/key->dtype,
                                              /*shape=*/kshape);
  TensorValue new_subkey = TensorValue::Assemble(/*dev=*/key->device,
                                                 /*dtype=*/key->dtype,
                                                 /*shape=*/kshape);
  call->out = TupleValue::make({new_key, new_subkey});
  call->device = key->device;
});

}  // namespace declare
}  // namespace op
}  // namespace raf
