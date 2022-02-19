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
 * \file src/op/declare/random.cc
 * \brief Declaration of random operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/random.h"
namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.threefry_generate", [](const CallValues& call) {
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

MNM_OP_DECLARE("mnm.op.threefry_split", [](const CallValues& call) {
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
}  // namespace mnm
