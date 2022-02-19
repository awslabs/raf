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
 * \file include/cutlass_ext/library/gemm_operation_ext.h
 * \brief Extentions for gemm operation
 */
#pragma once

#include "cutlass/library/library.h"

#include "gemm_operation.h"
#include "./library_ext.h"
#include "./library_internal_ext.h"

namespace cutlass {
namespace library {

/*! \brief Extention for GemmUniversalOperation, with epilogue operators supported */
template <typename Operator_>
class GemmUniversalOperationExt : public GemmUniversalOperation<Operator_> {
 public:
  using Operator = Operator_;
  using EpilogueOutputOp = typename Operator::EpilogueOutputOp;

 protected:
  /*! \brief Extended operator description, with epilogue operator information */
  GemmDescriptionExt description_ext_;

 public:
  GemmUniversalOperationExt(char const* name = "unknown_gemm")
      : GemmUniversalOperation<Operator_>(name),
        description_ext_(this->description_, EpilogueOpMap<EpilogueOutputOp>::kId) {
  }

  virtual OperationDescription const& description() const {
    return description_ext_;
  }
};

}  // namespace library
}  // namespace cutlass
