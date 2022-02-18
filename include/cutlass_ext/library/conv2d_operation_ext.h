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
 * \file include/cutlass_ext/library/conv2d_operation_ext.h
 * \brief Extentions for conv2d operation
 */
#pragma once

#include "cutlass/library/library.h"

#include "conv2d_operation.h"
#include "./library_internal_ext.h"
#include "./library_ext.h"

namespace cutlass {
namespace library {

/*! \brief Extention for Conv2dOperation, with epilogue operators supported */
template <typename Operator_>
class Conv2dOperationExt : public Conv2dOperation<Operator_> {
 public:
  using Operator = Operator_;
  using EpilogueOutputOp = typename Operator::EpilogueOutputOp;

 protected:
  /*! \brief Extended operator description, with epilogue operator information */
  ConvDescriptionExt description_ext_;

 public:
  Conv2dOperationExt(char const* name = "unknown_gemm")
      : Conv2dOperation<Operator_>(name),
        description_ext_(this->description_, EpilogueOpMap<EpilogueOutputOp>::kId) {
  }

  virtual OperationDescription const& description() const {
    return description_ext_;
  }
};

}  // namespace library
}  // namespace cutlass
