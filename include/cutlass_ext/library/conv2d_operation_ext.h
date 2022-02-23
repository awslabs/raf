/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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
