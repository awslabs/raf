/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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
