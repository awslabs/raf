/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/cutlass_ext/library/singleton_ext.h
 * \brief Extentions for cutlass singleton
 */
#pragma once

#include "cutlass/library/manifest.h"

#include "./operation_table_ext.h"

namespace cutlass {
namespace library {

/*! \brief Operation table singleton */
class SingletonExt {
 public:
  /*! \brief Manifest for all available operators */
  Manifest manifest;
  /*! \brief Operation table referencing the manifest */
  OperationTableExt operation_table;

 public:
  SingletonExt();
  static SingletonExt const& get();
};

}  // namespace library
}  // namespace cutlass
