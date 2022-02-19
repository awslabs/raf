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
