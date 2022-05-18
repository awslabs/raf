/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/profiler/scope_timer.cc
 * \brief Scope timer to measure execution time of a code scope.
 */
#include "raf/registry.h"
#include "raf/scope_timer.h"

namespace raf {
namespace scope_timer {

RAF_REGISTER_GLOBAL("raf.scope_timer.DumpReport").set_body_typed([]() {
  ScopeTimerPool::Get()->DumpReport();
});

RAF_REGISTER_GLOBAL("raf.scope_timer.Reset").set_body_typed([]() {
  ScopeTimerPool::Get()->Reset();
});

}  // namespace scope_timer
}  // namespace raf
