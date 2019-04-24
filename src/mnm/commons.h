#pragma once

#include <mutex>
#include <type_traits>

#define LOCKED_IF(cond, the_mutex, stmt)         \
  if (cond) {                                    \
    std::lock_guard<std::mutex> lock(the_mutex); \
    if (cond) {                                  \
      stmt;                                      \
    }                                            \
  }

#define ASSERT_SAME_CLASS(cls_a, cls_b)            \
  static_assert(std::is_same<cls_a, cls_b>::value, \
                "Assertion failed: " #cls_a " and " #cls_b " are not the same class.")

#define ASSERT_DERIVED_FROM(cls_derived, cls_base)             \
  static_assert(std::is_base_of<cls_base, cls_derived>::value, \
                "Assertion failed: " #cls_derived " is not derived from " #cls_base ".")
