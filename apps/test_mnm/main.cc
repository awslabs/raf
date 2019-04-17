#include <dmlc/logging.h>
#include <mnm/device_api.h>
#include <mnm/memory_pool.h>
#include <mnm/tensor.h>
#include <mnm/types.h>
#include <cassert>
#include <iostream>

void test_index_type() {
  using index_t = mnm::types::index_t;
  mnm::types::index_t a(-1);
  mnm::types::index_t b(-2);
  auto c = a + b;
  (a += b) += c;
  CHECK(std::string(a) == "-6");
  CHECK(std::string(b) == "-2");
  CHECK(std::string(c) == "-3");
}

int main() {
  test_index_type();
  return 0;
}
