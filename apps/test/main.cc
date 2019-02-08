#include <iostream>
#include <cassert>
#include <mnm/test.h>
#include <mnm/types.h>
#include <dmlc/logging.h>


void test_print_hello() {
  print_hello();
}

void test_dim_type() {
  using dim_t = mnm::types::dim_t;
  mnm::types::dim_t a(-1);
  mnm::types::dim_t b(-2);
  auto c = a + b;
  (a += b) += c;
  CHECK(std::string(a) == "-6");
  CHECK(std::string(b) == "-2");
  CHECK(std::string(c) == "-3");
}

void test_shape_type() {
  using dim_t = mnm::types::dim_t;
  using shape_t = mnm::types::shape_t;
  shape_t shape({1, 2, 3});
  std::vector<int> vshape2({1, 2, 3});
  shape_t shape2(vshape2);
  CHECK(std::string(shape) == "(1, 2, 3)");
  CHECK(std::string(shape2) == "(1, 2, 3)");
}

int main() {
  test_print_hello();
  test_dim_type();
  test_shape_type();
  return 0;
}
