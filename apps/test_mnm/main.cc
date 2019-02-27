#include <dmlc/logging.h>
#include <mnm/device_api.h>
#include <mnm/test.h>
#include <mnm/types.h>
#include <mnm/memory_pool.h>
#include <cassert>
#include <iostream>

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

void test_normalize_axis() {
  using shape_t = mnm::types::shape_t;
  shape_t shape({1, 2, 3});
  CHECK(std::string(shape[shape.NormalizeAxis(0)]) == "1");
  CHECK(std::string(shape[shape.NormalizeAxis(1)]) == "2");
  CHECK(std::string(shape[shape.NormalizeAxis(2)]) == "3");
  CHECK(std::string(shape[shape.NormalizeAxis(-1)]) == "3");
  CHECK(std::string(shape[shape.NormalizeAxis(-2)]) == "2");
  CHECK(std::string(shape[shape.NormalizeAxis(-3)]) == "1");
  bool caught = false;
  try {
    std::cout << shape[shape.NormalizeAxis(-4)] << std::endl;
  } catch (const dmlc::Error& e) {
    caught = true;
  }
  CHECK(caught);
}

void test_broadcast() {
  using shape_t = mnm::types::shape_t;
  shape_t shape_a({5, 15, 1});
  shape_t shape_b({10, 1, 15, 1});
  shape_t shape_c({});
  CHECK(std::string(shape_t::Broadcast(shape_a, shape_b)) == "(10, 5, 15, 1)");
  CHECK(std::string(shape_t::Broadcast(shape_a, shape_c)) == "(5, 15, 1)");
  CHECK(std::string(shape_t::Broadcast(shape_b, shape_c)) == "(10, 1, 15, 1)");
  CHECK(std::string(shape_t::Broadcast(shape_c, shape_c)) == "()");
}

int main() {
  test_print_hello();
  test_dim_type();
  test_shape_type();
  test_normalize_axis();
  test_broadcast();
  return 0;
}
