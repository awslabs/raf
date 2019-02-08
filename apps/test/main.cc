#include <iostream>
#include <cassert>
#include <mnm/test.h>
#include <mnm/types.h>


void test_print_hello() {
  print_hello();
}

void test_dim_type() {
  using dim_t = mnm::types::dim_t;
  mnm::types::dim_t a(-1);
  mnm::types::dim_t b(-2);
  auto c = a + b;
  (a += b) += c;
  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  assert(std::string(a) == "-6");
  assert(std::string(b) == "-2");
  assert(std::string(c) == "-3");
}

int main() {
  test_print_hello();
  test_dim_type();
  return 0;
}
