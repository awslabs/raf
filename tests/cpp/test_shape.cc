#include <gtest/gtest.h>
#include <mnm/device_api.h>
#include <mnm/test.h>
#include <mnm/types.h>
#include <cassert>
#include <iostream>

TEST(ShapeType, TextFormat) {
  using dim_t = mnm::types::dim_t;
  using shape_t = mnm::types::shape_t;
  shape_t shape({1, 2, 3});
  std::vector<int> vshape2({1, 2, 3});
  shape_t shape2(vshape2);
  ASSERT_STREQ(std::string(shape).c_str(), "(1, 2, 3)");
  ASSERT_STREQ(std::string(shape2).c_str(), "(1, 2, 3)");
}

TEST(ShapeType, Normalize) {
  using shape_t = mnm::types::shape_t;
  shape_t shape({1, 2, 3});
  ASSERT_STREQ(std::string(shape[shape.NormalizeAxis(0)]).c_str(), "1");
  ASSERT_STREQ(std::string(shape[shape.NormalizeAxis(1)]).c_str(), "2");
  ASSERT_STREQ(std::string(shape[shape.NormalizeAxis(2)]).c_str(), "3");
  ASSERT_STREQ(std::string(shape[shape.NormalizeAxis(-1)]).c_str(), "3");
  ASSERT_STREQ(std::string(shape[shape.NormalizeAxis(-2)]).c_str(), "2");
  ASSERT_STREQ(std::string(shape[shape.NormalizeAxis(-3)]).c_str(), "1");
  bool caught = false;
  try {
    std::cout << shape[shape.NormalizeAxis(-4)] << std::endl;
  } catch (const dmlc::Error& e) {
    caught = true;
  }
  ASSERT_TRUE(caught);
}

TEST(ShapeType, DimZero) {
  using shape_t = mnm::types::shape_t;
  shape_t shape_a({5, 15, 1});
  shape_t shape_b({10, 1, 15, 1});
  shape_t shape_c({});
  ASSERT_STREQ(std::string(shape_t::Broadcast(shape_a, shape_b)).c_str(), "(10, 5, 15, 1)");
  ASSERT_STREQ(std::string(shape_t::Broadcast(shape_a, shape_c)).c_str(), "(5, 15, 1)");
  ASSERT_STREQ(std::string(shape_t::Broadcast(shape_b, shape_c)).c_str(), "(10, 1, 15, 1)");
  ASSERT_STREQ(std::string(shape_t::Broadcast(shape_c, shape_c)).c_str(), "()");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
