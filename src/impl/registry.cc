#include <mnm/registry.h>

namespace mnm {
namespace registry {

const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

}  // namespace registry
}  // namespace mnm