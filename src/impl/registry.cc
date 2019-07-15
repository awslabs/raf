#include <mnm/registry.h>

namespace mnm {
namespace registry {

const tvm::runtime::PackedFunc& GetPackedFunc(const std::string& name) {
  const tvm::runtime::PackedFunc* pf = Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

}  // namespace registry
}  // namespace mnm
