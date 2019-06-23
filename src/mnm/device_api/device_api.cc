#include <mnm/device_api.h>
#include <mnm/registry.h>

namespace mnm {
namespace device_api {

using registry::Registry;

DeviceAPI* DeviceAPI::Create(DevType device_type, bool allow_missing) {
  thread_local char creator_name[128];
  sprintf(creator_name, "mnm.device_api.%s", device_type.c_str());
  auto creator = Registry::Get(creator_name);
  if (creator == nullptr) {
    CHECK(allow_missing) << "ValueError: DeviceAPI " << creator_name << " is not enabled.";
    return nullptr;
  }
  void* ret = (*creator)();
  return static_cast<DeviceAPI*>(ret);
}

DeviceAPI* DeviceAPIManager::GetAPI(DevType device_type, bool allow_missing) {
  APIPtr& api = api_[int(device_type)];
  if (api == nullptr) {
    api.reset(DeviceAPI::Create(device_type, allow_missing));
  }
  return api.get();
}

}  // namespace device_api
}  // namespace mnm
