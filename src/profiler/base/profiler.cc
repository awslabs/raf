/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/profiler.cc
 * \brief MNM profiler, a simple implementation
 */
#include "mnm/registry.h"
#include "mnm/profiler.h"

namespace mnm {
namespace profiler {

Profiler::Profiler() {
  init_time_ = ProfileStat::NowInMicrosec();
}

Profiler::~Profiler() {
}

Profiler* Profiler::Get(std::shared_ptr<Profiler>* sp) {
  static std::mutex mtx;
  static std::shared_ptr<Profiler> prof = nullptr;
  if (!prof) {
    std::unique_lock<std::mutex> lk(mtx);
    if (!prof) {
      prof = std::make_shared<Profiler>();
    }
  }
  if (sp) {
    *sp = prof;
  }
  return prof.get();
}

void Profiler::SetConfig(bool profiling = false) {
  profiling_ = std::move(profiling);
}

void Profiler::AddNewProfileStat(std::string categories, std::string name, uint64_t start_time,
                                 uint64_t end_time, const std::vector<std::string>& args) {
  std::unique_ptr<ProfileStat> stat =
      std::unique_ptr<ProfileStat>(new ProfileStat(categories, name, start_time, end_time, args));
  profile_stats_.opr_exec_stats_->enqueue(stat.release());
}

std::string Profiler::GetProfile() {
  std::lock_guard<std::recursive_mutex> lock{this->m_};
  std::stringstream ss;
  ss << "{" << std::endl;
  ss << "    \"traceEvents\": [" << std::endl;

  ProfileStat* stat;
  int stat_count = 0;
  while (profile_stats_.opr_exec_stats_->try_dequeue(stat)) {
    CHECK_NOTNULL(stat);
    std::unique_ptr<ProfileStat> profile_stat(stat);  // manage lifecycle
    CHECK_NE(profile_stat->categories_.c_str()[0], '\0') << "Category must be set";
    if (stat_count) {
      ss << ",\n";
    }
    profile_stat->EmitEvents(&ss);
    ++stat_count;
  }
  ss << "\n" << std::endl;
  ss << "    ]," << std::endl;
  ss << "    \"displayTimeUnit\": \"ms\"" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::vector<ProfileStat> Profiler::GetProfileStats() {
  std::lock_guard<std::recursive_mutex> lock{this->m_};
  std::vector<ProfileStat> results;

  ProfileStat* stat;
  while (profile_stats_.opr_exec_stats_->try_dequeue(stat)) {
    CHECK_NOTNULL(stat);
    results.push_back(*stat);
  }

  return results;
}

DeviceStats::~DeviceStats() {
  std::shared_ptr<TQueue> es = opr_exec_stats_;
  if (es) {
    ProfileStat* stat = nullptr;
    while (es->try_dequeue(stat)) {
      delete stat;
    }
  }
}

ProfileStat::ProfileStat(std::string categories, std::string name, uint64_t start_time,
                         uint64_t end_time, const std::vector<std::string>& args) {
  categories_ = categories;
  name_ = name;
  for (int i = 0; i < args.size(); i++) args_string += args[i] + ";";
  items_[kStart].enabled_ = items_[kStop].enabled_ = true;
  items_[kStart].event_type_ = kDurationBegin;
  items_[kStart].timestamp_ = start_time;
  items_[kStop].event_type_ = kDurationEnd;
  items_[kStop].timestamp_ = end_time;
}

void ProfileStat::EmitEvents(std::ostream* os) {
  size_t count = 0;
  for (size_t i = 0; i < sizeof(items_) / sizeof(items_[0]); ++i) {
    if (items_[i].enabled_) {
      if (count) {
        *os << ",\n";
      }
      *os << "    {\n"
          << "        \"name\": \"" << name_.c_str() << "\",\n"
          << "        \"cat\": "
          << "\"" << categories_.c_str() << "\",\n"
          << "        \"ph\": \"" << static_cast<char>(items_[i].event_type_) << "\",\n"
          << "        \"ts\": " << items_[i].timestamp_ << ",\n"
          << "        \"args\": { \""
          << "args_string"
          << "\": "
          << "\"" << args_string << "\"},\n";
      *os << "        \"pid\": " << process_id_
          << ",\n"
          // << "        \"tid\": " << std::hash<std::thread::id>{}(thread_id_) << "\n"
          << "        \"tid\": \"" << categories_.c_str() << "\"\n"
          << "    }\n";
      ++count;
    }
  }
}

void EnableProfiler() {
  Profiler::Get()->SetConfig(true);
}

void DisableProfiler() {
  Profiler::Get()->SetConfig(false);
}

std::string GetProfile() {
  return Profiler::Get()->GetProfile();
}

MNM_REGISTER_GLOBAL("mnm.profiler.EnableProfiler").set_body_typed(EnableProfiler);
MNM_REGISTER_GLOBAL("mnm.profiler.DisableProfiler").set_body_typed(DisableProfiler);
MNM_REGISTER_GLOBAL("mnm.profiler.GetProfile").set_body_typed(GetProfile);

}  // namespace profiler
}  // namespace mnm
