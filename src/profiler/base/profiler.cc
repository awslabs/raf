/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/profiler/base/profiler.cc
 * \brief MNM profiler, a simple implementation
 */
#include "mnm/registry.h"
#include "mnm/profiler.h"

namespace mnm {
namespace profiler {

Profiler::Profiler() {
}

Profiler::~Profiler() {
}

Profiler* Profiler::Get() {
  static Profiler prof;
  return &prof;
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
  if (!args.empty()) {
    for (int i = 0; i < args.size() - 1; i++) {
      args_string += args[i] + ";";
    }
    args_string += args[args.size() - 1];
  }
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

void EnableProfiler(int profile_level) {
  Profiler::Get()->set_profile_level(profile_level);
}

void DisableProfiler() {
  Profiler::Get()->set_profile_level(0);
}

void CollectBaseProfile() {
  Profiler::Get()->CollectStat();
}

std::string GetProfile() {
  return Profiler::Get()->GetProfile();
}

MNM_REGISTER_GLOBAL("mnm.profiler.EnableProfiler").set_body_typed(EnableProfiler);
MNM_REGISTER_GLOBAL("mnm.profiler.DisableProfiler").set_body_typed(DisableProfiler);
MNM_REGISTER_GLOBAL("mnm.profiler.CollectBaseProfile").set_body_typed(CollectBaseProfile);
MNM_REGISTER_GLOBAL("mnm.profiler.GetProfile").set_body_typed(GetProfile);

}  // namespace profiler
}  // namespace mnm
