// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "fts_indexer.h"

#include <algorithm>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include "fts_posting_codec.h"


namespace zvec {


using FILE = ailego::File;


FtsIndexer::~FtsIndexer() {
  rocksdb_context_.close();
  LOG_INFO("Closed %s", ID().c_str());
}


std::string FtsIndexer::ID() const {
  return "FtsIndexer[collection:" + collection_name_ + "|path:'" +
         working_dir_ + "']";
}


std::vector<std::string> FtsIndexer::collect_cf_names() const {
  std::vector<std::string> out;
  out.reserve(fields_.size() * 2);
  for (const auto &field : fields_) {
    out.push_back(FtsColumnIndexer::TermsCfName(field.name()));
    out.push_back(FtsColumnIndexer::DocLengthsCfName(field.name()));
  }
  return out;
}


FtsIndexer::Ptr FtsIndexer::CreateAndOpen(
    const std::string &collection_name, const std::string &working_dir,
    bool create_dir_if_missing, const std::vector<FieldSchema> &fields,
    bool read_only) {
  auto indexer =
      std::make_shared<FtsIndexer>(collection_name, working_dir, fields);
  if (indexer->open(create_dir_if_missing, read_only).ok()) {
    return indexer;
  }
  return nullptr;
}


Status FtsIndexer::open(bool create_dir_if_missing, bool read_only) {
  read_only_ = read_only;
  // Validate that every supplied field is actually FTS-indexed.
  for (const auto &field : fields_) {
    if (field.index_type() != IndexType::FTS) {
      LOG_ERROR("Field[%s] is not an FTS field in %s", field.name().c_str(),
                ID().c_str());
      return Status::InvalidArgument("non-FTS field passed to FtsIndexer: ",
                                     field.name());
    }
  }

  auto cf_names = collect_cf_names();
  auto merger = std::make_shared<FtsPostingMerger>();

  Status s;
  if (FILE::IsExist(working_dir_)) {
    if (!FILE::IsDirectory(working_dir_)) {
      LOG_ERROR("FtsIndexer path[%s] is not a directory", working_dir_.c_str());
      return Status::InvalidArgument("FtsIndexer path is not a directory: ",
                                     working_dir_);
    }
    s = rocksdb_context_.open(working_dir_, cf_names, read_only, merger);
  } else {
    if (!create_dir_if_missing) {
      LOG_ERROR("FtsIndexer path[%s] does not exist", working_dir_.c_str());
      return Status::NotFound("FtsIndexer path missing: ", working_dir_);
    }
    s = rocksdb_context_.create(working_dir_, cf_names, merger);
  }

  if (!s.ok()) {
    LOG_ERROR("Failed to open %s: %s", ID().c_str(), s.c_str());
    return s;
  }

  for (const auto &field : fields_) {
    auto column_indexer = FtsColumnIndexer::CreateAndOpen(
        collection_name_, field, rocksdb_context_, read_only);
    if (column_indexer == nullptr) {
      LOG_ERROR("Failed to create FtsColumnIndexer[%s]", field.name().c_str());
      return Status::InternalError("Failed to create FtsColumnIndexer for ",
                                   field.name());
    }
    indexers_.emplace(field.name(), std::move(column_indexer));
  }

  LOG_INFO("Opened %s", ID().c_str());
  return s;
}


Status FtsIndexer::flush() {
  for (auto &[_, indexer] : indexers_) {
    if (indexer->IsSealed()) {
      continue;
    }
    if (auto s = indexer->FlushSpecialValues(); !s.ok()) {
      LOG_ERROR("Failed to flush special values for %s: %s",
                indexer->ID().c_str(), s.c_str());
      return s;
    }
  }
  auto s = rocksdb_context_.flush();
  if (s.ok()) {
    LOG_INFO("Flushed %s", ID().c_str());
  } else {
    LOG_ERROR("Failed to flush %s: %s", ID().c_str(), s.c_str());
  }
  return s;
}


Status FtsIndexer::seal() {
  for (const auto &[_, indexer] : indexers_) {
    if (indexer->IsSealed()) {
      continue;
    }
    if (auto s = indexer->Seal(); !s.ok()) {
      LOG_ERROR("Failed to seal %s: %s", indexer->ID().c_str(), s.c_str());
      return s;
    }
  }
  if (auto s = flush(); !s.ok()) {
    return s;
  }
  if (auto s = rocksdb_context_.compact(); !s.ok()) {
    LOG_ERROR("Failed to compact %s during sealing: %s", ID().c_str(),
              s.c_str());
    return s;
  }
  LOG_INFO("Sealed %s", ID().c_str());
  return Status::OK();
}


Status FtsIndexer::create_column_indexer(const FieldSchema &field) {
  if (field.index_type() != IndexType::FTS) {
    return Status::InvalidArgument("non-FTS field: ", field.name());
  }
  auto it = std::find_if(fields_.begin(), fields_.end(),
                         [&field](const FieldSchema &cur) {
                           return cur.name() == field.name();
                         });
  if (it != fields_.end() || indexers_.count(field.name()) > 0) {
    LOG_ERROR("FtsColumnIndexer[%s] already exists in %s",
              field.name().c_str(), ID().c_str());
    return Status::AlreadyExists(field.name());
  }

  Status s;
  bool cf_terms_created = false;
  bool cf_doc_lengths_created = false;
  AILEGO_DEFER([&]() {
    if (s.ok()) {
      LOG_INFO("Created a new FtsColumnIndexer[%s] in %s",
               field.name().c_str(), ID().c_str());
    } else {
      if (cf_terms_created) {
        rocksdb_context_.drop_cf(FtsColumnIndexer::TermsCfName(field.name()));
      }
      if (cf_doc_lengths_created) {
        rocksdb_context_.drop_cf(
            FtsColumnIndexer::DocLengthsCfName(field.name()));
      }
      LOG_ERROR("Failed to create FtsColumnIndexer[%s] in %s: %s",
                field.name().c_str(), ID().c_str(), s.c_str());
    }
  });

  s = rocksdb_context_.create_cf(FtsColumnIndexer::TermsCfName(field.name()));
  if (!s.ok()) {
    return s;
  }
  cf_terms_created = true;
  s = rocksdb_context_.create_cf(
      FtsColumnIndexer::DocLengthsCfName(field.name()));
  if (!s.ok()) {
    return s;
  }
  cf_doc_lengths_created = true;

  auto column_indexer = FtsColumnIndexer::CreateAndOpen(
      collection_name_, field, rocksdb_context_, read_only_);
  if (column_indexer == nullptr) {
    s = Status::InternalError("Failed to open new FtsColumnIndexer for ",
                              field.name());
    return s;
  }
  fields_.push_back(field);
  indexers_.emplace(field.name(), std::move(column_indexer));
  s = Status::OK();
  return s;
}


Status FtsIndexer::remove_column_indexer(const std::string &field_name) {
  auto it = std::find_if(fields_.begin(), fields_.end(),
                         [&field_name](const FieldSchema &cur) {
                           return cur.name() == field_name;
                         });
  auto column_indexer = (*this)[field_name];
  if (it == fields_.end() && !column_indexer) {
    LOG_ERROR("FtsColumnIndexer[%s] doesn't exist in %s", field_name.c_str(),
              ID().c_str());
    return Status::NotFound(field_name);
  }
  if (it == fields_.end() || !column_indexer) {
    LOG_ERROR("%s is in corrupted state", ID().c_str());
    return Status::InternalError("FtsIndexer state inconsistent for ",
                                 field_name);
  }

  if (auto s = column_indexer->DropStorage(); !s.ok()) {
    LOG_ERROR("Failed to drop storage for FtsColumnIndexer[%s]: %s",
              field_name.c_str(), s.c_str());
    return s;
  }

  fields_.erase(it);
  indexers_.erase(field_name);
  LOG_INFO("Removed FtsColumnIndexer[%s] in %s", field_name.c_str(),
           ID().c_str());
  return Status::OK();
}


}  // namespace zvec
