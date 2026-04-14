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


#pragma once


#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <zvec/db/schema.h>
#include "db/common/rocksdb_context.h"
#include "fts_column_indexer.h"


namespace zvec {


/**
 * @brief Owner of an FTS RocksDB instance and its per-field column indexers.
 *
 * Mirrors @ref InvertedIndexer in role, but for full-text fields. Holds a
 * single RocksDB instance under @c working_dir with @ref FtsPostingMerger as
 * its merge operator, and one pair of column families (`<field>_fts_terms`,
 * `<field>_fts_doc_lengths`) per FTS-indexed field.
 *
 * Lifecycle (parallels InvertedIndexer):
 *   - CreateAndOpen() returns nullptr on failure.
 *   - flush() persists the in-memory atomics and durably flushes the DB.
 *   - seal() flushes, marks each column indexer sealed, and triggers a
 *     compaction.
 *   - create_column_indexer() / remove_column_indexer() add or drop FTS
 *     fields after the indexer is open.
 */
class FtsIndexer {
 public:
  using Ptr = std::shared_ptr<FtsIndexer>;

  FtsIndexer(const std::string &collection_name,
             const std::string &working_dir,
             const std::vector<FieldSchema> &fields)
      : collection_name_(collection_name),
        working_dir_(working_dir),
        fields_(fields) {}

  ~FtsIndexer();

  static Ptr CreateAndOpen(const std::string &collection_name,
                           const std::string &working_dir,
                           bool create_dir_if_missing,
                           const std::vector<FieldSchema> &fields,
                           bool read_only = false);

  /// Look up the column indexer for a given FTS field; nullptr if absent.
  FtsColumnIndexer::Ptr operator[](const std::string &field_name) const {
    auto it = indexers_.find(field_name);
    if (it != indexers_.end()) {
      return it->second;
    }
    return nullptr;
  }

  /// Persist atomic counters from each column indexer, then flush the DB.
  Status flush();

  /// flush + seal each column indexer + compact.
  Status seal();

  /// Add a new FTS-indexed field after the indexer is already open.
  Status create_column_indexer(const FieldSchema &field);

  /// Remove an FTS-indexed field (drops its CFs and special keys).
  Status remove_column_indexer(const std::string &field_name);

  std::string collection() const {
    return collection_name_;
  }
  std::string working_dir() const {
    return working_dir_;
  }
  std::string ID() const;

 private:
  Status open(bool create_dir_if_missing, bool read_only);

  // Collect the list of CF names for the currently configured fields.
  std::vector<std::string> collect_cf_names() const;

  std::string collection_name_;
  std::string working_dir_;
  std::vector<FieldSchema> fields_;

  std::unordered_map<std::string, FtsColumnIndexer::Ptr> indexers_;
  RocksdbContext rocksdb_context_{};
  bool read_only_{false};
};


}  // namespace zvec
