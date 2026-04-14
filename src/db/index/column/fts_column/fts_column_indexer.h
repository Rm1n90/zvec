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


#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "bm25_scorer.h"
#include "db/common/rocksdb_context.h"
#include "fts_posting_codec.h"
#include "fts_search_result.h"
#include "tokenizer.h"


namespace zvec {


/**
 * @brief Per-field full-text-search indexer.
 *
 * One instance per text field within a segment. All FtsColumnIndexers in a
 * segment share a single RocksDB instance owned by FtsIndexer; each field
 * gets its own pair of column families:
 *
 *   `<field>_fts_terms`        — term -> posting list (varint, see
 *                                 FtsPostingCodec). Writes go through the
 *                                 FtsPostingMerger so concurrent inserts on
 *                                 the same term accumulate correctly.
 *   `<field>_fts_doc_lengths`  — doc_id (4-byte BE) -> doc length in tokens.
 *
 * Plus three well-known keys in the default column family, prefixed with the
 * field name so multiple FTS fields do not collide:
 *
 *   `<field>_fts_total_docs`
 *   `<field>_fts_total_tokens`
 *   `<field>_fts_max_id`
 *   `<field>_fts_sealed`
 *
 * The indexer is thread-safe for concurrent insert + search, with the
 * following guarantees:
 *   - insert / search use RocksDB's own concurrency control.
 *   - The collection-stats counters (total_docs, total_tokens, max_id) are
 *     atomics; they are persisted on FlushSpecialValues / Seal.
 */
class FtsColumnIndexer {
 public:
  using Ptr = std::shared_ptr<FtsColumnIndexer>;

  /**
   * @brief Combinator applied across query terms.
   *
   *   OR  — a doc matches if it contains any query term; the BM25 scores
   *         for matched terms are summed.
   *   AND — a doc matches only if it contains every query term.
   */
  enum class MatchOp { OR, AND };

  /// Build (or reopen) an FTS column indexer for @p field.
  /// The field must declare an FtsIndexParams; @p context must already be
  /// opened with FtsPostingMerger as its merge operator and have the two
  /// column families (`<field>_fts_terms`, `<field>_fts_doc_lengths`)
  /// created.
  static Ptr CreateAndOpen(const std::string &collection_name,
                           const FieldSchema &field, RocksdbContext &context,
                           bool read_only = false);

  ~FtsColumnIndexer();

  FtsColumnIndexer(const FtsColumnIndexer &) = delete;
  FtsColumnIndexer &operator=(const FtsColumnIndexer &) = delete;
  FtsColumnIndexer(FtsColumnIndexer &&) = delete;
  FtsColumnIndexer &operator=(FtsColumnIndexer &&) = delete;

  // ===== indexing =====

  /// Tokenize @p text and write the resulting posting updates into RocksDB.
  /// Updates per-doc length and the collection-level counters atomically.
  Status Insert(uint32_t doc_id, std::string_view text);

  /// Record that @p doc_id has a null value for this field — the doc still
  /// counts toward total_docs (so other docs' BM25 lengths are normalized
  /// consistently) but contributes zero tokens and no postings.
  Status InsertNull(uint32_t doc_id);

  // ===== search =====

  /// Run a top-K BM25 search. Tokenizes @p query with the same tokenizer
  /// used at index time, looks up each unique term's posting list, scores
  /// candidate docs and returns the top @p topk by descending score.
  ///
  /// @p topk == 0 returns an empty result.
  Result<FtsSearchResult::Ptr> Search(std::string_view query, std::size_t topk,
                                      MatchOp op = MatchOp::OR) const;

  // ===== lifecycle =====

  /// Persist in-memory atomics (total_docs, total_tokens, max_id) to disk.
  /// Returns OK if there is nothing to persist (read-only is rejected).
  Status FlushSpecialValues();

  /// Persist atomics, then mark the indexer sealed (read-only).
  Status Seal();
  bool IsSealed() const {
    return sealed_.load();
  }

  /// Delete all of this indexer's CFs and special keys.
  Status DropStorage();

  // ===== introspection =====

  uint64_t TotalDocs() const {
    return total_docs_.load();
  }
  uint64_t TotalTokens() const {
    return total_tokens_.load();
  }
  uint32_t MaxId() const {
    return max_id_.load();
  }
  Bm25CollectionStats Stats() const {
    Bm25CollectionStats s;
    s.total_docs = total_docs_.load();
    s.total_tokens = total_tokens_.load();
    return s;
  }

  std::string ID() const;
  const FieldSchema &field() const {
    return field_;
  }
  const FtsTokenizer &tokenizer() const {
    return *tokenizer_;
  }
  const Bm25Params &bm25_params() const {
    return bm25_params_;
  }

  // CF name helpers — also used by FtsIndexer when creating / dropping CFs.
  static std::string TermsCfName(const std::string &field_name);
  static std::string DocLengthsCfName(const std::string &field_name);

 protected:
  FtsColumnIndexer(const std::string &collection_name, const FieldSchema &field,
                   RocksdbContext &context, bool read_only);

  Status open();
  Status load_state();
  void bump_max_id(uint32_t id);

  // Search helpers (defined in fts_column_indexer_search.cc).
  Status load_posting(const std::string &term,
                      std::vector<FtsPosting> *out) const;
  Status load_doc_length(uint32_t doc_id, uint32_t *out) const;
  void load_doc_lengths_batch(const std::vector<uint32_t> &doc_ids,
                              std::vector<uint32_t> *out) const;

  // Special-key helpers.
  std::string key_total_docs() const;
  std::string key_total_tokens() const;
  std::string key_max_id() const;
  std::string key_sealed() const;

 private:
  std::string collection_name_;
  FieldSchema field_;
  RocksdbContext &ctx_;
  rocksdb::ColumnFamilyHandle *cf_terms_{nullptr};
  rocksdb::ColumnFamilyHandle *cf_doc_lengths_{nullptr};
  bool read_only_{false};
  std::atomic<bool> sealed_{false};
  std::atomic<uint64_t> total_docs_{0};
  std::atomic<uint64_t> total_tokens_{0};
  std::atomic<uint32_t> max_id_{0};
  FtsTokenizer::Ptr tokenizer_{};
  Bm25Params bm25_params_{};
};


}  // namespace zvec
