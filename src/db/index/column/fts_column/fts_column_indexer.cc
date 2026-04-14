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


#include "fts_column_indexer.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <zvec/ailego/logger/logger.h>


namespace zvec {


namespace {

constexpr const char *kSuffixTerms = "_fts_terms";
constexpr const char *kSuffixDocLengths = "_fts_doc_lengths";
constexpr const char *kKeyTotalDocs = "_fts_total_docs";
constexpr const char *kKeyTotalTokens = "_fts_total_tokens";
constexpr const char *kKeyMaxId = "_fts_max_id";
constexpr const char *kKeySealed = "_fts_sealed";


// 4-byte big-endian doc-id key, so that RocksDB scans go in doc-id order.
inline std::string EncodeDocIdKey(uint32_t id) {
  std::string s(4, '\0');
  s[0] = static_cast<char>((id >> 24) & 0xFF);
  s[1] = static_cast<char>((id >> 16) & 0xFF);
  s[2] = static_cast<char>((id >> 8) & 0xFF);
  s[3] = static_cast<char>(id & 0xFF);
  return s;
}

// Fixed 4-byte little-endian uint32 value.
inline std::string EncodeUint32Le(uint32_t v) {
  std::string s(4, '\0');
  s[0] = static_cast<char>(v & 0xFF);
  s[1] = static_cast<char>((v >> 8) & 0xFF);
  s[2] = static_cast<char>((v >> 16) & 0xFF);
  s[3] = static_cast<char>((v >> 24) & 0xFF);
  return s;
}

inline bool DecodeUint32Le(const char *data, std::size_t size, uint32_t *out) {
  if (size != 4) {
    return false;
  }
  *out = (static_cast<uint32_t>(static_cast<uint8_t>(data[0]))) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[3])) << 24);
  return true;
}

inline std::string EncodeUint64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) {
    s[i] = static_cast<char>((v >> (i * 8)) & 0xFF);
  }
  return s;
}

inline bool DecodeUint64Le(const char *data, std::size_t size, uint64_t *out) {
  if (size != 8) {
    return false;
  }
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<uint64_t>(static_cast<uint8_t>(data[i])) << (i * 8);
  }
  *out = v;
  return true;
}

}  // namespace


std::string FtsColumnIndexer::TermsCfName(const std::string &field_name) {
  return field_name + kSuffixTerms;
}

std::string FtsColumnIndexer::DocLengthsCfName(const std::string &field_name) {
  return field_name + kSuffixDocLengths;
}


FtsColumnIndexer::FtsColumnIndexer(const std::string &collection_name,
                                   const FieldSchema &field,
                                   RocksdbContext &context, bool read_only)
    : collection_name_(collection_name),
      field_(field),
      ctx_(context),
      read_only_(read_only) {}


FtsColumnIndexer::~FtsColumnIndexer() = default;


FtsColumnIndexer::Ptr FtsColumnIndexer::CreateAndOpen(
    const std::string &collection_name, const FieldSchema &field,
    RocksdbContext &context, bool read_only) {
  if (field.index_type() != IndexType::FTS) {
    LOG_ERROR("Field[%s] is not an FTS field", field.name().c_str());
    return nullptr;
  }
  if (field.data_type() != DataType::STRING) {
    LOG_ERROR(
        "FTS field[%s] must have STRING data type (got data_type=%u)",
        field.name().c_str(), static_cast<uint32_t>(field.data_type()));
    return nullptr;
  }

  Ptr indexer(new FtsColumnIndexer(collection_name, field, context, read_only));
  auto s = indexer->open();
  if (!s.ok()) {
    LOG_ERROR("Failed to open %s: %s", indexer->ID().c_str(), s.c_str());
    return nullptr;
  }
  return indexer;
}


Status FtsColumnIndexer::open() {
  cf_terms_ = ctx_.get_cf(TermsCfName(field_.name()));
  if (cf_terms_ == nullptr) {
    return Status::InternalError("FTS terms CF missing for field ",
                                 field_.name());
  }
  cf_doc_lengths_ = ctx_.get_cf(DocLengthsCfName(field_.name()));
  if (cf_doc_lengths_ == nullptr) {
    return Status::InternalError("FTS doc-lengths CF missing for field ",
                                 field_.name());
  }

  auto params = std::dynamic_pointer_cast<FtsIndexParams>(field_.index_params());
  if (params == nullptr) {
    return Status::InvalidArgument(
        "FTS field is missing FtsIndexParams: ", field_.name());
  }
  tokenizer_ = CreateFtsTokenizer(params->tokenizer());
  if (tokenizer_ == nullptr) {
    return Status::InvalidArgument("Unknown FTS tokenizer: ",
                                   params->tokenizer());
  }
  bm25_params_.k1 = params->k1();
  bm25_params_.b = params->b();

  return load_state();
}


Status FtsColumnIndexer::load_state() {
  std::string value;
  rocksdb::Status s;

  s = ctx_.db_->Get(ctx_.read_opts_, key_total_docs(), &value);
  if (s.ok()) {
    uint64_t v = 0;
    if (!DecodeUint64Le(value.data(), value.size(), &v)) {
      return Status::InternalError("Corrupt total_docs for ", ID());
    }
    total_docs_.store(v);
  } else if (!s.IsNotFound()) {
    return Status::InternalError("Failed to read total_docs for ", ID(), ": ",
                                 s.ToString());
  }

  s = ctx_.db_->Get(ctx_.read_opts_, key_total_tokens(), &value);
  if (s.ok()) {
    uint64_t v = 0;
    if (!DecodeUint64Le(value.data(), value.size(), &v)) {
      return Status::InternalError("Corrupt total_tokens for ", ID());
    }
    total_tokens_.store(v);
  } else if (!s.IsNotFound()) {
    return Status::InternalError("Failed to read total_tokens for ", ID(), ": ",
                                 s.ToString());
  }

  s = ctx_.db_->Get(ctx_.read_opts_, key_max_id(), &value);
  if (s.ok()) {
    uint32_t v = 0;
    if (!DecodeUint32Le(value.data(), value.size(), &v)) {
      return Status::InternalError("Corrupt max_id for ", ID());
    }
    max_id_.store(v);
  } else if (!s.IsNotFound()) {
    return Status::InternalError("Failed to read max_id for ", ID(), ": ",
                                 s.ToString());
  }

  s = ctx_.db_->Get(ctx_.read_opts_, key_sealed(), &value);
  if (s.ok()) {
    sealed_.store(true);
  } else if (!s.IsNotFound()) {
    return Status::InternalError("Failed to read sealed flag for ", ID(), ": ",
                                 s.ToString());
  }

  return Status::OK();
}


Status FtsColumnIndexer::FlushSpecialValues() {
  if (read_only_) {
    return Status::PermissionDenied("FTS indexer is read-only");
  }

  rocksdb::Status s;
  s = ctx_.db_->Put(ctx_.write_opts_, key_total_docs(),
                    EncodeUint64Le(total_docs_.load()));
  if (!s.ok()) {
    return Status::InternalError("Failed to persist total_docs for ", ID(),
                                 ": ", s.ToString());
  }
  s = ctx_.db_->Put(ctx_.write_opts_, key_total_tokens(),
                    EncodeUint64Le(total_tokens_.load()));
  if (!s.ok()) {
    return Status::InternalError("Failed to persist total_tokens for ", ID(),
                                 ": ", s.ToString());
  }
  s = ctx_.db_->Put(ctx_.write_opts_, key_max_id(),
                    EncodeUint32Le(max_id_.load()));
  if (!s.ok()) {
    return Status::InternalError("Failed to persist max_id for ", ID(), ": ",
                                 s.ToString());
  }
  return Status::OK();
}


Status FtsColumnIndexer::Seal() {
  if (read_only_) {
    return Status::PermissionDenied("FTS indexer is read-only");
  }
  auto s = FlushSpecialValues();
  if (!s.ok()) {
    return s;
  }
  auto rs = ctx_.db_->Put(ctx_.write_opts_, key_sealed(), "1");
  if (!rs.ok()) {
    return Status::InternalError("Failed to seal ", ID(), ": ", rs.ToString());
  }
  sealed_.store(true);
  read_only_ = true;
  return Status::OK();
}


Status FtsColumnIndexer::DropStorage() {
  // Drop CFs (each may legitimately fail with NotFound after a partial init —
  // collect errors but keep going).
  Status final_status;
  if (auto s = ctx_.drop_cf(TermsCfName(field_.name())); !s.ok()) {
    final_status = s;
  } else {
    cf_terms_ = nullptr;
  }
  if (auto s = ctx_.drop_cf(DocLengthsCfName(field_.name())); !s.ok()) {
    final_status = s;
  } else {
    cf_doc_lengths_ = nullptr;
  }

  // Best-effort delete special keys.
  ctx_.db_->Delete(ctx_.write_opts_, key_total_docs());
  ctx_.db_->Delete(ctx_.write_opts_, key_total_tokens());
  ctx_.db_->Delete(ctx_.write_opts_, key_max_id());
  ctx_.db_->Delete(ctx_.write_opts_, key_sealed());

  total_docs_.store(0);
  total_tokens_.store(0);
  max_id_.store(0);
  sealed_.store(false);
  return final_status;
}


std::string FtsColumnIndexer::ID() const {
  return "FtsColumnIndexer[collection:" + collection_name_ +
         "|field:" + field_.name() + "]";
}


std::string FtsColumnIndexer::key_total_docs() const {
  return field_.name() + kKeyTotalDocs;
}
std::string FtsColumnIndexer::key_total_tokens() const {
  return field_.name() + kKeyTotalTokens;
}
std::string FtsColumnIndexer::key_max_id() const {
  return field_.name() + kKeyMaxId;
}
std::string FtsColumnIndexer::key_sealed() const {
  return field_.name() + kKeySealed;
}


void FtsColumnIndexer::bump_max_id(uint32_t id) {
  uint32_t prev = max_id_.load(std::memory_order_relaxed);
  while (id > prev &&
         !max_id_.compare_exchange_weak(prev, id, std::memory_order_relaxed)) {
    // CAS retried with the new prev value
  }
}


// ===== insert path =====
// Defined here (rather than in fts_column_indexer_write.cc) for now so the
// piece is small enough to read in one go.


Status FtsColumnIndexer::Insert(uint32_t doc_id, std::string_view text) {
  if (read_only_) {
    return Status::PermissionDenied("FTS indexer is read-only");
  }

  // Tokenize the text and compute per-term frequency in this document.
  std::vector<std::string> tokens;
  tokenizer_->Tokenize(text, &tokens);

  // No tokens means the field has no indexable content. Still treat as a doc
  // for collection-stats purposes so BM25 length normalization stays sane.
  if (tokens.empty()) {
    total_docs_.fetch_add(1, std::memory_order_relaxed);
    bump_max_id(doc_id);
    rocksdb::Status rs = ctx_.db_->Put(ctx_.write_opts_, cf_doc_lengths_,
                                       EncodeDocIdKey(doc_id), EncodeUint32Le(0));
    if (!rs.ok()) {
      return Status::InternalError("Failed to write doc length for ", ID(),
                                   ": ", rs.ToString());
    }
    return Status::OK();
  }

  // Group tokens by term, count frequencies.
  // (small map suffices — typical field has tens of terms)
  std::vector<std::pair<std::string, uint32_t>> term_freqs;
  term_freqs.reserve(tokens.size());
  std::sort(tokens.begin(), tokens.end());
  std::size_t i = 0;
  while (i < tokens.size()) {
    std::size_t j = i + 1;
    while (j < tokens.size() && tokens[j] == tokens[i]) {
      ++j;
    }
    term_freqs.emplace_back(std::move(tokens[i]),
                            static_cast<uint32_t>(j - i));
    i = j;
  }

  // Issue posting list merges in a single WriteBatch so they are atomic.
  rocksdb::WriteBatch batch;
  for (const auto &[term, tf] : term_freqs) {
    auto operand = FtsPostingCodec::EncodeSingle(doc_id, tf);
    auto rs = batch.Merge(cf_terms_, term, operand);
    if (!rs.ok()) {
      return Status::InternalError("Failed to stage posting merge for term ",
                                   term, ": ", rs.ToString());
    }
  }
  // Doc length goes into the same batch — written with Put (no merge).
  auto rs = batch.Put(cf_doc_lengths_, EncodeDocIdKey(doc_id),
                      EncodeUint32Le(static_cast<uint32_t>(tokens.size())));
  if (!rs.ok()) {
    return Status::InternalError("Failed to stage doc length for ", ID(), ": ",
                                 rs.ToString());
  }
  rs = ctx_.db_->Write(ctx_.write_opts_, &batch);
  if (!rs.ok()) {
    return Status::InternalError("Failed to commit posting batch for ", ID(),
                                 ": ", rs.ToString());
  }

  total_docs_.fetch_add(1, std::memory_order_relaxed);
  total_tokens_.fetch_add(tokens.size(), std::memory_order_relaxed);
  bump_max_id(doc_id);
  return Status::OK();
}


Status FtsColumnIndexer::InsertNull(uint32_t doc_id) {
  if (read_only_) {
    return Status::PermissionDenied("FTS indexer is read-only");
  }
  total_docs_.fetch_add(1, std::memory_order_relaxed);
  bump_max_id(doc_id);
  auto rs = ctx_.db_->Put(ctx_.write_opts_, cf_doc_lengths_,
                          EncodeDocIdKey(doc_id), EncodeUint32Le(0));
  if (!rs.ok()) {
    return Status::InternalError("Failed to write null doc length for ", ID(),
                                 ": ", rs.ToString());
  }
  return Status::OK();
}


// ===== read helpers (used by search) =====
// Implementation lives in fts_column_indexer_search.cc (see below).


}  // namespace zvec
