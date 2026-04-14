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


#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <zvec/ailego/logger/logger.h>
#include "fts_column_indexer.h"


namespace zvec {


namespace {

// Same encoding as in fts_column_indexer.cc — kept private here so the two
// translation units do not require a shared header just for this.
inline std::string EncodeDocIdKey(uint32_t id) {
  std::string s(4, '\0');
  s[0] = static_cast<char>((id >> 24) & 0xFF);
  s[1] = static_cast<char>((id >> 16) & 0xFF);
  s[2] = static_cast<char>((id >> 8) & 0xFF);
  s[3] = static_cast<char>(id & 0xFF);
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

}  // namespace


Status FtsColumnIndexer::load_posting(const std::string &term,
                                      std::vector<FtsPosting> *out) const {
  out->clear();
  std::string value;
  auto rs = ctx_.db_->Get(ctx_.read_opts_, cf_terms_, term, &value);
  if (rs.IsNotFound()) {
    return Status::OK();
  }
  if (!rs.ok()) {
    return Status::InternalError("Failed to read posting for term '", term,
                                 "' from ", ID(), ": ", rs.ToString());
  }
  return FtsPostingCodec::Decode(value.data(), value.size(), out);
}


Status FtsColumnIndexer::load_doc_length(uint32_t doc_id,
                                         uint32_t *out) const {
  std::string value;
  auto rs = ctx_.db_->Get(ctx_.read_opts_, cf_doc_lengths_,
                          EncodeDocIdKey(doc_id), &value);
  if (rs.IsNotFound()) {
    *out = 0;
    return Status::OK();
  }
  if (!rs.ok()) {
    return Status::InternalError("Failed to read doc length for id ",
                                 std::to_string(doc_id), " from ", ID(), ": ",
                                 rs.ToString());
  }
  if (!DecodeUint32Le(value.data(), value.size(), out)) {
    return Status::InternalError("Corrupt doc length value for id ",
                                 std::to_string(doc_id), " in ", ID());
  }
  return Status::OK();
}


void FtsColumnIndexer::load_doc_lengths_batch(
    const std::vector<uint32_t> &doc_ids, std::vector<uint32_t> *out) const {
  out->assign(doc_ids.size(), 0);
  if (doc_ids.empty()) {
    return;
  }

  std::vector<std::string> keys;
  keys.reserve(doc_ids.size());
  for (uint32_t id : doc_ids) {
    keys.push_back(EncodeDocIdKey(id));
  }
  std::vector<rocksdb::Slice> key_slices;
  key_slices.reserve(keys.size());
  for (const auto &k : keys) {
    key_slices.emplace_back(k);
  }
  std::vector<rocksdb::ColumnFamilyHandle *> cfs(doc_ids.size(),
                                                 cf_doc_lengths_);
  std::vector<std::string> values(doc_ids.size());
  auto statuses = ctx_.db_->MultiGet(ctx_.read_opts_, cfs, key_slices, &values);
  for (std::size_t i = 0; i < doc_ids.size(); ++i) {
    if (statuses[i].ok()) {
      uint32_t v = 0;
      if (DecodeUint32Le(values[i].data(), values[i].size(), &v)) {
        (*out)[i] = v;
      }
    }
  }
}


Result<FtsSearchResult::Ptr> FtsColumnIndexer::Search(std::string_view query,
                                                      std::size_t topk,
                                                      MatchOp op) const {
  if (topk == 0) {
    return std::make_shared<FtsSearchResult>(std::vector<FtsScoredHit>{});
  }

  // 1. Tokenize and deduplicate query terms (preserving each unique form).
  std::vector<std::string> raw_terms;
  tokenizer_->Tokenize(query, &raw_terms);
  std::sort(raw_terms.begin(), raw_terms.end());
  raw_terms.erase(std::unique(raw_terms.begin(), raw_terms.end()),
                  raw_terms.end());
  if (raw_terms.empty()) {
    return std::make_shared<FtsSearchResult>(std::vector<FtsScoredHit>{});
  }

  // 2. Load each term's posting list and compute its IDF.
  Bm25CollectionStats stats = Stats();
  Bm25Scorer scorer(bm25_params_, stats);

  struct TermData {
    std::string term;
    std::vector<FtsPosting> postings;
    float idf{0.0f};
  };
  std::vector<TermData> term_data;
  term_data.reserve(raw_terms.size());
  for (auto &term : raw_terms) {
    TermData td;
    td.term = std::move(term);
    auto s = load_posting(td.term, &td.postings);
    if (!s.ok()) {
      return tl::make_unexpected(s);
    }
    if (td.postings.empty()) {
      // For AND semantics, a missing term means zero matches overall.
      if (op == MatchOp::AND) {
        return std::make_shared<FtsSearchResult>(std::vector<FtsScoredHit>{});
      }
      continue;  // OR: just contributes nothing
    }
    const auto df = static_cast<uint32_t>(td.postings.size());
    td.idf = scorer.Idf(df);
    term_data.push_back(std::move(td));
  }
  if (term_data.empty()) {
    return std::make_shared<FtsSearchResult>(std::vector<FtsScoredHit>{});
  }

  // 3. Build a candidate score map. For OR, every doc in any posting list is
  //    a candidate. For AND, the candidate set is the intersection of all
  //    posting lists.
  std::unordered_map<uint32_t, float> partial_scores;
  std::unordered_map<uint32_t, std::size_t> match_counts;
  partial_scores.reserve(term_data.front().postings.size() * 2);
  match_counts.reserve(partial_scores.bucket_count());
  for (const auto &td : term_data) {
    for (const auto &p : td.postings) {
      partial_scores[p.doc_id] += td.idf * static_cast<float>(p.tf);
      match_counts[p.doc_id]++;
    }
  }

  // 4. Filter by AND if needed, gather the candidate doc-id list.
  std::vector<uint32_t> candidates;
  candidates.reserve(partial_scores.size());
  if (op == MatchOp::AND) {
    const std::size_t required = term_data.size();
    for (const auto &[doc_id, _] : partial_scores) {
      if (match_counts[doc_id] == required) {
        candidates.push_back(doc_id);
      }
    }
  } else {
    for (const auto &[doc_id, _] : partial_scores) {
      candidates.push_back(doc_id);
    }
  }
  if (candidates.empty()) {
    return std::make_shared<FtsSearchResult>(std::vector<FtsScoredHit>{});
  }

  // 5. Fetch doc lengths in a single MultiGet, then compute final BM25.
  std::vector<uint32_t> doc_lens;
  load_doc_lengths_batch(candidates, &doc_lens);

  // We accumulated `idf * tf` for each (doc, term) pair into partial_scores.
  // BM25 is `idf * tf * (k1+1) / (tf + k1 * norm)`. We have `idf * tf`; we now
  // need to apply the BM25 saturation per (doc, term). Re-walk posting lists
  // with the doc length in hand to compute the proper score.
  std::unordered_map<uint32_t, std::size_t> id_to_idx;
  id_to_idx.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    id_to_idx[candidates[i]] = i;
  }

  std::vector<float> final_scores(candidates.size(), 0.0f);
  for (const auto &td : term_data) {
    for (const auto &p : td.postings) {
      auto it = id_to_idx.find(p.doc_id);
      if (it == id_to_idx.end()) {
        continue;  // filtered out by AND
      }
      const std::size_t idx = it->second;
      final_scores[idx] += scorer.ScoreWithIdf(p.tf, td.idf, doc_lens[idx]);
    }
  }

  // 6. Top-K selection. Use partial_sort on (doc_id, score) pairs.
  std::vector<FtsScoredHit> hits;
  hits.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    hits.push_back(FtsScoredHit{candidates[i], final_scores[i]});
  }
  const std::size_t k = std::min(topk, hits.size());
  std::partial_sort(
      hits.begin(), hits.begin() + k, hits.end(),
      [](const FtsScoredHit &a, const FtsScoredHit &b) {
        if (a.score != b.score) {
          return a.score > b.score;
        }
        return a.doc_id < b.doc_id;
      });
  hits.resize(k);
  return std::make_shared<FtsSearchResult>(std::move(hits));
}


}  // namespace zvec
