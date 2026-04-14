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


#include <cstdint>


namespace zvec {


/**
 * @brief Collection-level statistics needed for BM25 scoring.
 *
 * These are maintained by the FTS indexer and flushed to RocksDB. At query
 * time they are loaded once per search and passed into Bm25Scorer.
 */
struct Bm25CollectionStats {
  /// Total number of live documents in the collection.
  uint64_t total_docs{0};
  /// Total number of tokens across all live documents.
  uint64_t total_tokens{0};

  /// Average document length (in tokens). Returns 0 when the collection is
  /// empty; callers should treat that as "no docs to score" rather than
  /// dividing by it.
  double AvgDocLen() const {
    if (total_docs == 0) {
      return 0.0;
    }
    return static_cast<double>(total_tokens) /
           static_cast<double>(total_docs);
  }
};


/**
 * @brief Tunable BM25 parameters.
 *
 * k1 controls term-frequency saturation (higher = less saturation).
 * b  controls length normalization (0 = none, 1 = full).
 * The defaults match Lucene's.
 */
struct Bm25Params {
  float k1{1.2f};
  float b{0.75f};
};


/**
 * @brief Scores individual (term, doc) pairs using Okapi BM25.
 *
 * Formula (Lucene-style, with +1 inside the log to guarantee non-negative
 * IDF):
 *
 *     idf(q)        = log( (N - df + 0.5) / (df + 0.5) + 1 )
 *     norm(d)       = 1 - b + b * (|d| / avgdl)
 *     score(q, d)   = idf(q) * tf * (k1 + 1) / (tf + k1 * norm(d))
 *
 * Multi-term queries sum per-term scores.
 *
 * The scorer is pure: it stores no per-doc state. Callers typically
 * precompute Idf() once per query term and then call ScoreWithIdf() inside
 * their per-doc loop.
 */
class Bm25Scorer {
 public:
  Bm25Scorer(const Bm25Params &params, const Bm25CollectionStats &stats)
      : params_(params), stats_(stats) {}

  /// Score a single (term, doc) pair.
  float Score(uint32_t tf, uint32_t df, uint32_t doc_len) const;

  /// Just the IDF component — useful to hoist out of per-doc loops.
  float Idf(uint32_t df) const;

  /// Score given a precomputed IDF. Equivalent to Score() but avoids
  /// recomputing the log.
  float ScoreWithIdf(uint32_t tf, float idf, uint32_t doc_len) const;

  const Bm25Params &params() const {
    return params_;
  }
  const Bm25CollectionStats &stats() const {
    return stats_;
  }

 private:
  Bm25Params params_;
  Bm25CollectionStats stats_;
};


}  // namespace zvec
